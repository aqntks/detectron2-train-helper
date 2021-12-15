import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import polygons_to_bitmask
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
setup_logger()


def polygon_to_rle(polygon, shape=(520, 704)):
    # print(polygon)
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1])

    rle = mask_util.encode(np.asfortranarray(mask))
    return rle


# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)


def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x: x['segmentation'], targ))
    enc_targs = [polygon_to_rle(enc_targ[0]) for enc_targ in enc_targs]
    ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']: item['annotations'] for item in dataset_dicts}

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)


def setup():
    dataDir = Path('LIVECell_dataset_2021/images')
    model_name = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    register_coco_instances('sartorius_train', {}, 'livecell_annotations_train.json', dataDir)
    register_coco_instances('sartorius_val', {}, 'livecell_annotations_val.json', dataDir)
    register_coco_instances('sartorius_test', {}, 'livecell_annotations_test.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("sartorius_train", "sartorius_test")
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 300000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(
        DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.TEST.EVAL_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(
        DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    return cfg


def main():
    cfg = setup()

    cfg.OUTPUT_DIR = "./transfer/tl2"
    cfg.MODEL.DEVICE = "cuda:2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()