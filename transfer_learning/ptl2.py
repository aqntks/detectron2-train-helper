import os
import copy
import torch
import numpy as np
from pathlib import Path
import pycocotools.mask as mask_util

from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import *
from detectron2.evaluation.evaluator import DatasetEvaluator

setup_logger()


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


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
                        T.RandomCrop(crop_type='relative_range', crop_size=[0.5, 0.5]),
                        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
                        T.RandomFlip(),
                        T.RandomApply(T.RandomRotation((-10, 10)), prob=0.2),
                        T.RandomApply(T.RandomSaturation(0.8, 1.2), prob=0.2),
                        T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.2),
                        T.RandomApply(T.RandomContrast(0.6, 1.3), prob=0.2),
                        T.RandomApply(T.RandomLighting(0.7), prob=0.2),
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")
    dataset_dict["instances"] = filter_empty_instances(instances)
    return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def setup():
    dataDir = Path('./')
    model_name = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    register_coco_instances('sartorius_train', {}, 'annotations_all.json', dataDir)
    register_coco_instances('sartorius_val', {}, 'annotations_val.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'transfer/tl2/model_0101135.pth'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    return cfg


def main():
    cfg = setup()

    cfg.OUTPUT_DIR = "./ptl/ptl2"
    cfg.MODEL.DEVICE = "cuda:0"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
