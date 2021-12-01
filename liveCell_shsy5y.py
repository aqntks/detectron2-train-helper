import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path
from pycocotools import _mask
from pycocotools.coco import COCO
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode

# Load LiveCell shsy5y data
live_cell_imgs_dir = Path('../input/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/images/livecell_train_val_images/SHSY5Y')

register_coco_instances('sartorius_live_cell_train',{}, '../input/sartorius-live-cell-shsy5y-coco/annotations_train.json', live_cell_imgs_dir)
register_coco_instances('sartorius_live_cell_val',{},'../input/sartorius-live-cell-shsy5y-coco/annotations_val.json', live_cell_imgs_dir)

live_cell_train_meta = MetadataCatalog.get('sartorius_live_cell_train')
live_cell_train_ds = DatasetCatalog.get('sartorius_live_cell_train')

live_cell_val_meta = MetadataCatalog.get('sartorius_live_cell_val')
live_cell_val_ds = DatasetCatalog.get('sartorius_live_cell_val')


# Read LiveCell shsy5y train and val data
with open('../input/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_train.json') as f:
  data_train = json.loads(f.read())

with open('../input/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_val.json') as f:
  data_val = json.loads(f.read())


# Add categories
categories = [{'name':'shsy5y', 'id':1}, {'name':'astro', 'id':2}, {'name':'cort', 'id':3}]

data_train['categories'] = categories
data_val['categories'] = categories

# Format to COCO RLE format
train_annotations = []
for key in data_train['annotations'].keys():
  rle = _mask.frPoly(data_train['annotations'][key]['segmentation'],520,704)
  data_train['annotations'][key]['segmentation'] = {'size': rle[0]['size'], 'counts':rle[0]['counts'].decode('utf-8')}
  train_annotations.append(data_train['annotations'][key])
data_train['annotations'] = train_annotations

val_annotations = []
for key in data_val['annotations'].keys():
  rle = _mask.frPoly(data_val['annotations'][key]['segmentation'],520,704)
  data_val['annotations'][key]['segmentation'] = {'size': rle[0]['size'], 'counts': rle[0]['counts'].decode('utf-8')}
  val_annotations.append(data_val['annotations'][key])
data_val['annotations'] = val_annotations

# Save formatted data to JSON
with open('annotations_train.json', 'w', encoding='utf-8') as f:
  json.dump(data_train, f, ensure_ascii=True, indent=4)

with open('annotations_val.json', 'w', encoding='utf-8') as f:
  json.dump(data_val, f, ensure_ascii=True, indent=4)