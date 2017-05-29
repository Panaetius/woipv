import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
import random
import os
import cv2
import scipy.misc
import pandas as pd
from collections import defaultdict

image_size = (288.0, 288.0)
min_bbox_size = 20

dataDir = "%s/../../data/raw/MSCOCO" % os.path.dirname(os.path.realpath(
    __file__))
processedDataDir = "%s/../../data/processed/MSCOCO" % os.path.dirname(
    os.path.realpath(
        __file__))
dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

coco = COCO(annFile)

categories = coco.loadCats(coco.getCatIds())
nms = [(cat['id'], cat['name']) for cat in categories]

catIds = coco.getCatIds(catNms=nms)

imgIds = coco.getImgIds()

images = coco.loadImgs(imgIds)

count = 0
cat_count = defaultdict(int)
pixel_count = defaultdict(int)

for img in images:
    path = '%s/%s/%s' % (dataDir, dataType, img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)

    cats = set()
    pixels = img['width'] * img['height']
    for ann in anns:
        mask = coco.annToMask(ann)
        unique, counts = np.unique(mask, return_counts=True)
        if unique.size < 2:
            continue
        cats.add(ann["category_id"])
        counts = dict(zip(unique, counts))
        pixel_count[ann["category_id"]] += counts[1]
        pixels -= counts[1]

    pixel_count[0] += pixels

    for c in cats:
        cat_count[c] += 1

    count += 1

    if count % 1000 == 0:
        print("Processed %s entries" % count)

print(cat_count)
print(pixel_count)
print(nms)
