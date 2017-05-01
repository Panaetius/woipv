import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
import random
import os
import cv2

image_size = (600.0, 600.0)
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
nms = [cat['name'] for cat in categories]

catIds = coco.getCatIds(catNms=nms)

imgIds = coco.getImgIds()

random.shuffle(imgIds)

train = []

writer = tf.python_io.TFRecordWriter(
    "%s/data.tfrecords" % processedDataDir)

images = coco.loadImgs(imgIds)

count = 0

for img in images:
    path = '%s/%s/%s' % (dataDir, dataType, img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    original_width = img['width']
    original_height = img['height']
    if original_width > original_height:
        scale = image_size[1]/ original_height
    else:
        scale = image_size[0]/ original_width

    target_width = int(original_width * scale)
    target_height = int(original_height * scale)

    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (target_width, target_height))
    img_data = img_data[...,::-1].copy()

    anns = [ann for ann in anns if ann['iscrowd'] == 0]

    if len(anns) == 0 or target_width > 1800 or target_height > 1800: # ignore images without boundingboxes or aspect ratios > 1:3 due to memory constraints
        continue

    annCatIds = [ann["category_id"] - 1 for ann in anns]

    annBBoxes = [np.multiply(np.asarray(ann["bbox"]), [scale, scale,
                                                      scale, scale])
                          for ann in anns]

    annBBoxes = [np.asarray([target_height - (ann[1] + ann[3]/2.0), ann[0] + ann[2]/2.0,
                             ann[3], ann[2]]).tolist()
                 for ann in
                 annBBoxes if ann[3] > 64.0 and ann[2] > 64.0]

    if len(annBBoxes) == 0:
        continue

    annBBoxes = np.asarray(annBBoxes).flatten().tolist()

    assert len(annBBoxes) % 4 == 0, "Bboxes not a multiple of 4"
    assert all(coord >= 0.0 for coord in annBBoxes), "Bboxes negative"

    example = tf.train.Example(features=tf.train.Features(feature={
        'categories': tf.train.Feature(int64_list=tf.train.Int64List(
            value=annCatIds)),
        'bboxes': tf.train.Feature(float_list=tf.train.FloatList(
            value=annBBoxes)),  # flatten the list so tf eats it
        'image_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[img['id']])),
        'image_raw': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[img_data.flatten().tostring()])),
        'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[target_width])),
        'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[target_height]))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()

    writer.write(serialized)

    count += 1

    if count % 1000 == 0:
        print("Processed %s entries" % count)

writer.close()
