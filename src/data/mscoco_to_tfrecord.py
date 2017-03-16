import tensorflow as tf
from pycocotools.coco import COCO
import random
import os
import cv2

image_size = (600, 600)

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

    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (image_size[0], image_size[1]))

    annCatIds = [ann["category_id"] for ann in anns]
    annBBoxes = [ann["bbox"] for ann in anns]

    example = tf.train.Example(features=tf.train.Features(feature={
        'categories': tf.train.Feature(int64_list=tf.train.Int64List(
            value=annCatIds)),
        'bboxes': tf.train.Feature(float_list=tf.train.FloatList(
            value=sum(annBBoxes, []))),  # flatten the list so tf eats it
        'image_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[img['id']])),
        'image_raw': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[img_data.flatten().tostring()]))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()

    writer.write(serialized)

    count += 1

    if count % 1000 == 0:
        print("Processed %s entries" % count)

writer.close()
