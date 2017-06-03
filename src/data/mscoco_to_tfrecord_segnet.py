import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
import random
import os
import cv2
import scipy.misc
import pandas as pd

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

# catIds = coco.getCatIds(catNms=nms)
# catIds = catIds[:32]

# catIds = []
catIdsMapping = {65: 1,
                59: 2,
                82: 3,
                6: 4,
                7: 5,
                62: 6,
                8: 7,
                3: 8,
                17: 9,
                63: 10,
                4: 11,
                22: 12,
                18: 13,
                61: 14,
                73: 15,
                54: 16}

catIds = list(catIdsMapping.keys())

imgIds = coco.getImgIds()

random.shuffle(imgIds)

train = []

writer = tf.python_io.TFRecordWriter(
    "%s/data.tfrecords" % processedDataDir)

images = coco.loadImgs(imgIds)

count = 0
c_freqs = np.zeros([16,2], dtype=np.int64)
tot_freqs = np.zeros([16,2], dtype=np.int64)

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

    largeCount = sum([1 if ann["bbox"][2] * scale * ann["bbox"][3] * scale > 50*50 else 0 for ann in anns]) #ignore images with only small annotations

    if len(anns) == 0 or target_width > 512 or target_height > 512 or largeCount == 0: # ignore images without boundingboxes or aspect ratios > 1:2 due to memory constraints
        continue

    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (target_width, target_height))
    img_data = img_data[...,::-1].copy()

    #anns = [ann for ann in anns if ann['iscrowd'] == 0 and ann["bbox"][3] * scale > 90.0 and ann["bbox"][2] * scale > 90.0] #Filter bboxes too small to get an IoU > 0.7

    annCatIds = [(catIdsMapping[ann["category_id"]]) for ann in anns]

    x = np.arange(0, target_width)
    y = np.arange(0, target_height)
    xv, yv = np.meshgrid(x, y)

    labels = [np.stack([yv, xv, np.round(scipy.misc.imresize(coco.annToMask(ann), scale, interp='nearest')) * (catIdsMapping[ann["category_id"]])], axis=2).reshape([-1, 3]) for ann in anns] 
    labels = np.concatenate(labels, axis=0)
    labels = labels[labels[..., 2] > 0]
    middle_labels = labels[(labels[..., 0] > target_height - image_size[1]) & (labels[..., 0] < image_size[1]) &  (labels[..., 1] > target_width - image_size[0]) & (labels[..., 1] < image_size[0])]

    if middle_labels.size < 3 * 2500:
        continue

    labels = pd.DataFrame(labels).drop_duplicates().values

    for i in range(16):
            f = np.sum(labels[..., 2] == i+1)

            c_freqs[i, 0] += target_height * target_width - f
            tot_freqs[i, 0] += target_height * target_width

            if f > 0:
                c_freqs[i, 1] += f
                tot_freqs[i, 1] += target_height * target_width

    labels[:, 2] -= 1
    labels = labels[labels[:,2].argsort()] # First sort doesn't need to be stable.
    labels = labels[labels[:,1].argsort(kind='mergesort')]
    labels = labels[labels[:,0].argsort(kind='mergesort')]
    labels = labels.astype(np.int16)

    size = labels.flatten().size

    #scipy.misc.imsave("%s/test.png"%processedDataDir, labels)

    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[labels.flatten().tostring()])),
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


f = c_freqs / (tot_freqs + 1e-10)
m = np.repeat(np.expand_dims(np.median(f, axis=1), axis=1), 2, axis=1)

print((m/f).transpose())

writer.close()