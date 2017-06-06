import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import scipy.misc
import tensorflow as tf

root_dir = '%s/../../data/raw/VOC2012/'%os.path.dirname(
    os.path.realpath(
        __file__))
img_dir = os.path.join(root_dir, 'JPEGImages/')
ann_dir = os.path.join(root_dir, 'SegmentationClass')
set_dir = os.path.join(root_dir, 'ImageSets', 'Segmentation')

image_size = (288.0, 288.0)

palette = {(128,   0,   0) : 1 ,
            (  0, 128,   0) : 2 ,
            (128, 128,   0) : 3 ,
            (  0,   0, 128) : 4 ,
            (128,   0, 128) : 5 ,
            (  0, 128, 128) : 6 ,
            (128, 128, 128) : 7 ,
            ( 64,   0,   0) : 8 ,
            (192,   0,   0) : 9 ,
            ( 64, 128,   0) : 10,
            (192, 128,   0) : 11,
            ( 64,   0, 128) : 12,
            (192,   0, 128) : 13,
            ( 64, 128, 128) : 14,
            (192, 128, 128) : 15,
            (  0,  64,   0) : 16,
            (128,  64,   0) : 17,
            (  0, 192,   0) : 18,
            (128, 192,   0) : 19,
            (  0,  64, 128) : 20,
            (  224,  224, 192) : 21  }


def convert_from_color_segmentation(arr_3d):
    result = np.ndarray(shape=arr_3d.shape[:2], dtype=np.uint8)
    result[:,:] = 0
    for rgb, idx in palette.items():
        result[(arr_3d==rgb).all(2)] = idx
    return result

annotation = os.path.join(set_dir, "trainval.txt")

processedDataDir = "%s/../../data/processed/pascal_voc" % os.path.dirname(
    os.path.realpath(
        __file__))

writer = tf.python_io.TFRecordWriter(
    "%s/data.tfrecords" % processedDataDir)

count = 0

num_classes = 20
c_freqs = np.zeros([num_classes+1, 2], dtype=np.int64)
tot_freqs = np.zeros([num_classes+1, 2], dtype=np.int64)

with open(annotation, 'r') as f:
    for img_name in f:
        img_base_name = img_name.strip()
        img_name = os.path.join(img_dir, img_base_name + ".jpg")
        img = imread(img_name)

        segname = os.path.join(ann_dir, img_base_name + ".png")
        seg = imread(segname)
        seg = convert_from_color_segmentation(seg)

        original_width = img.shape[1]
        original_height = img.shape[0]

        if original_width > original_height:
            scale = image_size[1]/ original_height
        else:
            scale = image_size[0]/ original_width

        target_width = int(original_width * scale)
        target_height = int(original_height * scale)

        img = scipy.misc.imresize(img, (target_height, target_width), interp='bilinear')
        seg = scipy.misc.imresize(seg, (target_height, target_width), interp='nearest')

        x = np.arange(0, target_width)
        y = np.arange(0, target_height)
        xx, yy = np.meshgrid(x, y)

        seg = np.stack([yy, xx, seg], axis=2).reshape([-1, 3])

        labels = seg[(seg[..., 2] > 0) & (seg[..., 2] <= num_classes + 1)]
        middle_labels = labels[(labels[..., 0] > target_height - image_size[1]) & (labels[..., 0] < image_size[1]) &  (labels[..., 1] > target_width - image_size[0]) & (labels[..., 1] < image_size[0]) & (labels[..., 2] <= num_classes)]

        if middle_labels.size < 3 * 2500:
            continue

        if labels.size == 0:
            continue

        total_pixels = target_height * target_width

        c_freqs[0, 1] += target_height * target_width - labels[..., 2].size
        c_freqs[0, 0] += labels[..., 2].size
        tot_freqs[0, 0] += total_pixels
        tot_freqs[0, 1] += total_pixels

        for i in range(num_classes):
                f = np.sum(labels[..., 2] == i+1)
                tot_freqs[i+1, 0] += total_pixels
                tot_freqs[i+1, 1] += total_pixels

                if f > 0:
                    c_freqs[i+1, 1] += f
                    c_freqs[i+1, 0] += total_pixels - f

        labels[:, 2] -= 1
        labels = labels[labels[:,2].argsort()] # First sort doesn't need to be stable.
        labels = labels[labels[:,1].argsort(kind='mergesort')]
        labels = labels[labels[:,0].argsort(kind='mergesort')]
        labels = labels.astype(np.int16)

        example = tf.train.Example(features=tf.train.Features(feature={
            'labels': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[labels.flatten().tostring()])),
            'image_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[img.flatten().tostring()])),
            'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[target_width])),
            'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[target_height]))}))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()

        writer.write(serialized)

        count += 1

        if count %100 == 0:
            break

        if count % 1000 == 0:
            print("Processed %s entries" % count)

f = c_freqs / (tot_freqs + 1e-10)
m = np.repeat(np.expand_dims(np.median(f, axis=1), axis=1), 2, axis=1)

print((m/f).transpose())

writer.close()