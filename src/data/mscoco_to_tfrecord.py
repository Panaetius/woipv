import tensorflow as tf
from pycocotools.coco import COCO
import random

image_size = (600,600)

dataDir=os.path.dirname(os.path.realpath(__file__)) + "../../data/raw/MSCOCO"
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

coco=COCO(annFile)

categories = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in categories]

catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds )

random.shuffle(imgIds)

train = []

writer = tf.python_io.TFRecordWriter("%s/processed/data.tfrecords"%dataDir)

for img in imgIds:
    path = '%s/images/%s/%s'%(dataDir,dataType,img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (image_size[0], image_size[1]))
    
    annCatIds = [ann["category_id"] for ann in anns]
    annBBoxes = [ann["bbox"] for ann in anns]

    example = tf.train.Example(features=tf.train.Features(feature={
        'categories': tf.train.Feature(int64_list=tf.train.Int64List(
            value=annCatIds)),
        'bboxes': tf.train.Feature(int64_list=tf.train.Int64List(
            value=annBBoxes)),
        'image_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img_data))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    
    writer.write(serialized)
    
writer.close()