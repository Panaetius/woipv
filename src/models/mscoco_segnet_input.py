import tensorflow as tf


class MSCOCOSegnetInputProducer(object):
    def __init__(self, config):
        self.path = config.path
        self.tfrecords_filename = 'data.tfrecords'
        self.num_examples_per_epoch = config.num_examples_per_epoch
        self.width = config.width
        self.height = config.height
        self.image_size = self.width * self.height * 3
        self.num_preprocess_threads = 16
        self.num_classes = config.num_classes

    def __read(self, filename_queue):
        class CocoRecord(object):
            image_raw = []
            bboxes = []
            categories = []
            image_id = -1
            pass

        result = CocoRecord()

        reader = tf.TFRecordReader()

        _, value = reader.read(filename_queue)

        features = tf.parse_single_example(
            value,
            features={
                'categories': tf.VarLenFeature(tf.int64),
                'labels': tf.FixedLenFeature([1], tf.string),
                'image_id': tf.FixedLenFeature([1], tf.int64),
                'image_raw': tf.FixedLenFeature([1], tf.string),
                'image_width': tf.FixedLenFeature([1], tf.int64),
                'image_height': tf.FixedLenFeature([1], tf.int64)
            })

        result.categories = features['categories']
        result.labels = tf.decode_raw(features['labels'], tf.uint8)
        result.image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
        result.image_id = features['image_id']
        result.width = features['image_width']
        result.height = features['image_height']

        return result

    def __put_bboxes_on_image(self, images, boxes, scale_x, scale_y):

        output = []

        polys1 = tf.sparse_tensor_to_dense(polys, default_value=-1)
        mask = bboxes1 >= 0
        bboxes1 = tf.boolean_mask(bboxes1, mask)
        bboxes = tf.reshape(bboxes1, [1, -1, 4])

        bboxes = bboxes * [[scale_y, scale_x, scale_y, scale_x]]
        shape = tf.shape(bboxes)
        bboxes = self.__clip_bboxes(tf.reshape(bboxes, [-1, 4]), 1.0, 1.0)
        y, x, h, w = tf.split(bboxes, 4, axis=1)
        bboxes = tf.concat([1.0 - (y + h / 2.0) - 0.001, x - w / 2.0 - 0.001,
                            1.0 - (y - h / 2.0) + 0.001,
                            x + w / 2.0 + 0.001],
                            axis=1)
        bboxes = tf.reshape(bboxes, shape)
        bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

        image = tf.cond(tf.size(bboxes1) > 0,
                        lambda: tf.image.draw_bounding_boxes(images,
                                                            bboxes),
                        lambda: images)

        return image

    def __clip_bboxes(self, bboxes, max_width, max_height):
        """ Clips a list of bounding boxes (m,4) to be within an image
        region"""
        with tf.variable_scope('clip_bboxes'):
            max_width = tf.cast(max_width, tf.float32)
            max_height = tf.cast(max_height, tf.float32)

            y, x, h, w = tf.split(bboxes, 4, axis=1)

            minx = tf.minimum(tf.maximum(x - w / 2.0, 0.0), max_width)
            maxx = tf.minimum(tf.maximum(x + w / 2.0, 0.0), max_width)
            miny = tf.minimum(tf.maximum(y - h / 2.0, 0.0), max_height)
            maxy = tf.minimum(tf.maximum(y + h / 2.0, 0.0), max_height)

            width = maxx - minx + 1e-10
            x = (minx + maxx) / 2.0
            height = maxy - miny + 1e-10
            y = (miny + maxy) / 2.0

            bboxes = tf.concat([y, x, height, width],
                               axis=1)

            return bboxes

    def inputs(self):
        filename_queue = tf.train.string_input_producer(
            [self.path + self.tfrecords_filename])

        result = self.__read(filename_queue)

        labels_shape = tf.cast(tf.concat([result.height, result.width], 0), tf.int32)
        labels = tf.reshape(result.labels, labels_shape)

        distorted_image = result.image_raw

        distorted_image = tf.cast(distorted_image, tf.float32)

        target_shape = tf.cast(tf.concat([result.height, result.width, [3]], 0), tf.int32)
        distorted_image = tf.reshape(distorted_image, target_shape)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        #distorted_image = tf.image.random_brightness(distorted_image,
        #                                             max_delta=35)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.4, upper=1.4)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.03)

        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_standardization(distorted_image)

        # Ensure that the random shuffling has good mixing properties.

        distorted_image = tf.expand_dims(distorted_image, axis=0)

        # preview_images = self.__put_bboxes_on_image(distorted_image, result.bboxes,
        #                                             scale_x=1.0/tf.cast(result.width[0], tf.float32), scale_y=1.0/tf.cast(result.height[0], tf.float32))
        preview_labels = tf.cast(labels, tf.float32)
        preview_labels = preview_labels / self.num_classes

        tf.summary.image('images', distorted_image, max_outputs=16)
        tf.summary.image('labels', tf.expand_dims(tf.expand_dims(preview_labels, 2), 0), max_outputs=16)

        return distorted_image, result.categories, labels
