import tensorflow as tf


class MSCOCOInputProducer(object):
    def __init__(self, config):
        self.path = config.path
        self.batch_size = config.batch_size
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
                'bboxes': tf.VarLenFeature(tf.float32),
                'image_raw': tf.FixedLenFeature([1], tf.string),
                'image_id': tf.FixedLenFeature([1], tf.int64)
            })

        result.categories = features['categories']
        result.bboxes = features['bboxes']
        result.image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
        result.image_id = features['image_id']

        return result

    def __put_bboxes_on_image(self, images, boxes, scale):
        images = tf.split(images, self.batch_size, axis=0)
        boxes = tf.sparse_split(sp_input=boxes,
                                        num_split=self.batch_size, axis=0)

        output = []

        for i in range(self.batch_size):
            bboxes1 = boxes[i]
            bboxes1 = tf.sparse_tensor_to_dense(bboxes1, default_value=-1)
            mask = bboxes1 >= 0
            bboxes1 = tf.boolean_mask(bboxes1, mask)
            bboxes = tf.reshape(bboxes1, [1, -1, 4])

            bboxes = bboxes * scale
            shape = tf.shape(bboxes)
            bboxes = self.__clip_bboxes(tf.reshape(bboxes, [-1, 4]), 1.0, 1.0)
            x, y, w, h = tf.split(bboxes, 4, axis=1)
            bboxes = tf.concat([y - h / 2.0 - 0.001, x - w / 2.0 - 0.001,
                                y + h / 2.0 + 0.001,
                                x + w / 2.0 + 0.001],
                               axis=1)
            bboxes = tf.reshape(bboxes, shape)
            bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

            image = tf.cond(tf.size(bboxes1) > 0,
                            lambda: tf.image.draw_bounding_boxes(images[i],
                                                              bboxes),
                            lambda: images[i])

            output.append(image)

        return tf.concat(output, axis=0)

    def __clip_bboxes(self, bboxes, max_width, max_height):
        """ Clips a list of bounding boxes (m,4) to be within an image
        region"""
        with tf.variable_scope('clip_bboxes'):
            x, y, w, h = tf.split(bboxes, 4, axis=1)
            minx = tf.minimum(x - w / 2.0, 0)
            maxx = tf.maximum(x + w / 2.0, max_width) - max_width
            miny = tf.minimum(y - h / 2.0, 0)
            maxy = tf.maximum(y + h / 2.0, max_height) - max_height

            width_delta = minx - maxx
            x_delta = -minx / 2.0 - maxx / 2.0
            height_delta = miny - maxy
            y_delta = -miny / 2.0 - maxy / 2.0

            delta = tf.concat([x_delta, y_delta, width_delta, height_delta],
                              axis=1)

            return bboxes + delta

    def inputs(self):
        filename_queue = tf.train.string_input_producer(
            [self.path + self.tfrecords_filename])

        result = self.__read(filename_queue)

        distorted_image = tf.cast(result.image_raw, tf.float32)
        distorted_image = tf.reshape(distorted_image,
                                     [self.width, self.height, 3])

        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=35)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.4, upper=1.4)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.01)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.03
        min_queue_examples = int(self.num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        print('Filling queue with %d images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        images, category_batch, bbox_batch = tf.train.shuffle_batch(
            [float_image, result.categories, result.bboxes],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=min_queue_examples)

        preview_images = self.__put_bboxes_on_image(images, bbox_batch,
                                                    scale=1.0/600)
        tf.summary.image('images', preview_images, max_outputs=16)

        return images, category_batch, bbox_batch
