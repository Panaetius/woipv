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

        tf.summary.image('images', images, max_outputs=16)

        return images, category_batch, bbox_batch
