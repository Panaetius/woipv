import tensorflow as tf

class WoipvModel(object):

    def __init__(self, is_training, config):
        self.istraining = config.istraining
        self.num_classes = config.num_classes

    def __reslayer(self, input, in_filters, out_filters, stride=1):
        """ A regular resnet block """
        with tf.variable_scope('sub1'):
            kernel = tf.get_variable('weights', [3, 3, in_filters, out_filters],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights', [3, 3, out_filters, out_filters],
                                      initializer=tf.contrib.layers.xavier_initializer(
                                          dtype=tf.float32),
                                      dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                 name='conv1')
            bias = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])

        with tf.variable_scope('subadd'):
            if in_filters != out_filters:
                input = tf.nn.avg_pool(input, stride, stride, 'VALID')
                input = tf.pad(input, [[0, 0], [0, 0], [0, 0],
                             [(out_filters - in_filters) // 2,
                              (out_filters - in_filters) // 2]])
            bias += input
            conv = tf.nn.elu(bias, 'elu')

        return conv

    def __batch_norm_wrapper(self, inputs, decay=0.999, shape=[0]):
        """ Batch Normalization """
        epsilon = 1e-3
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              trainable=False)

        if self.istraining:
            batch_mean, batch_var = tf.nn.moments(inputs, shape, name="moments")
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta,
                                                 scale,
                                                 epsilon, name="batch_norm")
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale,
                                             epsilon, name="batch_norm")

    def inference(self, input):
        with tf.variable_scope('first'):
            kernel = tf.get_variable('weights', [7, 7, 3, 64],
                                     initializer=tf.contrib.layers.xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input, kernel, [1, 2, 2, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')
            input = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool')

        for i in range(3):
            with tf.variable_scope('reslayer_64_%d' % i):
                input = self.__reslayer(input, 64, 64)

        with tf.variable_scope('reslayer_downsample_128'):
            input = self.__reslayer(input, 64, 128, stride=2)

        for i in range(3):
            with tf.variable_scope('reslayer_128_%d' % i):
                input = self.__reslayer(input, 128, 128)

        with tf.variable_scope('reslayer_downsample_256'):
            input = self.__reslayer(input, 128, 256, stride=2)

        for i in range(5):
            with tf.variable_scope('reslayer_256_%d' % i):
                input = self.__reslayer(input, 256, 256)

        with tf.variable_scope('reslayer_downsample_512'):
            input = self.__reslayer(input, 256, 512, stride=2)

        for i in range(2):
            with tf.variable_scope('reslayer_512_%d' % i):
                input = self.__reslayer(input, 512, 512)

        with tf.variable_scope('global_average_pool'):
            input = tf.reduce_mean(input, [1, 2])

        with tf.variable_scope('softmax_linear'):
            weights = tf.get_variable('weights', [512, self.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            softmax_linear = self.__batch_norm_wrapper(tf.matmul(input,
                                                                weights))

        return softmax_linear

        pass

    def loss(self, logits):
        pass