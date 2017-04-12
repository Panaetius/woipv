import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import xavier_initializer
from roi_pooling import roi_pooling

from enum import Enum

class NetworkType(Enum):
    RESNET34 = 1
    RESNET50 = 2


class WoipvModel(object):
    def __init__(self, config):
        self.is_training = config.is_training
        self.num_classes = config.num_classes
        self.num_examples_per_epoch = config.num_examples_per_epoch
        self.num_epochs_per_decay = config.num_epochs_per_decay
        self.initial_learning_rate = config.initial_learning_rate
        self.learning_rate_decay_factor = config.learning_rate_decay_factor
        self.batch_size = config.batch_size
        self.adam_epsilon = 0.0001
        self.moving_average_decay = 0.9999
        self.width = config.width
        self.height = config.height
        self.min_box_size = config.min_box_size
        self.rcnn_cls_loss_weight = config.rcnn_cls_loss_weight
        self.rcnn_reg_loss_weight = config.rcnn_reg_loss_weight
        self.rpn_cls_loss_weight = config.rpn_cls_loss_weight
        self.rpn_reg_loss_weight = config.rpn_reg_loss_weight
        self.background_weight = config.background_weight
        self.dropout_prob = config.dropout_prob
        self.weight_decay = config.weight_decay
        self.net = config.net

        if self.net == NetworkType.RESNET34:
            self.conv_feature_count = 512
        elif self.net == NetworkType.RESNET50:
            self.conv_feature_count = 2048

    def __reslayer(self, inputs, in_filters, out_filters, stride=1):
        """ A regular resnet block """
        with tf.variable_scope('sub1'):
            kernel = tf.get_variable('weights', [3, 3, in_filters, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights',
                                     [3, 3, out_filters, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv1')
            bias = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])

        with tf.variable_scope('subadd'):
            if in_filters != out_filters:
                # inputs = tf.nn.avg_pool(inputs, (1, stride, stride, 1),
                #                         (1, stride, stride, 1), 'SAME')
                # inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                #                          [(out_filters - in_filters) // 2,
                #                           (out_filters - in_filters) // 2]])
                kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
                inputs = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
            bias += inputs
            conv = tf.nn.elu(bias, 'elu')

            num = np.power(2, np.floor(np.log2(out_filters) / 2))

            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(out_filters /
                                                             num)))
            tf.summary.image('sub2/activations', grid, max_outputs=1)

        return conv

    def __reslayer_bottleneck(self, inputs, in_filters, out_filters, stride=1):
        """ A regular resnet block """
        with tf.variable_scope('sub1'):
            kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters/4],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights',
                                     [3, 3, out_filters/4, out_filters/4],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv1')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub3'):
            kernel = tf.get_variable('weights', [1, 1, out_filters/4, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])

        with tf.variable_scope('subadd'):
            if in_filters != out_filters:
                # inputs = tf.nn.avg_pool(inputs, (1, stride, stride, 1),
                #                         (1, stride, stride, 1), 'SAME')
                # inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                #                          [(out_filters - in_filters) // 2,
                #                           (out_filters - in_filters) // 2]])
                kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
                inputs = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
            batch_norm += inputs
            conv = tf.nn.elu(batch_norm, 'elu')

            num = np.power(2, np.floor(np.log2(out_filters) / 2))

            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(out_filters /
                                                             num)))
            tf.summary.image('sub3/activations', grid, max_outputs=1)

        return conv


    def __batch_norm_wrapper(self, inputs, decay=0.999, shape=None):
        """ Batch Normalization """
        if shape is None:
            shape = [0]

        epsilon = 1e-3
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              trainable=False)

        if self.is_training:
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



    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in ip5wke model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
          total_loss: Total loss from loss().
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """

        # Compute the moving average of all individual losses and the total loss
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        accuracies = tf.get_collection('accuracies')
        for a in accuracies:
            tf.summary.scalar('accuracy', a)

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of
            # the loss as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def __put_kernels_on_grid(self, kernel, grid, pad=1):
        """Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):shape of the grid. Require: NumKernels == grid_Y * grid_X
                            User is responsible of how to break into two multiples.
          pad:              number of black pixels around each filter (between them)

        Return:
          Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
        """
        grid_Y, grid_X = grid
        # pad X and Y
        x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + pad
        X = kernel.get_shape()[1] + pad

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, 3]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, 3]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 1]
        # x_min = tf.reduce_min(x7)
        # x_max = tf.reduce_max(x7)
        # x8 = (x7 - x_min) / (x_max - x_min)

        return x7

    def __put_activations_on_grid(self, activations, grid, pad=1):
        """Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):shape of the grid. Require: NumKernels == grid_Y * grid_X
                            User is responsible of how to break into two multiples.
          pad:              number of black pixels around each filter (between them)

        Return:
          Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
        """
        grid_Y, grid_X = grid
        # get first image in batch to make things simpler
        activ = activations[0, :]
        # greyscale
        activ = tf.expand_dims(activ, 2)
        # pad X and Y
        x1 = tf.pad(activ, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

        # X and Y dimensions, w.r.t. padding
        Y = activ.get_shape()[0] + pad
        X = activ.get_shape()[1] + pad

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, 1]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, 1]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 255.0]
        x_min = tf.reduce_min(x7)
        x_max = tf.reduce_max(x7)
        x7 = 255.0 * (x7 - x_min) / (x_max - x_min)

        # return x8
        return x7



    def inference(self, inputs):
        # resnet
        with tf.variable_scope('first_layer'):
            kernel = tf.get_variable('weights', [7, 7, 3, 64],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

            grid = self.__put_kernels_on_grid(kernel, (8, 8))
            tf.summary.image('conv1/features', grid, max_outputs=1)
            grid = self.__put_activations_on_grid(conv, (8, 8))
            tf.summary.image('conv1/activations', grid, max_outputs=1)

            inputs = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME',
                                    name='pool')

        if self.net == NetworkType.RESNET34:
            inputs = self.__resnet34(inputs)
        elif self.net == NetworkType.RESNET50:
            inputs = self.__resnet50(inputs)

        # classify regions and add final region adjustments
        with tf.variable_scope('fully_connected'):
            fc = tf.reduce_mean(inputs, [1, 2])
            class_weights = tf.get_variable('class_weights',
                                            [self.conv_feature_count,
                                             self.num_classes],
                                            initializer=xavier_initializer(
                                                dtype=tf.float32),
                                            dtype=tf.float32)
            class_bias = tf.get_variable("class_bias", [
                self.num_classes],
                initializer=tf.constant_initializer(
                0.1),
                dtype=tf.float32)

            class_score = tf.matmul(fc, class_weights)
            class_score = tf.nn.bias_add(class_score, class_bias)
            

        return class_score

    def __resnet34(self, inputs):
        for i in range(3):
            with tf.variable_scope('reslayer_64_%d' % i):
                inputs = self.__reslayer(inputs, 64, 64)

        with tf.variable_scope('reslayer_downsample_128'):
            inputs = self.__reslayer(inputs, 64, 128, stride=2)

        for i in range(3):
            with tf.variable_scope('reslayer_128_%d' % i):
                inputs = self.__reslayer(inputs, 128, 128)

        with tf.variable_scope('reslayer_downsample_256'):
            inputs = self.__reslayer(inputs, 128, 256, stride=2)

        for i in range(5):
            with tf.variable_scope('reslayer_256_%d' % i):
                inputs = self.__reslayer(inputs, 256, 256)

        with tf.variable_scope('reslayer_downsample_512'):
            inputs = self.__reslayer(inputs, 256, 512, stride=2)

        for i in range(2):
            with tf.variable_scope('reslayer_512_%d' % i):
                inputs = self.__reslayer(inputs, 512, 512)
        return inputs


    def __resnet50(self, inputs):
        with tf.variable_scope('reslayer_downsample_256'):
                inputs = self.__reslayer_bottleneck(inputs, 64, 256, stride=1)

        for i in range(2):
            with tf.variable_scope('reslayer_256_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 256, 256)

        with tf.variable_scope('reslayer_downsample_512'):
            inputs = self.__reslayer_bottleneck(inputs, 256, 512, stride=2)

        for i in range(3):
            with tf.variable_scope('reslayer_512_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 512, 512)

        with tf.variable_scope('reslayer_downsample_1024'):
            inputs = self.__reslayer_bottleneck(inputs, 512, 1024, stride=2)

        for i in range(5):
            with tf.variable_scope('reslayer_1024_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 1024, 1024)

        with tf.variable_scope('reslayer_downsample_2048'):
            inputs = self.__reslayer_bottleneck(inputs, 1024, 2048, stride=2)

        for i in range(2):
            with tf.variable_scope('reslayer_2048_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 2048, 2048)
        return inputs


    def loss(self, class_scores, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.one_hot(labels, self.num_classes),
            logits=class_scores)
        loss = tf.reduce_mean(loss, name='cross_entropy')

        tf.add_to_collection('losses', tf.identity(loss,
                                                   name="loss"))

        labels = tf.cast(labels, tf.int64)

        correct_prediction = tf.equal(tf.argmax(class_scores, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection('accuracies', accuracy)

        curr_conf_matrix = tf.cast(
            tf.contrib.metrics.confusion_matrix(tf.argmax(class_scores, 1), labels,
                                                num_classes=self.num_classes),
            tf.float32)
        conf_matrix = tf.get_variable('conf_matrix', dtype=tf.float32,
                                    initializer=tf.zeros(
                                        [self.num_classes, self.num_classes],
                                        tf.float32),
                                    trainable=False)

        # make old values decay so early errors don't distort the confusion matrix
        conf_matrix.assign(tf.multiply(conf_matrix, 0.97))

        conf_matrix = conf_matrix.assign_add(curr_conf_matrix)

        tf.summary.image('Confusion Matrix',
                        tf.reshape(tf.clip_by_norm(conf_matrix, 1, axes=[0]),
                                    [1, self.num_classes, self.num_classes, 1]))
        with tf.control_dependencies([conf_matrix]):
            return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, loss, global_step):
        num_batches_per_epoch = self.num_examples_per_epoch / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.learning_rate_decay_factor,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr, epsilon=self.adam_epsilon)
            grads = opt.compute_gradients(loss)
            grads = [
                (tf.clip_by_norm(tf.where(tf.is_nan(grad), tf.zeros_like(grad),
                                          grad), 5.0),
                 var)
                for grad,
                    var in
                grads]

        # Apply gradients.
        # grad_check = tf.check_numerics(grads, "NaN or Inf gradients found: ")
        # with tf.control_dependencies([grad_check]):
        apply_gradient_op = opt.apply_gradients(grads,
                                                    global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies(
                [apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op
