import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from pycocotools.coco import COCO

from tensorflow.contrib.layers import xavier_initializer
from enum import Enum


@ops.RegisterGradient("GuidedElu")
def _GuidedEluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._elu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))


class NetworkType(Enum):
    RESNET34 = 1
    RESNET50 = 2
    VGGNET = 3
    PRETRAINED = 4


class WoipvSegnetModel(object):
    def __init__(self, config):
        self.is_training = config.is_training
        self.num_classes = config.num_classes
        self.num_examples_per_epoch = config.num_examples_per_epoch
        self.num_epochs_per_decay = config.num_epochs_per_decay
        self.initial_learning_rate = config.initial_learning_rate
        self.learning_rate_decay_factor = config.learning_rate_decay_factor
        self.adam_epsilon = 0.1
        self.moving_average_decay = 0.9999
        self.width = config.width
        self.height = config.height
        self.dropout_prob = config.dropout_prob
        self.weight_decay = config.weight_decay
        self.restore_from_chkpt = config.restore_from_chkpt
        self.background_weight = 0.5

        self.graph = config.graph

        self.interactive_sess = tf.InteractiveSession()

    def __batch_norm_wrapper(self, inputs, decay=0.999, shape=None):
        """ Batch Normalization """
        if shape is None:
            shape = [inputs.get_shape()[-1]]

        epsilon = 1e-3
        scale = tf.Variable(tf.ones(shape), name="scale")
        beta = tf.Variable(tf.zeros(shape), name="beta")
        pop_mean = tf.Variable(tf.zeros(shape),
                               trainable=False, name="pop_mean")
        pop_var = tf.Variable(tf.ones(shape),
                              trainable=False, name="pop_var")

        if self.is_training:
            output, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs,
                                              mean=None, variance=None, offset=beta,
                                              scale=scale,
                                              epsilon=epsilon, is_training=self.is_training, name="batch_norm")

            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                output = tf.identity(output)

            return output
        else:
            output, _, _ =  tf.nn.fused_batch_norm(inputs,
                                          mean=pop_mean, variance=pop_var, offset=beta, scale=scale,
                                          epsilon=epsilon, name="batch_norm")

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in ip5wke model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
          total_loss: Total loss from loss().
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """

        # Compute the moving average of all individual losses and the total
        # loss
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
            tf.summary.scalar(l.op.name + ' (raw)',
                              tf.where(tf.is_nan(l), 0.0, l))
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

        # scale to [0, 255.0]
        mean, var = tf.nn.moments(activ,axes=[0, 1])
        activ = (activ - mean) / tf.maximum(var, 1.0/tf.sqrt(tf.cast(tf.size(activ), tf.float32)))

        x_min = tf.reduce_min(activ, axis=[0, 1])
        x_max = tf.reduce_max(activ, axis=[0, 1])
        activ = (activ - x_min) / (x_max - x_min)

        # greyscale
        activ = tf.expand_dims(activ, 2)
        # pad X and Y
        x1 = tf.pad(activ, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

        # X and Y dimensions, w.r.t. padding
        Y = tf.shape(activ)[0] + pad
        X = tf.shape(activ)[1] + pad

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

        # return x8
        return x7

    def __unpool(self, updates, mask, ksize=[1, 2, 2, 1], output_shape=None, name=''):
        with tf.variable_scope(name):
            mask = tf.cast(mask, tf.int32)
            input_shape = tf.shape(updates, out_type=tf.int32)
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype=tf.int32)
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
            feature_range = tf.range(output_shape[3], dtype=tf.int32)
            f = one_like_mask * feature_range
            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def __pad_to_size(self, input, target_shape):
        input_shape = tf.shape(input)
        difference = target_shape - input_shape
        offset = tf.zeros_like(difference)
        padding = tf.concat([tf.expand_dims(difference, axis=1), tf.expand_dims(offset, axis=1)], axis=1)

        return tf.pad(input, padding)

    def inference(self, inputs):

        # Encoder
        # -------------------------------------------------------------------------
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 3, 64],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv1 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_kernels_on_grid(kernel, (8, 8))
            tf.summary.image('conv1/features', grid, max_outputs=1)
            grid = self.__put_activations_on_grid(conv, (8, 8))
            tf.summary.image('conv1/activations', grid, max_outputs=1)

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 64, 64],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv2 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (8, 8))
            tf.summary.image('conv2/activations', grid, max_outputs=1)

        pool2, argmax2 = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm2')

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 64, 128],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv3 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (8, 16))
            tf.summary.image('conv3/activations', grid, max_outputs=1)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 128, 128],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv4 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (8, 16))
            tf.summary.image('conv4/activations', grid, max_outputs=1)

        pool4, argmax4 = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        # norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm4')

        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 128, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv5 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 16))
            tf.summary.image('conv5/activations', grid, max_outputs=1)

        with tf.variable_scope('conv6') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv6 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 16))
            tf.summary.image('conv6/activations', grid, max_outputs=1)

        with tf.variable_scope('conv7') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv7 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 16))
            tf.summary.image('conv7/activations', grid, max_outputs=1)

        pool7, argmax7 = tf.nn.max_pool_with_argmax(conv7, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool7')
        # norm7 = tf.nn.lrn(pool7, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm7')

        with tf.variable_scope('conv8') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(pool7, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv8 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv8/activations', grid, max_outputs=1)

        with tf.variable_scope('conv9') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv9 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv9/activations', grid, max_outputs=1)

        with tf.variable_scope('conv10') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv9, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv10 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv10/activations', grid, max_outputs=1)

        pool10, argmax10 = tf.nn.max_pool_with_argmax(conv10, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME', name='pool10')
        # norm10 = tf.nn.lrn(pool10, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm10')

        with tf.variable_scope('conv11') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(pool10, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv11 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv11/activations', grid, max_outputs=1)

        with tf.variable_scope('conv12') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv11, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv12 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv12/activations', grid, max_outputs=1)

        with tf.variable_scope('conv13') as scope:
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(conv12, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.__batch_norm_wrapper(conv)
            conv13 = tf.nn.elu(bias, name=scope.name)
            grid = self.__put_activations_on_grid(conv, (16, 32))
            tf.summary.image('conv13/activations', grid, max_outputs=1)

        pool13, argmax13 = tf.nn.max_pool_with_argmax(conv13, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME', name='pool13')

        # Decoder
        # -------------------------------------------------------------------------
        unpool13 = self.__unpool(pool13, argmax13, output_shape=tf.shape(conv13), name='unpool13')

        with tf.variable_scope('deconv13'):
            #unpool13 = self.__pad_to_size(unpool13, tf.shape(conv12))
            target_shape = tf.shape(conv13)
            unpool13 = tf.image.resize_image_with_crop_or_pad(unpool13, target_shape[1], target_shape[2])
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv13 = tf.nn.conv2d_transpose(unpool13, kernel, tf.shape(conv12),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv13, (16, 32))
            tf.summary.image('deconv13/activations', grid, max_outputs=1)

            deconv13 = self.__batch_norm_wrapper(deconv13, shape=[512])

        with tf.variable_scope('deconv12'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv12 = tf.nn.conv2d_transpose(deconv13, kernel, tf.shape(conv11),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv12, (16, 32))
            tf.summary.image('deconv12/activations', grid, max_outputs=1)

            deconv12 = self.__batch_norm_wrapper(deconv12, shape=[512])

        with tf.variable_scope('deconv11'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv11 = tf.nn.conv2d_transpose(deconv12, kernel, tf.shape(pool10),
                                                strides=[1, 1, 1, 1], padding='SAME')

            deconv11 = self.__batch_norm_wrapper(deconv11, shape=[512])
            grid = self.__put_activations_on_grid(deconv11, (16, 32))
            tf.summary.image('deconv10/activations', grid, max_outputs=1)

        unpool10 = self.__unpool(deconv11, argmax10, output_shape=tf.shape(conv10), name='unpool10')
        grid = self.__put_activations_on_grid(unpool10, (16, 32))
        tf.summary.image('unpool10/activations', grid, max_outputs=1)

        with tf.variable_scope('deconv10'):
            target_shape = tf.shape(conv10)
            unpool10 = tf.image.resize_image_with_crop_or_pad(unpool10, target_shape[1], target_shape[2])
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv10 = tf.nn.conv2d_transpose(unpool10, kernel, tf.shape(conv9),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv10, (16, 32))
            tf.summary.image('deconv10/activations', grid, max_outputs=1)

            deconv10 = self.__batch_norm_wrapper(deconv10, shape=[512])

        with tf.variable_scope('deconv9'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 512, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv9 = tf.nn.conv2d_transpose(deconv10, kernel, tf.shape(conv8),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv9, (16, 32))
            tf.summary.image('deconv9/activations', grid, max_outputs=1)

            deconv9 = self.__batch_norm_wrapper(deconv9, shape=[512])

        with tf.variable_scope('deconv8'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 512],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv8 = tf.nn.conv2d_transpose(deconv9, kernel, tf.shape(pool7),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv8, (16, 16))
            tf.summary.image('deconv8/activations', grid, max_outputs=1)

            deconv8 = self.__batch_norm_wrapper(deconv8, shape=[256])

        unpool7 = self.__unpool(deconv8, argmax7, output_shape=tf.shape(conv7), name='unpool7')

        with tf.variable_scope('deconv7'):
            target_shape = tf.shape(conv7)
            unpool7 = tf.image.resize_image_with_crop_or_pad(unpool7, target_shape[1], target_shape[2])
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv7 = tf.nn.conv2d_transpose(unpool7, kernel, tf.shape(conv6),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv7, (16, 16))
            tf.summary.image('deconv7/activations', grid, max_outputs=1)

            deconv7 = self.__batch_norm_wrapper(deconv7, shape=[256])

        with tf.variable_scope('deconv6'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 256, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv6 = tf.nn.conv2d_transpose(deconv7, kernel, tf.shape(conv6),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv6, (16, 16))
            tf.summary.image('deconv6/activations', grid, max_outputs=1)

            deconv6 = self.__batch_norm_wrapper(deconv6, shape=[256])

        with tf.variable_scope('deconv5'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 128, 256],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv5 = tf.nn.conv2d_transpose(deconv6, kernel, tf.shape(pool4),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv5, (8, 16))
            tf.summary.image('deconv5/activations', grid, max_outputs=1)

            deconv5 = self.__batch_norm_wrapper(deconv5, shape=[128])

        unpool4 = self.__unpool(deconv5, argmax4, output_shape=tf.shape(conv4), name='unpool4')

        with tf.variable_scope('deconv4'):
            target_shape = tf.shape(conv4)
            unpool4 = tf.image.resize_image_with_crop_or_pad(unpool4, target_shape[1], target_shape[2])
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 128, 128],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv4 = tf.nn.conv2d_transpose(unpool4, kernel, tf.shape(conv3),
                                                strides=[1, 1, 1, 1], padding='SAME')

            deconv4 = self.__batch_norm_wrapper(deconv4, shape=[128])

        with tf.variable_scope('deconv3'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 64, 128],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv3 = tf.nn.conv2d_transpose(deconv4, kernel, tf.shape(pool2),
                                                strides=[1, 1, 1, 1], padding='SAME')

            deconv3 = self.__batch_norm_wrapper(deconv3, shape=[64])

        unpool2 = self.__unpool(deconv3, argmax2, output_shape=tf.shape(conv2), name='unpool2')

        with tf.variable_scope('deconv2'):
            target_shape = tf.shape(conv2)
            unpool2 = tf.image.resize_image_with_crop_or_pad(unpool2, target_shape[1], target_shape[2])
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 64, 64],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv2 = tf.nn.conv2d_transpose(unpool2, kernel, tf.shape(conv1),
                                                strides=[1, 1, 1, 1], padding='SAME')

            deconv2 = self.__batch_norm_wrapper(deconv2, shape=[64])

        with tf.variable_scope('deconv1'):
            kernel = tf.get_variable('weights',
                                                shape=[3, 3, 64, 64],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            deconv1 = tf.nn.conv2d_transpose(deconv2, kernel, tf.shape(conv1),
                                                strides=[1, 1, 1, 1], padding='SAME')
            grid = self.__put_activations_on_grid(deconv1, (8, 8))
            tf.summary.image('deconv1/activations0', grid, max_outputs=1)

            deconv1 = self.__batch_norm_wrapper(deconv1, shape=[64])
            grid = self.__put_activations_on_grid(deconv1, (8, 8))
            tf.summary.image('deconv1/activations', grid, max_outputs=1)

        with tf.variable_scope('final_softmax'):
            kernel = tf.get_variable('weights',
                                                shape=[1, 1, 64, self.num_classes+1],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            softmax = tf.nn.conv2d(deconv1, kernel,
                                                strides=[1, 1, 1, 1], padding='SAME')

        return softmax

    def loss(self, class_scores, labels, images):
        # polys = tf.sparse_tensor_to_dense(polys, default_value=-1)
        # mask = polys >= 0
        # polys = tf.boolean_mask(polys, mask)
        labels = tf.cast(tf.expand_dims(labels, axis=0), tf.int32)
        score_amax = tf.argmax(class_scores, axis=3)
        scores = tf.expand_dims(tf.cast(score_amax, tf.float32), axis=3)
        scores = scores / self.num_classes
        tf.summary.image("predicted_seg", scores)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=class_scores, name="loss")

        weights = tf.concat([[self.background_weight], tf.ones([self.num_classes])], axis=0)

        weights = tf.gather(weights, tf.reshape(labels, [-1]))

        weights = tf.reshape(weights, tf.shape(labels))

        loss = loss * weights

        loss = tf.reduce_mean(loss)

        tf.add_to_collection('losses', tf.identity(loss,
                                                   name="losses"))

        return tf.add_n(tf.get_collection('losses'), name='total_loss')



    def train(self, loss, global_step):
        num_batches_per_epoch = self.num_examples_per_epoch
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
        with tf.variable_scope('calculate_gradients'):
            with tf.control_dependencies([loss_averages_op]):
                opt = tf.train.AdamOptimizer(lr, epsilon=self.adam_epsilon)
                grads = opt.compute_gradients(loss)
                # grads = [
                #     (tf.clip_by_value(tf.where(tf.is_nan(grad), tf.zeros_like(grad),
                # grad), -1000.0, 1000.0), var) if grad is not None else
                # (tf.zeros_like(var), var) for grad, var in grads]

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
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     self.moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(
        #     tf.trainable_variables())

        with tf.control_dependencies(
                [apply_gradient_op]):#, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op
