import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import io
from math import log
import gc

from tensorflow.contrib.layers import xavier_initializer
from enum import Enum
from memory_profiler import profile


@ops.RegisterGradient("GuidedElu")
def _GuidedEluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._elu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
#     return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
#                                                  grad,
#                                                  op.outputs[1],
#                                                  op.get_attr("ksize"),
#                                                  op.get_attr("strides"),
#                                                  padding=op.get_attr("padding"))


class NetworkType(Enum):
    RESNET34 = 1
    RESNET50 = 2
    VGGNET = 3
    PRETRAINED = 4


class WoipvFRRUModel(object):
    def __init__(self, config):
        self.is_training = config.is_training
        self.num_classes = config.num_classes
        self.lanes = config.lanes
        self.base_channels = 48
        self.num_examples_per_epoch = config.num_examples_per_epoch
        self.num_epochs_per_decay = config.num_epochs_per_decay
        self.initial_learning_rate = config.initial_learning_rate
        self.learning_rate_decay_factor = config.learning_rate_decay_factor
        self.adam_epsilon = 0.1
        self.moving_average_decay = 0.9999
        self.batch_size = config.batch_size
        self.width = config.width
        self.height = config.height
        self.dropout_prob = config.dropout_prob
        self.weight_decay = config.weight_decay
        self.restore_from_chkpt = config.restore_from_chkpt
        lab_weights =  [3.84082892, 7.45042275, 3.84551333, 4.16781683, 4.84447082, 1.92947501, 3.18845224, 2.23213208, 4.64877237, 2.62109963, 2.72254321, 2.77499254, 2.88145541, 2.85779791, 3.19611563, 4.54788916]
        self.background_weights = [0.57483173, 0.53596904, 0.57472695, 0.56816044, 0.55754441, 0.67488938, 0.59299031, 0.64433079, 0.56025879, 0.61786339, 0.61248375, 0.60989047, 0.60497782, 0.60603114, 0.59272599, 0.56176058] # 0.15480931



        # mu = 0.01 # smaller = bigger imbalance
        # self.label_weights = [log(1 + mu * 1.0 / w) for w in lab_weights]
        self.label_weights = lab_weights

        if config.exclude_class is not None:
            self.label_weights[config.exclude_class] = 0.0
        self.exclude_class = config.exclude_class

        self.graph = config.graph


    def __batch_norm_wrapper(self, inputs, decay=0.999, shape=None, norm_shape=None):
        """ Batch Normalization """
        if shape is None:
            shape = [inputs.get_shape()[-1]]

        if norm_shape is None:
            norm_shape = [0, 1, 2]

        epsilon = 1e-3
        scale = tf.get_variable("scale", shape, initializer=tf.ones_initializer())#tf.Variable(tf.ones(shape), name="scale")
        beta = tf.get_variable("beta", shape, initializer=tf.zeros_initializer())#tf.Variable(tf.zeros(shape), name="beta")
        pop_mean = tf.get_variable("pop_mean", shape, initializer=tf.zeros_initializer())#tf.Variable(tf.zeros(shape),
                               #trainable=False, name="pop_mean")
        pop_var = tf.get_variable("pop_var", shape, initializer=tf.ones_initializer())#tf.Variable(tf.ones(shape),
                              #trainable=False, name="pop_var")

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
            return output


    def __conv(self, input, kernel, strides=[1, 1, 1, 1], nonlinearity=True, batch_norm=True, name="conv"):
        with tf.variable_scope(name) as scope:
            kernel = tf.get_variable('weights',
                                                shape=kernel,
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)
            conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')

            if batch_norm:
                conv = self.__batch_norm_wrapper(conv)

            if nonlinearity:
                conv = tf.nn.elu(conv, name=scope.name)

            return conv


    def __residual_unit(self, input, channels_in, channels_out, name="residual_unit"):
        with tf.variable_scope(name) as scope:
            original = input

            if channels_in != channels_out:
                original = self.__conv(original, [1, 1, channels_in, channels_out], nonlinearity=False, batch_norm=False)

            input = self.__conv(input, [3, 3, channels_in, channels_out], name="conv1")
            input = self.__conv(input, [3, 3, channels_out, channels_out], nonlinearity=False, name="conv2")

            input = input + original

            return input


    def __full_res_residual_network_unit(self, inputs, highway, pooling, in_channels=None, multiplier=None, name="frru"):
        with tf.variable_scope(name) as scope:
            if multiplier is None:
                multiplier = pooling

            if in_channels is None:
                in_channels = self.base_channels * multiplier

            num_channels = self.base_channels * multiplier

            original_highway = highway
            highway_shape = tf.shape(original_highway)

            if pooling > 1:
                highway = tf.nn.max_pool(highway, [1, pooling, pooling, 1], [1, pooling, pooling, 1], padding="SAME")

            inputs = tf.concat([inputs, highway], axis=3)

            inputs = self.__conv(inputs, [3, 3, in_channels + self.lanes, num_channels], name="conv1")
            inputs = self.__conv(inputs, [3, 3, num_channels, num_channels], name="conv2")

            highway_out = self.__conv(inputs, [1, 1, num_channels, self.lanes], nonlinearity=False, batch_norm=False, name="highway_conv")

            if pooling > 1:
                highway_out = tf.image.resize_nearest_neighbor(highway_out, [highway_shape[1], highway_shape[2]])

            highway_out = highway_out + original_highway

            return inputs, highway_out



    def inference(self, inputs):

        inputs = self.__conv(inputs, [3, 3, 3, self.base_channels], name="conv1")

        inputs = self.__residual_unit(inputs, self.base_channels, self.base_channels, name="residual_unit1")
        inputs = self.__residual_unit(inputs, self.base_channels, self.base_channels, name="residual_unit2")
        inputs = self.__residual_unit(inputs, self.base_channels, self.base_channels, name="residual_unit3")

        highway = self.__conv(inputs, [1, 1, 48, self.lanes], nonlinearity=False, name="highway_conv")

        # Encoder
        ##########################################

        pool1_shape = tf.shape(inputs)

        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="pool1")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2, self.base_channels, name="frru1_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2, name="frru1_2")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2, name="frru1_3")

        pool2_shape = tf.shape(inputs)


        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="pool2")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, self.base_channels * 2, name="frru2_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, name="frru2_2")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, name="frru2_3")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, name="frru2_4")

        pool3_shape = tf.shape(inputs)

        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="pool3")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 3, self.base_channels * 2 ** 2, name="frru3_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 3, name="frru3_2")

        pool4_shape = tf.shape(inputs)

        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="pool4")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 4, multiplier=2 ** 3, name="frru4_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 4, multiplier=2 ** 3, name="frru4_2")

        # Decoder
        ###########################################


        inputs = tf.image.resize_bilinear(inputs, [pool4_shape[1], pool4_shape[2]], name="unpool5")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 3, self.base_channels * 2 ** 3, multiplier= 2 ** 2, name="frru5_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 3, multiplier= 2 ** 2, name="frru5_2")

        inputs = tf.image.resize_bilinear(inputs, [pool3_shape[1], pool3_shape[2]], name="unpool6")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, name="frru6_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2 ** 2, name="frru6_2")

        inputs = tf.image.resize_bilinear(inputs, [pool2_shape[1], pool2_shape[2]], name="unpool7")

        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2, self.base_channels * 2 ** 2, name="frru7_1")
        inputs, highway = self.__full_res_residual_network_unit(inputs, highway, 2, name="frru7_2")

        inputs = tf.image.resize_bilinear(inputs, [pool1_shape[1], pool1_shape[2]], name="unpool8")

        inputs = tf.concat([inputs, highway], axis=3)

        inputs = self.__residual_unit(inputs, self.base_channels * 2 + self.lanes, 64, name="residual_unit4")
        inputs = self.__residual_unit(inputs, 64, 64, name="residual_unit5")
        inputs = self.__residual_unit(inputs, 64, 64, name="residual_unit6")

        inputs = self.__conv(inputs, [1, 1, 64, self.num_classes * 2], nonlinearity=False, batch_norm=False, name="softmax_layer")

        return inputs

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
        #for a in accuracies:
            #tf.summary.scalar('accuracy', a)

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            #Name each loss as '(raw)' and name the moving average version of
            #the loss as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)',
                              tf.where(tf.is_nan(l), 0.0, l))
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op


    def loss(self, class_scores, labels, images):
        # polys = tf.sparse_tensor_to_dense(polys, default_value=-1)
        # mask = polys >= 0
        # polys = tf.boolean_mask(polys, mask)

        # labels_oh = tf.one_hot(labels, self.num_classes+1)

        # new_balance = tf.reduce_sum(labels_oh, axis=[0, 1])/tf.reduce_sum(labels_oh)

        # class_balance = tf.Variable(tf.zeros([self.num_classes+1]),
        #                        trainable=False, name="class_balance")
        # balance = tf.assign(class_balance,
        #                            class_balance * 0.999 + new_balance * (1 - 0.999))

        # labels = tf.Print(labels, [balance], "balance", summarize=100)

        labels = tf.cast(labels, tf.int64)
        

        if self.exclude_class is not None:
            m = tf.cast(tf.not_equal(labels, tf.cast(self.exclude_class, tf.int64)), tf.int64)
            labels_without_exclude = labels * m
            labs = tf.one_hot(labels_without_exclude, self.num_classes)

        else:
            labels_without_exclude = labels
            labs = tf.one_hot(labels, self.num_classes + 1)

        labels_without_exclude = tf.reshape(labels_without_exclude, [self.batch_size, self.height, self.width, self.num_classes])
            
        cls_scores = tf.reshape(class_scores, [self.batch_size, self.height, self.width, self.num_classes, 2])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_without_exclude, logits=cls_scores, name="loss")
        # loss = self.__softmax_crossentropy(class_scores, labs)

        #weights = tf.gather(self.label_weights, tf.reshape(labels_without_exclude, [-1]))
        weights = tf.tile([[[self.label_weights]]], [self.batch_size, self.height, self.width, 1])
        weights = weights * tf.cast(labels_without_exclude, dtype=tf.float32)
        backgroundweights = tf.tile([[[self.background_weights]]], [self.batch_size, self.height, self.width, 1])
        weights = tf.where(tf.equal(weights, 0), tf.ones_like(weights) * backgroundweights, weights)

        # weights = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_max(tf.reshape(weights, [-1, self.num_classes]), axis=1), axis=1), [1, self.num_classes]), [self.batch_size, self.height, self.width, self.num_classes])

        #weights = tf.Print(weights, [tf.shape(weights), weights], "weights", summarize=1024)

        loss = loss * weights

        if self.exclude_class is not None:
            loss = tf.where(tf.equal(labels, tf.cast(self.exclude_class, tf.int64)), tf.zeros_like(loss, dtype=tf.float32), loss)#tf.boolean_mask(loss, tf.not_equal(labels, tf.cast(self.exclude_class, tf.int64)))

        loss2 = tf.reduce_sum(loss)

        tf.add_to_collection('losses', tf.identity(loss2,
                                                   name="losses"))

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), tf.nn.softmax(cls_scores)[0], labels_without_exclude[0], loss[0]


    
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
        # for var in tf.trainable_variables():
        #     #tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         #tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     self.moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(
        #     tf.trainable_variables())

        with tf.control_dependencies(
                [apply_gradient_op]):#, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op
