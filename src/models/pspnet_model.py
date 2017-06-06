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


class WoipvPspNetModel(object):
    def __init__(self, config):
        self.is_training = config.is_training
        self.num_classes = config.num_classes
        self.base_channels = 48
        self.last_layer_features = 36
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
        lab_weights = [0.76383808, 2.8615545, 5.07328564, 3.12578502, 3.52720184, 8.56995074, 1.6714053, 1.75974135, 1.72642916, 8.02411964, 2.83844672, 1.98107858, 3.43759644, 3.60653777, 2.07062979, 2.5808826, 20.19650681, 2.20710592, 3.47251928, 1.36609693, 3.22695245]
        self.background_weights = [1.44755085, 0.60586247, 0.55466529, 0.59520962, 0.58258452, 0.53097912, 0.71341887, 0.69845344, 0.70384382, 0.53322648, 0.60690857, 0.6687959, 0.58510359, 0.58047544, 0.65917182, 0.62014133, 0.51269261, 0.64644668, 0.58410374, 0.78865129, 0.59167743]



        # mu = 0.01 # smaller = bigger imbalance
        # self.label_weights = [log(1 + mu * 1.0 / w) for w in lab_weights]
        self.label_weights = lab_weights

        # if config.exclude_class is not None:
        #     self.label_weights[config.exclude_class] = 0.0
        self.exclude_class = config.exclude_class

        self.graph = config.graph


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
            batch_norm = self.__batch_norm_wrapper(conv, decay=0.9999)
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights',
                                     [3, 3, out_filters, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv1')
            bias = self.__batch_norm_wrapper(conv, decay=0.9999)

        with tf.variable_scope('subadd'):
            if in_filters != out_filters:
                kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
                inputs = tf.nn.conv2d(
                    inputs, kernel, [1, stride, stride, 1], padding='SAME')
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
            kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters / 4],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, decay=0.9999)
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights',
                                     [3, 3, out_filters / 4, out_filters / 4],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv1')
            batch_norm = self.__batch_norm_wrapper(conv, decay=0.9999)
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub3'):
            kernel = tf.get_variable('weights', [1, 1, out_filters / 4, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, decay=0.9999)

        with tf.variable_scope('subadd'):
            if in_filters != out_filters:
                kernel = tf.get_variable('weights', [1, 1, in_filters, out_filters],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
                inputs = tf.nn.conv2d(
                    inputs, kernel, [1, stride, stride, 1], padding='SAME')
            batch_norm += inputs
            conv = tf.nn.elu(batch_norm, 'elu')

            num = np.power(2, np.floor(np.log2(out_filters) / 2))

            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(out_filters /
                                                             num)))
            tf.summary.image('sub3/activations', grid, max_outputs=1)

        return conv


    def __resnet50(self, inputs):
        with tf.variable_scope('reslayer_downsample_256'):
            inputs = self.__reslayer_bottleneck(inputs, 64, 256)

        for i in range(2):
            with tf.variable_scope('reslayer_256_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 256, 256)
            
            res256 = inputs

        with tf.variable_scope('reslayer_downsample_512'):
            inputs = self.__reslayer_bottleneck(inputs, 256, 512, stride=2)

        for i in range(3):
            with tf.variable_scope('reslayer_512_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 512, 512)
            
            res512 = inputs

        with tf.variable_scope('reslayer_downsample_1024'):
            inputs = self.__reslayer_bottleneck(inputs, 512, 1024, stride=1)

        for i in range(5):
            with tf.variable_scope('reslayer_1024_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 1024, 1024)
            
            res1024 = inputs

        with tf.variable_scope('reslayer_downsample_2048'):
            inputs = self.__reslayer_bottleneck(inputs, 1024, 2048, stride=1)

        for i in range(2):
            with tf.variable_scope('reslayer_2048_%d' % i):
                inputs = self.__reslayer_bottleneck(inputs, 2048, 2048)
        return inputs, res512, res256


    def __put_activations_on_grid(self, activations, grid, pad=1, normalize=True):
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
        if normalize:
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


    def inference(self, inputs):

        with tf.variable_scope('first_layer'):
                kernel = tf.get_variable('weights', [3, 3, 3, 64],
                                         initializer=xavier_initializer(
                    dtype=tf.float32),
                    dtype=tf.float32)
                conv = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1],
                                    padding='SAME',
                                    name='conv')
                batch_norm = self.__batch_norm_wrapper(
                    conv, decay=0.9999)
                conv = tf.nn.elu(batch_norm, 'elu')

                grid = self.__put_activations_on_grid(conv, (8, 8))
                tf.summary.image('conv1/activations', grid, max_outputs=1)

                first_layer = conv

                conv = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1], padding='SAME',
                                      name='pool')

        conv, res512, res256 = self.__resnet50(conv)

        with tf.variable_scope('pyramid_1'):
            kernel = tf.get_variable('weights', [1, 1, 2048, 512],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            p1 = tf.nn.avg_pool(conv, [1, self.last_layer_features, self.last_layer_features, 1], [1, self.last_layer_features, self.last_layer_features, 1], padding="SAME")

            p1 = tf.nn.conv2d(p1, kernel, [1, 1, 1, 1], padding="SAME")
            p1 = self.__batch_norm_wrapper(p1)
            p1 = tf.nn.elu(p1)

            num = np.power(2, np.floor(np.log2(512) / 2))

            grid = self.__put_activations_on_grid(p1, (int(num),
                                                         int(512 /
                                                             num)), normalize=False)
            tf.summary.image('pyramid_1', grid, max_outputs=1)
            
            p1 = tf.image.resize_bilinear(p1, [self.last_layer_features, self.last_layer_features])

        with tf.variable_scope('pyramid_2'):
            kernel = tf.get_variable('weights', [1, 1, 2048, 512],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            p2 = tf.nn.avg_pool(conv, [1, self.last_layer_features//2, self.last_layer_features//2, 1], [1, self.last_layer_features//2, self.last_layer_features//2, 1], padding="SAME")
            p2 = tf.nn.conv2d(p2, kernel, [1, 1, 1, 1], padding="SAME")
            p2 = self.__batch_norm_wrapper(p2)
            p2 = tf.nn.elu(p2)
            p2 = tf.image.resize_bilinear(p2, [self.last_layer_features, self.last_layer_features])

        with tf.variable_scope('pyramid_3'):
            kernel = tf.get_variable('weights', [1, 1, 2048, 512],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            p3 = tf.nn.avg_pool(conv, [1, self.last_layer_features//3, self.last_layer_features//3, 1], [1, self.last_layer_features//3, self.last_layer_features//3, 1], padding="SAME")
            p3 = tf.nn.conv2d(p3, kernel, [1, 1, 1, 1], padding="SAME")
            p3 = self.__batch_norm_wrapper(p3)
            p3 = tf.nn.elu(p3)
            p3 = tf.image.resize_bilinear(p3, [self.last_layer_features, self.last_layer_features])

        with tf.variable_scope('pyramid_4'):
            kernel = tf.get_variable('weights', [1, 1, 2048, 512],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            p4 = tf.nn.avg_pool(conv, [1, self.last_layer_features//6, self.last_layer_features//6, 1], [1, self.last_layer_features//6, self.last_layer_features//6, 1], padding="SAME")
            p4 = tf.nn.conv2d(p4, kernel, [1, 1, 1, 1], padding="SAME")
            p4 = self.__batch_norm_wrapper(p4)
            p4 = tf.nn.elu(p4)
            p4 = tf.image.resize_bilinear(p4, [self.last_layer_features, self.last_layer_features])

        with tf.variable_scope('pyramid_concat'):
            conv = tf.concat([conv, p1, p2, p3, p4], axis=3, name="pyramid_concat")

            num = np.power(2, np.floor(np.log2(4096) / 2))

            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(4096 /
                                                             num)))
            tf.summary.image('pyramid_concat', grid, max_outputs=1)

        with tf.variable_scope('final_layer1'):

            kernel = tf.get_variable('weights', [3, 3, 4096, 512],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding="SAME")
            conv = self.__batch_norm_wrapper(conv)
            conv = tf.nn.elu(conv)
            conv = tf.nn.dropout(conv, self.dropout_prob)

        #     res512_shape = tf.shape(res512)
        #     conv = tf.image.resize_bilinear(conv, [res512_shape[1], res512_shape[2]])

        #     conv = tf.concat([conv, res512], axis=3)

        # with tf.variable_scope('final_layer2'):

        #     kernel = tf.get_variable('weights', [3, 3, 1024, 512],
        #                                  initializer=xavier_initializer(
        #                                  dtype=tf.float32),
        #                                  dtype=tf.float32)
        #     conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding="SAME")
        #     conv = self.__batch_norm_wrapper(conv)
        #     conv = tf.nn.elu(conv)
        #     conv = tf.nn.dropout(conv, self.dropout_prob)

        #     res256_shape = tf.shape(res256)
        #     conv = tf.image.resize_bilinear(conv, [res256_shape[1], res256_shape[2]])

        #     conv = tf.concat([conv, res256], axis=3)

        # with tf.variable_scope('final_layer3'):

        #     kernel = tf.get_variable('weights', [3, 3, 768, 512],
        #                                  initializer=xavier_initializer(
        #                                  dtype=tf.float32),
        #                                  dtype=tf.float32)
        #     conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding="SAME")
        #     conv = self.__batch_norm_wrapper(conv)
        #     conv = tf.nn.elu(conv)
        #     conv = tf.nn.dropout(conv, self.dropout_prob)

        #     first_layer_shape = tf.shape(first_layer)
        #     conv = tf.image.resize_bilinear(conv, [first_layer_shape[1], first_layer_shape[2]])

        #     conv = tf.concat([conv, first_layer], axis=3)

        # with tf.variable_scope('final_layer4'):

        #     kernel = tf.get_variable('weights', [3, 3, 576, 512],
        #                                  initializer=xavier_initializer(
        #                                  dtype=tf.float32),
        #                                  dtype=tf.float32)
        #     conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding="SAME")
        #     conv = self.__batch_norm_wrapper(conv)
        #     conv = tf.nn.elu(conv)
        #     conv = tf.nn.dropout(conv, self.dropout_prob)

        with tf.variable_scope('softmax'):
            bias = tf.get_variable('bias', [self.num_classes + 1],
                                         initializer=tf.constant_initializer(value=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
            kernel2 = tf.get_variable('weights', [1, 1, 512, self.num_classes + 1],
                                         initializer=xavier_initializer(
                                         dtype=tf.float32),
                                         dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel2, [1, 1, 1, 1], padding="SAME", name="softmax")
            conv = tf.nn.bias_add(conv, bias)

            conv = tf.image.resize_bilinear(conv, [self.height, self.width])

        return conv

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
            labels_without_exclude = tf.slice(labels, [0, 0, 0, 0], [-1, -1, -1, self.num_classes])
            exclude_labels = tf.slice(labels, [0, 0, 0, self.num_classes], [-1, -1, -1, 1])
            #labs = tf.one_hot(labels_without_exclude, self.num_classes)
            labs = tf.expand_dims(1 - tf.reduce_max(labels_without_exclude, axis=3), axis=3)
            labs = tf.concat([labs, labels_without_exclude], axis=3)
        else:
            labels_without_exclude = labels
            labs = tf.one_hot(labels, self.num_classes + 1)

        labels_without_exclude = tf.reshape(labs, [self.batch_size, self.height, self.width, self.num_classes+1])
            
        cls_scores = tf.reshape(class_scores, [self.batch_size, self.height, self.width, self.num_classes+1])

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_scores, labels=tf.cast(labels_without_exclude, tf.float32))
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_without_exclude, logits=cls_scores, name="loss")
        # loss = self.__softmax_crossentropy(class_scores, labs)

        #weights = tf.gather(self.label_weights, tf.reshape(labels_without_exclude, [-1]))
        weights = tf.tile([[[self.label_weights]]], [self.batch_size, self.height, self.width, 1])
        weights = weights * tf.cast(labs, dtype=tf.float32)
        # weights = tf.ones([self.batch_size, self.height, self.width, self.num_classes+1])
        backgroundweights = tf.tile([[[self.background_weights]]], [self.batch_size, self.height, self.width, 1])
        weights = tf.where(tf.equal(weights, 0), tf.ones_like(weights) * backgroundweights, weights)

        if self.exclude_class is not None:
            exclude_labels = 1 - tf.tile(exclude_labels, [1, 1, 1, self.num_classes+1])
            weights = weights * tf.cast(exclude_labels, tf.float32)

        # weights = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_max(tf.reshape(weights, [-1, self.num_classes]), axis=1), axis=1), [1, self.num_classes]), [self.batch_size, self.height, self.width, self.num_classes])

        #weights = tf.Print(weights, [tf.shape(weights), weights], "weights", summarize=1024)

        loss = loss * weights

        # if self.exclude_class is not None:
        #     loss = tf.where(tf.equal(labels, tf.cast(self.exclude_class, tf.int64)), tf.zeros_like(loss, dtype=tf.float32), loss)#tf.boolean_mask(loss, tf.not_equal(labels, tf.cast(self.exclude_class, tf.int64)))

        loss2 = tf.reduce_mean(tf.reduce_sum(loss, axis=3), axis=[0, 1, 2])

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
