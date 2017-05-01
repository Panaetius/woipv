import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from tensorflow.contrib.layers import xavier_initializer
from enum import Enum


@ops.RegisterGradient("GuidedElu")
def _GuidedEluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._elu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))


class NetworkType(Enum):
    RESNET34 = 1
    RESNET50 = 2
    PRETRAINED = 3


class WoipvModel(object):
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
        self.min_box_size = config.min_box_size
        self.rcnn_cls_loss_weight = config.rcnn_cls_loss_weight
        self.rcnn_reg_loss_weight = config.rcnn_reg_loss_weight
        self.rpn_cls_loss_weight = config.rpn_cls_loss_weight
        self.rpn_reg_loss_weight = config.rpn_reg_loss_weight
        self.dropout_prob = config.dropout_prob
        self.weight_decay = config.weight_decay
        self.restore_from_chkpt = config.restore_from_chkpt
        self.net = config.net
        self.num_proposals = 6000
        self.num_proposals_after_nms = 300

        if self.net == NetworkType.RESNET34:
            self.conv_feature_count = 256
        elif self.net == NetworkType.RESNET50:
            self.conv_feature_count = 1024
        elif self.net == NetworkType.PRETRAINED:
            self.conv_feature_count = 1024

        self.graph = config.graph

        self.interactive_sess = tf.InteractiveSession()

        self.num_anchors = 9

        if self.net == NetworkType.RESNET34:
            self.num_features = 38
        elif self.net == NetworkType.RESNET50:
            self.num_features = 38
        elif self.net == NetworkType.PRETRAINED:
            self.num_features = 38

        self.feat_stride = int(round(self.width/self.num_features))

        self.__create_anchors((128.0, 256.0, 512.0),
                              ((2.0, 1.0), (1.0, 1.0), (1.0, 2.0)))

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
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub2'):
            kernel = tf.get_variable('weights',
                                     [3, 3, out_filters / 4, out_filters / 4],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv1')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('sub3'):
            kernel = tf.get_variable('weights', [1, 1, out_filters / 4, out_filters],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])

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

    def __region_proposals(self, inputs, input_size, output_size, k):
        with tf.variable_scope('common_roi'):
            kernel = tf.get_variable('weights', [3, 3, input_size, output_size],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')

            num = np.power(2, np.floor(np.log2(output_size) / 2))
            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(output_size /
                                                             num)))
            tf.summary.image('common_roi/activations', grid, max_outputs=1)

            conv = tf.nn.dropout(conv, self.dropout_prob)

        with tf.variable_scope('cls_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 2 * k],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv_cls = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv')
            conv_shape = tf.shape(conv_cls)
            softmax_conv = tf.reshape(tf.slice(tf.nn.softmax(tf.reshape(conv_cls, [1, conv_shape[1], conv_shape[2], k, 2])), [
                                      0, 0, 0, 0, 1], [1, conv_shape[1], conv_shape[2], k, 1]), [1, conv_shape[1], conv_shape[2], k])
            grid = self.__put_activations_on_grid(softmax_conv, (3, 3))
            tf.summary.image('conv_cls/activations', grid, max_outputs=1)

        with tf.variable_scope('reg_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 4 * k],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv_regions = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv')
            grid = self.__put_activations_on_grid(conv_regions, (6, 6))
            tf.summary.image('conv_regions/activations', grid, max_outputs=1)

            coords, size = tf.split(tf.reshape(tf.clip_by_value(conv_regions, -0.2, 0.2),
                                               [-1, 4]), 2,
                                    axis=1)

            size = tf.exp(size)

            conv_regions = tf.reshape(tf.concat([coords, size], axis=1),
                                      [1, conv_shape[1], conv_shape[2], 4 * k])

        return conv_cls, conv_regions

    def __roi_pooling(self, inputs, boxes, indices, pool_height, pool_width):
        """ Performs RoI Pooling """

        inputs_shape = tf.cast(tf.shape(inputs), tf.float32)

        num_boxes = tf.shape(boxes)[0]

        width = self.feat_stride * inputs_shape[2]
        height = self.feat_stride * inputs_shape[1]

        with tf.control_dependencies([tf.assert_greater(boxes[:, 2], boxes[:, 0],
                                                        message="y1 > y2 in roi pool0",
                                                        summarize=100),
                                      tf.assert_greater(boxes[:, 3], boxes[:, 1],
                                                        message="x1 > x2 in roi pool0",
                                                        summarize=100)]):
            boxes = tf.identity(boxes)

        boxes = boxes * tf.tile([[1.0 / height, 1.0 / width,
                                  1.0 / height, 1.0 / width]], [num_boxes, 1])
        

        with tf.control_dependencies([tf.assert_greater(boxes[:, 2], boxes[:, 0],
                                                        message="y1 > y2 in roi pool1",
                                                        summarize=100),
                                      tf.assert_greater(boxes[:, 3], boxes[:, 1],
                                                        message="x1 > x2 in roi pool1",
                                                        summarize=100)]):
            boxes = tf.identity(boxes)

        boxes = tf.concat([tf.expand_dims(tf.clip_by_value(boxes[:, 0], 0.0, 1.0 - 1e-5), 1),
                           tf.expand_dims(tf.clip_by_value(boxes[:, 1], 0.0, 1.0 - 1e-5), 1),
                           tf.expand_dims(tf.clip_by_value(boxes[:, 2], 1e-5, 1.0), 1),
                           tf.expand_dims(tf.clip_by_value(boxes[:, 3], 1e-5, 1.0), 1)],
                          axis=1)

        with tf.control_dependencies([tf.assert_greater(boxes[:, 2], boxes[:, 0],
                                                        message="y1 > y2 in roi pool2",
                                                        summarize=100),
                                      tf.assert_greater(boxes[:, 3], boxes[:, 1],
                                                        message="x1 > x2 in roi pool2",
                                                        summarize=100)]):
            boxes = tf.identity(boxes)

        roi_pool = tf.image.crop_and_resize(
            inputs, boxes, indices, [pool_height, pool_width], extrapolation_value=0.0)

        return roi_pool

    def __process_rois(self, regions, class_scores):
        """ get relevant regions, with non-maximum suppression and clipping """
        region_shape = tf.shape(regions)

        width = region_shape[2] * self.feat_stride
        height = region_shape[1] * self.feat_stride

        anchors = self.get_tiled_anchors_for_shape(width, height)

        region_boxes = self.adjust_bbox(tf.reshape(regions,
                                                                [-1, 4]), anchors)

        class_scores = tf.reshape(class_scores, [-1, 2])

        if self.is_training:
            # ignore boxes that cross the image boundary
            self.outside_anchors = tf.logical_and(tf.logical_and(anchors[:, 0] - anchors[:, 2] / 2.0 >= -self.feat_stride, anchors[:, 1] - anchors[:, 3] / 2.0 >= -self.feat_stride), tf.logical_and(
                anchors[:, 0] + anchors[:, 2] / 2.0 < tf.cast(height, tf.float32) + self.feat_stride, anchors[:, 1] + anchors[:, 3] / 2.0 < tf.cast(width, tf.float32) + self.feat_stride))
            region_boxes = tf.boolean_mask(region_boxes, self.outside_anchors)
            class_scores = tf.boolean_mask(class_scores, self.outside_anchors)

        shape = tf.shape(region_boxes)
        mask = tf.logical_or(region_boxes[:, 2] >
                                        self.min_box_size,
                                        region_boxes[:,
                                                    3] > self.min_box_size)
        region_box = tf.boolean_mask(region_boxes, mask)
        class_score = tf.boolean_mask(class_scores, mask)

        region_box = self.clip_bboxes(region_box, width, height)

        class_score = tf.nn.softmax(class_score)

        class_score = tf.slice(class_score, [0, 1], [-1, 1])
        class_score = tf.reshape(class_score, [-1])

        _, idx = tf.nn.top_k(class_score, k=tf.minimum(self.num_proposals, tf.shape(class_score)[0]))
        class_score = tf.gather(class_score, idx)
        region_box = tf.gather(region_box, idx)

        with tf.variable_scope('non_maximum_supression'):
            bboxes2 = self.bboxes_to_yxyx(
                region_box, tf.cast(height, tf.float32))
            idx = tf.image.non_max_suppression(bboxes2, class_score,
                                               self.num_proposals_after_nms,
                                               0.7)
            region_box = tf.gather(region_box, idx)

        return region_box

    def get_tiled_anchors_for_shape(self, width, height):
        """ creates/tiles anchors for a width x height image/feature map,
            producing coordinates from [0, width) and [0, height) for the
            resulting bounding boxes, according to the feature stride
            of the last conv layer """
        anchors = tf.expand_dims(self.anchors, axis=0)
        feat_height = tf.cast(tf.ceil(height/self.feat_stride), tf.int32)
        feat_width = tf.cast(tf.ceil(width/self.feat_stride), tf.int32)
        anchor_shape = [feat_height * feat_width, 1, 1]
        anchors = tf.tile(anchors, tf.stack(anchor_shape))

        x = tf.range(0.0, feat_width * self.feat_stride, self.feat_stride)
        y = tf.range(0.0, feat_height * self.feat_stride, self.feat_stride)

        X, Y = tf.meshgrid(x, y)
        X = tf.expand_dims(X, 2)
        Y = tf.expand_dims(Y, 2)

        shift = tf.reshape(tf.concat([Y, X, tf.zeros_like(X), tf.zeros_like(X)], 2), [-1, 1, 4])

        shift = tf.tile(shift, [1, self.num_anchors, 1])

        anchors = tf.cast(anchors, tf.float32) + shift
        return tf.reshape(anchors, [-1, 4])

    def __box_ious(self, boxes_a, boxes_b):
        """ Calculate intersection over union of two bounding boxes """
        with tf.variable_scope('box_ious'):
            xA = tf.maximum(boxes_a[:, 1],
                            boxes_b[:, 1], name="xA")
            yA = tf.maximum(boxes_a[:, 0],
                            boxes_b[:, 0], name="yA")
            xB = tf.minimum(boxes_a[:, 3],
                            boxes_b[:, 3], name="xB")
            yB = tf.minimum(boxes_a[:, 2],
                            boxes_b[:, 2], name="yB")

            with tf.variable_scope('intersection_area'):
                intersectionArea = tf.maximum(0.0, (xB - xA)) * tf.maximum(
                    0.0, (yB - yA))
            with tf.variable_scope('box_area'):
                boxesAArea = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
                boxesBArea = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

            with tf.variable_scope('iou'):
                ious = intersectionArea / (
                    boxesAArea + boxesBArea - intersectionArea)

            return ious

    def __scale_to_range(self, x, min_val, max_val):
        current_min = tf.reduce_min(x)
        current_max = tf.reduce_max(x)

        # scale to [0; 1]
        x = (x - current_min) / (current_max - current_min + 1e-10)

        # scale to [min_val; max_val]
        x = x * (max_val - min_val) + min_val
        return x

    def __create_anchors(self, sizes, aspects):
        """ Creates the anchors of the shape (feature_size, feature_size,
        len(sizes) * len(aspects) * 4)"""
        k = len(sizes) * len(aspects)
        img_anchors = []
        for i in sizes:
            for j in aspects:
                img_anchors.append(
                    [0, 0, 2 * i * j[0] / (j[0] + j[1]), 2 * i * j[1] / (j[0] + j[1])])

        self.anchors = np.asarray(img_anchors)

    def __batch_norm_wrapper(self, inputs, decay=0.999, shape=None):
        """ Batch Normalization """
        if shape is None:
            shape = [0]

        epsilon = 1e-3
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name="scale")
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name="beta")
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               trainable=False, name="pop_mean")
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              trainable=False, name="pop_var")

        if self.is_training:
            batch_mean, batch_var = tf.nn.moments(
                inputs, shape, name="moments")
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

    def __scale_bboxes(self, bboxes, scale_x, scale_y):
        """ Scales a list of bounding boxes (m,4) according to the scale
        factors"""
        with tf.variable_scope('scale_bboxes'):
            return tf.multiply(bboxes, tf.tile([[scale_y, scale_x, scale_y,
                                                 scale_x]],
                                               [tf.shape(bboxes)[0], 1]))

    def clip_bboxes(self, bboxes, max_width, max_height):
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

    def adjust_bbox(self, deltas, boxes):
        new_x = tf.reshape(deltas[:, 1] * boxes[:, 3] + boxes[:, 1], [-1, 1])
        new_y = tf.reshape(deltas[:, 0] * boxes[:, 2] + boxes[:, 0], [-1, 1])
        new_w = tf.reshape(deltas[:, 3] * boxes[:, 3], [-1, 1])
        new_h = tf.reshape(deltas[:, 2] * boxes[:, 2], [-1, 1])

        new_boxes = tf.concat([new_y, new_x, new_h, new_w], axis=1)

        return new_boxes

    def bounding_box_loss(self, predicted_boxes, label_boxes, anchors):
        """ Calculate the loss for predicted and ground truth boxes. Boxes
        should all be (n,4), and weights should be (n) """

        yp, xp, hp, wp = tf.split(predicted_boxes, 4, axis=1)
        yl, xl, hl, wl = tf.split(label_boxes, 4, axis=1)
        ya, xa, ha, wa = tf.split(anchors, 4, axis=1)

        tx = (xp - xa) / wa
        ty = (yp - ya) / ha
        tw = tf.log(wp / wa)
        th = tf.log(hp / ha)

        tlx = (xl - xa) / wa
        tly = (yl - ya) / ha
        tlw = tf.log(wl / wa)
        tlh = tf.log(hl / ha)

        t = tf.concat([ty, tx, th, tw], axis=1)
        tl = tf.concat([tly, tlx, tlh, tlw], axis=1)

        loss = self.__smooth_l1_loss(tl, t)

        return loss

    def __smooth_l1_loss(self, label_regions, predicted_regions):
        """Smooth/Robust l1 loss"""

        tensor = tf.abs(predicted_regions - label_regions)

        loss = tf.where(tensor < 1, x=tf.square(tensor) / 2, y=tensor - 0.5)

        return loss

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

        # scale to [0, 255.0]
        # x_min = tf.reduce_min(x7, axis=3)
        # x_max = tf.reduce_max(x7, axis=3)
        # x7 = 255.0 * (x7 - x_min) / (x_max - x_min)

        # return x8
        return x7

    def __put_bboxes_on_image(self, images, boxes, scale_x, scale_y):

        output = []
        bboxes = tf.reshape(boxes, [1, -1, 4])

        num_boxes = tf.shape(bboxes)[1]

        bboxes = bboxes * tf.expand_dims(tf.tile([[scale_y, scale_x, scale_y, scale_x]],
                                                    [num_boxes, 1]), axis=0)

        shape = tf.shape(bboxes)
        bboxes = self.clip_bboxes(tf.reshape(bboxes, [-1, 4]), 1.0, 1.0)
        bboxes = self.bboxes_to_yxyx(bboxes, 1.0)
        bboxes = tf.reshape(bboxes, [1, -1, 4])
        bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

        return tf.image.draw_bounding_boxes(images, bboxes)

    def __grad_cam(self, loss, images, bbox, feature_width, feature_height, image_width, image_height):
        # not regular Grad-CAM, we need to adjust and mask the image since the
        # gradients go through ROI pooling and don't apply to the whole image

        with self.graph.gradient_override_map({'Elu': 'GuidedElu'}):
            image = tf.split(images, 1, axis=0)[0]
            target_conv_layer_grad = tf.reshape(tf.split(tf.gradients(loss, self.last_conv_layer)[
                                                0], 1, axis=0)[0], [feature_height, feature_width, self.conv_feature_count])

            h_scale = feature_height / image_height
            w_scale = feature_width / image_width

            bbox = bbox * tf.cast(tf.concat([[h_scale], [w_scale], [h_scale], [w_scale]], 0), tf.float32)

            x = tf.minimum(tf.maximum(
                tf.cast(bbox[1] - bbox[3] / 2, tf.int32), 0), feature_width - 1)
            y = tf.minimum(tf.maximum(
                tf.cast((bbox[0] - bbox[2] / 2), tf.int32), 0), feature_height - 1)
            w = tf.maximum(tf.minimum(
                tf.cast(bbox[3], tf.int32), feature_width - x), 1)
            h = tf.maximum(tf.minimum(
                tf.cast(bbox[2], tf.int32), feature_height - y), 1)

            target_conv_layer_grad = tf.image.crop_to_bounding_box(
                target_conv_layer_grad, y, x, h, w)

            grads_val = tf.div(target_conv_layer_grad, tf.sqrt(
                tf.reduce_mean(tf.square(target_conv_layer_grad))) + tf.constant(1e-5))
            output = tf.reshape(tf.split(self.last_conv_layer, 1, axis=0)[
                                0], [feature_height, feature_width, self.conv_feature_count])
            weights = tf.reduce_mean(grads_val, axis=[0, 1])
            cam = tf.ones([feature_height, feature_width],
                          dtype=tf.float32)

            weights = tf.tile(tf.reshape(
                weights, [1, 1, self.conv_feature_count]), [feature_height, feature_width, 1])

            cam += tf.reduce_sum(weights * output, axis=2)

            x = tf.range(tf.cast(bbox[1] - bbox[3] / 2, tf.int32),
                         tf.cast(bbox[1] + bbox[3] / 2, tf.int32))
            y = tf.range(tf.cast(bbox[0] - bbox[2] / 2, tf.int32),
                         tf.cast(bbox[0] + bbox[2] / 2, tf.int32))
            x_size = tf.size(x)
            x = tf.expand_dims(tf.tile(tf.expand_dims(
                x, axis=1), [1, tf.size(y)]), axis=2)
            y = tf.expand_dims(
                tf.tile(tf.expand_dims(y, axis=0), [x_size, 1]), axis=2)
            indices = tf.reshape(tf.concat([y, x], axis=2), [-1, 2])
            bbox_filter = tf.scatter_nd(
                indices, tf.ones([tf.shape(indices)[0]]), [feature_height, feature_width])
            cam = cam * bbox_filter
            # Passing through ReLU
            cam = tf.nn.relu(cam)  # tf.nn.elu(cam)
            cam = self.__scale_to_range(cam, 0.0, 1.0)
            cam = tf.image.resize_images(tf.reshape(cam, [feature_height, feature_width, 1]), [
                                         image_height, image_width])
            cam2 = tf.tile([cam], [1, 1, 1, 3])

            img = tf.image.resize_images(tf.cast(image, tf.float32), [
                                         image_height, image_width])
            img = self.__scale_to_range(img, 0.0, 1.0)

            grad_cam = (img * 0.5 + cam2 * 0.5)

            gb_grads = tf.split(tf.gradients(loss, images)[
                                0], 1, axis=0)[0]

            gb_viz = tf.image.resize_images(
                gb_grads, [image_height, image_width])

            gb_viz = tf.reshape(gb_viz, [image_height, image_width, 3])

            gb_viz = self.__scale_to_range(gb_viz, 0.0, 1.0)

            cam = tf.reshape(cam, [image_height, image_width])

            gd_gb = tf.reshape(gb_viz, [image_height, image_width, 3])

            gd_gb = tf.concat([
                tf.reshape(gb_viz[:, :, 0] * cam,
                           [image_height, image_width, 1]),
                tf.reshape(gb_viz[:, :, 1] * cam,
                           [image_height, image_width, 1]),
                tf.reshape(gb_viz[:, :, 2] * cam, [image_height, image_width, 1])], axis=2)

            return tf.concat([grad_cam, tf.reshape(gb_viz, [1, image_height, image_width, 3]), tf.reshape(gd_gb, [1, image_height, image_width, 3])], axis=0)

    def bboxes_to_yxyx(self, bboxes, max_height=None):
        """ Transforms coordinates to tensorflow coordinates
        since tensorflow image ops retardedly have (0,0) in the bottom left,
        even though tensors are indexed from the top left,
        we need to mirror y coordinates with the max_height parameter"""
        if max_height is not None:
            y, x, h, w = tf.split(bboxes, 4, axis=1)
            bboxes = tf.concat([max_height - (y + h / 2.0),
                                x - w / 2.0,
                                max_height - (y - h / 2.0),
                                x + w / 2.0],
                               axis=1)
        else:
            y, x, h, w = tf.split(bboxes, 4, axis=1)
            bboxes = tf.concat([y - h / 2.0,
                                x - w / 2.0,
                                y + h / 2.0,
                                x + w / 2.0],
                               axis=1)
        with tf.control_dependencies([tf.assert_greater_equal(bboxes[:, 2], bboxes[:, 0],
                                                              message="y1 > y2 in bboxes_to_yxyx",
                                                              summarize=100),
                                      tf.assert_greater_equal(bboxes[:, 3], bboxes[:, 1],
                                                              message="x1 > x2 in bboxes_to_yxyx",
                                                              summarize=100)]):
            bboxes = tf.identity(bboxes)
        return bboxes

    def __bboxes_xyxy_to_regular(self, bboxes):
        y1, x1, y2, x2 = tf.split(bboxes, 4, axis=1)
        bboxes = tf.concat([(y1 + y2) / 2.0, (x1 + x2) / 2.0,
                            y2 - y1,
                            x2 - x1],
                           axis=1)
        return bboxes

    def inference(self, inputs, conv_output=None):
        # resnet

        self.inputs = inputs

        if self.net == NetworkType.RESNET34 or self.net == NetworkType.RESNET50:
            with tf.variable_scope('first_layer'):
                kernel = tf.get_variable('weights', [7, 7, 3, 64],
                                         initializer=xavier_initializer(
                    dtype=tf.float32),
                    dtype=tf.float32)
                conv = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1],
                                    padding='SAME',
                                    name='conv')
                batch_norm = self.__batch_norm_wrapper(
                    conv, shape=[0, 1, 2, 3])
                conv = tf.nn.elu(batch_norm, 'elu')

                grid = self.__put_kernels_on_grid(kernel, (8, 8))
                tf.summary.image('conv1/features', grid, max_outputs=1)
                grid = self.__put_activations_on_grid(conv, (8, 8))
                tf.summary.image('conv1/activations', grid, max_outputs=1)

                conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME',
                                      name='pool')

        if self.net == NetworkType.RESNET34:
            inputs = self.__resnet34(conv)
        elif self.net == NetworkType.RESNET50:
            inputs = self.__resnet50(conv)
        elif self.net == NetworkType.PRETRAINED:
            inputs = conv_output

        # with tf.variable_scope('global_average_pool'):
        #     inputs = tf.reduce_mean(inputs, [1, 2])

        if self.restore_from_chkpt or self.net == NetworkType.PRETRAINED:
            # don't change weights if model is pretrained
            inputs = tf.stop_gradient(inputs)

        # get roi's
        with tf.variable_scope('region_proposal_network'):
            conv_cls, conv_regions = self.__region_proposals(inputs, self.conv_feature_count,
                                                             512, self.num_anchors)

            reg_shape = tf.shape(conv_regions)

            all_boxes = self.__process_rois(conv_regions, conv_cls)
            boxes_shape = tf.shape(all_boxes)

            with tf.control_dependencies([tf.assert_positive(all_boxes[:, 2],
                                                                message="height > 0",
                                                                summarize=100),
                                            tf.assert_positive(all_boxes[:, 3],
                                                                message="width > 0",
                                                                summarize=100)]):
                boxes = tf.identity(all_boxes)

            boxes = self.bboxes_to_yxyx(boxes) #, tf.cast(reg_shape[2], tf.float32))

            roi_indices = tf.tile([0], [tf.shape(boxes)[0]])

            inputs2 = inputs

            self.last_conv_layer = inputs2
            pooled_region = self.__roi_pooling(inputs2, boxes, roi_indices, 7, 7)

            num = np.power(2, np.floor(
                np.log2(self.conv_feature_count) / 2))

            grid = self.__put_activations_on_grid(pooled_region, (int(num),
                                                                    int(self.conv_feature_count /
                                                                        num)))
            tf.summary.image('roi_pooling', grid, max_outputs=15)

        # classify regions and add final region adjustments
        with tf.variable_scope('region_classification'):
            class_weights = tf.get_variable('class_weights',
                                            [self.conv_feature_count,
                                             self.num_classes + 1],
                                            initializer=xavier_initializer(uniform=True,
                                                                           dtype=tf.float32),
                                            dtype=tf.float32)
            class_bias = tf.get_variable("class_bias", [
                self.num_classes + 1],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)

            region_weights = tf.get_variable('region_weights',
                                             [self.conv_feature_count, self.num_classes *
                                              4],
                                             initializer=xavier_initializer(uniform=True,
                                                                            dtype=tf.float32),
                                             dtype=tf.float32)

            region_bias = tf.get_variable("region_bias", [
                self.num_classes * 4],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)

            class_scores = []
            region_scores = []

            with tf.variable_scope('f'):
                fc = tf.reduce_mean(pooled_region, [1, 2])

            with tf.variable_scope('rcn_class'):
                class_score = tf.matmul(fc, class_weights)
                class_score = tf.nn.bias_add(class_score, class_bias)

            with tf.variable_scope('rcn_region'):
                region_score = tf.matmul(fc, region_weights)
                region_score = tf.nn.bias_add(region_score, region_bias)

                region_score = tf.clip_by_value(region_score, -0.2, 0.2)
                shape = tf.shape(region_score)

                coords, size = tf.split(tf.reshape(region_score, [-1, 4]),
                                        2,
                                        axis=1)
                size = tf.exp(size)
                region_score = tf.reshape(tf.concat([coords, size], axis=1),
                                            shape)

        return class_score, region_score, conv_cls, conv_regions, all_boxes

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

        # with tf.variable_scope('reslayer_downsample_512'):
        #     inputs = self.__reslayer(inputs, 256, 512, stride=2)

        # for i in range(2):
        #     with tf.variable_scope('reslayer_512_%d' % i):
        #         inputs = self.__reslayer(inputs, 512, 512)
        return inputs

    def __resnet50(self, inputs):
        with tf.variable_scope('reslayer_downsample_256'):
            inputs = self.__reslayer_bottleneck(inputs, 64, 256)

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

        # with tf.variable_scope('reslayer_downsample_2048'):
        #     inputs = self.__reslayer_bottleneck(inputs, 1024, 2048, stride=2)

        # for i in range(2):
        #     with tf.variable_scope('reslayer_2048_%d' % i):
        #         inputs = self.__reslayer_bottleneck(inputs, 2048, 2048)
        return inputs

    def process_iou_score(self, conv_region, label_region):
        """ Calculates IoU scores for two lists of regions (m,4) and (n,4) """
        with tf.variable_scope('process_iou_score'):
            return self.__box_ious(tf.cast(conv_region, tf.float32),
                                   tf.cast(label_region, tf.float32))

    def loss(self, class_scores, region_scores, conv_cls, conv_regions, labels,
             label_regions, proposed_boxes, images):
        #conv_regions = tf.reshape(conv_regions, [self.batch_size, -1, 4])
        label_regions = tf.sparse_split(sp_input=label_regions,
                                        num_split=1, axis=0)
        labels = tf.sparse_split(sp_input=labels,
                                 num_split=1, axis=0)

        pop_iou_distribution = tf.Variable(tf.ones([self.num_anchors]),
                                           trainable=False)

        conv_regs = conv_regions
        conv_shape = tf.shape(conv_regs)
        images_shape = tf.shape(images)
        conv_regs = tf.reshape(conv_regs, [-1, 4])
        conv_classes = tf.reshape(conv_cls, [-1, 2])
        num_regions = tf.shape(conv_regs)[0]
        label_regs = label_regions[0]
        label = labels[0]
        label_regs = tf.reshape(
            tf.sparse_tensor_to_dense(label_regs), [-1, 4])
        label = tf.reshape(tf.sparse_tensor_to_dense(label,
                                                        default_value=-1),
                            [-1])

        mask = label >= 0

        label = tf.boolean_mask(label, mask)
        label_regs = tf.boolean_mask(label_regs, mask)

        num_labels = tf.shape(label_regs)[0]

        # If an image doesn't have annotations in MSCOCO, use dummy ones
        label = tf.cond(num_labels > 0, lambda: label, lambda: tf.constant([
            -1], dtype=tf.int64))
        label_regs = tf.cond(num_labels > 0, lambda: label_regs, lambda:
                                tf.constant([[0.0, 0.0, 0.01, 0.01]]))

        proposed_box = proposed_boxes
        proposed_box = tf.cond(tf.size(proposed_box) > 0,
                                lambda: proposed_box, lambda:
                                tf.constant([[0.0, 0.0, 0.01, 0.01]]))

        reg_scores = region_scores
        reg_scores = tf.cond(tf.size(reg_scores) > 0, lambda: reg_scores,
                                lambda: tf.constant([[0.0, 0.0, 0.01, 0.01]]))

        # calculate rpn loss
        with tf.variable_scope('rpn_loss'):
            label_regs2 = tf.reshape(tf.tile(label_regs,
                                                [num_regions, 1]),
                                        [num_regions, num_labels, 4])

            feat_anchors = self.get_tiled_anchors_for_shape(
                conv_shape[2] * self.feat_stride, conv_shape[1] * self.feat_stride)

            tiled_anchors = tf.tile(tf.cast(tf.reshape(
                feat_anchors, [-1, 1, 4]),
                tf.float32), [1, num_labels, 1])

            ious = self.process_iou_score(self.bboxes_to_yxyx(tf.reshape(
                tiled_anchors, [-1, 4])), self.bboxes_to_yxyx(tf.reshape(label_regs2, [-1, 4])))

            ious = tf.reshape(ious, [num_regions, num_labels])

            ious = tf.stop_gradient(ious)

            highest_overlap = tf.reshape(tf.argmax(ious, axis=0), [-1, 1])

            highest_overlap = tf.concat([highest_overlap, tf.cast(tf.reshape(tf.range(tf.size(highest_overlap)), [-1, 1]),
                                                                    tf.int64)], axis=1)

            highest_overlap_delta = tf.scatter_nd(highest_overlap, tf.tile(
                [0.70001], [num_labels]), tf.cast([num_regions, num_labels], tf.int64))

            ious = ious + highest_overlap_delta

            max_ious = tf.reduce_max(ious, axis=1)
            max_ids = tf.argmax(ious, axis=1)

            # mask regions
            label_regs3 = label_regs
            target_regions = tf.gather(label_regs3, max_ids)

            target_mask = tf.where(max_ious > 0.3,
                                    tf.where(max_ious > 0.7,
                                            tf.tile([True], [
                                                num_regions]),
                                            tf.tile([False], [
                                                num_regions])),
                                    tf.tile([True], [num_regions]))

            target_mask = tf.logical_and(
                tf.cast(target_mask, tf.bool), tf.reshape(self.outside_anchors, [-1]))

            conv_classes = tf.boolean_mask(conv_classes, target_mask)

            target_labels = tf.where(max_ious > 0.7,
                                        tf.tile([[0, 1]],
                                                        [
                                                            num_regions, 1]),
                                        tf.tile([[1, 0]],
                                                        [num_regions, 1]))
            target_labels = tf.boolean_mask(tf.reshape(
                tf.cast(target_labels, tf.float32),
                [-1, 2]),
                target_mask)

            # sample positive/negatives to balance classes
            max_ious = tf.boolean_mask(max_ious, target_mask)
            positives = tf.cast(max_ious > 0.7, tf.int32)
            negatives = tf.cast(max_ious < 0.3, tf.int32)

            rpn_num_positives = tf.reduce_sum(positives)

            rand = tf.random_uniform(tf.shape(max_ious))

            _, pos_mask_idx = tf.nn.top_k(
                tf.cast(positives, tf.float32) * rand, k=tf.minimum(rpn_num_positives, 128))
            pos_mask = tf.scatter_nd(tf.reshape(pos_mask_idx, [-1, 1]), tf.tile(
                [1], [tf.size(pos_mask_idx)]), [tf.size(max_ious)])
            pos_mask = tf.cast(pos_mask, tf.bool)

            _, neg_mask_idx = tf.nn.top_k(
                tf.cast(negatives, tf.float32) * rand, k=tf.minimum(256, tf.maximum(2, rpn_num_positives * 2)) - tf.minimum(rpn_num_positives, 128))
            neg_mask = tf.scatter_nd(tf.reshape(neg_mask_idx, [-1, 1]), tf.tile(
                [1], [tf.size(neg_mask_idx)]), [tf.size(max_ious)])
            neg_mask = tf.cast(neg_mask, tf.bool)

            sample_mask = tf.logical_or(pos_mask, neg_mask)

            conv_classes2 = tf.boolean_mask(conv_classes, sample_mask)
            target_labels = tf.boolean_mask(target_labels, sample_mask)

            with tf.variable_scope('label_loss'):
                conv_labels_loss = \
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=target_labels, logits=conv_classes2)

                conv_labels_loss = tf.reduce_mean(conv_labels_loss,
                                                    name="conv_label_loss")

                correct_prediction = tf.equal(tf.argmax(conv_classes, 1),
                                                tf.where(max_ious > 0.7,
                                                        tf.ones_like(
                                                            max_ious,
                                                            dtype=tf.int64),
                                                        tf.zeros_like(
                                                            max_ious,
                                                            dtype=tf.int64)))
                correct_prediction = tf.boolean_mask(correct_prediction,
                                                        max_ious > 0.7)
                rpn_accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32))

            with tf.variable_scope('region_loss'):

                conv_regs = self.adjust_bbox(conv_regs, tf.cast(
                    tf.reshape(
                        feat_anchors, [-1, 4]),
                    tf.float32))

                # clip bboxes because sometimes a bounding box bigger than
                # the image is correct bould would lead to a big loss
                conv_regs = self.clip_bboxes(conv_regs, tf.cast(
                    images_shape[2], tf.float32), tf.cast(images_shape[1], tf.float32))

                sample_mask2 = pos_mask

                conv_regs2 = tf.boolean_mask(conv_regs, target_mask)

                conv_regs2 = tf.reshape(tf.boolean_mask(
                    conv_regs2, sample_mask2), [-1, 4])

                target_regions2 = tf.boolean_mask(tf.cast(target_regions, tf.float32), target_mask)
                target_regions2 = tf.reshape(tf.boolean_mask(
                    target_regions2, sample_mask2), [-1, 4])

                anchors = tf.cast(tf.reshape(feat_anchors, [-1, 4]),
                                    tf.float32)
                anchors = tf.boolean_mask(anchors, target_mask)
                anchors = tf.boolean_mask(anchors, sample_mask2)

                conv_region_loss = self.bounding_box_loss(
                    conv_regs2,
                    target_regions2,
                    anchors)

                conv_region_loss = tf.reduce_mean(conv_region_loss, axis=1)

                conv_region_loss = tf.cond(tf.equal(tf.size(
                    conv_region_loss), 0), lambda: tf.constant(0.0),
                    lambda: tf.reduce_mean(conv_region_loss))

            idx = tf.argmax(conv_classes, dimension=1)

            _, prediction_values = tf.split(conv_classes, 2, axis=1)

            # filter regions with background class predicted
            mask = tf.reshape(prediction_values > 0.7, [-1])

            prediction_values = tf.cond(tf.size(prediction_values) > 0,
                                        lambda: tf.boolean_mask(
                                            prediction_values, mask),
                                        lambda: tf.constant([0.0]))

            proposed_regions = tf.cond(tf.size(prediction_values) > 0,
                                        lambda: tf.boolean_mask(tf.boolean_mask(
                                            conv_regs, target_mask), mask),
                                        lambda: tf.constant([[0.0,
                                                                0.0,
                                                                0.0, 0.0]]))

            _, ids = tf.nn.top_k(tf.reshape(prediction_values, [-1]), k=tf.minimum(
                tf.cast(tf.size(
                    prediction_values) / 4, tf.int32), 10))

            proposed_regions = tf.gather(proposed_regions, ids)
            proposed_regions = tf.cond(tf.size(idx) > 0, lambda:
                                        proposed_regions, lambda: tf.constant([0.0, 0.0, 0.0, 0.0]))
            rpn_proposed_regions = proposed_regions
            conv_region_losses = tf.where(tf.logical_or(tf.is_nan(conv_region_loss), tf.logical_or(tf.is_inf(conv_region_loss), conv_region_loss == 0)), 1e-5, conv_region_loss)

        # calculate rcnn loss
        with tf.variable_scope('rcnn_loss'):
            proposed_box2 = tf.expand_dims(proposed_box, axis=1)

            background = tf.zeros([self.num_classes])
            background = tf.reshape(
                tf.concat([[1], background], axis=0),
                [self.num_classes + 1])

            cls_scores = class_scores
            num_regions = tf.shape(proposed_box)[0]
            regions = tf.reshape(reg_scores, [num_regions, -1, 4])

            num_labs = tf.shape(label_regs3)[0]

            label_regs3 = tf.reshape(tf.tile(label_regs3, [num_regions, 1]),
                                        [num_regions, num_labs, 4])

            regions2 = tf.reshape(tf.tile(proposed_box2, [1, num_labs, 1]),
                                    [num_regions, num_labs, 4])

            ious = self.process_iou_score(self.bboxes_to_yxyx(tf.reshape(
                regions2, [-1, 4])), self.bboxes_to_yxyx(tf.reshape(label_regs3, [-1, 4])))

            ious = tf.reshape(ious, [num_regions, num_labs])

            ious = tf.stop_gradient(ious)

            max_ious = tf.reduce_max(ious, axis=1)
            max_ids = tf.cast(tf.argmax(ious, axis=1), tf.int32)

            # get labels
            target_labels = tf.gather(label, max_ids)

            target_labels2 = tf.where(max_ious > 0.5,
                                        tf.one_hot(target_labels + 1,
                                                    self.num_classes + 1),
                                        tf.tile([background],
                                                            [num_regions, 1]))

            # get regions
            target_mask = tf.where(max_ious > 0.5,
                                    tf.tile([True], [
                                        num_regions]),
                                    tf.tile([False], [
                                        num_regions]))
            target_regions = tf.gather_nd(label_regs3, tf.concat([
                tf.reshape(tf.range(num_regions), [-1, 1]),
                tf.reshape(max_ids, [-1, 1])],
                axis=1))
            region_idx = tf.concat([
                tf.cast(tf.reshape(tf.range(num_regions), [-1, 1]),
                        tf.int64),
                tf.reshape(tf.maximum(target_labels, 0), [-1, 1])],
                axis=1)
            region_adjustment = tf.gather_nd(regions, region_idx)
            region_adjustment = tf.reshape(region_adjustment, [-1, 4])

            proposed_regions = self.adjust_bbox(region_adjustment,
                                                    proposed_box)

            # sample positive/negatives to balance classes
            positives = tf.cast(max_ious > 0.5, tf.int32)
            negatives = tf.cast(tf.logical_and(
                max_ious < 0.5, max_ious > 0.1), tf.int32)
            rcnn_num_positives = tf.reduce_sum(positives)
            num_tot = tf.maximum(4, tf.cast(tf.minimum(3 * rcnn_num_positives / 2, 256), tf.int32))

            rand = tf.random_uniform(tf.shape(max_ious))

            _, pos_mask_idx = tf.nn.top_k(
                tf.cast(positives, tf.float32) * rand, k=tf.minimum(rcnn_num_positives, tf.cast(3 * num_tot / 2, tf.int32)))
            pos_mask = tf.scatter_nd(tf.reshape(pos_mask_idx, [-1, 1]), tf.tile(
                [1], [tf.size(pos_mask_idx)]), [tf.size(max_ious)])
            pos_mask = tf.cast(pos_mask, tf.bool)

            _, neg_mask_idx = tf.nn.top_k(
                tf.cast(negatives, tf.float32) * rand, k=num_tot - tf.minimum(rcnn_num_positives, tf.cast(3 * num_tot / 2, tf.int32)))
            neg_mask = tf.scatter_nd(tf.reshape(neg_mask_idx, [-1, 1]), tf.tile(
                [1], [tf.size(neg_mask_idx)]), [tf.size(max_ious)])
            neg_mask = tf.cast(neg_mask, tf.bool)

            sample_mask = tf.logical_or(pos_mask, neg_mask)

            positive_propositions = tf.boolean_mask(proposed_box, pos_mask)

            cls_scores2 = tf.boolean_mask(cls_scores, sample_mask)
            target_labels3 = tf.boolean_mask(target_labels2, sample_mask)

            with tf.variable_scope('label_loss'):
                rcnn_label_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.cast(target_labels3, tf.float32), logits=cls_scores2)

                loss_idx = tf.argmax(target_labels3, axis=1)
                score = target_labels3 * cls_scores2
                grad_loss2 = tf.boolean_mask(
                    tf.reduce_mean(score, axis=1), loss_idx > 0)
                grad_loss3 = tf.reduce_max(grad_loss2)

                grad_idx = tf.argmax(tf.cond(tf.size(
                    grad_loss2) > 0, lambda: grad_loss2, lambda: tf.reduce_mean(score, axis=1)))
                boxes = tf.cond(tf.size(grad_loss2) > 0, lambda: tf.boolean_mask(tf.boolean_mask(
                    proposed_box, sample_mask), loss_idx > 0), lambda: tf.boolean_mask(proposed_box, sample_mask))[tf.cast(grad_idx, tf.int32)]

                min_tot = tf.reduce_max(tf.reduce_mean(score, axis=1))

                grad_loss = tf.cond(
                    tf.size(grad_loss3) > 0, lambda: grad_loss3, lambda: min_tot)

                grad_cam_image = self.__grad_cam(
                    grad_loss, self.inputs, boxes, conv_shape[2], conv_shape[1], images_shape[2], images_shape[1])

                rcnn_label_loss = tf.reduce_mean(rcnn_label_loss)

                correct_prediction = tf.equal(tf.argmax(cls_scores, 1),
                                                target_labels + 1)
                rcnn_accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32))

            with tf.variable_scope('region_loss'):
                target_regions = tf.boolean_mask(target_regions, target_mask)
                proposed_regions2 = tf.boolean_mask(proposed_regions, target_mask)
                l2_loss = self.__smooth_l1_loss(target_regions,
                                                proposed_regions2)

                l2_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=1))

                rcnn_loss = tf.where(tf.logical_or(tf.is_nan(
                    l2_loss), tf.logical_or(tf.size(l2_loss) == 0,
                                            num_labels == 0)), 0.0,
                    l2_loss)

            cls_scores = tf.nn.softmax(cls_scores)

            idx = tf.argmax(cls_scores, dimension=1)

            # filter regions with background class predicted
            mask = idx > 0

            prediction_values = tf.gather_nd(cls_scores, tf.concat([
                tf.cast(tf.reshape(tf.range(tf.size(idx)), [-1, 1]),
                        tf.int64), tf.reshape(idx, [-1, 1])],
                axis=1))

            prediction_values = tf.boolean_mask(prediction_values,
                                                mask)

            prediction_values = tf.cond(tf.size(prediction_values) > 0,
                                        lambda: prediction_values,
                                        lambda: tf.constant([0.0]))

            proposed_regions = tf.cond(tf.size(prediction_values) > 0,
                                        lambda: tf.boolean_mask(
                                            proposed_regions, mask),
                                        lambda: tf.constant([[0.0,
                                                                0.0,
                                                                0.0, 0.0]]))

            _, ids = tf.nn.top_k(prediction_values, k=tf.minimum(
                tf.cast(tf.size(
                    prediction_values), tf.int32), 30))
            proposed_regions = tf.gather(proposed_regions, ids)
            proposed_regions = tf.cond(tf.size(idx) > 0, lambda:
                                        proposed_regions, lambda: tf.constant([0.0, 0.0, 0.0, 0.0]))
            final_proposed_regions = proposed_regions
            positive_propositions = tf.cond(tf.shape(positive_propositions)[0] > 0, lambda: positive_propositions, lambda: tf.constant([[0.0, 0.0, 0.0, 0.0]]))
            positive_proposed_regions = positive_propositions

        # if no box was proposed -> no loss
        rcnn_label_loss = tf.where(tf.shape(proposed_box)[0]
                                    == 0,
                                    1e-5, rcnn_label_loss)
        rcnn_loss = tf.where(tf.shape(proposed_box)[0] == 0,
                                1e-5, rcnn_loss)

        images_shape = tf.cast(images_shape, tf.float32)

        rpn_images = self.__put_bboxes_on_image(images,
                                                rpn_proposed_regions, scale_x=1.0 / images_shape[2], scale_y=1.0 / images_shape[1])

        tf.summary.image("rpn_bbox_predictions", rpn_images, max_outputs=16)

        final_images = self.__put_bboxes_on_image(images,
                                            final_proposed_regions, scale_x=1.0 / images_shape[2], scale_y=1.0 / images_shape[1])

        tf.summary.image("bbox_predictions", final_images, max_outputs=16)

        positive_images = self.__put_bboxes_on_image(images,
                                            positive_proposed_regions, scale_x=1.0 / images_shape[2], scale_y=1.0 / images_shape[1])

        tf.summary.image("positive_propositions", positive_images, max_outputs=16)

        tf.summary.image("grad_cam_images", grad_cam_image, max_outputs=16)
        
        anchors = self.get_tiled_anchors_for_shape(images_shape[2], images_shape[1])
        anchors = anchors + [[images_shape[1], images_shape[2], 0.0, 0.0]]
        anchor_images = tf.image.resize_image_with_crop_or_pad(images, tf.cast(images_shape[1] * 3, tf.int32), tf.cast(images_shape[2] * 3, tf.int32))
        anchor_images = self.__put_bboxes_on_image(anchor_images,
                                            anchors, scale_x=0.333 / images_shape[2], scale_y=0.333 / images_shape[1])

        

        tf.summary.image("anchors", anchor_images, max_outputs=16)

        anchors = self.get_tiled_anchors_for_shape(images_shape[2], images_shape[1])
        anchor_mask = tf.logical_and(tf.logical_and(anchors[:, 0] - anchors[:, 2] / 2.0 >= -self.feat_stride, anchors[:, 1] - anchors[:, 3] / 2.0 >= -self.feat_stride), tf.logical_and(
                anchors[:, 0] + anchors[:, 2] / 2.0 < tf.cast(images_shape[1], tf.float32) + self.feat_stride, anchors[:, 1] + anchors[:, 3] / 2.0 < tf.cast(images_shape[2], tf.float32) + self.feat_stride))
        anchors = tf.boolean_mask(anchors, anchor_mask)
        anchors = anchors + [[images_shape[1], images_shape[2], 0.0, 0.0]]
        anchor_images = tf.image.resize_image_with_crop_or_pad(images, tf.cast(images_shape[1] * 3, tf.int32), tf.cast(images_shape[2] * 3, tf.int32))
        anchor_images = self.__put_bboxes_on_image(anchor_images,
                                            anchors, scale_x=0.333 / images_shape[2], scale_y=0.333 / images_shape[1])

        tf.summary.image("anchors_cropped", anchor_images, max_outputs=16)

        tf.summary.scalar('rcnn/num_positives',
                          tf.reduce_mean(rcnn_num_positives))
        tf.summary.scalar('rpn/num_positives',
                          tf.reduce_mean(rpn_num_positives))

        conv_labels_losses = tf.reduce_mean(conv_labels_loss,
                                            name="conv_labels_loss")
        conv_region_losses = tf.cond(tf.equal(tf.reduce_mean(conv_region_losses,
                                                             name="conv_region_loss"), 0.0), lambda: tf.constant(1e-5), lambda:
                                     tf.reduce_mean(conv_region_losses,
                                                    name="conv_region_loss"))
        rcnn_label_losses = tf.cond(tf.equal(tf.reduce_mean(
            rcnn_label_loss), 0.0), lambda: tf.constant(1e-5), lambda:
            tf.reduce_mean(rcnn_label_loss, name="rcnn_label_loss"))
        rcnn_losses = tf.cond(tf.equal(tf.reduce_mean(
            rcnn_loss), 0.0), lambda: tf.constant(1e-5), lambda:
            tf.reduce_mean(rcnn_loss, name="rcnn_region_loss"))

        tf.add_to_collection('losses', tf.identity(self.rpn_cls_loss_weight *
                                                   conv_labels_losses,
                                                   name="conv_labels_losses"))
        tf.add_to_collection('losses', tf.identity(self.rpn_reg_loss_weight *
                                                   conv_region_losses,
                                                   name="conv_region_losses"))
        tf.add_to_collection('losses', tf.identity(self.rcnn_cls_loss_weight *
                                                   rcnn_label_losses,
                                                   name="rcnn_label_losses"))
        tf.add_to_collection('losses', tf.identity(self.rcnn_reg_loss_weight *
                                                   rcnn_losses,
                                                   name="rcnn_losses"))

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), \
            rcnn_accuracy, rpn_accuracy

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
        variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies(
                [apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op
