import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import xavier_initializer
from roi_pooling import roi_pooling


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

        self.interactive_sess = tf.InteractiveSession()

        self.__create_anchors(600, 19, (128, 256, 512),
                              ((2, 1), (1, 1), (1, 2)))

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
                inputs = tf.nn.avg_pool(inputs, (1, stride, stride, 1),
                                        (1, stride, stride, 1), 'SAME')
                inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                                         [(out_filters - in_filters) // 2,
                                          (out_filters - in_filters) // 2]])
            bias += inputs
            conv = tf.nn.elu(bias, 'elu')

            num = np.power(2, np.floor(np.log2(out_filters) / 2))

            grid = self.__put_activations_on_grid(conv, (int(num),
                                                         int(out_filters /
                                                             num)))
            tf.summary.image('sub2/activations', grid, max_outputs=1)

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

        with tf.variable_scope('cls_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 2 * k],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv_cls = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv')
            # batch_norm = self.__batch_norm_wrapper(conv_cls, shape=[0, 1, 2, 3])
            # conv_cls = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('reg_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 4 * k],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv_regions = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv')
            # batch_norm = self.__batch_norm_wrapper(conv_regions,
            #                                        shape=[0, 1, 2, 3])
            # conv_regions = tf.nn.relu(batch_norm, 'relu')

            conv_regions = tf.clip_by_value(conv_regions, -0.2, 0.2)

            coords, size = tf.split(tf.reshape(conv_regions, [-1, 4]), 2,
                                    axis=1)

            size = tf.exp(size)

            conv_regions = tf.reshape(tf.concat([coords, size], axis=1),
                                      [self.batch_size, 19, 19, 4 * k])

        return conv_cls, conv_regions

    def __roi_pooling(self, inputs, boxes, pool_height, pool_width):

        roi_pool = roi_pooling(inputs, boxes, pool_height, pool_width)

        return roi_pool

    def __process_rois(self, regions, class_scores):
        """ get relevant regions, with non-maximum suppression and clipping """
        region_boxes = tf.reshape(self.__adjust_bbox(tf.reshape(regions,
                                                                [-1, 4]),
                                                     tf.cast(
                                                         np.repeat(
                                                             self.feat_anchors.reshape(
                                                                 1,
                                                                 -1,
                                                                 4),
                                                             self.batch_size,
                                                             axis=0).reshape(-1,
                                                                             4),
                                                         tf.float32)),
                                  [self.batch_size,
                                   -1, 4])

        class_scores = tf.reshape(class_scores, (self.batch_size, 19, 19, 9, 2))

        if self.is_training:
            # ignore boxes that cross the image boundary
            region_boxes = tf.reshape(tf.boolean_mask(region_boxes,
                                                      tf.tile(tf.cast(
                                                          self.outside_feat_anchors.reshape(
                                                              1, -1, 4),
                                                          tf.bool),
                                                          [self.batch_size, 1,
                                                           1])),
                                      [self.batch_size, -1, 4])
            class_scores = tf.reshape(tf.boolean_mask(class_scores,
                                                      tf.tile(tf.cast(
                                                          self.outside_score_anchors.reshape(
                                                              1, 19, 19, 9, 2),
                                                          tf.bool),
                                                          [self.batch_size, 1,
                                                           1,
                                                           1, 1])),
                                      [self.batch_size, -1, 2])

        # filter too small boxes
        region_boxes = tf.unstack(region_boxes)
        class_scores = tf.unstack(class_scores)

        bbox_list = []

        for i, region_box in enumerate(region_boxes):
            class_score = class_scores[i]
            shape = tf.shape(region_box)
            mask = tf.where(tf.logical_or(region_box[:, 2] <
                                          self.min_box_size,
                                          region_box[:,
                                          3] < self.min_box_size), tf.tile([
                False], [shape[0]]), tf.tile([True],
                                             [shape[0]]))
            region_box = tf.boolean_mask(region_box,
                                         mask)
            class_score = tf.reshape(tf.boolean_mask(class_score,
                                                     mask),
                                     [-1, 2])

            class_score = tf.nn.softmax(class_score)

            # since we don't know the exact dimensions due to possible masking,
            # just flatten everything and and use the scores/boxes as a list
            class_score = tf.reshape(class_score, [-1, 2])
            region_box = tf.reshape(region_box, [-1, 4])

            #class_score = tf.split(class_score, 2, axis=1)[1]
            class_score = tf.slice(class_score, [0, 1], [-1, 1])
            # ignore taking top k regions for now
            # # get top regions
            # indices = np.argpartition(tf.split(class_scores, 2, axis=4)[1],
            #                           -6000)[-6000:]
            # region_boxes = region_boxes[indices]
            # class_scores = class_scores[indices]

            with tf.variable_scope('non_maximum_supression'):
                # filter boxes to have class score > 0.5
                filter_indices = tf.where(tf.greater(tf.squeeze(class_score), 0.5))
                bboxes = tf.reshape(tf.gather(region_box, filter_indices),
                                    [-1, 4])
                cur_class_scores = tf.reshape(tf.gather(class_score,
                                                        filter_indices), [-1])

                bboxes2 = self.__bboxes_to_xyxy(bboxes)
                idx = tf.image.non_max_suppression(bboxes2, cur_class_scores,
                                                   256,
                                                   0.7)
                bbox_list.append(tf.reshape(tf.gather(bboxes, idx), [-1, 4]))

        return bbox_list

    def __box_ious(self, boxes_a, boxes_b):
        """ Calculate intersection over union of two bounding boxes """
        with tf.variable_scope('box_ious'):
            xA = tf.maximum(boxes_a[:, 0] - boxes_a[:, 2] / 2,
                            boxes_b[:, 0] - boxes_b[:, 2] / 2, name="xA")
            yA = tf.maximum(boxes_a[:, 1] - boxes_a[:, 3] / 2,
                            boxes_b[:, 1] - boxes_b[:, 3] / 2, name="yA")
            xB = tf.minimum(boxes_a[:, 0] + boxes_a[:, 2] / 2,
                            boxes_b[:, 0] + boxes_b[:, 2] / 2, name="xB")
            yB = tf.minimum(boxes_a[:, 1] + boxes_a[:, 3] / 2,
                            boxes_b[:, 1] + boxes_b[:, 3] / 2, name="yB")

            with tf.variable_scope('intersection_area'):
                intersectionArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(
                    0.0, (yB - yA + 1))
            with tf.variable_scope('box_area'):
                boxesAArea = boxes_a[:, 2] * boxes_a[:, 3]
                boxesBArea = boxes_b[:, 2] * boxes_b[:, 3]

            with tf.variable_scope('iou'):
                ious = intersectionArea / tf.cast(
                    boxesAArea + boxesBArea - intersectionArea, \
                    tf.float32)

            return ious

    def __create_anchors(self, image_size, feature_size, sizes, aspects):
        """ Creates the anchors of the shape (feature_size, feature_size,
        len(sizes) * len(aspects) * 4)"""
        k = len(sizes) * len(aspects)
        img_anchors = []
        for i in sizes:
            for j in aspects:
                img_anchors.append(
                    [i * j[0] / (j[0] + j[1]), i * j[1] / (j[0] + j[1])])

        img_anchors = np.asarray(img_anchors)

        anchors = img_anchors * feature_size / image_size

        feat_sizes = np.tile(anchors, (feature_size,
                                       feature_size, 1, 1))
        img_sizes = np.tile(img_anchors, (feature_size, feature_size, 1, 1))

        x_coords = np.array(range(feature_size))
        img_x_coords = x_coords * image_size / feature_size

        feat_coords = np.tile(np.array(np.meshgrid(x_coords, x_coords)).T,
                              (1, 1, k)).reshape(feature_size, feature_size,
                                                 k, 2)
        img_coords = np.tile(
            np.array(np.meshgrid(img_x_coords, img_x_coords)).T,
            (1, 1, k)).reshape(feature_size, feature_size,
                               k, 2)

        self.feat_anchors = np.concatenate((feat_coords, feat_sizes), axis=3)
        self.img_anchors = np.concatenate((img_coords, img_sizes), axis=3)

        outside_anchors = np.ones((feature_size, feature_size, k))

        outside_anchors[np.where(
            self.feat_anchors[..., 0] - (self.feat_anchors[..., 2] / 2) < 0)] \
            = 0
        outside_anchors[np.where(
            self.feat_anchors[..., 1] - self.feat_anchors[..., 3] / 2 < 0)] = 0
        outside_anchors[np.where(self.feat_anchors[..., 0]
                                 + self.feat_anchors[..., 2] / 2 >
                                 feature_size)] = 0
        outside_anchors[np.where(self.feat_anchors[..., 1]
                                 + self.feat_anchors[..., 3] / 2 >
                                 feature_size)] = 0

        self.outside_score_anchors = np.tile(outside_anchors,
                                             (1, 1, 1, 2))

        self.outside_feat_anchors = self.outside_image_anchors = np.tile(
            outside_anchors, (1, 1, 1, 4))

        self.masked_anchors = np.ma.masked_array(self.feat_anchors,
                                                 mask=self.outside_feat_anchors)

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

    def __scale_bboxes(self, bboxes, scale_x, scale_y):
        """ Scales a list of bounding boxes (m,4) according to the scale
        factors"""
        with tf.variable_scope('scale_bboxes'):
            return tf.multiply(bboxes, tf.tile([[scale_x, scale_y, scale_x,
                                                 scale_y]],
                                               [tf.shape(bboxes)[0], 1]))

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

    def __adjust_bbox(self, deltas, boxes):
        new_x = tf.reshape(deltas[:, 0] * boxes[:, 2] + boxes[:, 0], [-1, 1])
        new_y = tf.reshape(deltas[:, 1] * boxes[:, 3] + boxes[:, 1], [-1, 1])
        new_w = tf.reshape(deltas[:, 2] * boxes[:, 2], [-1, 1])
        new_h = tf.reshape(deltas[:, 3] * boxes[:, 3], [-1, 1])

        new_boxes = tf.concat([new_x, new_y, new_w, new_h], axis=1)

        return new_boxes

    def __bounding_box_loss(self, predicted_boxes, label_boxes, anchors,
                            weights):
        """ Calculate the loss for predicted and ground truth boxes. Boxes
        should all be (n,4), and weights should be (n) """

        xp, yp, wp, hp = tf.split(predicted_boxes, 4, axis=1)
        xl, yl, wl, hl = tf.split(label_boxes, 4, axis=1)
        xa, ya, wa, ha = tf.split(anchors, 4, axis=1)

        tx = (xp - xa) / wa
        ty = (yp - ya) / ha
        tw = tf.log(wp / wa)
        th = tf.log(hp / ha)

        tlx = (xl - xa) / wa
        tly = (yl - ya) / ha
        tlw = tf.log(wl / wa)
        tlh = tf.log(hl / ha)

        t = tf.concat([tx, ty, tw, th], axis=1)
        tl = tf.concat([tlx, tly, tlw, tlh], axis=1)

        loss = self.__smooth_l1_loss(tl, t, weights)

        return loss

    def __smooth_l1_loss(self, label_regions, predicted_regions, weights):
        """Smooth/Robust l1 loss"""

        tensor = tf.abs(predicted_regions - label_regions)

        loss = tf.multiply(
            tf.where(tensor < 1, x=tf.square(tensor) / 2, y=tensor - 0.5),
            tf.cast(weights, tf.float32))

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
        # x_min = tf.reduce_min(x7, axis=3)
        # x_max = tf.reduce_max(x7, axis=3)
        # x7 = 255.0 * (x7 - x_min) / (x_max - x_min)

        # return x8
        return x7

    def __put_bboxes_on_image(self, images, boxes, scale):
        images = tf.split(images, self.batch_size, axis=0)

        output = []

        for i, bboxes in enumerate(boxes):
            bboxes = tf.reshape(bboxes, [1, -1, 4])

            bboxes = bboxes * scale
            shape = tf.shape(bboxes)
            bboxes = self.__clip_bboxes(tf.reshape(bboxes, [-1, 4]), 1.0, 1.0)
            bboxes = self.__bboxes_to_xyxy(bboxes)
            bboxes = tf.reshape(bboxes, shape)
            bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

            output.append(tf.image.draw_bounding_boxes(images[i], bboxes))

        return tf.concat(output, axis=0)

    def __bboxes_to_xyxy(self, bboxes):
        x, y, w, h = tf.split(bboxes, 4, axis=1)
        bboxes = tf.concat([y - h / 2.0, x - w / 2.0,
                            y + h / 2.0,
                            x + w / 2.0],
                           axis=1)
        return bboxes

    def __bboxes_xyxy_to_regular(self, bboxes):
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        bboxes = tf.concat([(x1 + x2) / 2.0, (y1 + y2) / 2.0,
                            x2 - x1,
                            y2 - y1],
                           axis=1)
        return bboxes

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

        # with tf.variable_scope('global_average_pool'):
        #     inputs = tf.reduce_mean(inputs, [1, 2])

        # get roi's
        with tf.variable_scope('region_proposal_network'):
            conv_cls, conv_regions = self.__region_proposals(inputs, 512,
                                                             512, 9)

            all_boxes = self.__process_rois(conv_regions, conv_cls)

            pooled_regions = []

            for j, boxes in enumerate(all_boxes):
                boxes = all_boxes[j]
                boxes_shape = tf.shape(boxes)
                boxes = tf.reshape(boxes, [-1, 4])
                boxes = self.__clip_bboxes(boxes, 19, 19)
                boxes = tf.reshape(boxes, boxes_shape)

                boxes = tf.cast(tf.round(boxes), tf.int32)
                roi_indices = tf.reshape(tf.range(tf.shape(boxes)[0]),
                                         [-1, 1])

                boxes = tf.concat([roi_indices, boxes], axis=1)
                pooled_region = self.__roi_pooling(inputs, boxes, 7, 7)
                pooled_region = tf.transpose(pooled_region, [0, 2, 3, 1])
                grid = self.__put_activations_on_grid(pooled_region, (16, 32))
                tf.summary.image('roi_pooling', grid, max_outputs=15)
                pooled_region = tf.reshape(pooled_region, [-1, 512 * 7 * 7])
                pooled_regions.append(pooled_region)

        # classify regions and add final region adjustments
        with tf.variable_scope('region_classification'):
            common_weights1 = tf.get_variable('common_weights1',
                                              [7 * 7 * 512, 4096],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)

            common_weights2 = tf.get_variable('common_weights2',
                                              [4096, 4096],
                                              initializer=xavier_initializer(
                                                  dtype=tf.float32),
                                              dtype=tf.float32)

            class_weights = tf.get_variable('class_weights',
                                            [4096,
                                             self.num_classes + 1],
                                            initializer=xavier_initializer(
                                                dtype=tf.float32),
                                            dtype=tf.float32)
            class_bias = tf.get_variable("class_bias", [
                self.num_classes + 1],
                                         initializer=tf.constant_initializer(
                                             0.1),
                                         dtype=tf.float32)

            region_weights = tf.get_variable('region_weights',
                                             [4096, self.num_classes *
                                              4],
                                             initializer=xavier_initializer(
                                                 dtype=tf.float32),
                                             dtype=tf.float32)

            region_bias = tf.get_variable("region_bias", [
                self.num_classes * 4],
                                          initializer=tf.constant_initializer(
                                              0.1),
                                          dtype=tf.float32)

            class_scores = []
            region_scores = []

            for j, batch in enumerate(pooled_regions):
                with tf.variable_scope('fc6'):
                    fc6 = tf.matmul(batch, common_weights1)
                    fc6 = self.__batch_norm_wrapper(fc6)
                    fc6 = tf.nn.elu(fc6)
                    fc6 = tf.nn.dropout(fc6, self.dropout_prob)

                with tf.variable_scope('fc7'):
                    fc7 = tf.matmul(fc6, common_weights2)
                    fc7 = self.__batch_norm_wrapper(fc7)
                    fc7 = tf.nn.elu(fc7)
                    fc7 = tf.nn.dropout(fc7, self.dropout_prob)

                with tf.variable_scope('rcn_class'):
                    class_score = tf.matmul(fc7, class_weights)
                    class_score = tf.nn.bias_add(class_score, class_bias)

                    class_score = tf.clip_by_value(class_score, -1000, 1000)

                with tf.variable_scope('rcn_region'):
                    region_score = tf.matmul(fc7, region_weights)
                    region_score = tf.nn.bias_add(region_score, region_bias)

                    region_score = tf.clip_by_value(region_score, -0.2, 0.2)
                    shape = tf.shape(region_score)

                    coords, size = tf.split(tf.reshape(region_score, [-1, 4]),
                                            2,
                                            axis=1)
                    size = tf.exp(size)
                    region_score = tf.reshape(tf.concat([coords, size], axis=1),
                                              shape)

                class_scores.append(class_score)
                region_scores.append(region_score)

        return class_scores, region_scores, conv_cls, conv_regions, all_boxes

    def __process_iou_score(self, conv_region, label_region):
        """ Calculates IoU scores for two lists of regions (m,4) and (n,4) """
        with tf.variable_scope('process_iou_score'):
            return self.__box_ious(tf.cast(conv_region, tf.float32),
                                   tf.cast(label_region, tf.float32))

    def loss(self, class_scores, region_scores, conv_cls, conv_regions, labels,
             label_regions, proposed_boxes, images):
        conv_regions = tf.reshape(conv_regions, [self.batch_size, -1, 4])
        label_regions = tf.sparse_split(sp_input=label_regions,
                                        num_split=self.batch_size, axis=0)
        labels = tf.sparse_split(sp_input=labels,
                                 num_split=self.batch_size, axis=0)

        conv_labels_losses = []
        conv_region_losses = []
        rcnn_label_losses = []
        rcnn_losses = []
        rcnn_accuracies = []
        rpn_accuracies = []

        final_proposed_regions = []

        for b in range(self.batch_size):
            conv_regs = conv_regions[b]
            conv_classes = tf.reshape(conv_cls[b], [-1, 2])
            num_regions = tf.shape(conv_regs)[0]
            label_regs = label_regions[b]
            label = labels[b]
            label_regs = tf.reshape(tf.sparse_tensor_to_dense(label_regs), [-1])
            label = tf.reshape(tf.sparse_tensor_to_dense(label,
                                                         default_value=-1),
                               [-1])

            mask = label >= 0

            label = tf.boolean_mask(label, mask)
            label_regs = tf.boolean_mask(tf.reshape(label_regs, [-1, 4]), mask)
            label_regs = tf.reshape(label_regs, [-1, 4])

            num_labels = tf.shape(label_regs)[0]

            # If an image doesn't have annotations in MSCOCO, use dummy ones
            label = tf.cond(num_labels > 0, lambda: label, lambda: tf.constant([
                -1], dtype=tf.int64))
            label_regs = tf.cond(num_labels > 0, lambda: label_regs, lambda:
            tf.constant([[0.0, 0.0, 0.01, 0.01]]))

            proposed_box = proposed_boxes[b]
            proposed_box = tf.cond(tf.size(proposed_box) > 0,
                                   lambda: proposed_box, lambda:
                                   tf.constant([[0.0, 0.0, 0.01, 0.01]]))

            reg_scores = region_scores[b]
            reg_scores = tf.cond(tf.size(reg_scores) > 0, lambda: reg_scores,
                                 lambda: tf.constant([[0.0, 0.0, 0.01, 0.01]]))

            lab_reg_shape = tf.shape(label_regs)
            label_regs = self.__scale_bboxes(tf.reshape(label_regs, [-1,
                                                                     4]),
                                             19.0 / self.width,
                                             19.0 / self.height)
            label_regs = tf.reshape(label_regs, lab_reg_shape)

            # calculate rpn loss
            with tf.variable_scope('rpn_loss'):
                label_regs2 = tf.reshape(tf.tile(label_regs,
                                                 [num_regions, 1]),
                                         [num_regions, -1, 4])

                num_labs = tf.shape(label_regs2)[1]

                conv_regs2 = tf.reshape(tf.tile(tf.cast(tf.reshape(
                    self.feat_anchors, [-1, 4]),
                    tf.float32), [1, num_labs]),
                    [num_regions, num_labs, 4])
                ious = tf.map_fn(
                    lambda x: self.__process_iou_score(x[0], x[1]),
                    [conv_regs2, label_regs2], dtype=tf.float32,
                    name="calculate_ious")

                max_ious = tf.reduce_max(ious, axis=1)
                max_ids = tf.argmax(ious, axis=1)

                target_labels = tf.where(max_ious > 0.7,
                                         tf.reshape(tf.tile([0, 1],
                                                            [
                                                                num_regions]),
                                                    [num_regions, 2]),
                                         tf.reshape(tf.tile([1, 0],
                                                            [num_regions]),
                                                    [num_regions, 2]))

                target_mask = tf.where(max_ious > 0.3,
                                       tf.where(max_ious > 0.7,
                                                tf.tile([1], [
                                                    num_regions]),
                                                tf.tile([0], [
                                                    num_regions])),
                                       tf.tile([1], [num_regions]))

                target_mask2 = tf.reshape(tf.where(max_ious > 0.7,
                                                   tf.tile([1], [
                                                       num_regions]),
                                                   tf.tile([0], [
                                                       num_regions])), [-1, 1])

                label_regs3 = tf.reshape(label_regs,
                                         [-1, 4])
                target_regions = tf.gather(label_regs3, max_ids)

                with tf.variable_scope('label_loss'):
                    conv_labels_loss = \
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.reshape(
                                tf.cast(target_labels, tf.float32),
                                [-1, 2]), logits=conv_classes)

                    # adjust for class imbalance of positive/negative IoUs
                    weights = tf.where(max_ious > 0.7,
                                       tf.tile([4.53], [
                                           num_regions]),
                                       tf.tile([0.75], [
                                           num_regions]))

                    conv_labels_loss = tf.multiply(conv_labels_loss, weights)

                    conv_labels_loss = tf.boolean_mask(conv_labels_loss,
                                                       tf.cast(target_mask,
                                                               tf.bool))

                    conv_labels_loss = tf.reduce_mean(conv_labels_loss,
                                                      name="conv_label_loss")

                    conv_classes = tf.nn.softmax(conv_classes)

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
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                      tf.float32))
                    rpn_accuracies.append(accuracy)

                with tf.variable_scope('region_loss'):
                    conv_regs = self.__adjust_bbox(conv_regs, tf.cast(
                        tf.reshape(
                            self.feat_anchors, [-1, 4]),
                        tf.float32))

                    # conv_regs = tf.Print(conv_regs, [conv_regs], "conv_regs",
                    #                      summarize=2048)
                    #
                    # target_regions = tf.Print(target_regions,
                    #                           [target_regions],
                    #                           "target_regions", summarize=2048)

                    conv_region_loss = self.__bounding_box_loss(
                        conv_regs,
                        tf.cast(target_regions, tf.float32),
                        tf.cast(tf.reshape(self.feat_anchors, [-1, 4]),
                                tf.float32),
                        tf.tile(target_mask2, [1, 4]))

                    conv_region_loss = tf.reshape(tf.boolean_mask(
                        conv_region_loss,
                        tf.cast(tf.tile(
                            target_mask2, [1,
                                           4]), tf.bool)), [-1, 4])

                    conv_region_loss = tf.cond(tf.equal(tf.size(
                        conv_region_loss), 0), lambda: tf.constant(0.0),
                        lambda: tf.reduce_mean(
                            tf.reduce_sum(
                                conv_region_loss, axis=1)))

                conv_labels_losses.append(conv_labels_loss)
                conv_region_losses.append(conv_region_loss)

            # calculate rcnn loss
            with tf.variable_scope('rcnn_loss'):
                proposed_box = tf.reshape(proposed_box, [-1, 4])

                background = tf.zeros([self.num_classes])
                background = tf.reshape(
                    tf.concat([[1], background], axis=0),
                    [self.num_classes + 1])

                cls_scores = class_scores[b]
                num_regions = tf.shape(proposed_box)[0]
                regions = tf.reshape(reg_scores, [num_regions, -1, 4])

                label_regs3 = tf.reshape(tf.tile(label_regs3, [num_regions, 1]),
                                         [num_regions, -1, 4])

                num_labs = tf.shape(label_regs3)[1]

                regions2 = tf.reshape(tf.tile(proposed_box, [1, num_labs]),
                                      [num_regions, num_labs, 4])

                ious = tf.map_fn(
                    lambda x: self.__process_iou_score(x[0], x[1]),
                    [regions2, label_regs3], dtype=tf.float32,
                    name="calculate_ious")

                max_ious = tf.reduce_max(ious, axis=1)
                max_ids = tf.cast(tf.argmax(ious, axis=1), tf.int32)

                target_labels = tf.gather(label, max_ids)
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

                target_labels2 = tf.where(max_ious > 0.5,
                                          tf.one_hot(target_labels + 1,
                                                     self.num_classes + 1),
                                          tf.reshape(tf.tile(background,
                                                             [num_regions]),
                                                     [num_regions,
                                                      self.num_classes + 1]))

                target_mask = tf.where(max_ious > 0.5,
                                       tf.tile([1], [
                                           num_regions]),
                                       tf.tile([0], [
                                           num_regions]))

                proposed_regions = self.__adjust_bbox(region_adjustment,
                                                      proposed_box)
                proposed_regions = tf.clip_by_value(proposed_regions, -1000,
                                                    1000)

                with tf.variable_scope('label_loss'):
                    # cls_scores = tf.nn.softmax(cls_scores)

                    rcnn_label_loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=
                        tf.cast(target_labels2, tf.float32), logits=cls_scores)

                    weights = tf.where(max_ious > 0.5,
                                       tf.tile([4.53], [
                                           num_regions]),
                                       tf.tile([0.75], [
                                           num_regions]))

                    rcnn_label_loss = tf.multiply(rcnn_label_loss, weights)
                    rcnn_label_loss = tf.reduce_mean(rcnn_label_loss)

                    # rcnn_label_loss = tf.boolean_mask(rcnn_label_loss,
                    #                                    tf.cast(target_mask,
                    #                                            tf.bool))

                    # rcnn_label_loss = tf.losses.log_loss(tf.reshape(
                    #     tf.cast(target_labels2, tf.float32),
                    #     [-1, self.num_classes + 1]),
                    #     cls_scores,
                    #     weights=tf.concat([[[self.background_weight]], tf.ones(
                    #         [1, 90])], axis=1))

                    correct_prediction = tf.equal(tf.argmax(cls_scores, 1),
                                                  target_labels + 1)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                      tf.float32))
                    rcnn_accuracies.append(accuracy)

                with tf.variable_scope('region_loss'):
                    # d = target_regions - proposed_regions
                    # l2_loss = tf.reduce_sum(d * d, axis=1)

                    l2_loss = self.__smooth_l1_loss(target_regions,
                                                    proposed_regions,
                                                    tf.tile(
                                                        tf.reshape(
                                                            target_mask, [-1,
                                                                          1]),
                                                        [1,
                                                         4]))

                    l2_loss = tf.reshape(tf.boolean_mask(l2_loss, tf.cast(
                        tf.tile(
                            tf.reshape(
                                target_mask, [-1,
                                              1]), [1,
                                                    4]),
                        tf.bool)),
                                         [-1, 4])

                    l2_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=1))

                    rcnn_loss = tf.where(tf.logical_or(tf.is_nan(
                        l2_loss), tf.logical_or(tf.size(l2_loss) == 0,
                                                num_labels == 0)), 0.0,
                        l2_loss)

                cls_scores = tf.nn.softmax(cls_scores)

                idx = tf.argmax(cls_scores, dimension=1)

                #filter regions with background class predicted
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

                #hide regions with a score <= 0.1
                mask = prediction_values > 0.1

                proposed_regions = tf.boolean_mask(proposed_regions, mask)

                _, ids = tf.nn.top_k(prediction_values, k=tf.minimum(
                    tf.cast(tf.size(
                        prediction_values) / 4, tf.int32), 20))
                proposed_regions = tf.gather(proposed_regions, ids)
                proposed_regions = tf.cond(tf.size(idx) > 0, lambda:
                proposed_regions, lambda: tf.constant([0.0, 0.0, 0.0, 0.0]))
                final_proposed_regions.append(proposed_regions)

            # if no box was proposed -> no loss
            rcnn_label_loss = tf.where(tf.logical_or(tf.shape(proposed_box)[0]
                                                     == 0,
                                                     tf.shape(proposed_regions)[
                                                         0]
                                                     == 0),
                                       0.0, rcnn_label_loss)
            rcnn_loss = tf.where(tf.shape(proposed_box)[0] == 0,
                                 0.0, rcnn_loss)

            rcnn_label_losses.append(rcnn_label_loss)
            rcnn_losses.append(rcnn_loss)

        images = self.__put_bboxes_on_image(images,
                                            final_proposed_regions, 1.0 / 19)

        tf.summary.image("bbox_predictions", images, max_outputs=16)

        conv_labels_losses = tf.reduce_mean(conv_labels_losses,
                                            name="conv_labels_loss")
        conv_region_losses = tf.reduce_mean(conv_region_losses,
                                            name="conv_region_loss")
        rcnn_label_losses = tf.cond(tf.equal(tf.reduce_sum(
            rcnn_label_losses), 0.0), lambda: tf.constant(0.0), lambda: \
            tf.reduce_mean(rcnn_label_losses, name="rcnn_label_loss"))
        rcnn_losses = tf.cond(tf.equal(tf.reduce_sum(
            rcnn_losses), 0.0), lambda: tf.constant(0.0), lambda:
            tf.reduce_mean(rcnn_losses, name="rcnn_region_loss"))

        conv_labels_losses = tf.Print(conv_labels_losses,
                                      [conv_labels_losses],
                                      "conv_labels_losses", summarize=2048)
        conv_region_losses = tf.Print(conv_region_losses,
                                      [conv_region_losses],
                                      "conv_region_losses", summarize=2048)
        rcnn_label_losses = tf.Print(rcnn_label_losses,
                                     [rcnn_label_losses],
                                     "rcnn_label_losses", summarize=2048)
        rcnn_losses = tf.Print(rcnn_losses,
                               [rcnn_losses],
                               "rcnn_losses", summarize=2048)

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

        rcn_accuracy = tf.reduce_mean(rcnn_accuracies)
        rpn_accuracy = tf.reduce_mean(rpn_accuracies)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), \
               rcn_accuracy, rpn_accuracy

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
        with tf.variable_scope('calculate_gradients'):
            with tf.control_dependencies([loss_averages_op]):
                opt = tf.train.AdamOptimizer(lr, epsilon=self.adam_epsilon)
                grads = opt.compute_gradients(loss)
                grads = [
                    (tf.clip_by_value(tf.where(tf.is_nan(grad), tf.zeros_like(grad),
                                              grad), -5.0, 5.0),
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
