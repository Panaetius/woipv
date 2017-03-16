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
            batch_norm = self.__batch_norm_wrapper(conv_cls, shape=[0, 1, 2, 3])
            conv_cls = tf.nn.elu(batch_norm, 'elu')

        with tf.variable_scope('reg_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 4 * k],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv_regions = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv')
            batch_norm = self.__batch_norm_wrapper(conv_regions,
                                                   shape=[0, 1, 2, 3])
            conv_regions = tf.nn.elu(batch_norm, 'elu')

        return conv_cls, conv_regions

    def __roi_pooling(self, inputs, boxes, pool_height, pool_width):

        return roi_pooling(inputs, boxes, pool_height, pool_width)

    def __process_rois(self, regions, class_scores):
        """ get relevant regions, with non-maximum suppression and clipping """
        region_boxes = self.feat_anchors.reshape(1,19,19,9,
                                                 4) * tf.reshape(regions,
                                                                 (1,19,19,9,4))

        class_scores = tf.reshape(class_scores, (1, 19, 19, 9, 2))

        if self.is_training:
            # ignore boxes that cross the image boundary
            region_boxes = tf.boolean_mask(region_boxes,
                                           tf.cast(
                                               self.outside_feat_anchors.reshape(
                                               1, 19, 19, 9, 4), tf.bool))
            class_scores = tf.boolean_mask(class_scores,
                                           tf.cast(
                                               self.outside_score_anchors.reshape(
                                               1, 19, 19, 9, 2), tf.bool))

        class_scores = tf.nn.softmax(class_scores)

        # since we don't know the exact dimensions due to possible masking,
        # just flatten everything and and use the scores/boxes as a list
        class_scores = tf.reshape(class_scores, [self.batch_size, -1, 2])
        region_boxes = tf.reshape(region_boxes, [self.batch_size, -1, 4])

        class_scores = tf.split(class_scores, 2, axis=2)[1]
        # ignore takingtop k regions for now
        # # get top regions
        # indices = np.argpartition(tf.split(class_scores, 2, axis=4)[1],
        #                           -6000)[-6000:]
        # region_boxes = region_boxes[indices]
        # class_scores = class_scores[indices]

        bbox_list = []

        for i, bboxes in enumerate(tf.unstack(region_boxes, axis=0)):
            # filter boxes to have class score > 0.5
            filter_indices = tf.where(tf.greater(class_scores[i], 0.5))
            bboxes = tf.gather(bboxes, filter_indices)
            cur_class_scores = tf.gather(class_scores[i], filter_indices)
            idx = tf.image.non_max_suppression(bboxes, cur_class_scores, 256,
                                               0.7)
            bbox_list.append(tf.gather(bboxes, idx))

        return bbox_list

    def __box_ious(self, boxes_a, boxes_b):
        """ Calculate intersetion over union of two bounding boxes """
        xA = np.maximum(boxes_a[:, 0] - boxes_a[:, 2] / 2,
                        boxes_b[:, 0] - boxes_b[:, 2] / 2)
        yA = np.maximum(boxes_a[:, 1] - boxes_a[:, 3] / 2,
                        boxes_b[:, 1] - boxes_b[:, 3] / 2)
        xB = np.minimum(boxes_a[:, 0] + boxes_a[:, 2] / 2,
                        boxes_b[:, 0] + boxes_b[:, 2] / 2)
        yB = np.minimum(boxes_a[:, 1] + boxes_a[:, 3] / 2,
                        boxes_b[:, 1] + boxes_b[:, 3] / 2)

        intersectionArea = (xB - xA + 1) * (yB - yA + 1)

        boxesAArea = boxes_a[:, 2] * boxes_a[:, 3]
        boxesBArea = boxes_b[:, 2] * boxes_b[:, 3]

        ious = intersectionArea / float(
            boxesAArea + boxesBArea - intersectionArea)

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
            self.feat_anchors[..., 0] - (self.feat_anchors[..., 2] / 2) < 0)]\
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

    def __smooth_l1_loss(self, label_regions, predicted_regions, weights):
        """Smooth/Robust l1 loss"""
        tensor = tf.abs(predicted_regions - label_regions)

        return tf.multiply(
            tf.where(tensor < 1, x=tf.square(tensor) / 2, y=tensor - 0.5))

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

    def inference(self, inputs):
        # resnet
        with tf.variable_scope('first'):
            kernel = tf.get_variable('weights', [7, 7, 3, 64],
                                     initializer=xavier_initializer(
                                         dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1],
                                padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')
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
                                                             256, 9)
            boxes = self.__process_rois(conv_regions, conv_cls)
            pooled_regions = self.__roi_pooling(inputs, boxes, 7, 7)

        # classify regions and add final region adjustments
        with tf.variable_scope('region_classification'):
            class_weights = tf.get_variable('class_weights',
                                            [7 * 7 * 512, self.num_classes],
                                            initializer=xavier_initializer(
                                                dtype=tf.float32),
                                            dtype=tf.float32)
            class_scores = tf.matmul(pooled_regions, class_weights)

            region_weights = tf.get_variable('region_weights',
                                             [7 * 7 * 512, self.num_classes *
                                              4],
                                             initializer=xavier_initializer(
                                                 dtype=tf.float32),
                                             dtype=tf.float32)
            region_scores = tf.matmul(pooled_regions, region_weights)

        return class_scores, region_scores, conv_cls, conv_regions

    def loss(self, class_scores, region_scores, conv_cls, conv_regions, labels,
             label_regions):
        region_count = conv_cls.shape[0]
        label_region_count = label_regions.shape[0]

        rpn_score = np.zeros(region_count)
        negative_rpn_score = np.ones(region_count)

        rpn_label_regions = np.repeat([0, 0, 0, 0], region_count, axis=0)

        for i in range(label_region_count):
            ious = self.__box_ious(self.feat_anchors,
                                   np.repeat(label_regions[i], region_count,
                                             axis=0))

            positive_ious = np.where(ious > 0.7)

            if (positive_ious.size > 0):
                rpn_score[positive_ious] = 1
                rpn_label_regions[positive_ious] = label_regions[
                    i]  # TODO: Make it so an existing best match target region only gets replaced if the IoU is higher
            else:
                rpn_score[np.argmax(ious, axis=0)] = 1
                rpn_label_regions[np.argmax(ious, axis=0)] = label_regions[i]

            negative_rpn_score[np.where(ious > 0.3)] = 0

        rpn_score[np.where(negative_rpn_score > 0)] = -1
        rpn_score[np.where(self.outside_anchors == 0)] = 0

        target_labels = np.repeat([1, 0], region_count, axis=0)
        target_labels[np.where(rpn_score > 0)] = [0, 1]

        weights = np.ones(region_count)
        weights[np.where(rpn_score == 0)] = 0

        conv_labels_losses = tf.losses.log_loss(target_labels, conv_cls,
                                                weights=weights)

        conv_region_losses = self.__smooth_l1_loss(rpn_label_regions,
                                                   self.feat_anchors * conv_regions,
                                                   weights)

        region_count = class_scores.shape[0]
        label_region_count = label_regions.shape[0]

        rpn_score = np.zeros(region_count)
        negative_rpn_score = np.ones(region_count)

        rpn_label_regions = np.repeat([0, 0, 0, 0], region_count, axis=0)

        for i in range(label_region_count):
            ious = self.__box_ious(self.feat_anchors,
                                   np.repeat(label_regions[i], region_count,
                                             axis=0))

            positive_ious = np.where(ious > 0.7)

            if positive_ious.size > 0:
                rpn_score[positive_ious] = 1
                rpn_label_regions[positive_ious] = label_regions[
                    i]  # TODO: Make it so an existing best match target region only gets replaced if the IoU is higher
            else:
                rpn_score[np.argmax(ious, axis=0)] = 1
                rpn_label_regions[np.argmax(ious, axis=0)] = label_regions[i]

            negative_rpn_score[np.where(ious > 0.3)] = 0

        rpn_score[np.where(negative_rpn_score > 0)] = -1
        rpn_score[np.where(self.outside_anchors == 0)] = 0

        # TODO: one hot encode actual labels
        target_labels = np.repeat([1, 0], region_count, axis=0)
        target_labels[np.where(rpn_score > 0)] = [0, 1]

        weights = np.ones(region_count)
        weights[np.where(rpn_score == 0)] = 0

        score_labels_losses = tf.losses.log_loss(target_labels, class_scores,
                                                 weights=weights)

        score_region_losses = self.__smooth_l1_loss(rpn_label_regions,
                                                    region_scores, weights)

        return score_labels_losses + score_region_losses + conv_labels_losses + conv_region_losses

    def train(self, total_loss, global_step):
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
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr, epsilon=self.adam_epsilon)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

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
