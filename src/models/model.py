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
            conv_regions = tf.nn.relu(batch_norm, 'relu')

        return conv_cls, conv_regions

    def __roi_pooling(self, inputs, boxes, pool_height, pool_width):

        roi_pool = roi_pooling(inputs, boxes, pool_height, pool_width)

        return roi_pool

    def __process_rois(self, regions, class_scores):
        """ get relevant regions, with non-maximum suppression and clipping """
        region_boxes = self.feat_anchors.reshape(1, 19, 19, 9,
                                                 4) * tf.reshape(regions,
                                                                 (1, 19, 19, 9,
                                                                  4))

        class_scores = tf.reshape(class_scores, (1, 19, 19, 9, 2))

        if self.is_training:
            # ignore boxes that cross the image boundary
            region_boxes = tf.reshape(tf.boolean_mask(region_boxes,
                                                      tf.cast(
                                                          self.outside_feat_anchors.reshape(
                                                              1, 19, 19, 9, 4),
                                                          tf.bool)),
                                      [self.batch_size, -1, 4])
            class_scores = tf.reshape(tf.boolean_mask(class_scores,
                                                      tf.cast(
                                                          self.outside_score_anchors.reshape(
                                                              1, 19, 19, 9, 2),
                                                          tf.bool)),
                                      [self.batch_size, -1, 2])
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

        with tf.variable_scope('non_maximum_supression'):
            for i, bboxes in enumerate(tf.unstack(region_boxes, axis=0)):
                # filter boxes to have class score > 0.5
                filter_indices = tf.where(tf.greater(tf.reshape(class_scores[i],
                                                                [-1]), 0.5))
                bboxes = tf.reshape(tf.gather(bboxes, filter_indices), [-1, 4])
                cur_class_scores = tf.reshape(tf.gather(class_scores[i],
                                                        filter_indices), [-1])
                idx = tf.image.non_max_suppression(bboxes, cur_class_scores,
                                                   256,
                                                   0.7)
                bbox_list.append(tf.reshape(tf.gather(bboxes, idx), [-1, 4]))

        return bbox_list

    def __box_ious(self, boxes_a, boxes_b):
        """ Calculate intersetion over union of two bounding boxes """
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
                intersectionArea = (xB - xA + 1) * (yB - yA + 1)
            with tf.variable_scope('box_area'):
                boxesAArea = boxes_a[:, 2] * boxes_a[:, 3]
                boxesBArea = boxes_b[:, 2] * boxes_b[:, 3]

            with tf.variable_scope('iou'):
                ious = intersectionArea / tf.cast(
                    boxesAArea + boxesBArea - intersectionArea + 1e-7, \
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

    def __scale_bboxes(self,bboxes, scale_x, scale_y):
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
            x,y,w,h = tf.split(bboxes, 4, axis=1)
            minx = tf.minimum(x - w / 2.0, 0)
            maxx = tf.maximum(x + w / 2.0, max_width) - max_width
            miny = tf.minimum(y - h / 2.0, 0)
            maxy = tf.maximum(y + h / 2.0, max_height) - max_height

            width_delta = minx - maxx
            x_delta = -minx / 2.0 - maxx / 2.0
            height_delta = miny - maxy
            y_delta = -miny / 2.0 - maxy / 2.0

            delta = tf.concat([x_delta, y_delta, width_delta, height_delta], axis=1)

            return bboxes + delta

    def __bounding_box_loss(self, predicted_boxes, label_boxes, anchors,
                            weights, epsilon=1e-7):
        """ Calculate the loss for predicted and ground truth boxes. Boxes
        should all be (n,4), and weights should be (n) """
        predicted_boxes = tf.Print(predicted_boxes, [predicted_boxes],
                                   "predicted_boxes: ", summarize=1024)
        label_boxes = tf.Print(label_boxes, [label_boxes],
                                   "label_boxes: ", summarize=1024)
        anchors = tf.Print(anchors, [anchors],
                                   "anchors: ", summarize=1024)

        xp, yp, wp, hp = tf.split(predicted_boxes, 4, axis=1)
        xl, yl, wl, hl = tf.split(label_boxes, 4, axis=1)
        xa, ya, wa, ha = tf.split(anchors, 4, axis=1)

        tx = (xp - xa) / (wa + epsilon)
        ty = (yp - ya) / (ha + epsilon)
        tw = tf.log((wp + epsilon) / (wa + epsilon))
        th = tf.log((hp + epsilon) / (ha + epsilon))

        tlx = (xl - xa) / (wa + epsilon)
        tly = (yl - ya) / (ha + epsilon)
        tlw = tf.log((wl + epsilon) / (wa + epsilon))
        tlh = tf.log((hl + epsilon) / (ha + epsilon))

        t = tf.concat([tx, ty, tw, th], axis=1)
        tl = tf.concat([tlx, tly, tlw, tlh], axis=1)

        loss = self.__smooth_l1_loss(tl, t, weights)

        loss = tf.Print(loss, [loss],
                           "bbloss: ", summarize=1024)

        return loss

    def __smooth_l1_loss(self, label_regions, predicted_regions, weights):
        """Smooth/Robust l1 loss"""

        tensor = tf.abs(predicted_regions - label_regions)

        loss = tf.multiply(
            tf.where(tensor < 1, x=tf.square(tensor) / 2, y=tensor - 0.5),
            tf.cast(weights, tf.float32))

        # loss = tf.where(tf.is_nan(loss), tf.Print(loss, [label_regions,
        #                                             predicted_regions,
        #                                                  tensor,
        #                                                  loss,
        #                            weights], "NaN l1 loss: ", summarize=10000),
        #                  loss)

        return loss# tf.where(tf.is_nan(loss), tf.ones_like(loss) * 1000, loss)

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

            boxes = tf.Print(boxes, [boxes], "Boxes: ", summarize=40)

            boxes_shape = tf.shape(boxes)
            boxes = tf.reshape(boxes, [-1, 4])
            boxes = self.__clip_bboxes(boxes, 19, 19)
            boxes = tf.reshape(boxes, boxes_shape)

            boxes = tf.Print(boxes, [boxes], "Boxes2: ", summarize=1024)
            inputs = tf.Print(inputs, [tf.shape(inputs)], "inputs: ")

            pooled_regions = []

            for i, bboxes in enumerate(tf.unstack(boxes, axis=0)):
                bboxes = tf.cast(tf.round(bboxes), tf.int32)
                roi_indices = tf.reshape(tf.range(tf.shape(bboxes)[0]),
                                         [-1, 1])

                bboxes = tf.concat([roi_indices, bboxes], axis=1)
                pooled_region = self.__roi_pooling(inputs, bboxes, 7, 7)
                # pooled_region = tf.transpose(pooled_region, [0, 3, 1, 2])
                pooled_region = tf.reshape(pooled_region, [-1, 512 * 7 * 7])
                pooled_regions.append(pooled_region)

        # classify regions and add final region adjustments
        with tf.variable_scope('region_classification'):
            class_weights = tf.get_variable('class_weights',
                                            [7 * 7 * 512, self.num_classes + 1],
                                            initializer=xavier_initializer(
                                                dtype=tf.float32),
                                            dtype=tf.float32)

            region_weights = tf.get_variable('region_weights',
                                             [7 * 7 * 512, self.num_classes *
                                              4],
                                             initializer=xavier_initializer(
                                                 dtype=tf.float32),
                                             dtype=tf.float32)
            class_scores = []
            region_scores = []

            for i, batch in enumerate(tf.unstack(pooled_regions, axis=0)):
                class_score = tf.matmul(batch, class_weights)
                class_score = tf.nn.elu(class_score, 'elu')
                region_score = tf.matmul(batch, region_weights)
                region_score = tf.nn.relu(region_score, 'relu')

                class_scores.append(class_score)
                region_scores.append(region_score)

        return class_scores, region_scores, conv_cls, conv_regions, boxes

    def __get_iou_scores(self, i, label_regions, negative_rpn_score,
                         region_count, rpn_label_regions, rpn_score):
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

    def __process_iou_score(self, conv_region, label_region):
        """ Calculates IoU scores for two lists of regions (m,4) and (n,4) """
        with tf.variable_scope('process_iou_score'):
            return self.__box_ious(tf.cast(conv_region, tf.float32),
                                   tf.cast(label_region, tf.float32))

    def loss(self, class_scores, region_scores, conv_cls, conv_regions, labels,
             label_regions, proposed_boxes):
        label_regions = tf.sparse_tensor_to_dense(label_regions)
        labels = tf.sparse_tensor_to_dense(labels)
        conv_regions = tf.reshape(conv_regions, [self.batch_size, -1, 4])

        conv_labels_losses = []
        conv_region_losses = []
        rcnn_label_losses = []
        rcnn_losses = []

        for b in range(self.batch_size):
            label_regs = label_regions[b]
            lab_reg_shape = tf.shape(label_regs)
            label_regs = self.__scale_bboxes(tf.reshape(label_regs, [-1,
                                                                     4]),
                                             19.0 / self.width,
                                             19.0 / self.height)
            label_regs = tf.reshape(label_regs, lab_reg_shape)
            conv_regs = conv_regions[b]
            conv_classes = tf.reshape(conv_cls[b], [-1, 2])
            num_regions = tf.shape(conv_regs)[0]

            # calculate rpn loss
            with tf.variable_scope('rpn_loss'):
                label_regs2 = tf.reshape(tf.tile(label_regs,
                                                 [num_regions]),
                                        [num_regions, -1, 4])

                num_labs = tf.shape(label_regs2)[1]

                conv_regs2 = tf.reshape(tf.tile(conv_regs, [1, num_labs]),
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
                                                                     num_regions]), [num_regions, 2]),
                                         tf.reshape(tf.tile([1, 0],
                                                      [num_regions]),
                                                    [num_regions, 2]))

                target_mask = tf.where(max_ious > 0.3,
                                       tf.where(max_ious > 0.7,
                                                tf.tile([0], [
                                                    num_regions]),
                                                tf.tile([1], [
                                                    num_regions])),
                                       tf.tile([0], [num_regions]))
                target_mask = tf.reshape(target_mask, [-1, 1])

                label_regs = tf.reshape(label_regs,
                                         [-1, 4])

                target_regions = tf.gather(label_regs, max_ids)

                conv_classes = tf.nn.softmax(conv_classes)

                with tf.variable_scope('label_loss'):
                    conv_labels_loss = tf.losses.log_loss(tf.reshape(
                        tf.cast(target_labels, tf.float32),
                        [-1, 2]),
                        conv_classes,
                        weights=1 - target_mask)

                with tf.variable_scope('region_loss'):
                    conv_region_loss = self.__bounding_box_loss(
                        tf.cast(tf.reshape(self.feat_anchors, [-1, 4]),
                                tf.float32)
                        * conv_regs,
                    tf.cast(target_regions, tf.float32),
                        tf.cast(tf.reshape(self.feat_anchors, [-1, 4]),
                                tf.float32),
                        1 - target_mask)

                conv_labels_losses.append(conv_labels_loss)
                conv_region_losses.append(conv_region_loss)

            # calculate rcnn loss
            with tf.variable_scope('rcnn_loss'):
                label_regs = label_regions[b]
                lab_reg_shape = tf.shape(label_regs)
                label_regs = self.__scale_bboxes(tf.reshape(label_regs, [-1,
                                                                         4]),
                                                 19.0 / self.width,
                                                 19.0 / self.height)
                label_regs = tf.reshape(label_regs, lab_reg_shape)
                label = labels[b]
                proposed_box = proposed_boxes[b]
                proposed_box = tf.reshape(proposed_box, [-1, 4])

                cls_scores = class_scores[b]
                num_regions = tf.shape(proposed_box)[0]
                regions = tf.reshape(region_scores[b], [num_regions, -1, 4])
                label_regs = tf.reshape(tf.tile(label_regs, [num_regions]),
                                        [num_regions, -1, 4])

                num_labs = tf.shape(label_regs)[1]

                regions2 = tf.reshape(tf.tile(proposed_box, [1, num_labs]),
                                     [num_regions, num_labs, 4])

                ious = tf.map_fn(
                    lambda x: self.__process_iou_score(x[0], x[1]),
                    [regions2, label_regs], dtype=tf.float32,
                    name="calculate_ious")

                max_ious = tf.reduce_max(ious, axis=1)
                max_ids = tf.cast(tf.argmax(ious, axis=1), tf.int32)

                target_labels = tf.gather(label, max_ids)
                target_regions = tf.gather_nd(label_regs, tf.concat([
                    tf.reshape(tf.range(num_regions),[-1, 1]),
                    tf.reshape(max_ids, [-1, 1])],
                    axis=1))
                region_adjustment = tf.gather_nd(regions, tf.concat([
                    tf.cast(tf.reshape(tf.range(num_regions),[-1, 1]),
                            tf.int64),
                    tf.reshape(target_labels, [-1, 1])],
                    axis=1))
                region_adjustment = tf.reshape(region_adjustment, [-1, 4])

                background = tf.zeros([self.num_classes])
                background = tf.reshape(
                    tf.concat([[1], background], axis=0),
                    [self.num_classes + 1])

                target_labels = tf.where(max_ious > 0.5,
                                         tf.one_hot(target_labels + 1,
                                                    self.num_classes + 1),
                                         tf.reshape(tf.tile(background,
                                                            [num_regions]),
                                                    [num_regions,
                                                     self.num_classes + 1]))

                target_mask = tf.where(max_ious > 0.5,
                                       tf.tile([0], [
                                           num_regions]),
                                       tf.tile([1], [
                                           num_regions]))
                target_mask = tf.reshape(target_mask, [-1, 1])

                with tf.variable_scope('label_loss'):
                    cls_scores = tf.nn.softmax(cls_scores)
                    rcnn_label_loss = tf.losses.log_loss(tf.reshape(
                        tf.cast(target_labels, tf.float32),
                        [-1, self.num_classes + 1]),
                        cls_scores)

                with tf.variable_scope('region_loss'):
                    l1_loss = tf.reshape(self.__bounding_box_loss(
                        proposed_box * region_adjustment,
                        tf.cast(target_regions, tf.float32),
                        proposed_box,
                        1 - target_mask), [-1, 4])

                    zero_loss = tf.reshape(tf.tile([0.0, 0.0, 0.0, 0.0],
                                                   [num_regions]), [-1, 4])

                    rcnn_loss = tf.reduce_mean(tf.where(max_ious > 0.5,
                                                    l1_loss, zero_loss),
                                               axis=0)

            #if no box was proposed -> no loss
            rcnn_label_loss = tf.where(tf.shape(proposed_box)[0] == 0,
                                       0.0, rcnn_label_loss)
            rcnn_loss = tf.where(tf.shape(proposed_box)[0] == 0,
                                       [0.0, 0.0, 0.0, 0.0],
            rcnn_loss)

            rcnn_label_losses.append(rcnn_label_loss)
            rcnn_losses.append(rcnn_loss)

        conv_labels_losses = tf.Print(conv_labels_losses,
                                      [conv_labels_losses],
                                      "conv_labels_losses", summarize=1024)
        conv_region_losses = tf.Print(conv_region_losses,
                                      [conv_region_losses],
                                      "conv_region_losses", summarize=1024)
        rcnn_label_losses = tf.Print(rcnn_label_losses,
                                      [rcnn_label_losses],
                                      "rcnn_label_losses", summarize=1024)
        rcnn_losses = tf.Print(rcnn_losses,
                                      [rcnn_losses],
                                      "rcnn_losses", summarize=1024)

        return tf.reduce_mean(conv_labels_losses) + tf.reduce_mean(
            conv_region_losses) + tf.reduce_mean(rcnn_label_losses) + \
               tf.reduce_mean(rcnn_losses)

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

        # Apply gradients.
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
