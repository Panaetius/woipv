import tensorflow as tf

class WoipvModel(object):

    def __init__(self, is_training, config):
        self.istraining = config.istraining
        self.num_classes = config.num_classe
        
        self.__create_anchors(self, 600, 48, (128, 256, 512), ((2,1), (1,1), (1,2)))

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
        
    def __region_proposals(self, input, input_size, output_size, k):
        with tf.variable_scope('common_roi'):
            kernel = tf.get_variable('weights', [3, 3, input_size, output_size],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv, shape=[0, 1, 2, 3])
            conv = tf.nn.elu(batch_norm, 'elu')
        
        with tf.variable_scope('cls_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 2 * k],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            conv_cls = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv_cls, shape=[0, 1, 2, 3])
            conv_cls = tf.nn.elu(batch_norm, 'elu')
            
        with tf.variable_scope('reg_roi'):
            kernel = tf.get_variable('weights', [1, 1, output_size, 4 * k],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            conv_regions = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME',
                                name='conv')
            batch_norm = self.__batch_norm_wrapper(conv_regions, shape=[0, 1, 2, 3])
            conv_regions = tf.nn.elu(batch_norm, 'elu')
            
        return conv_cls, conv_regions
        
    def __roi_pooling(self, input, boxes, pool_height, pool_width):
    
        return tf.roi_pooling(input, boxes, pool_height, pool_width)
        
    def __process_rois(self, regions, class_scores):
        """ get relevant regions, with non-maximum supression and clipping """
        region_boxes = self.feat_anchors * regions
        if self.istraining:
            #ignore boxes that cross the image boundary
            region_boxes = np.ma.masked_array(region_boxes, mask=self.outside_feat_anchors)
            class_scores = np.ma.masked_array(class_scores, mask=self.outside_feat_anchors)
        
        #get top regions
        indices = np.argpartition(class_scores[:, 1] - class_scores[:, 0], -6000)[-6000:]
        region_boxes = region_boxes[indices]
        class_scores = class_scores[indices]
        
        count, scores, boxes = self.__non_maximum_supression(region_boxes, class_scores, 0.7, 0.5, 256)
        
        return boxes
        
    def __non_maximum_supression(self, boxes, scores, overlapThreshold, scoreThreshold, maxCount):
        """ calculate nms on the boxes """
         n = len(scores)

        idx = np.argsort(scores)[::-1]
        sorted_scores = scores[idx]
        sorted_boxes = boxes[idx]
        
        top_k_ids = []
        size = 0
        i = 0

        while i < n and size < maxCount:
            if sorted_scores[i] < score_threshold:
                break
            top_k_ids.append(i)
            size += 1
            i += 1
            while i < n:
                tiled_bbox_i = np.tile(sorted_bboxes[i], (size, 1)) 
                ious = self.__box_ious(tiled_bbox_i, sorted_bboxes[top_k_ids])
                max_iou = np.max(ious)
                if max_iou > iou_threshold:
                    i += 1
                else:
                    break

        return size, sorted_scores[top_k_ids], sorted_bboxes[top_k_ids]
        
    def __box_ious(self, boxesA, boxesB):
        """ Calculate intersetion over union of two bounding boxes """
        xA = np.maximum(boxesA[:, 0] - boxesA[:, 2]/2, boxesB[:, 0] - boxesB[:, 2]/2)
        yA = np.maximum(boxesA[:, 1] - boxesA[:, 3]/2, boxesB[:, 1] - boxesB[:, 3]/2)
        xB = np.minimum(boxesA[:, 0] + boxesA[:, 2]/2, boxesB[:, 0] + boxesB[:, 2]/2)
        yB = np.minimum(boxesA[:, 1] + boxesA[:, 3]/2, boxesB[:, 1] + boxesB[:, 3]/2)
        
        intersectionArea = (xB - xA + 1) * (yB - yA + 1)
        
        boxesAArea = boxesA[:, 2] * boxesA[:, 3]
        boxesBArea = boxesB[:, 2] * boxesB[:, 3]
        
        ious = intersectionArea / float(boxesAArea + boxesBArea - interArea)
        
        return ious
        
        
    def __create_anchors(self, image_size, feature_size, sizes, aspects):
        """ Creates the anchors of the shape (feature_size, feature_size, len(sizes) * len(aspects) * 4)"""
        anchors = []
        for i in sizes:
            for j in aspects:
                anchors.append([i * j[0] / (j[0] + j[1]), i * j[1] / (j[0] + j[1])])
        
        img_anchors = anchors * image_size / feature_size
        
        feat_sizes = np.tile(anchors, (feature_size, feature_size))
        img_sizes = np.tile(img_anchors, (feature_size, feature_size))
        
        x_coords = np.array(range(feature_size))
        img_x_coords = x_coords * image_size/feature_size
        
        feat_coords = np.array(np.meshgrid(x_coords, x_coords)).T.reshape(-1,2)
        img_coords = np.array(np.meshgrid(img_x_coords, img_x_coords)).T.reshape(-1,2)
        
        self.feat_anchors = np.concatenate((feat_coords, feat_sizes), axis = 2)
        self.img_anchors = np.concatenate((img_coords, img_sizes), axis = 2)
        
        outside_anchors = np.ones((feature_size, feature_size))
        
        outside_anchors[np.where(self.feat_anchors[:, 0] - self.feat_anchors[:, 2] / 2 < 0)] = 0
        outside_anchors[np.where(self.feat_anchors[:, 1] - self.feat_anchors[:, 3] / 2 < 0)] = 0
        outside_anchors[np.where(self.feat_anchors[:, 0] + self.feat_anchors[:, 2] / 2 > feature_size)] = 0
        outside_anchors[np.where(self.feat_anchors[:, 1] + self.feat_anchors[:, 3] / 2 > feature_size)] = 0
        
        self.outside_feat_anchors = self.outside_image_anchors = outside_anchors
        
        self.masked_anchors = np.ma.masked_array(self.feat_anchors, mask=self.outside_feat_anchors)
        

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
                                             
    def __smooth_l1_loss(tensor, weights):
        """Smooth/Robust l1 loss"""
        tensor = tf.abs(tensor)
        
        return tf.multiply(tf.where(tensor < 1, tf.square(tensor) / 2, tensor - 0.5))

    def inference(self, input):
        #resnet
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
            
        #get roi's
        with tf.variable_scope('region_proposal_network'):
            conv_cls, conv_regions = self.__region_proposals(self, input, 512, 256, 9)
            boxes = self.__process_rois(self, conv_regions, conv_cls)
            pooled_regions = __roi_pooling(self, input, boxes, 7, 7)
        
        #classify regions and add final region adjustments
        with tf.variable_scope('region_classification'):
            class_weights = tf.get_variable('class_weights', [7 * 7 * 512, self.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            class_scores = tf.matmul(pooled_regions, class_weights)
            
            region_weights = tf.get_variable('region_weights', [7 * 7 * 512, self.num_classes * 4],
                                     initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                     dtype=tf.float32)
            region_scores = tf.matmul(pooled_regions, region_weights)

        return class_scores, region_scores, conv_cls, conv_regions

    def loss(self, class_scores, region_scores, conv_cls, conv_regions, labels, label_regions):
        region_count = conv_cls.shape[0]
        label_region_count = label_regions.shape[0]
        
        rpn_score = np.zeros(region_count)
        negative_rpn_score = np.ones(region_count)
        
        rpn_label_regions = np.repeat([0,0,0,0], region_count, axis=0)
                
        for i in range(label_region_count):
            ious = self.__box_ious(self.feat_anchors, np.repeat(label_regions[i], region_count, axis=0))
            
            positive_ious = np.where(ious > 0.7)
            
            if(positive_ious.size > 0):
                rpn_score[positive_ious] = 1
                rpn_label_regions[positive_ious] = label_regions[i] #TODO: Make it so an existing region only gets replaced if the IoU is higher
            else:
                rpn_score[np.argmax(ious, axis=0)] = 1
                rpn_label_regions[np.argmax(ious, axis=0)] = label_regions[i]
                
            negative_rpn_score[np.where(ious > 0.3)] = 0
            
        rpn_score[np.where(negative_rpn_score > 0)] = -1
        rpn_score[np.where(outside_anchors == 0)] = 0
        
        target_labels = np.repeat([1,0], region_count, axis=0)
        target_labels[np.where(rpn_score > 0)] = [0, 1]
        
        weights = np.ones(region_count)
        weights[np.where(rpn_score == 0)] = 0
        
        conv_labels_losses = tf.losses.log_loss(target_labels, conv_cls, weights = weights)        
        
        rpn_region_scores = conv_regions - rpn_label_regions
        
        conv_region_losses = self.__smooth_l1_loss(rpn_region_scores, weights)
        