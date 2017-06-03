import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import psutil
from os import getpid
import math
import gc

from frru_model import WoipvFRRUModel, NetworkType
from mscoco_segnet_input import MSCOCOSegnetInputProducer
from pascalvoc_segnet_input import PascalVocSegnetInputProducer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/training/woipv_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

class Config(object):
    path = "%s/../../data/processed/pascal_voc/" % os.path.dirname(
            os.path.realpath(__file__))
    chkpt_path = "%s/../../models/transfer_chkpt/" % os.path.dirname(
            os.path.realpath(__file__))
    num_examples_per_epoch = 72000
    num_epochs_per_decay = 5
    is_training = True
    batch_size = 4
    num_classes = 16
    lanes = 32
    exclude_class = None #index of class to ignore/not contribute to loss, -1 = last class, None = don't use
    initial_learning_rate = 1e-8
    learning_rate_decay_factor = 0.5
    width = 288
    height = 288
    min_box_size = 10
    rcnn_cls_loss_weight = 95.0 / (256)
    rcnn_reg_loss_weight = 0.005
    rpn_cls_loss_weight = 2.0
    rpn_reg_loss_weight = 25.0
    dropout_prob = 0.5
    weight_decay = 0.0001
    net = NetworkType.RESNET50
    pretrained_checkpoint_path = "%s/../../models/pretrained/"% os.path.dirname(
            os.path.realpath(__file__))
    pretrained_checkpoint_meta = "ResNet-L50.meta"
    restore_from_chkpt = False
    resnet_34_variables_to_restore = ['first_layer/weights:0', 'first_layer/Variable:0', 'first_layer/Variable_1:0', 'first_layer/Variable_2:0', 'first_layer/Variable_3:0', 'reslayer_64_0/sub1/weights:0', 'reslayer_64_0/sub1/Variable:0', 'reslayer_64_0/sub1/Variable_1:0', 'reslayer_64_0/sub1/Variable_2:0', 'reslayer_64_0/sub1/Variable_3:0', 'reslayer_64_0/sub2/weights:0', 'reslayer_64_0/sub2/Variable:0', 'reslayer_64_0/sub2/Variable_1:0', 'reslayer_64_0/sub2/Variable_2:0', 'reslayer_64_0/sub2/Variable_3:0', 'reslayer_64_1/sub1/weights:0', 'reslayer_64_1/sub1/Variable:0', 'reslayer_64_1/sub1/Variable_1:0', 'reslayer_64_1/sub1/Variable_2:0', 'reslayer_64_1/sub1/Variable_3:0', 'reslayer_64_1/sub2/weights:0', 'reslayer_64_1/sub2/Variable:0', 'reslayer_64_1/sub2/Variable_1:0', 'reslayer_64_1/sub2/Variable_2:0', 'reslayer_64_1/sub2/Variable_3:0', 'reslayer_64_2/sub1/weights:0', 'reslayer_64_2/sub1/Variable:0', 'reslayer_64_2/sub1/Variable_1:0', 'reslayer_64_2/sub1/Variable_2:0', 'reslayer_64_2/sub1/Variable_3:0', 'reslayer_64_2/sub2/weights:0', 'reslayer_64_2/sub2/Variable:0', 'reslayer_64_2/sub2/Variable_1:0', 'reslayer_64_2/sub2/Variable_2:0', 'reslayer_64_2/sub2/Variable_3:0', 'reslayer_downsample_128/sub1/weights:0', 'reslayer_downsample_128/sub1/Variable:0', 'reslayer_downsample_128/sub1/Variable_1:0', 'reslayer_downsample_128/sub1/Variable_2:0', 'reslayer_downsample_128/sub1/Variable_3:0', 'reslayer_downsample_128/sub2/weights:0', 'reslayer_downsample_128/sub2/Variable:0', 'reslayer_downsample_128/sub2/Variable_1:0', 'reslayer_downsample_128/sub2/Variable_2:0', 'reslayer_downsample_128/sub2/Variable_3:0', 'reslayer_128_0/sub1/weights:0', 'reslayer_128_0/sub1/Variable:0', 'reslayer_128_0/sub1/Variable_1:0', 'reslayer_128_0/sub1/Variable_2:0', 'reslayer_128_0/sub1/Variable_3:0', 'reslayer_128_0/sub2/weights:0', 'reslayer_128_0/sub2/Variable:0', 'reslayer_128_0/sub2/Variable_1:0', 'reslayer_128_0/sub2/Variable_2:0', 'reslayer_128_0/sub2/Variable_3:0', 'reslayer_128_1/sub1/weights:0', 'reslayer_128_1/sub1/Variable:0', 'reslayer_128_1/sub1/Variable_1:0', 'reslayer_128_1/sub1/Variable_2:0', 'reslayer_128_1/sub1/Variable_3:0', 'reslayer_128_1/sub2/weights:0', 'reslayer_128_1/sub2/Variable:0', 'reslayer_128_1/sub2/Variable_1:0', 'reslayer_128_1/sub2/Variable_2:0', 'reslayer_128_1/sub2/Variable_3:0', 'reslayer_128_2/sub1/weights:0', 'reslayer_128_2/sub1/Variable:0', 'reslayer_128_2/sub1/Variable_1:0', 'reslayer_128_2/sub1/Variable_2:0', 'reslayer_128_2/sub1/Variable_3:0', 'reslayer_128_2/sub2/weights:0', 'reslayer_128_2/sub2/Variable:0', 'reslayer_128_2/sub2/Variable_1:0', 'reslayer_128_2/sub2/Variable_2:0', 'reslayer_128_2/sub2/Variable_3:0', 'reslayer_downsample_256/sub1/weights:0', 'reslayer_downsample_256/sub1/Variable:0', 'reslayer_downsample_256/sub1/Variable_1:0', 'reslayer_downsample_256/sub1/Variable_2:0', 'reslayer_downsample_256/sub1/Variable_3:0', 'reslayer_downsample_256/sub2/weights:0', 'reslayer_downsample_256/sub2/Variable:0', 'reslayer_downsample_256/sub2/Variable_1:0', 'reslayer_downsample_256/sub2/Variable_2:0', 'reslayer_downsample_256/sub2/Variable_3:0', 'reslayer_256_0/sub1/weights:0', 'reslayer_256_0/sub1/Variable:0', 'reslayer_256_0/sub1/Variable_1:0', 'reslayer_256_0/sub1/Variable_2:0', 'reslayer_256_0/sub1/Variable_3:0', 'reslayer_256_0/sub2/weights:0', 'reslayer_256_0/sub2/Variable:0', 'reslayer_256_0/sub2/Variable_1:0', 'reslayer_256_0/sub2/Variable_2:0', 'reslayer_256_0/sub2/Variable_3:0', 'reslayer_256_1/sub1/weights:0', 'reslayer_256_1/sub1/Variable:0', 'reslayer_256_1/sub1/Variable_1:0', 'reslayer_256_1/sub1/Variable_2:0', 'reslayer_256_1/sub1/Variable_3:0', 'reslayer_256_1/sub2/weights:0', 'reslayer_256_1/sub2/Variable:0', 'reslayer_256_1/sub2/Variable_1:0', 'reslayer_256_1/sub2/Variable_2:0', 'reslayer_256_1/sub2/Variable_3:0', 'reslayer_256_2/sub1/weights:0', 'reslayer_256_2/sub1/Variable:0', 'reslayer_256_2/sub1/Variable_1:0', 'reslayer_256_2/sub1/Variable_2:0', 'reslayer_256_2/sub1/Variable_3:0', 'reslayer_256_2/sub2/weights:0', 'reslayer_256_2/sub2/Variable:0', 'reslayer_256_2/sub2/Variable_1:0', 'reslayer_256_2/sub2/Variable_2:0', 'reslayer_256_2/sub2/Variable_3:0', 'reslayer_256_3/sub1/weights:0', 'reslayer_256_3/sub1/Variable:0', 'reslayer_256_3/sub1/Variable_1:0', 'reslayer_256_3/sub1/Variable_2:0', 'reslayer_256_3/sub1/Variable_3:0', 'reslayer_256_3/sub2/weights:0', 'reslayer_256_3/sub2/Variable:0', 'reslayer_256_3/sub2/Variable_1:0', 'reslayer_256_3/sub2/Variable_2:0', 'reslayer_256_3/sub2/Variable_3:0', 'reslayer_256_4/sub1/weights:0', 'reslayer_256_4/sub1/Variable:0', 'reslayer_256_4/sub1/Variable_1:0', 'reslayer_256_4/sub1/Variable_2:0', 'reslayer_256_4/sub1/Variable_3:0', 'reslayer_256_4/sub2/weights:0', 'reslayer_256_4/sub2/Variable:0', 'reslayer_256_4/sub2/Variable_1:0', 'reslayer_256_4/sub2/Variable_2:0', 'reslayer_256_4/sub2/Variable_3:0', 'reslayer_downsample_512/sub1/weights:0', 'reslayer_downsample_512/sub1/Variable:0', 'reslayer_downsample_512/sub1/Variable_1:0', 'reslayer_downsample_512/sub1/Variable_2:0', 'reslayer_downsample_512/sub1/Variable_3:0', 'reslayer_downsample_512/sub2/weights:0', 'reslayer_downsample_512/sub2/Variable:0', 'reslayer_downsample_512/sub2/Variable_1:0', 'reslayer_downsample_512/sub2/Variable_2:0', 'reslayer_downsample_512/sub2/Variable_3:0', 'reslayer_512_0/sub1/weights:0', 'reslayer_512_0/sub1/Variable:0', 'reslayer_512_0/sub1/Variable_1:0', 'reslayer_512_0/sub1/Variable_2:0', 'reslayer_512_0/sub1/Variable_3:0', 'reslayer_512_0/sub2/weights:0', 'reslayer_512_0/sub2/Variable:0', 'reslayer_512_0/sub2/Variable_1:0', 'reslayer_512_0/sub2/Variable_2:0', 'reslayer_512_0/sub2/Variable_3:0', 'reslayer_512_1/sub1/weights:0', 'reslayer_512_1/sub1/Variable:0', 'reslayer_512_1/sub1/Variable_1:0', 'reslayer_512_1/sub1/Variable_2:0', 'reslayer_512_1/sub1/Variable_3:0', 'reslayer_512_1/sub2/weights:0', 'reslayer_512_1/sub2/Variable:0', 'reslayer_512_1/sub2/Variable_1:0', 'reslayer_512_1/sub2/Variable_2:0', 'reslayer_512_1/sub2/Variable_3:0']
    resnet_50_variables_to_restore = ['first_layer/weights:0', 'first_layer/Variable:0', 'first_layer/Variable_1:0', 'first_layer/Variable_2:0', 'first_layer/Variable_3:0', 'reslayer_64_0/sub1/weights:0', 'reslayer_64_0/sub1/Variable:0', 'reslayer_64_0/sub1/Variable_1:0', 'reslayer_64_0/sub1/Variable_2:0', 'reslayer_64_0/sub1/Variable_3:0', 'reslayer_64_0/sub2/weights:0', 'reslayer_64_0/sub2/Variable:0', 'reslayer_64_0/sub2/Variable_1:0', 'reslayer_64_0/sub2/Variable_2:0', 'reslayer_64_0/sub2/Variable_3:0', 'reslayer_64_1/sub1/weights:0', 'reslayer_64_1/sub1/Variable:0', 'reslayer_64_1/sub1/Variable_1:0', 'reslayer_64_1/sub1/Variable_2:0', 'reslayer_64_1/sub1/Variable_3:0', 'reslayer_64_1/sub2/weights:0', 'reslayer_64_1/sub2/Variable:0', 'reslayer_64_1/sub2/Variable_1:0', 'reslayer_64_1/sub2/Variable_2:0', 'reslayer_64_1/sub2/Variable_3:0', 'reslayer_64_2/sub1/weights:0', 'reslayer_64_2/sub1/Variable:0', 'reslayer_64_2/sub1/Variable_1:0', 'reslayer_64_2/sub1/Variable_2:0', 'reslayer_64_2/sub1/Variable_3:0', 'reslayer_64_2/sub2/weights:0', 'reslayer_64_2/sub2/Variable:0', 'reslayer_64_2/sub2/Variable_1:0', 'reslayer_64_2/sub2/Variable_2:0', 'reslayer_64_2/sub2/Variable_3:0', 'reslayer_downsample_128/sub1/weights:0', 'reslayer_downsample_128/sub1/Variable:0', 'reslayer_downsample_128/sub1/Variable_1:0', 'reslayer_downsample_128/sub1/Variable_2:0', 'reslayer_downsample_128/sub1/Variable_3:0', 'reslayer_downsample_128/sub2/weights:0', 'reslayer_downsample_128/sub2/Variable:0', 'reslayer_downsample_128/sub2/Variable_1:0', 'reslayer_downsample_128/sub2/Variable_2:0', 'reslayer_downsample_128/sub2/Variable_3:0', 'reslayer_128_0/sub1/weights:0', 'reslayer_128_0/sub1/Variable:0', 'reslayer_128_0/sub1/Variable_1:0', 'reslayer_128_0/sub1/Variable_2:0', 'reslayer_128_0/sub1/Variable_3:0', 'reslayer_128_0/sub2/weights:0', 'reslayer_128_0/sub2/Variable:0', 'reslayer_128_0/sub2/Variable_1:0', 'reslayer_128_0/sub2/Variable_2:0', 'reslayer_128_0/sub2/Variable_3:0', 'reslayer_128_1/sub1/weights:0', 'reslayer_128_1/sub1/Variable:0', 'reslayer_128_1/sub1/Variable_1:0', 'reslayer_128_1/sub1/Variable_2:0', 'reslayer_128_1/sub1/Variable_3:0', 'reslayer_128_1/sub2/weights:0', 'reslayer_128_1/sub2/Variable:0', 'reslayer_128_1/sub2/Variable_1:0', 'reslayer_128_1/sub2/Variable_2:0', 'reslayer_128_1/sub2/Variable_3:0', 'reslayer_128_2/sub1/weights:0', 'reslayer_128_2/sub1/Variable:0', 'reslayer_128_2/sub1/Variable_1:0', 'reslayer_128_2/sub1/Variable_2:0', 'reslayer_128_2/sub1/Variable_3:0', 'reslayer_128_2/sub2/weights:0', 'reslayer_128_2/sub2/Variable:0', 'reslayer_128_2/sub2/Variable_1:0', 'reslayer_128_2/sub2/Variable_2:0', 'reslayer_128_2/sub2/Variable_3:0', 'reslayer_downsample_256/sub1/weights:0', 'reslayer_downsample_256/sub1/Variable:0', 'reslayer_downsample_256/sub1/Variable_1:0', 'reslayer_downsample_256/sub1/Variable_2:0', 'reslayer_downsample_256/sub1/Variable_3:0', 'reslayer_downsample_256/sub2/weights:0', 'reslayer_downsample_256/sub2/Variable:0', 'reslayer_downsample_256/sub2/Variable_1:0', 'reslayer_downsample_256/sub2/Variable_2:0', 'reslayer_downsample_256/sub2/Variable_3:0', 'reslayer_256_0/sub1/weights:0', 'reslayer_256_0/sub1/Variable:0', 'reslayer_256_0/sub1/Variable_1:0', 'reslayer_256_0/sub1/Variable_2:0', 'reslayer_256_0/sub1/Variable_3:0', 'reslayer_256_0/sub2/weights:0', 'reslayer_256_0/sub2/Variable:0', 'reslayer_256_0/sub2/Variable_1:0', 'reslayer_256_0/sub2/Variable_2:0', 'reslayer_256_0/sub2/Variable_3:0', 'reslayer_256_1/sub1/weights:0', 'reslayer_256_1/sub1/Variable:0', 'reslayer_256_1/sub1/Variable_1:0', 'reslayer_256_1/sub1/Variable_2:0', 'reslayer_256_1/sub1/Variable_3:0', 'reslayer_256_1/sub2/weights:0', 'reslayer_256_1/sub2/Variable:0', 'reslayer_256_1/sub2/Variable_1:0', 'reslayer_256_1/sub2/Variable_2:0', 'reslayer_256_1/sub2/Variable_3:0', 'reslayer_256_2/sub1/weights:0', 'reslayer_256_2/sub1/Variable:0', 'reslayer_256_2/sub1/Variable_1:0', 'reslayer_256_2/sub1/Variable_2:0', 'reslayer_256_2/sub1/Variable_3:0', 'reslayer_256_2/sub2/weights:0', 'reslayer_256_2/sub2/Variable:0', 'reslayer_256_2/sub2/Variable_1:0', 'reslayer_256_2/sub2/Variable_2:0', 'reslayer_256_2/sub2/Variable_3:0', 'reslayer_256_3/sub1/weights:0', 'reslayer_256_3/sub1/Variable:0', 'reslayer_256_3/sub1/Variable_1:0', 'reslayer_256_3/sub1/Variable_2:0', 'reslayer_256_3/sub1/Variable_3:0', 'reslayer_256_3/sub2/weights:0', 'reslayer_256_3/sub2/Variable:0', 'reslayer_256_3/sub2/Variable_1:0', 'reslayer_256_3/sub2/Variable_2:0', 'reslayer_256_3/sub2/Variable_3:0', 'reslayer_256_4/sub1/weights:0', 'reslayer_256_4/sub1/Variable:0', 'reslayer_256_4/sub1/Variable_1:0', 'reslayer_256_4/sub1/Variable_2:0', 'reslayer_256_4/sub1/Variable_3:0', 'reslayer_256_4/sub2/weights:0', 'reslayer_256_4/sub2/Variable:0', 'reslayer_256_4/sub2/Variable_1:0', 'reslayer_256_4/sub2/Variable_2:0', 'reslayer_256_4/sub2/Variable_3:0', 'reslayer_downsample_512/sub1/weights:0', 'reslayer_downsample_512/sub1/Variable:0', 'reslayer_downsample_512/sub1/Variable_1:0', 'reslayer_downsample_512/sub1/Variable_2:0', 'reslayer_downsample_512/sub1/Variable_3:0', 'reslayer_downsample_512/sub2/weights:0', 'reslayer_downsample_512/sub2/Variable:0', 'reslayer_downsample_512/sub2/Variable_1:0', 'reslayer_downsample_512/sub2/Variable_2:0', 'reslayer_downsample_512/sub2/Variable_3:0', 'reslayer_512_0/sub1/weights:0', 'reslayer_512_0/sub1/Variable:0', 'reslayer_512_0/sub1/Variable_1:0', 'reslayer_512_0/sub1/Variable_2:0', 'reslayer_512_0/sub1/Variable_3:0', 'reslayer_512_0/sub2/weights:0', 'reslayer_512_0/sub2/Variable:0', 'reslayer_512_0/sub2/Variable_1:0', 'reslayer_512_0/sub2/Variable_2:0', 'reslayer_512_0/sub2/Variable_3:0', 'reslayer_512_1/sub1/weights:0', 'reslayer_512_1/sub1/Variable:0', 'reslayer_512_1/sub1/Variable_1:0', 'reslayer_512_1/sub1/Variable_2:0', 'reslayer_512_1/sub1/Variable_3:0', 'reslayer_512_1/sub2/weights:0', 'reslayer_512_1/sub2/Variable:0', 'reslayer_512_1/sub2/Variable_1:0', 'reslayer_512_1/sub2/Variable_2:0', 'reslayer_512_1/sub2/Variable_3:0']
    graph = tf.Graph()
    continuous = True

def train():
    """Train ip5wke for a number of steps."""
    print("Building graph %.3f" % time.time())

    plt.ion()

    cfg = Config()

    with cfg.graph.as_default():
        finished = True
        global_step = tf.Variable(0, trainable=False, name="global_step")

        # Get images and labels
        input_producer = MSCOCOSegnetInputProducer(cfg)
        #input_producer = PascalVocSegnetInputProducer(cfg)
        images, labels, original_images = input_producer.inputs()

        model = WoipvFRRUModel(cfg)

        
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        class_scores = model.inference(images)

        # Calculate loss.
        loss = model.loss(class_scores, labels, images)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss[0], global_step)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        print("Running init %.3f" % time.time())

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(init)

        if cfg.continuous and os.path.isdir(FLAGS.train_dir):
            # restore variables (for transfer learning)
            print("Restoring checkpoint for transfer learning %.3f" %
                time.time())
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            
            variables_to_restore = tf.global_variables()
            chkpt_saver = tf.train.Saver(variables_to_restore,
                            write_version=tf.train.SaverDef.V2)
            chkpt_saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint restored %.3f" % time.time())

        if cfg.restore_from_chkpt:
            # restore variables (for transfer learning)
            print("Restoring checkpoint for transfer learning %.3f" %
                time.time())
            ckpt = tf.train.get_checkpoint_state(cfg.chkpt_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if self.net == NetworkType.RESNET34:
                variables_to_restore = self.resnet_34_variables_to_restore
            elif self.net == NetworkType.RESNET50:
                variables_to_restore = self.resnet_50_variables_to_restore
            variables_to_restore = [v for v in tf.global_variables() if v.name
                                    in cfg.variables_to_restore]
            chkpt_saver = tf.train.Saver(variables_to_restore,
                            write_version=tf.train.SaverDef.V2)
            chkpt_saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint restored %.3f" % time.time())

        if not tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.MakeDirs(FLAGS.train_dir)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),
                            write_version=tf.train.SaverDef.V2)

        sess.graph.finalize()

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        process = psutil.Process(getpid())
        # before = process.memory_percent()

        # run_metadata = tf.RunMetadata()
        print("Started training %.3f" % time.time())
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, image = sess.run([train_op, loss, original_images])
                                    #  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                    #  run_metadata=run_metadata)

            loss_value, predictions, labs, ls = loss_value

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            #tf.contrib.tfprof.model_analyzer.print_model_analysis(
            #    tf.get_default_graph(),
            #    run_meta=run_metadata,
            #    tfprof_options={
            #        'max_depth': 10000,
            #        'min_bytes': 1,  # Only >=1
            #        'min_micros': 1,  # Only >=1
            #        'min_params': 0,
            #        'min_float_ops': 0,
            #        'device_regexes': ['.*'],
            #        'order_by': 'name',
            #        'account_type_regexes': ['.*'],
            #        'start_name_regexes': ['.*'],
            #        'trim_name_regexes': [],
            #        'show_name_regexes': ['.*'],
            #        'hide_name_regexes': [],
            #        'account_displayed_op_only': True,
            #        'select': ['micros'],
            #        'viz': False,
            #        'dump_to_file': ''
            #    })

            #return

            if step % 25 == 0:
                # after = process.memory_percent()
                if step % 100 == 0:
                    plt.clf()
                    plt.figure(1, figsize=(15,15))
                    plt.gcf().canvas.set_window_title("Image Gen: %d" % step)
                    plt.imshow(image[0]/255.0)

                    plt.figure(2, figsize=(15,15))
                    plt.gcf().canvas.set_window_title("Predictions Gen: %d" % step)

                    predictions = np.transpose(predictions[..., 1], [2, 0, 1])
                    dim_a = math.ceil(math.sqrt(cfg.num_classes))
                    plt.title('Predictions')
                    for i in range(cfg.num_classes):

                        plt.subplot(dim_a, dim_a, i + 1)
                        plt.imshow(predictions[i], cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

                    plt.figure(3, figsize=(15,15))
                    plt.gcf().canvas.set_window_title("Labels Gen: %d" % step)

                    
                    labs = np.transpose(labs, [2, 0, 1])
                    dim_a = math.ceil(math.sqrt(cfg.num_classes))
                    for i in range(cfg.num_classes):

                        plt.subplot(dim_a, dim_a, i + 1)
                        plt.imshow(labs[i], cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

                    plt.figure(4, figsize=(15,15))
                    plt.gcf().canvas.set_window_title("Loss Gen: %d" % step)

                    
                    ls = np.transpose(ls, [2, 0, 1])
                    dim_a = math.ceil(math.sqrt(cfg.num_classes))
                    for i in range(cfg.num_classes):

                        plt.subplot(dim_a, dim_a, i + 1)
                        plt.imshow(np.clip(ls[i], 0.0, 5.0), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=5.0)

                    plt.figure(5, figsize=(15,15))
                    plt.gcf().canvas.set_window_title("Predictions rnd Gen: %d" % step)

                    dim_a = math.ceil(math.sqrt(cfg.num_classes))
                    plt.title('Predictions')
                    for i in range(cfg.num_classes):

                        plt.subplot(dim_a, dim_a, i + 1)
                        plt.imshow((predictions[i] > 0.6).astype(int), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

                    plt.pause(0.05)

                examples_per_sec = cfg.batch_size / duration
                sec_per_batch = float(duration)
                # correct_prediction = tf.equal(tf.argmax(logits, 1),
                #                               tf.cast(labels, tf.int64))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                #                                   tf.float32))
                # train_acc = sess.run(accuracy)
                # tf.summary.scalar('accuracy', accuracy)
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # trace_file = open('timeline%d.ctf.json'%step, 'w')
                # trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
                # trace_file.close()
                # dur = int(min(100 * duration * (100 - after) / (1e-10 + after - before), 362439))
                # m, s = divmod(dur, 60)
                # h, m = divmod(m, 60)
                # print("%s: step %d, %.4f -> %.4f (%.0f:%.0f:%.0f)"%(datetime.now(), step, before, after, h, m, s))

                
                # before = process.memory_percent()

                format_str = ('%s: step %d, loss = %.3f(%.1f examples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if process.memory_percent() > 80:
                print("restoring session to free up memory")
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                break
        # When done, ask the threads to stop.
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)


# noinspection PyUnusedLocal
def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
