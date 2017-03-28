import tensorflow as tf
from tensorflow.python.client import timeline
import os
import time
from datetime import datetime
import numpy as np

from model import WoipvModel
from mscoco_input import MSCOCOInputProducer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/woipv_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

class Config(object):
    path = "%s/../../data/processed/MSCOCO/" % os.path.dirname(
            os.path.realpath(__file__))
    chkpt_path = "%s/../../models/transfer_chkpt/" % os.path.dirname(
            os.path.realpath(__file__))
    batch_size = 4
    num_examples_per_epoch = 8000
    num_epochs_per_decay = 1
    is_training = True
    num_classes = 90
    initial_learning_rate = 1e-5
    learning_rate_decay_factor = 0.5
    width = 600
    height = 600
    min_box_size = 1
    rcnn_cls_loss_weight = 0.1
    rcnn_reg_loss_weight = 0.1
    rpn_cls_loss_weight = 1.0
    rpn_reg_loss_weight = 1.0
    background_weight = 0.2
    dropout_prob = 0.5 # not used yet
    weight_decay = 0.0001
    restore_from_chkpt = True
    variables_to_restore = ['first_layer/weights:0', 'first_layer/Variable:0', 'first_layer/Variable_1:0', 'first_layer/Variable_2:0', 'first_layer/Variable_3:0', 'reslayer_64_0/sub1/weights:0', 'reslayer_64_0/sub1/Variable:0', 'reslayer_64_0/sub1/Variable_1:0', 'reslayer_64_0/sub1/Variable_2:0', 'reslayer_64_0/sub1/Variable_3:0', 'reslayer_64_0/sub2/weights:0', 'reslayer_64_0/sub2/Variable:0', 'reslayer_64_0/sub2/Variable_1:0', 'reslayer_64_0/sub2/Variable_2:0', 'reslayer_64_0/sub2/Variable_3:0', 'reslayer_64_1/sub1/weights:0', 'reslayer_64_1/sub1/Variable:0', 'reslayer_64_1/sub1/Variable_1:0', 'reslayer_64_1/sub1/Variable_2:0', 'reslayer_64_1/sub1/Variable_3:0', 'reslayer_64_1/sub2/weights:0', 'reslayer_64_1/sub2/Variable:0', 'reslayer_64_1/sub2/Variable_1:0', 'reslayer_64_1/sub2/Variable_2:0', 'reslayer_64_1/sub2/Variable_3:0', 'reslayer_64_2/sub1/weights:0', 'reslayer_64_2/sub1/Variable:0', 'reslayer_64_2/sub1/Variable_1:0', 'reslayer_64_2/sub1/Variable_2:0', 'reslayer_64_2/sub1/Variable_3:0', 'reslayer_64_2/sub2/weights:0', 'reslayer_64_2/sub2/Variable:0', 'reslayer_64_2/sub2/Variable_1:0', 'reslayer_64_2/sub2/Variable_2:0', 'reslayer_64_2/sub2/Variable_3:0', 'reslayer_downsample_128/sub1/weights:0', 'reslayer_downsample_128/sub1/Variable:0', 'reslayer_downsample_128/sub1/Variable_1:0', 'reslayer_downsample_128/sub1/Variable_2:0', 'reslayer_downsample_128/sub1/Variable_3:0', 'reslayer_downsample_128/sub2/weights:0', 'reslayer_downsample_128/sub2/Variable:0', 'reslayer_downsample_128/sub2/Variable_1:0', 'reslayer_downsample_128/sub2/Variable_2:0', 'reslayer_downsample_128/sub2/Variable_3:0', 'reslayer_128_0/sub1/weights:0', 'reslayer_128_0/sub1/Variable:0', 'reslayer_128_0/sub1/Variable_1:0', 'reslayer_128_0/sub1/Variable_2:0', 'reslayer_128_0/sub1/Variable_3:0', 'reslayer_128_0/sub2/weights:0', 'reslayer_128_0/sub2/Variable:0', 'reslayer_128_0/sub2/Variable_1:0', 'reslayer_128_0/sub2/Variable_2:0', 'reslayer_128_0/sub2/Variable_3:0', 'reslayer_128_1/sub1/weights:0', 'reslayer_128_1/sub1/Variable:0', 'reslayer_128_1/sub1/Variable_1:0', 'reslayer_128_1/sub1/Variable_2:0', 'reslayer_128_1/sub1/Variable_3:0', 'reslayer_128_1/sub2/weights:0', 'reslayer_128_1/sub2/Variable:0', 'reslayer_128_1/sub2/Variable_1:0', 'reslayer_128_1/sub2/Variable_2:0', 'reslayer_128_1/sub2/Variable_3:0', 'reslayer_128_2/sub1/weights:0', 'reslayer_128_2/sub1/Variable:0', 'reslayer_128_2/sub1/Variable_1:0', 'reslayer_128_2/sub1/Variable_2:0', 'reslayer_128_2/sub1/Variable_3:0', 'reslayer_128_2/sub2/weights:0', 'reslayer_128_2/sub2/Variable:0', 'reslayer_128_2/sub2/Variable_1:0', 'reslayer_128_2/sub2/Variable_2:0', 'reslayer_128_2/sub2/Variable_3:0', 'reslayer_downsample_256/sub1/weights:0', 'reslayer_downsample_256/sub1/Variable:0', 'reslayer_downsample_256/sub1/Variable_1:0', 'reslayer_downsample_256/sub1/Variable_2:0', 'reslayer_downsample_256/sub1/Variable_3:0', 'reslayer_downsample_256/sub2/weights:0', 'reslayer_downsample_256/sub2/Variable:0', 'reslayer_downsample_256/sub2/Variable_1:0', 'reslayer_downsample_256/sub2/Variable_2:0', 'reslayer_downsample_256/sub2/Variable_3:0', 'reslayer_256_0/sub1/weights:0', 'reslayer_256_0/sub1/Variable:0', 'reslayer_256_0/sub1/Variable_1:0', 'reslayer_256_0/sub1/Variable_2:0', 'reslayer_256_0/sub1/Variable_3:0', 'reslayer_256_0/sub2/weights:0', 'reslayer_256_0/sub2/Variable:0', 'reslayer_256_0/sub2/Variable_1:0', 'reslayer_256_0/sub2/Variable_2:0', 'reslayer_256_0/sub2/Variable_3:0', 'reslayer_256_1/sub1/weights:0', 'reslayer_256_1/sub1/Variable:0', 'reslayer_256_1/sub1/Variable_1:0', 'reslayer_256_1/sub1/Variable_2:0', 'reslayer_256_1/sub1/Variable_3:0', 'reslayer_256_1/sub2/weights:0', 'reslayer_256_1/sub2/Variable:0', 'reslayer_256_1/sub2/Variable_1:0', 'reslayer_256_1/sub2/Variable_2:0', 'reslayer_256_1/sub2/Variable_3:0', 'reslayer_256_2/sub1/weights:0', 'reslayer_256_2/sub1/Variable:0', 'reslayer_256_2/sub1/Variable_1:0', 'reslayer_256_2/sub1/Variable_2:0', 'reslayer_256_2/sub1/Variable_3:0', 'reslayer_256_2/sub2/weights:0', 'reslayer_256_2/sub2/Variable:0', 'reslayer_256_2/sub2/Variable_1:0', 'reslayer_256_2/sub2/Variable_2:0', 'reslayer_256_2/sub2/Variable_3:0', 'reslayer_256_3/sub1/weights:0', 'reslayer_256_3/sub1/Variable:0', 'reslayer_256_3/sub1/Variable_1:0', 'reslayer_256_3/sub1/Variable_2:0', 'reslayer_256_3/sub1/Variable_3:0', 'reslayer_256_3/sub2/weights:0', 'reslayer_256_3/sub2/Variable:0', 'reslayer_256_3/sub2/Variable_1:0', 'reslayer_256_3/sub2/Variable_2:0', 'reslayer_256_3/sub2/Variable_3:0', 'reslayer_256_4/sub1/weights:0', 'reslayer_256_4/sub1/Variable:0', 'reslayer_256_4/sub1/Variable_1:0', 'reslayer_256_4/sub1/Variable_2:0', 'reslayer_256_4/sub1/Variable_3:0', 'reslayer_256_4/sub2/weights:0', 'reslayer_256_4/sub2/Variable:0', 'reslayer_256_4/sub2/Variable_1:0', 'reslayer_256_4/sub2/Variable_2:0', 'reslayer_256_4/sub2/Variable_3:0', 'reslayer_downsample_512/sub1/weights:0', 'reslayer_downsample_512/sub1/Variable:0', 'reslayer_downsample_512/sub1/Variable_1:0', 'reslayer_downsample_512/sub1/Variable_2:0', 'reslayer_downsample_512/sub1/Variable_3:0', 'reslayer_downsample_512/sub2/weights:0', 'reslayer_downsample_512/sub2/Variable:0', 'reslayer_downsample_512/sub2/Variable_1:0', 'reslayer_downsample_512/sub2/Variable_2:0', 'reslayer_downsample_512/sub2/Variable_3:0', 'reslayer_512_0/sub1/weights:0', 'reslayer_512_0/sub1/Variable:0', 'reslayer_512_0/sub1/Variable_1:0', 'reslayer_512_0/sub1/Variable_2:0', 'reslayer_512_0/sub1/Variable_3:0', 'reslayer_512_0/sub2/weights:0', 'reslayer_512_0/sub2/Variable:0', 'reslayer_512_0/sub2/Variable_1:0', 'reslayer_512_0/sub2/Variable_2:0', 'reslayer_512_0/sub2/Variable_3:0', 'reslayer_512_1/sub1/weights:0', 'reslayer_512_1/sub1/Variable:0', 'reslayer_512_1/sub1/Variable_1:0', 'reslayer_512_1/sub1/Variable_2:0', 'reslayer_512_1/sub1/Variable_3:0', 'reslayer_512_1/sub2/weights:0', 'reslayer_512_1/sub2/Variable:0', 'reslayer_512_1/sub2/Variable_1:0', 'reslayer_512_1/sub2/Variable_2:0', 'reslayer_512_1/sub2/Variable_3:0']

def train():
    """Train ip5wke for a number of steps."""
    print("Building graph %.3f" % time.time())
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        cfg = Config()

        # Get images and labels for ip5wke.
        input_producer = MSCOCOInputProducer(cfg)
        images, categories, bboxes = input_producer.inputs()

        model = WoipvModel(cfg)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        class_scores, region_scores, rpn_class_scores, rpn_region_scores, \
        boxes = \
            model.inference(images)

        # Calculate loss.
        loss, rcn_accuracy, rpn_accuracy = model.loss(class_scores,
                                                     region_scores,
                                   rpn_class_scores,
                          rpn_region_scores, categories, bboxes, boxes, images)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),
                               write_version=tf.train.SaverDef.V2)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        # Start running operations on the Graph.
        print("Running init %.3f" % time.time())
        sess = tf.Session(config=config)
        sess.run(init)

        if cfg.restore_from_chkpt:
            # restore variables (for transfer learning)
            print("Restoring checkpoint for transfer learning %.3f" %
                  time.time())
            ckpt = tf.train.get_checkpoint_state(cfg.chkpt_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            variables_to_restore = [v for v in tf.global_variables() if v.name
                                    in cfg.variables_to_restore]
            chkpt_saver = tf.train.Saver(variables_to_restore,
                               write_version=tf.train.SaverDef.V2)
            chkpt_saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint restored %.3f" % time.time())

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # run_metadata = tf.RunMetadata()
        print("Started training %.3f" % time.time())
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
                                     # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                     # run_metadata=run_metadata)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 25 == 0:
                num_examples_per_step = cfg.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                # correct_prediction = tf.equal(tf.argmax(logits, 1),
                #                               tf.cast(labels, tf.int64))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                #                                   tf.float32))
                # train_acc = sess.run(accuracy)
                # tf.summary.scalar('accuracy', accuracy)
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # trace_file = open('timeline.ctf.json', 'w')
                # trace_file.write(trace.generate_chrome_trace_format())
                # trace_file.close()

                rcn_acc, rpn_acc = sess.run([rcn_accuracy, rpn_accuracy])

                format_str = ('%s: step %d, loss = %.2f, rcn_accuracy = %.3f '
                              ' rpn_acc = %.3f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    rcn_acc, rpn_acc,
                                    examples_per_sec, sec_per_batch))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # When done, ask the threads to stop.
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)


# noinspection PyUnusedLocal
def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
