import tensorflow as tf
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
tf.app.flags.DEFINE_float('dropout_keep_probability', 0.5,
                          """How many nodes to keep during dropout""")

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")

class Config(object):
    path = os.path.dirname(
            os.path.realpath(__file__))
    batch_size = 1
    num_examples_per_epoch = 8000
    num_epochs_per_decay = 40
    is_training = True
    num_classes = 90
    initial_learning_rate = 0.01
    learning_rate_decay_factor = 0.5


def train():
    """Train ip5wke for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        config = Config()

        # Get images and labels for ip5wke.
        input_producer = MSCOCOInputProducer(config)
        images, categories, bboxes = input_producer.inputs()

        model = WoipvModel(config)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        class_scores, region_scores, rpn_class_scores, rpn_region_scores = \
            model.inference(images)

        # Calculate loss.
        loss = model.loss(class_scores, region_scores, rpn_class_scores,
                          rpn_region_scores, categories, bboxes)

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
        sess = tf.Session(config=config)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 25 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                # correct_prediction = tf.equal(tf.argmax(logits, 1),
                #                               tf.cast(labels, tf.int64))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                #                                   tf.float32))
                # train_acc = sess.run(accuracy)
                # tf.summary.scalar('accuracy', accuracy)

                format_str = ('%s: step %d, loss = %.2f, '  # accuracy = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    # train_acc,
                                    examples_per_sec, sec_per_batch))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


# noinspection PyUnusedLocal
def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
