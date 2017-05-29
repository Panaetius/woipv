import argparse
import psutil

from os import getpid
import tensorflow as tf
import numpy as np

def create_model(input_size, output_size):
    # model placeholders:
    shape = tf.clip_by_value(tf.cast(tf.random_normal([2]) * 38.0 + 64.0, tf.int32), 38, 120)
    shape = tf.concat([[1], shape, [512]], axis=0)

    return tf.cast(tf.ones(shape, dtype=tf.int64), tf.int32)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=7000)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=100)
    parser.add_argument('--device', type=str, default="gpu:0")
    return parser.parse_args(args=args)

def main():
    args = parse_args()
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    np.random.seed(1234)
    process = psutil.Process(getpid())

    with tf.Session(config=session_conf) as session, tf.device(args.device):
        op = create_model(args.input_size, args.output_size)
        session.run(tf.global_variables_initializer())
        before = process.memory_percent()

        for epoch in range(args.max_epochs):
            session.run(op)
            
            if epoch % 100 == 0:
                after = process.memory_percent()
                print("MEMORY CHANGE %.4f -> %.4f" % (before, after))
                before = after

if __name__ == "__main__":
    main()