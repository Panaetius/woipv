import tensorflow as tf

def main():
    with tf.Session() as session:
        input = tf.get_variable('weights',
                                                shape=[1, 301, 201, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=0.5, dtype=tf.float32),
                                              dtype=tf.float32)

        val, idx = tf.nn.max_pool_with_argmax(input-5, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME") #padding will turn dimensions to 202x302

        y1 = idx // 201
        x1 = idx % 201

        y2 = idx // 202
        x2 = idx % 202

        max_x1 = tf.reduce_max(x1)
        max_y1 = tf.reduce_max(y1)
        max_x2 = tf.reduce_max(x2)
        max_y2 = tf.reduce_max(y2)


        session.run(tf.global_variables_initializer())
        m_x1, m_y1, m_x2, m_y2 = session.run([max_x1, max_y1, max_x2, max_y2])

        print("%d, %d, %d, %d"%(m_x1, m_y1, m_x2, m_y2))

if __name__ == "__main__":
    main()