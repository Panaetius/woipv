import bottle
import base64
import pickle
import scipy.misc

import json
import numpy as np
import os
import tensorflow as tf
from pspnet_model import WoipvPspNetModel, NetworkType
from train_pspnet import Config

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "ip5wke", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


class Inference:
    def __init__(self, host, port):
        serv_host = FLAGS.host
        serv_port = FLAGS.port
        model_name = FLAGS.model_name
        model_version = FLAGS.model_version
        self.request_timeout = FLAGS.request_timeout

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        init = tf.global_variables_initializer()

        config = Config()
        self.woipv = WoipvPspNetModel(config)

        self.input_tensor = tf.placeholder(tf.string, name="input_tensor")
        processed = tf.cast(tf.decode_raw(self.input_tensor, tf.uint8), tf.float32)
        processed = tf.reshape(processed, [288, 288, 3])
        self.image_op = processed
        processed = tf.image.per_image_standardization(processed)
        processed = tf.expand_dims(processed, axis=0)
        self.result_op = tf.nn.sigmoid(self.woipv.inference(processed))
        self.sess.run(init)

        ckpt = tf.train.get_checkpoint_state("/training/woipv_train")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        
        variables_to_restore = tf.global_variables()
        chkpt_saver = tf.train.Saver(variables_to_restore,
                        write_version=tf.train.SaverDef.V2)
        chkpt_saver.restore(self.sess, ckpt.model_checkpoint_path)

        self._host = host
        self._port = port
        self._app = bottle.Bottle()
        self._route()

    def _route(self):
        self._app.route('/', method="POST", callback=self._POST)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def _POST(self):
        file_data = base64.b64decode(bottle.request.json['file'])

        result = self.sess.run([self.result_op], feed_dict={self.input_tensor: file_data})
        #scipy.misc.imsave('current.jpg', img)
        res = json.dumps(result[0].tolist())

        return {"success": True, "result": res}


if __name__ == '__main__':
    server = Inference(host='0.0.0.0', port=8888)
    server.start()