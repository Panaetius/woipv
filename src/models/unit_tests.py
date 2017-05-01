import tensorflow as tf
from model import WoipvModel, NetworkType
from train_model import Config
import numpy as np

class clip_bboxes_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)

    def testClip(self):
        a = tf.constant([[1.0, 1.0, 2.0, 2.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[0.5, 0.5, 1.0, 1.0]])

        a = tf.constant([[1.0, 1.0, 4.0, 4.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[0.5, 0.5, 1.0, 1.0]])

        a = tf.constant([[0.5, 0.25, 1.0, 1.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[0.5, 0.375, 1.0, 0.75]])

        a = tf.constant([[0.1, 0.5, 1.0, 1.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllCloseAccordingToType(clipped.eval(), [[0.3, 0.5, 0.6, 1.0]])

        a = tf.constant([[0.5, 0.75, 1.0, 1.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[0.5, 0.625, 1.0, 0.75]])

        a = tf.constant([[0.9, 0.5, 1.0, 1.0]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllCloseAccordingToType(clipped.eval(), [[0.7, 0.5, 0.6, 1.0]])

        a = tf.constant([[0.4, 0.6, 0.5, 0.5]])
        clipped = self.model.clip_bboxes(a, 1.0, 1.0)

        with self.test_session():
            self.assertAllCloseAccordingToType(clipped.eval(), [[0.4, 0.6, 0.5, 0.5]])

class bboxes_yxyx_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)

    def testToYXYX(self):
        a = tf.constant([[2.0, 1.0, 3.0, 2.0]])
        clipped = self.model.bboxes_to_yxyx(a)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[0.5, 0.0, 3.5, 2.0]])

        a = tf.constant([[2.0, 1.0, 3.0, 2.0]])
        clipped = self.model.bboxes_to_yxyx(a, max_height=5.0)

        with self.test_session():
            self.assertAllEqual(clipped.eval(), [[1.5, 0.0, 4.5, 2.0]])

class adjust_bbox_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)

    def testAdjustBbox(self):
        box = tf.constant([[2.0, 1.0, 3.0, 2.5]])
        delta = tf.constant([[0.2, -0.1, 0.8, 1.2]])
        adjusted = self.model.adjust_bbox(delta, box)

        with self.test_session():
            self.assertAllEqual(adjusted.eval(), np.float32([[2.6, 0.75, 2.4, 3.0]]))

class bbox_loss_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)

    def testBboxLoss(self):
        np.set_printoptions(precision=100)
        predicted = tf.constant([[1.0, 2.0, 3.0, 20.0]])
        label = tf.constant([[2.0, 3.0, 4.0, 5.0]])
        anchors = tf.constant([[1.5, 2.5, 3.5, 2.5]])
        loss = self.model.bounding_box_loss(predicted, label, anchors)

        with self.test_session():
            self.assertAllCloseAccordingToType(loss.eval(), np.float32([[0.0408163, 0.08, 0.04138048, 0.88629436]]))

class generate_anchors_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)

    def testAnchors(self):
        anchors = self.model.get_tiled_anchors_for_shape(800, 600)
        num_anchors = tf.shape(anchors)[0]
        count = np.ceil(800 / self.model.feat_stride) * np.ceil(600/self.model.feat_stride) * self.model.num_anchors

        x,y,w,h = tf.split(anchors, 4, axis=1)

        with self.test_session():
            self.assertEqual(num_anchors.eval(), count)

            self.assertTrue(np.all(x.eval() >= 0.0))
            self.assertTrue(np.all(y.eval() >= 0.0))

            self.assertTrue(np.all(x.eval() <= 600))
            self.assertTrue(np.all(y.eval() <= 800))

            self.assertTrue(np.all(w.eval() >= 0.0))
            self.assertTrue(np.all(h.eval() >= 0.0))

            self.assertTrue(np.all(w.eval() < 1.5 * 512))
            self.assertTrue(np.all(h.eval() < 1.5 * 512))

class iou_test(tf.test.TestCase):
    def setUp(self):
        config = Config()
        self.model = WoipvModel(config)
        

    def testDisjunct(self):
        a = tf.constant([[0, 0, 1, 1]])
        b = tf.constant([[1, 1, 2, 2]])
        ious = self.model.process_iou_score(a, b)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [0])
        
        a = tf.constant([[0, 0, 1, 1]])
        b = tf.constant([[1, 1, 2, 2]])
        ious = self.model.process_iou_score(b, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [0])

        a = tf.constant([[-1, -1, 0, 0]])
        b = tf.constant([[0, 0, 1, 1]])
        ious = self.model.process_iou_score(b, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [0])

        a = tf.constant([[-1, -1, 0, 0]])
        b = tf.constant([[-2, -2, -1, -1]])
        ious = self.model.process_iou_score(b, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [0])

    def testIdentity(self):
        a = tf.constant([[0, 0, 1, 1]])
        ious = self.model.process_iou_score(a, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [1])

        a = tf.constant([[-1, -1, 1, 1]])
        ious = self.model.process_iou_score(a, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [1])

        a = tf.constant([[-2, -2, -1, -1]])
        ious = self.model.process_iou_score(a, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [1])

    def testUnion(self):
        a = tf.constant([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        b = tf.constant([[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5], [0.5, 0.0, 1.0, 1.0]])
        ious = self.model.process_iou_score(a, b)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [np.float32(0.25), np.float32(0.25/1.75), np.float32(0.5)])

        a = tf.constant([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        b = tf.constant([[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5], [0.5, 0.0, 1.0, 1.0]])
        ious = self.model.process_iou_score(b, a)

        with self.test_session():
            self.assertAllEqual(ious.eval(), [np.float32(0.25), np.float32(0.25/1.75), np.float32(0.5)])

if __name__ == '__main__':
    tf.test.main()
