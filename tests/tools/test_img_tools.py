import unittest

import numpy as np
import tensorflow as tf

from tf2rl.tools.img_tools import random_crop, grayscale


class TestDiscountCumSum(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_size = 4
        cls.input_shape = (cls.batch_size, 10, 10, 9)
        cls.output_shape = (cls.batch_size, 8, 8, 9)

    def test_random_crop(self):
        np_input_imgs = np.zeros(shape=self.input_shape)
        np_cropped_imgs = random_crop(np_input_imgs, output_size=self.output_shape[1])
        self.assertEqual(np_cropped_imgs.shape, self.output_shape)
        self.assertEqual(type(np_cropped_imgs), np.ndarray)

        tf_input_imgs = tf.zeros(shape=self.input_shape)
        tf_cropped_imgs = random_crop(tf_input_imgs, output_size=self.output_shape[1], is_tf=True)
        self.assertEqual(tf_cropped_imgs.shape, self.output_shape)
        self.assertEqual(type(tf_cropped_imgs), tf.python.framework.ops.EagerTensor)

        # @tf.function
        # def wrap_random_crop(*args, **kwargs):
        #     return random_crop(*args, **kwargs)
        # from time import perf_counter
        # start = perf_counter()
        # for _ in range(10000):
        #     random_crop(np_input_imgs, output_size=output_shape[1])
        # print(perf_counter() - start)
        #
        # wrap_random_crop(tf_input_imgs, output_size=output_shape[1], is_tf=True)
        # start = perf_counter()
        # for _ in range(10000):
        #     wrap_random_crop(tf_input_imgs, output_size=output_shape[1], is_tf=True)
        # print(perf_counter() - start)

    def test_grayscale(self):
        np_input_imgs = np.zeros(shape=self.input_shape)
        gradscaled_imgs = grayscale(np_input_imgs)
        assert np_input_imgs.shape == gradscaled_imgs.shape


if __name__ == '__main__':
    unittest.main()
