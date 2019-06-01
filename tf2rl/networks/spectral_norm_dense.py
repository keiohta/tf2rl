import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn


class SNDense(Dense):
    def __init__(
            self,
            units,
            trainable=True,
            class_name="SNDense",
            **kwargs):
        super().__init__(units, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        super().build(input_shape)
        self.u = K.zeros(
            shape=(self.units,),
            name='u',
            dtype=tf.float32)
        self.u_kernel = self.add_weight(
            shape=(1, self.kernel.shape.as_list()[-1]),
            initializer=tf.truncated_normal_initializer(),
            name='u_kernel',
            trainable=False)

    def compute_spectral_norm(self):
        u_hat = self.u_kernel
        v_hat = None

        def l2_norm(val, eps=1e-12):
            return val / (tf.reduce_sum(val ** 2) ** 0.5 + eps)

        v_ = tf.matmul(u_hat, tf.transpose(self.kernel))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, self.kernel)
        u_hat = l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, self.kernel), tf.transpose(u_hat))
        w_norm = self.kernel / sigma
        with tf.control_dependencies([self.u_kernel.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, self.kernel.shape.as_list())
        return w_norm

    def call(self, inputs):
        w = self.compute_spectral_norm()
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {'class_name': "SNDense"}
        base_config = super(SNDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
