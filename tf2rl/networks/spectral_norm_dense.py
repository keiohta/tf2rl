import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K


class SNDense(Dense):
    def __init__(
            self,
            units,
            trainable=True,
            u_kernel_initializer=tf.keras.initializers.TruncatedNormal,
            **kwargs):
        super().__init__(units, **kwargs)
        self.trainable = trainable
        self.u_kernel_initializer = u_kernel_initializer

    def build(self, input_shape):
        assert len(input_shape) >= 2
        super().build(input_shape)
        self.u_kernel = self.add_weight(
            shape=(1, self.units),
            initializer=self.u_kernel_initializer(),
            name='u_kernel',
            trainable=False)

    def compute_spectral_norm(self):
        u_hat = self.u_kernel

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
        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [w.shape[-1]]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, w)
            else:
                outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {
            "u_kernel_initializer": self.u_kernel_initializer,
            "trainable": self.trainable}
        base_config = super(SNDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
