# This is an implementation of noisy linear layer defined in following paper:
# Noisy Networks for Exploration, https://arxiv.org/abs/1706.10295
# Forked from https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# Fixed a bug by Kei Ohta

import tensorflow as tf

from tensorflow.keras import backend as K


class NoisyDense(tf.keras.layers.Layer):
    def __init__(
            self,
            units,
            sigma_init=0.017,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = int(units)
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(
            shape=[self.input_dim, self.units],
            initializer=tf.keras.initializers.Orthogonal(),
            name='kernel',
            dtype=tf.float32,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.trainable)

        self.sigma_kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.keras.initializers.Constant(value=self.sigma_init),
            name='sigma_kernel',
            trainable=self.trainable)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=self.trainable)

            self.sigma_bias = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(
                    value=self.sigma_init),
                name='sigma_bias',
                trainable=self.trainable)

        else:
            self.bias = None
            self.epsilon_bias = None

        super().build(input_shape)

    def call(self, inputs):
        # Implement Eq.(9)
        perturbed_kernel = (self.kernel
                            + self.sigma_kernel
                            * K.random_uniform(shape=self.kernel_shape))
        outputs = K.dot(inputs, perturbed_kernel)
        if self.use_bias:
            perturbed_bias = (self.bias
                              + self.sigma_bias
                              * K.random_uniform(shape=self.bias_shape))
            outputs = K.bias_add(outputs, perturbed_bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
