# This is an implementation of noisy linear layer defined in following paper:
# Noisy Networks for Exploration, https://arxiv.org/abs/1706.10295
# Forked from https://github.com/KNakane/tensorflow/blob/master/network/eager_module.py
# Edited by Kei Ohta

import numpy as np
import tensorflow as tf


class NoisyDense(tf.keras.layers.Layer):
    def __init__(
            self, 
            units,
            sigma_init=0.02,
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
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=[self.input_dim, self.units],
            initializer=tf.initializers.orthogonal(dtype=tf.float32),
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

        self.epsilon_kernel = tf.keras.backend.zeros(
            shape=(self.input_dim, self.units), name='epsilon_kernel')

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
                initializer=tf.keras.initializers.Constant(value=self.sigma_init),
                name='sigma_bias',
                trainable=self.trainable)

            self.epsilon_bias = tf.keras.backend.zeros(
                shape=(self.units,), name='epsilon_bias')

        else:
            self.bias = None
            self.epsilon_bias = None

        if self.trainable:
            self.sample_noise()
        else:
            self.remove_noise()

        super().build(input_shape)

    def call(self, inputs, test=False):
        # Implement Eq.(9)
        perturbed_kernel = self.kernel + \
            self.sigma_kernel * self.epsilon_kernel
        outputs = tf.keras.backend.dot(inputs, perturbed_kernel)
        if self.use_bias and not test:
            perturbed_bias = self.bias + self.sigma_bias * self.epsilon_bias
            outputs = tf.keras.backend.bias_add(outputs, perturbed_bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def sample_noise(self):
        self.epsilon_kernel.assign(np.random.normal(0, 1, (self.input_dim, self.units)))
        if self.use_bias:
            self.epsilon_bias.assign(np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        self.epsilon_kernel.assign(np.zeros(shape=(self.input_dim, self.units)))
        if self.use_bias:
            self.epsilon_bias.assign(self.epsilon_bias, np.zeros(shape=self.units,))
