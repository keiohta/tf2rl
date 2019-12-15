import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class Network(tf.keras.Model):
    def __init__(self,input_shape,units,activation,name):
        super().__init__(name=name)

        self.L = []
        for i in range(len(units)):
            self.L.append(Dense(units[i]),
                          activation=activation[i],
                          name=f"L{i}")

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(input_shape, dtype=np.float32)))

    def call(self, inputs):
        for l in self.L:
            inputs = l(inputs)
        return inputs

class StateActionNetwork(Network):
    def call(self, inputs):
        states, action = inputs
        return super().call(tf.concat([states, action], axis=1))
