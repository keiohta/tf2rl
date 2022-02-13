import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose
from tensorflow.keras.layers import LayerNormalization


class Encoder(tf.keras.Model):
    def __init__(self,
                 obs_shape=(84, 84, 9),
                 feature_dim=50,
                 n_conv_layers=4,
                 n_conv_filters=32,
                 name="curl_encoder"):
        super().__init__(name=name)

        assert len(obs_shape) == 3
        assert obs_shape[0] == 64 or obs_shape[0] == 84

        self.convs = []
        for layer_idx in range(n_conv_layers):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1
            self.convs.append(
                Conv2D(n_conv_filters, kernel_size=(3, 3), strides=(stride, stride), padding='valid',
                       activation='relu'))

        self.flatten = Flatten()
        self.fc = Dense(feature_dim)
        self.layer_norm = LayerNormalization()
        self.feature_dim = feature_dim

        dummy_obs = np.zeros(shape=(1,) + obs_shape, dtype=np.int)
        with tf.device("/cpu:0"):
            self(tf.constant(dummy_obs))

    def call(self, inputs, stop_q_grad=False):
        features = tf.divide(tf.cast(inputs, tf.float32),
                             tf.constant(255.))

        for conv in self.convs:
            features = conv(features)
        if stop_q_grad:
            features = tf.stop_gradient(features)
        features = self.flatten(features)
        features = self.fc(features)
        features = self.layer_norm(features)
        return features


class Decoder(tf.keras.Model):
    def __init__(self,
                 last_conv_dim=35,
                 n_deconv_layers=4,
                 n_deconv_filters=32,
                 n_output_channel=9,
                 name="curl_decoder"):
        super().__init__(name=name)

        self.n_layers = n_deconv_layers

        self.fc = Dense(n_deconv_filters * last_conv_dim * last_conv_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(last_conv_dim, last_conv_dim, n_deconv_filters))

        self.deconvs = []
        for layer_idx in range(n_deconv_layers - 1):
            self.deconvs.append(Conv2DTranspose(n_deconv_filters, 3, strides=(1, 1), activation='relu'))
        self.deconvs.append(Conv2DTranspose(n_output_channel, 3, strides=(2, 2), output_padding=1))

    def call(self, inputs):
        features = self.fc(inputs)
        features = self.reshape(features)

        for deconv in self.deconvs:
            features = deconv(features)
        return features


if __name__ == "__main__":
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)

    obs_shape = (84, 84, 9)
    encoder = Encoder(obs_shape=obs_shape)
    dummy_input = np.zeros((1,) + obs_shape, dtype=np.uint8)
    latent_vars = encoder(dummy_input)
    decoder = Decoder()
    reconstructed = decoder(latent_vars)
    assert obs_shape == reconstructed.shape[1:]
