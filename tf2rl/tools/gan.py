import numpy as np
import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, latent_dim, generator, discriminator):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.generator = generator
        self.discriminator = discriminator

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def reconstruct(self, x, apply_sigmoid=False):
        logits = self.generator(x)
        if apply_sigmoid:
            images = tf.sigmoid(logits)
            return images

        return logits

    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_imgs = self.generator(x)
            fake_output = self.discriminator(generated_imgs)
            real_output = self.discriminator(x)

            cross_ent = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            real_loss = cross_ent(tf.ones_like(real_output), real_output)
            fake_loss = cross_ent(tf.zeros_like(fake_output), fake_output)

            discriminator_loss = real_loss + fake_loss
            generator_loss = cross_ent(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


if __name__ == "__main__":
    import time
    import glob
    import matplotlib.pyplot as plt
    import imageio

    (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100

    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    epochs = 100
    latent_dim = 50
    num_examples_to_generate = 16


    class Generator(tf.keras.Model):
        def __init__(self, input_shape, latent_dim):
            super().__init__(name="Generator")
            self.conv1 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2))
            self.leaky_relu1 = tf.keras.layers.LeakyReLU()
            self.dropout1 = tf.keras.layers.Dropout(0.3)
            self.conv2 = tf.keras.layers.Conv2D(
                filters=128, kernel_size=5, strides=(2, 2))
            self.leaky_relu2 = tf.keras.layers.LeakyReLU()
            self.dropout2 = tf.keras.layers.Dropout(0.3)
            # No activation
            self.flatten = tf.keras.layers.Flatten()  # (batch, X)
            self.latent_vars = tf.keras.layers.Dense(latent_dim)

            # input: (batch, latent_dim)
            self.fc1 = tf.keras.layers.Dense(units=7 * 7 * 256, use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.leaky_relu3 = tf.keras.layers.LeakyReLU()
            self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 256))

            self.deconv1 = tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=5, strides=(1, 1), padding="same", use_bias=False)
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.leaky_relu4 = tf.keras.layers.LeakyReLU()

            self.deconv2 = tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=5, strides=(2, 2), padding="same", use_bias=False)
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.leaky_relu5 = tf.keras.layers.LeakyReLU()
            # No activation
            self.deconv3 = tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=5, strides=(2, 2), padding="same", use_bias=False, activation='tanh')

            dummy_input = tf.constant(
                np.zeros(shape=(1,) + input_shape, dtype=np.float32))
            with tf.device("/cpu:0"):
                self(dummy_input)

        def call(self, inputs):
            latent_vars = self.encode(inputs)
            reconstructed = self.decode(latent_vars)
            return reconstructed

        def encode(self, inputs):
            features = self.conv1(inputs)
            features = self.leaky_relu1(features)
            features = self.dropout1(features)
            features = self.conv2(features)
            features = self.leaky_relu2(features)
            features = self.dropout2(features)
            features = self.flatten(features)
            latent_vars = self.latent_vars(features)
            return latent_vars

        def decode(self, latent_vars):
            features = self.fc1(latent_vars)
            features = self.bn1(features)
            features = self.leaky_relu3(features)
            features = self.reshape(features)
            features = self.deconv1(features)
            features = self.bn2(features)
            features = self.leaky_relu4(features)
            features = self.deconv2(features)
            features = self.bn3(features)
            features = self.leaky_relu5(features)
            reconstructed = self.deconv3(features)
            return reconstructed


    class Discriminator(tf.keras.Model):
        def __init__(self, input_shape):
            super().__init__(name="Discriminator")
            self.conv1 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2))
            self.leaky_relu1 = tf.keras.layers.LeakyReLU()
            self.dropout1 = tf.keras.layers.Dropout(0.3)
            self.conv2 = tf.keras.layers.Conv2D(
                filters=128, kernel_size=5, strides=(2, 2))
            self.leaky_relu2 = tf.keras.layers.LeakyReLU()
            self.dropout2 = tf.keras.layers.Dropout(0.3)
            # No activation
            self.flatten = tf.keras.layers.Flatten()  # (batch, X)
            self.logit = tf.keras.layers.Dense(1)

            dummy_input = tf.constant(
                np.zeros(shape=(1,) + input_shape, dtype=np.float32))
            with tf.device("/cpu:0"):
                self(dummy_input)

        def call(self, inputs):
            features = self.conv1(inputs)
            features = self.leaky_relu1(features)
            features = self.dropout1(features)
            features = self.conv2(features)
            features = self.leaky_relu2(features)
            features = self.dropout2(features)
            features = self.flatten(features)
            logit = self.logit(features)
            return logit


    # inference_net = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    #         tf.keras.layers.Conv2D(
    #             filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
    #         tf.keras.layers.Conv2D(
    #             filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
    #         tf.keras.layers.Flatten(),
    #         # No activation
    #         tf.keras.layers.Dense(latent_dim),
    #     ]
    # )

    # generative_net = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    #         tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
    #         tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64,
    #             kernel_size=3,
    #             strides=(2, 2),
    #             padding="SAME",
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=32,
    #             kernel_size=3,
    #             strides=(2, 2),
    #             padding="SAME",
    #             activation='relu'),
    #         # No activation
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
    #     ]
    # )

    generator = Generator(input_shape=(28, 28, 1), latent_dim=64)
    discriminator = Discriminator(input_shape=(28, 28, 1))

    model = GAN(latent_dim, generator, discriminator)


    def generate_and_save_images(model, epoch):
        plt.close()
        fig = plt.figure(figsize=(4, 4))

        for i in range(8):
            restored_img = model.reconstruct(np.expand_dims(test_images[i], axis=0), apply_sigmoid=True)

            plt.subplot(4, 4, 2 * i + 1)
            plt.imshow(test_images[i, :, :, 0], cmap='gray')
            plt.tick_params(axis='both', labelsize=0, length=0)
            plt.subplot(4, 4, 2 * i + 2)
            plt.imshow(restored_img[0, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


    # generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            model.compute_apply_gradients(train_x)
        end_time = time.time()

        if epoch % 1 == 0:
            # loss = tf.keras.metrics.Mean()
            # for test_x in test_dataset:
            #     loss(model.compute_loss(test_x))
            # elbo = -loss.result()
            # print('Epoch: {}, Test set ELBO: {}, '
            #       'time elapse for current epoch {}'.format(epoch,
            #                                                 elbo,
            #                                                 end_time - start_time))
            generate_and_save_images(model, epoch)

    anim_file = 'vae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
