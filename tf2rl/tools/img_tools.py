import numpy as np
import tensorflow as tf


def random_crop(input_imgs, output_size, is_tf=False):
    """

    Args:
        input_imgs: np.ndarray or tf.Tensor
            Images whose shape is (batch_size, width, height, channels)
        output_size: Int
            Output width and height size.

    Returns:
        cropped_imgs: np.ndarray or tf.python.framework.ops.EagerTensor
            Cropped images whose shape is (batch_size, output_size, output_size, channels)

    """
    if is_tf:
        assert tf.rank(input_imgs).numpy() == 4, f"The dimension of input images must be 4, not {tf.rank(input_imgs)}"
    else:
        assert input_imgs.ndim == 4, f"The dimension of input images must be 4, not {len(input_imgs)}"

    batch_size, img_size, _, channels = input_imgs.shape

    assert img_size > output_size
    crop_max = img_size - output_size + 1

    if is_tf:
        cropped_imgs = []
        for cur_img in input_imgs:
            cropped_imgs.append(tf.image.random_crop(cur_img, size=[output_size, output_size, channels]))
        return tf.stack(cropped_imgs, axis=0)
    else:
        topleft_xs = np.random.randint(0, crop_max, batch_size)
        topleft_ys = np.random.randint(0, crop_max, batch_size)

        cropped_imgs = np.empty((batch_size, output_size, output_size, channels), dtype=input_imgs.dtype)

        for i, (cur_img, topleft_x, topleft_y) in enumerate(zip(input_imgs, topleft_xs, topleft_ys)):
            cropped_imgs[i] = cur_img[topleft_x:topleft_x + output_size, topleft_y:topleft_y + output_size, :]
        return cropped_imgs


def center_crop(img, output_size):
    """

    Args:
        img: np.ndarray
            Input image array. The shape is (batch_size, width, height, channel)
            It also accepts single image, i.e., (width, height, channel) image
        output_size: int
            Width and height size for output image

    Returns:

    """
    is_single_img = img.ndim == 3

    h, w = img.shape[:2] if is_single_img else img.shape[1:3]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if is_single_img:
        return img[top:top + new_h, left:left + new_w, :]
    else:
        return img[:, top:top + new_h, left:left + new_w, :]


def grayscale(input_imgs):
    # imgs: b x c x h x w
    batch_size, height, width, channels = input_imgs.shape
    frames = channels // 3

    output_imgs = np.empty((batch_size, height, width, channels), dtype=input_imgs.dtype)
    for frame in range(frames):
        grayscale_img = np.dot(input_imgs[..., 3 * frame: 3 * (frame + 1)], [0.2989, 0.5870, 0.1140])
        stacked_grayscale_imgs = np.stack((grayscale_img,) * 3, axis=-1)
        output_imgs[..., 3 * frame: 3 * (frame + 1)] = stacked_grayscale_imgs

    return output_imgs


def preprocess_img(img, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    if bits < 8:
        obs = tf.cast(tf.floor(img / 2 ** (8 - bits)), dtype=tf.float32)
    obs = obs / bins
    obs = obs + tf.random.uniform(shape=obs.shape) / bins
    obs = obs - 0.5
    return obs


if __name__ == "__main__":
    batch_size = 64
    channels = 9
    w, h = 100, 100
    output_size = 84
    imgs = np.zeros(shape=(64, w, h, channels), dtype=np.float32)
    randomly_cropped_imgs = random_crop(imgs, output_size)
    print(imgs.shape, randomly_cropped_imgs.shape)

    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)

    img = np.zeros(shape=(64, 84, 84, 9), dtype=np.uint8)
    preprocess_img(img)
