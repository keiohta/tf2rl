import numpy as np
from skimage.util.shape import view_as_windows


def random_crop(input_imgs, output_size):
    """

    Args:
        input_imgs: np.ndarray
            Images whose shape is (batch_size, width, height, channels)
        output_size: Int
            Output width and height size.

    Returns:

    """
    assert input_imgs.ndim == 4, f"The dimension of input images must be 4, not {len(input_imgs)}"

    batch_size = input_imgs.shape[0]
    img_size = input_imgs.shape[1]

    assert img_size > output_size
    crop_max = img_size - output_size

    topleft_x = np.random.randint(0, crop_max, batch_size)
    topleft_y = np.random.randint(0, crop_max, batch_size)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        input_imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(batch_size), topleft_x, topleft_y]
    return np.transpose(cropped_imgs, (0, 2, 3, 1))


def center_crop(img, output_size):
    """

    Args:
        img: np.ndarray
            Input image array. The shape is (width, height, channel)
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


if __name__ == "__main__":
    batch_size = 64
    channels = 9
    w, h = 100, 100
    output_size = 84
    imgs = np.zeros(shape=(64, w, h, channels), dtype=np.float32)
    randomly_cropped_imgs = random_crop(imgs, output_size)
    print(imgs.shape, randomly_cropped_imgs.shape)
