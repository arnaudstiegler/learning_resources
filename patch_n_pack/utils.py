import numpy as np
import math
import cv2
import torch


def resize_with_aspect_ratio(
        pixel_values: np.ndarray, effective_resolution: int
):
    """

    :param pixel_values: of shape (batch_size, num_channels, x_size, y_size)
    :return:
    """

    # TODO: might have to be moved to the preprocessor
    _, _, x_size, y_size = pixel_values.shape

    aspect_ratio = x_size / y_size

    new_y = np.sqrt(effective_resolution ** 2 / aspect_ratio)
    new_x = new_y * aspect_ratio

    rounded_y = math.floor(new_y)
    rounded_x = math.floor(new_x)
    res = cv2.resize(
        torch.permute(pixel_values.squeeze(0), (1,2,0)).numpy(),
        dsize=(rounded_y, rounded_x),
        interpolation=cv2.INTER_CUBIC,
    )
    img = (
        torch.tensor(res, dtype=torch.float)
        .view(3, rounded_x, rounded_y)
        .unsqueeze(0)
    )
    return img