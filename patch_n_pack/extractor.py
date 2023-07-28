from transformers import ViTImageProcessor
import numpy as np
import math
import cv2
import PIL


def to_numpy_array(img) -> np.ndarray:
    if isinstance(img, PIL.Image.Image):
        return np.array(img)
    # if it is a torch tensor
    return img.detach().cpu().numpy()


class PatchPackProcessor(ViTImageProcessor):
    def __init__(self):
        super().__init__()

    def resize_with_aspect_ratio(self, pixel_values: np.ndarray, effective_resolution: int):
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
        res = cv2.resize(pixel_values.numpy(), dsize=(rounded_y, rounded_x),
                         interpolation=cv2.INTER_CUBIC)
        img = torch.tensor(res, dtype=torch.float).view(3, rounded_x, rounded_y).unsqueeze(0)
        return img

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ):
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size_dict, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
