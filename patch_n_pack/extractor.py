from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.image_processing_utils import BatchFeature


class PatchPackProcessor(ViTImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_resize = False

    def __call__(self, images, **kwargs) -> BatchFeature:
        images = super().preprocess(images=images, **kwargs)
        return images
