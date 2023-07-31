from patch_n_pack.extractor import PatchPackProcessor
from patch_n_pack.model import PatchPackModelImageClassification
from PIL import Image
from transformers import ViTConfig


def set_up_model():
    config = ViTConfig()
    model = PatchPackModelImageClassification(config)
    processor = PatchPackProcessor()
    return processor, model


def run_inference(extractor, model, image):
    # TODO: need to redefine the extractor so that it tries to match the aspect ratio of the image
    # and packs images
    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    return logits


if __name__ == "__main__":
    image = Image.open("patch_n_pack/images/10003.jpeg")
    processor, model = set_up_model()
    run_inference(processor, model, image)
