from PIL import Image
from patch_n_pack.utils import resize_with_aspect_ratio, set_up_model


def run_inference(extractor, model, image):
    # TODO: need to redefine the extractor so that it tries to match the aspect ratio of the image
    # and packs images
    inputs = extractor(images=image, return_tensors="pt")

    x = resize_with_aspect_ratio(inputs.pixel_values, 512)

    outputs = model(pixel_values=x)
    logits = outputs.logits
    return logits


if __name__ == "__main__":
    image = Image.open("patch_n_pack/images/10003.jpeg")
    processor, model = set_up_model()
    run_inference(processor, model, image)