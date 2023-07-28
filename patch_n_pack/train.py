from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image


def set_up_model():
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return extractor, model


def run_inference(extractor, model, image):
    # TODO: need to redefine the extractor so that it tries to match the aspect ratio of the image
    # and packs images
    inputs = extractor(images=image, return_tensors="pt")
    import ipdb; ipdb.set_trace()
    outputs = model(**inputs)
    logits = outputs.logits
    return logits


if __name__ == "__main__":
    image = Image.open('patch_n_pack/images/10003.jpeg')
    extractor, model = set_up_model()
    run_inference(extractor, model, image)