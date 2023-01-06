import click
from clip_training.datagen import validate_image
from transformers import ViTFeatureExtractor


@click.command()
@click.option('--images_folder')
def main(images_folder: str) -> None:
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    validate_image(feature_extractor, images_folder)


if __name__ == '__main__':
    main()
