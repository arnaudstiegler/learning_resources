import os

import click
import pandas as pd
from transformers import ViTFeatureExtractor
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from clip_training.utils import validate_image, download_img


def download_data_locally(df: pd.DataFrame, dest: str) -> None:
    with Pool(30) as p:
        func = partial(download_img, dest)
        r = list(tqdm.tqdm(p.imap(func, df.iterrows()), total=df.shape[0]))

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    validate_image(feature_extractor, dest)


def read_parquet_data(local_path: str) -> pd.DataFrame:
    df = pd.read_parquet(local_path)
    df = df.loc[df['NSFW'] == 'UNLIKELY']
    return df


def get_filtered_df_for_training(filepath: str, images_path: str) -> pd.DataFrame:
    df = read_parquet_data(filepath)
    df = df.reset_index()
    df = df.rename(columns={"index": "image_index"})

    # Filter with the image present in the images folder
    images = [int(elem.replace('.jpg', '')) for elem in os.listdir(images_path) if
              elem != '.ipynb_checkpoints']
    df['has_image'] = df['SAMPLE_ID'].isin(images)

    # Final filter
    df = df.loc[df.has_image]
    df = df.reset_index(drop=True)

    return df


@click.command()
@click.option('--filepath')
@click.option('--dest')
def main(filepath: str, dest: str) -> None:
    data_df = read_parquet_data(filepath)
    download_data_locally(data_df, dest)


if __name__ == '__main__':
    main()
