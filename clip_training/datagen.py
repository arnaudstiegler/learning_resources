import os
import signal
import urllib
from contextlib import contextmanager

import click
import pandas as pd
from torchvision.io import read_image
from transformers import ViTFeatureExtractor
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def validate_image(feature_extractor: ViTFeatureExtractor, dest: str) -> None:
    # sanity check on downloaded images
    for file in tqdm(os.listdir(dest)):
        if file.endswith('.jpg'):
            try:
                image = read_image(os.path.join(dest, file))
                _ = feature_extractor(image)
            except Exception as e:
                print(e)
                print(f'Deleting: images/{file}')
                os.remove(os.path.join(dest, f'{file}'))


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def download_img(dest, row_tuple):
    _, row = row_tuple
    if not os.path.exists(os.path.join(dest, f"{int(row['SAMPLE_ID'])}.jpg")):
        try:
            with time_limit(1):
                urllib.request.urlretrieve(row['URL'], os.path.join(dest, f"{int(row['SAMPLE_ID'])}.jpg"))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            pass
    return


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
