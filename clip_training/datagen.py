import os
import signal
import urllib
from contextlib import contextmanager
import click
import pandas as pd
from torchvision.io import read_image
from transformers import ViTFeatureExtractor


def validate_image(feature_extractor: ViTFeatureExtractor, dest: str) -> None:
    # sanity check on downloaded images
    for file in os.listdir(dest):
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


def download_data_locally(df: pd.DataFrame, dest: str) -> None:
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    for i, row in df.iterrows():
        if not os.path.exists(os.path.join(dest, f"{i}.jpg")):
            try:
                with time_limit(10):
                    urllib.request.urlretrieve(row['URL'], os.path.join(dest, f"{i}.jpg"))
            except TimeoutException:
                print("Timed out!")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'{row["URL"]} failed with exception={e}')

        validate_image(feature_extractor, dest)


def read_parquet_data(local_path: str) -> pd.DataFrame:
    df = pd.read_parquet(local_path)
    df = df.loc[df['NSFW'] == 'UNLIKELY']
    return df


@click.command()
@click.argument('filepath')
@click.argument('dest')
def main(filepath: str, dest: str) -> None:
    data_df = read_parquet_data(filepath)
    download_data_locally(data_df, dest)


if __name__ == '__main__':
    main()
