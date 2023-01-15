import os
import signal
import urllib
from contextlib import contextmanager

from torchvision.io import read_image
from tqdm import tqdm
from transformers import ViTFeatureExtractor


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
