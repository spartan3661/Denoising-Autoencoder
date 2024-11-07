import os
import requests
from tqdm import tqdm
import zipfile

train_image_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
valid_image_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'

train_image_file = 'DIV2K_train_HR.zip'
valid_image_file = 'DIV2K_valid_HR.zip'


def download_file(url, output_path):
    if not os.path.isfile(output_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-lenth', 0))

        chunk_size = 1024

        with open(output_path, 'wb') as file, tqdm(
            desc=f"Downloading {output_path}",
            total = total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"{zip_path} does not exist")
        return
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"extracted {zip_path} to {extract_to}")




download_file(train_image_url, train_image_file)
download_file(valid_image_url, valid_image_file)

extract_zip(train_image_file, "./")
extract_zip(valid_image_file, "./")

