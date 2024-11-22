import os
import random
import torch
import zipfile
import urllib.request
from pycocotools.coco import COCO
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from tqdm import tqdm
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
BASE_DIR = config["base_dir"]
TRAIN_LIMIT = config["train_limit"]
VALID_LIMIT = config["valid_limit"]
RESIZE_SHAPE = tuple(config["resize_shape"])

TRAIN_CLEAN_DIR = config["train_clean_dir"]
TRAIN_NOISY_DIR = config["train_noisy_dir"]
VALID_CLEAN_DIR = config["valid_clean_dir"]
VALID_NOISY_DIR = config["valid_noisy_dir"]
ANNOTATIONS_DIR = config["annotations_dir"]
ANNOTATIONS_ZIP_URL = config["annotations_zip_url"]

NOISE_PARAMETERS = config["noise_parameters"]

os.makedirs(TRAIN_CLEAN_DIR, exist_ok=True)
os.makedirs(TRAIN_NOISY_DIR, exist_ok=True)
os.makedirs(VALID_CLEAN_DIR, exist_ok=True)
os.makedirs(VALID_NOISY_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


def add_noise(img, noise_type=NOISE_PARAMETERS["noise_type"], stddev=NOISE_PARAMETERS["stddev"], sp_prob=NOISE_PARAMETERS["sp_prob"]):
    if noise_type == 'gaussian':
        noise = torch.randn_like(img) * stddev
        noisy_img = img + noise
    elif noise_type == 'artefacts':
        artefact_mask = (torch.randn_like(img) > 0.95).float()
        artefact_noise = torch.randn_like(img) * artefact_mask
        noisy_img = img + artefact_noise
    elif noise_type == 'salt_and_pepper':
        salt_pepper_mask = torch.randn_like(img)
        salt_mask = (salt_pepper_mask < (sp_prob / 2)).float()
        pepper_mask = (salt_pepper_mask > (1 - sp_prob / 2)).float()
        noisy_img = img * (1 - salt_mask - pepper_mask) + salt_mask
    else:
        raise ValueError(f"Unknown noise type {noise_type}. Set blank to use gaussian.")
    return torch.clamp(noisy_img, 0.0, 1.0)

def download_annotations():
    annotations_zip_path = os.path.join(ANNOTATIONS_DIR, 'annotations_trainval2017.zip')

    if not os.path.exists(annotations_zip_path):
        print("Downloading annotations...")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def reporthook(block_num, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_num * block_size - pbar.n)

            urllib.request.urlretrieve(ANNOTATIONS_ZIP_URL, annotations_zip_path, reporthook=reporthook)
        print("Annotations downloaded.")

    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        print("Extracting annotations...")
        zip_ref.extractall(ANNOTATIONS_DIR)
        print("Annotations extracted.")

def download_and_process_coco(dataset_type='train', limit=100):
    if dataset_type == 'train':
        ann_file = os.path.join(ANNOTATIONS_DIR, 'annotations', 'instances_train2017.json')
    else:
        ann_file = os.path.join(ANNOTATIONS_DIR, 'annotations', 'instances_val2017.json')

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    img_ids = img_ids[:limit]

    transform = transforms.Compose([
        transforms.Resize(RESIZE_SHAPE),
        transforms.ToTensor()
    ])

    if dataset_type=="test" and not os.path.exists(dataset_type):
        clean_img_dir = os.path.join(dataset_type, "clean")
        noisy_img_dir = os.path.join(dataset_type, "noisy")

        os.makedirs(clean_img_dir)
        os.makedirs(noisy_img_dir)

    print(f"Processing {dataset_type} images...")
    for img_id in tqdm(img_ids, desc=f"Downloading and processing {dataset_type} images"):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']

        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_tensor = transform(img)
        except Exception as e:
            print(f"Could not load image {img_url}: {e}")
            continue

        if dataset_type == "test":
            clean_img_path = os.path.join(dataset_type, "clean", f"{img_id}.png")
            transforms.ToPILImage()(img_tensor).save(clean_img_path)
            
            noisy_img_tensor = add_noise(img_tensor)
            noisy_img_path = os.path.join(dataset_type, "noisy", f"{img_id}.png")
            transforms.ToPILImage()(noisy_img_tensor).save(noisy_img_path)

        else:

            if random.random() < 0.8:
                clean_dir = TRAIN_CLEAN_DIR
                noisy_dir = TRAIN_NOISY_DIR
            else:
                clean_dir = VALID_CLEAN_DIR
                noisy_dir = VALID_NOISY_DIR

            clean_img_path = os.path.join(clean_dir, f"{img_id}.png")
            transforms.ToPILImage()(img_tensor).save(clean_img_path)

            noisy_img_tensor = add_noise(img_tensor)
            noisy_img_path = os.path.join(noisy_dir, f"{img_id}.png")
            transforms.ToPILImage()(noisy_img_tensor).save(noisy_img_path)
if __name__=="__main__":
        
    download_annotations()
    download_and_process_coco(dataset_type='train', limit=TRAIN_LIMIT)
    download_and_process_coco(dataset_type='val', limit=VALID_LIMIT)

    print("Processing complete.")
