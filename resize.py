from argparse import ArgumentParser
import os
import random
from tqdm import tqdm
from PIL import Image
import shutil


def get_args() -> dict:
    parser = ArgumentParser(
        prog="resize images",
        description="resize images for designated size",
    )

    parser.add_argument("--input", "-i", type=str, default="data")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--size", "-s", type=int, default=128)

    return parser.parse_args()


def prepare_train_test_images(src: str, dst: str, size: int, train_ratio: float=.8):

    images = list()
    if os.path.exists(dst):
        shutil.rmtree(dst)

    for label in os.listdir(src):

        label_images = list(map(str, list(os.listdir(f'{src}/{label}'))))
        random.shuffle(label_images)
        split_index = int(len(label_images)*train_ratio)
        images.extend([f'{dst}/train/{label}/{image}' for image in label_images[:split_index]])
        images.extend([f'{dst}/test/{label}/{image}' for image in label_images[split_index:]])

        os.makedirs(f'{dst}/train/{label}', exist_ok=True)
        os.makedirs(f'{dst}/test/{label}', exist_ok=True)

    pbar = tqdm(images)
    for image in pbar:
        image_org = f'{src}/{image.split("/",2)[2]}'
        try:
            pbar.set_postfix(org=image_org, dst=image)
            Image.open(image_org).resize((size,size)).save(image)

        except OSError as e:
            print(e, ' :: ', image_org)
    pbar.close()


if __name__ == '__main__':
    args = get_args()
    prepare_train_test_images(args.input, args.output, args.size)
