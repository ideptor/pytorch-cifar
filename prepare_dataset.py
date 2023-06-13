from argparse import ArgumentParser
from collections import defaultdict
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil
import pandas as pd

def get_args() -> dict:
    parser = ArgumentParser(
        prog="resize images",
        description="resize images for designated size",
    )

    parser.add_argument("--input", "-i", type=str, default="data")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--size", "-s", type=int, default=128)
    parser.add_argument("--augment", "-a", type=int, default=1)
    parser.add_argument("--train-ratio", "-r", type=float, default=.8)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def prepare_train_test_images(src: str, dst: str, size: int, train_ratio: float=.8, augment_size: int=1, verbose: bool=False):

    summary_report = dict()

    images = list()
    if os.path.exists(dst):
        shutil.rmtree(dst)

    for label in os.listdir(src):

        label_images = list(map(str, list(os.listdir(f'{src}/{label}'))))
        random.shuffle(label_images)
        split_index = int(len(label_images)*train_ratio)
        images.extend([f'{dst}/train/{label}/{image}' for image in label_images[:split_index]])
        images.extend([f'{dst}/val/{label}/{image}' for image in label_images[split_index:]])

        os.makedirs(f'{dst}/train/{label}', exist_ok=True)
        os.makedirs(f'{dst}/val/{label}', exist_ok=True)

    

    summary_report['total_images'] = len(images)
    summary_report['train'] = defaultdict(int)
    summary_report['val'] = defaultdict(int)

    pbar = tqdm(images)
    for image in pbar:
        image_org = f'{src}/{image.split("/",2)[2]}'
        try:
            pbar.set_description(image_org[:35])
            _, phase, label, _  = image.split("/", 4)
            
            if phase == "train":
                for i in range(augment_size):
                    Image.open(image_org).resize((size,size)).save(image.rsplit(".",2)[0]+f"_{i}.{image.rsplit('.',2)[1]}")
                    summary_report[phase][label] += 1
            else:
                Image.open(image_org).resize((size,size)).save(image)
                summary_report[phase][label] += 1

        except OSError as e:
            print(e, ' :: ', image_org)
    pbar.close()

    data = list()
    if verbose:
        print(
            f"Total images: {summary_report['total_images']}"
        )
        for phase in ['train', 'val']:
            for label, count in summary_report[phase].items():
                data.append(
                    dict(
                    phase=phase,
                    label=label,
                    count=count,
                    )
                )
        df = pd.DataFrame(data)
        print(pd.crosstab(df.label, df.phase, values=df['count'], aggfunc=np.sum, margins=True))

if __name__ == '__main__':
    args = get_args()
    prepare_train_test_images(args.input, args.output, args.size, args.train_ratio, args.augment, args.verbose)
