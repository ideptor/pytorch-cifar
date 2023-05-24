from argparse import ArgumentParser
import os
import pathlib
from tqdm import tqdm



def get_args() -> dict:
    parser = ArgumentParser(
        prog="resize images",
        description="resize images for designated size",
    )

    parser.add_argument("--input", "-i", type=str, default="data")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--size", "-s", type=int, default=128)

    return parser.parse_args()

def resize(src: str, dst: str, size: int):
    pbar = tqdm(os.listdir(src))
    for label in pbar:
        os.makedirs(f'{dst}/{label}', exist_ok=True)
        for image in os.listdir(f'{src}/{label}'):
            img_src_path = f'{src}/{label}/{image}'
            img_dst_path = f'{dst}/{label}/{image}'.replace('.jpeg', '.jpg')
            pbar.set_description(f'{img_src_path}>{img_dst_path}')

if __name__ == '__main__':
    args = get_args()
    resize(args.input, args.output, args.size)
    print(args)
