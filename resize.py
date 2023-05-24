from argparse import ArgumentParser
import os
import pathlib


def get_args() -> dict:
    parser = ArgumentParser(
        prog="resize images",
        description="resize images for designated size",
    )

    parser.add_argument("--input", "-i", type=str, default="data")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--size", "-s", type=int, default=128)

    # for label in os.listdir(root):
    #     for image in os.listdir(f'{root}/{label}'):
    #         img_path = f'{root}/{label}/{image}'
    #         print(img_path)
                                
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)                        