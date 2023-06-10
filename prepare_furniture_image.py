import pandas as pd 
import os
from PIL import Image
from tqdm import tqdm

def run(src:str, dst:str):
    df = pd.read_csv(f'{src}/furniture_data_img.csv')
    df.key = df.Image_File.str.replace("/furniture_images/","")
    df.value = df.Furniture_Type.map(lambda x: x.split("/")[0].strip())
    label_dict = dict(zip(list(df.key), list(df.value)))
    print(label_dict)

    files = os.listdir(f"{src}/furniture_images")
    pbar = tqdm(files)
    for file in pbar:
        path = f'{dst}/{label_dict.get(file)}'
        pbar.set_description(f'{file} / {path}')
        os.makedirs(path, exist_ok=True)
        Image.open(f'{src}/furniture_images/{file}').save(f'{path}/{file}')
        # print(file, label_dict.get(file))

    pbar.close()


if __name__ == '__main__':
    run('furniture_images_origin', 'furniture_images')
