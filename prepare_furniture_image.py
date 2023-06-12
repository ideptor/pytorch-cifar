import pandas as pd 
import os
from PIL import Image
from tqdm import tqdm

def run(src:str, dst:str):
    df = pd.read_csv(f'{src}/furniture_data_img.csv')
    df.key = df.Image_File.str.replace("/furniture_images/","")
    df.value = df.Furniture_Type.map(lambda x: x.split("/")[0].strip())
    label_dict = dict(zip(list(df.key), list(df.value)))
    # print(label_dict)

    files = os.listdir(f"{src}/furniture_images")
    pbar = tqdm(files)  
    for file in pbar:
        label = label_dict.get(file)
        path = f'{dst}/{label}'
        pbar.set_description(f'{file} / {path}')

         
        if label.lower() == "sofa":
            for keyword in [
                "Bar Stool","Uncommon Display Rack","cupboard" ,"cabinet","mdf display unit","steel almirah code" ,
                "display unit", "Bookshelf","Cupboards", "Cabinet", "display unit", "chair", "Almirah"
                ]:
                if keyword.lower() in file.lower():
                    continue
            
        elif label.lower() == "table":
            for keyword in [
                "chair", "wardrobe",
            ]:
                if keyword.lower() in file.lower():
                    continue
 
        elif label.lower() == "bed":
            for keyword in [
                "cupboard", "wardrobe", "dressing table","board code","cupboard",
                "Door Based Almary", "Mahagony", "teak wood furnitures", "TEAK BED SNP",
                "Door Base Almary", "Box Teak", "Teak Cushioned", "BedNew Teak",
                "Almare", "Bedrooms Items", "Plywood Doors","Bed Sheets",
                "TEAK BED SP","Cabinet", "BOX BED", "pillow", "Cube Diy Plastic",
                "Blanket"
            ]:
                if keyword.lower() in file.lower():
                    continue

        
        
        os.makedirs(path, exist_ok=True)
        Image.open(f'{src}/furniture_images/{file}').save(f'{path}/{file}')
        # print(file, label_dict.get(file))

    pbar.close()


if __name__ == '__main__':
    run('furniture_images_origin', 'furniture_images')
