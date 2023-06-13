from argparse import ArgumentParser
import pandas as pd 
import os
from PIL import Image
from tqdm import tqdm
import shutil
from collections import defaultdict

def run(src:str, dst:str, dropped:bool=False):
    
    if os.path.exists(dst):
        shutil.rmtree(dst)
    df = pd.read_csv(f'{src}/furniture_data_img.csv')
    df.key = df.Image_File.str.replace("/furniture_images/","")
    df.value = df.Furniture_Type.map(lambda x: x.split("/")[0].strip())
    label_dict = dict(zip(list(df.key), list(df.value)))
    # print(label_dict)

    files = os.listdir(f"{src}/furniture_images")
    pbar = tqdm(files)  
    img_cnt = defaultdict(int)
    for file in pbar:
        label = label_dict.get(file)
        
        skip = False
        if label.lower() == "sofa":
            for keyword in set([
                "Bar Stool","Uncommon Display Rack","cupboard" ,"cabinet","mdf display unit","steel almirah code" ,
                "display unit", "Bookshelf","Cupboards", "Cabinet", "display unit",  "Almirah",
                "Dining Set teak Wood", "Carpet",  "1634012078367_", "Pillow", "Wooden Chairs","furniture", "wooden sofa", "teak living room chairs",
                "1634018211017","1634018211855","1634019835766","1634019809189","Shoe Rack", "1634011857519",
                "Dressing Table", "Glass Tabel", "Table Mate II Adjustable Folding",  "1634020222501",
                "1634020591362_", "1634022196877", "1634022289230", "Pallet Furniture Set",  "Cupbord",
                "Sofa Set with table", "1634026897006", "1634027114883", "telephone stand", "Dressing Table",
                "Original Teak Wood", "Portable Folding Laptop Table", "1634028127674", "Cushion Covers", "1634028652232",
                "Laptop Cooling Fan", "Study Table", "Table Mate", "1634014958500", "1634016221480", "folding laptop reading table",
                "Dinning","Laptop Table","1634017166949","1634017417813","Tv Stand","Mirror", "1634021379361","Drawyers",
                "Photo Stand","Dining","Display Rack","1634023768602",  "Folding Laptop Reading Table","Coner Stand",
                "1634025375002","1634025456471","Wardrobe","Back Mesh Chairs","1634021286664","Cushion Cover",
                "1634022534118","1634022571110","folding laptop","1634023700540","1634025684993","1634027513536","1634027768532","1634027918004","Swing Chairs"
            ]):
                if keyword.lower() in file.lower():
                    label=f"dropped/{label}"
                    break
            
        elif label.lower() == "table":
            for keyword in set([
                "chair", "wardrobe", "Bar Stool", "Sofa Set", "Reception Table", "Plastic Stool", "Rattan Stool",
                "Table Mate", "Stool for sale", "sofa set", "1634019132210","1634019488708", "Dining Table",
                "1634021033941","1634021036659","1634021037152", "1634021210500", "1634021388343", "1634023466906", 
                "1634024280163", "1634025321910", "1634025456144", "1634025409898", "1634025456851", "Bar Stool" ,
                "1634026127875", "1634026897341", "1634027084498", "Tablemate II ", "Table mate II",
                "1634027268480", "Office Furniture", "1634027583375", "1634027847262", "Almary", "1634028512878",
                "Used Furnitures", "1634028714623", "Dining", "1634016333748", "bed", "cupboards","Laptop Stand",
                "Wall lamp","1634018665604","1634019658949","1634020334009","1634020557428", "Dinning", "Almirah",
                "1634021755657","1634023883037","Laptop Desk","1634024014338","1634024160432","1634024456264",
                "1634024663830","1634024530231","1634025322280","1634025571912","1634025571577","1634025820111","1634026273990","1634027002245",
                "1634027042302","1634027336034","1634027579263","1634027875004","1634028472938","1634028579569",
            ]):
                if keyword.lower() in file.lower():
                    label=f"dropped/{label}"
                    break
 
        elif label.lower() == "bed":
            for keyword in set([
                "cupboard", "wardrobe", "dressing table","board code","cupboard",
                "Door Based Almary", "Mahagony", "teak wood furnitures", "TEAK BED SNP",
                "Door Base Almary", "Box Teak", "Teak Cushioned", "BedNew Teak",
                "Almare", "Bedrooms Items", "Plywood Doors","Bed Sheets",
                "TEAK BED SP","Cabinet", "pillow", "Cube Diy Plastic",
                "Blanket", "Cushion Set", "Multi wood kanappu",
                "room side Stools", "Telephone Stand", "1634025941244", "1634026778343",
                "Mirror", "1634012202907", "Teak Bedroom Set", "1634015102208","1634015539313",
                "Steel Bunker Beds", "1634015884661", "Bunk Bed", "Almirah", "Bedroom Set Theak",
                "Almari", "Bedroom Set", "1634016544660", "1634016544660", "1634016885504", "Almary"
                "Almirah", "1634016886223", "Steel Bed", "Table Mate 2", "1634017496124","Table Mate II",
                "1634017703462","1634017743328","1634017744798","1634017920857","1634017996331",
                "1634018060215", "1634018029708", "1634018060215", "Foldable Towel racks","Arch Teak 6x",
                "1634018330765","1634018365833","1634018366402","1634018509146","1634018550837",
                "1634018589785","1634018589963","1634018590485","1634018591272","1634018591817","1634018551577",
                "1634018593610", "1634018594136", "1634018589270","Almarry", "1634019062415", "1634019063445",
                "1634019094264", "1634019095607", "1634019096059", "1634019097426", "1634019134826", "1634019268145",
                "1634019437165", "1634019552968", "1634019555110", "Almariya", "1634019587177", "1634019806130",
                "Heavy Modern Bedroom Set","1634019913152","1634019912212","1634019951723", "1634019953005", "Cube Storage",
                "Bedsheet","Teak Bedroom SetPackage", "cushion chairs", "1634020331888", "1634020923382",
                "1634021001483", "1634021062355","1634021063790","1634013323734","1634013325825","1634014855776","1634014920227",
                "1634014960634", "1634014994607","1634014996851","Steel Bunker Bed","single steel","1634015380231",
                "Bed Sheet","1634015466090","1634015464373","1634015494290","Wooden Clothes Rack","1634015582873","1634015630591",
                "1634015769974","Steel Plywood Bed","1634015850836","Steel Bunker Bed", "steel single", "1634016116880","1634016038572",
                "Study Table", "1634016328620", "1634016449993","Almary","1634017122018","1634017122982", "1634017162425",
                "1634017163483", "1634017163886", "Sofa", "1634017277448",
                "1634017420499","almarh","Bedroom Furniture","1634017419221","1634017420306","1634017420499","1634017421031",
                "1634017536429", "1634017564150","1634017640734","1634018329445","1634018330065", "1634018329858", "1634018330273",
                "1634018330976", "1634018552252","Iron Bed","1634018365243","1634018552252","1634018991861","1634019311669","Bed Room Set",
                "1634020144438","1634019909779","1634019912620","Bedroom Furniture Set","1634020413637","1634020144438","1634020370339",
                "Theme Room Set","steel single bed","1634020922417","1634020927098","1634020928815","1634020996725","1634021099258",
                "Foldable Bed", "1634021283638", "1634021677844","1634021755460", "Folding Laptop Reading Table","1634021878610",
                "BED Sheet","matress","1634022526754","1634022452413","1634022568565","Bed Frames","Bed Steel",
                "1634022644967", "1634022680155","1634022682220","steel single", "1634022719333","1634022790911","1634022680155",
                "1634022682220","Bag Ball","Water bed", "Wood Cloth Rack","1634023157542","Steel Double Bed","1634023313344","Pilliows",
                "1634023848839","1634023543569","1634024126354","1634024131819","1634024167028","1634024168167","Bed Room Set",
                "1634011898011","1634012116273","Folding Bed","1634014855776","Portable Travel Baby Crib","1634014960634","1634014994607",
                "1634014998763","Steel Bunker Bed","Bed Sheet","single steel","1634015466090","Clothes Rack",
                "Summer Duvet","Folding Bed","Room Furnitures","1634015705442", "matterss","1634015996589", "Wardobes","1634016223137","1634016255558",
                "1634016408984","1634016579149","book rack","1634019198473","1634019484166","Chest of drawares","1634020698130",
                "1634020698883","1634020699281","House furnitures","1634020925772","Double Bunker Bed","Dressing Tabel","Towel Rack",
                "Bunker Bed","1634022964968","1634023942260","1634024013335","1634024080938","1634024168993","1634024201064","1634024272594","Kids Furnitures",
                "Chairs","1634024426443","Almaira","1634024460270","1634024487817","Single BedIron","1634024944494","1634024979767","1634024983117","1634024983572",
                "1634025022114","1634025096995","1634025138917","Bedroom Furnitture","1634025248624","Nightstands","1634025566292","1634025607948","Platform Bed",
                "1634025730170","1634026088490","1634026088690","1634026089107","1634026131906","1634026167070","1634026202405","Bedroom Items","1634026351223",
                "1634026351586", "1634026351958","1634026353847","1634026354511","1634026400603","1634026466640","Dressing Tabel","1634026511936","1634026515176","1634026548251",
                "Tables","Air Bed","Bed Lamps","Dining Table","1634026963957","coffee table","Cloth Rack","Bed Iron","1634011860134","1634016255558","1634016408984",
                "1634016579149","1634020591901","Chest of drawares","1634020626372",
                "1634012708522","Dressings Table","1634016579149","Wadrobe","Table","book rack", "Metress","1634019029721","1634019350189","1634019484166","1634020698130","1634020698883",
                "1634020699281","1634020925772","Folding Laptop Table","Dressing Tabel","1634022791683","1634022833083","Cloth Rack","1634022833083","1634022860577",
                "1634022964968","Bunker Bed","Furniture","1634025537557","1634025943973","steel bunker","Furniture","1634027480794","warobers","1634027618857",
                "1634027691286","1634027625219","1634027619762","1634027689745","1634027691286","1634027726792","1634027801493","1634027847046_","Wardrob","1634027954002",
                "1634027955691","1634027956114","1634027957434","1634028048217","1634028086585","1634028089295","1634028089840","1634028090385","1634028278001",
                "1634028280513", "1634028280181","1634028574142","1634028656010","1634028611070","1634028713351","1634018176816","1634018136332","1634021385079",
                "1634022194992", "1634012031356","1634012032953","1634013229820","1634013420250","Hayleys Spring Mettress","1634013466483","Double Layer Mattress 63",
                "1634014184982","1634015306029","1634015340298","1634015630154","1634017207731","1634018032446","1634021352221","1634021344417","1634021381028","1634021382698",
                "1634021386721","1634021451682","1634021451906","1634021481113","1634021481291","1634021518700","1634021519115","1634021843513","1634022196456","1634022937361",
                "1634023395258","1634023396192","1634023396489","1634023465115","1634023665636","1634023804514","1634023805501","1634023977268","1634024395336_","1634024454995",
                "1634024460609","1634024489950","1634025105484","1634025143779","1634025328370","1634025690595","1634025863743","1634025823727","1634025940694",
                "1634025943132","1634026009981","1634026087186","1634026512621","1634026514391","1634026514758","1634026515995","1634026737424","1634026998698","1634027081193",
                "1634027083193","1634027083509","1634027115197","1634027117886","1634027119137","1634027152184","1634027156147","1634027156348","1634027189683","1634027262234",
                "1634027227653","1634027262438","1634027263549","1634027264291","1634027334562","1634027542688","1634027479262","towal rack","Towel Stand","1634027621324",
                "1634027688109","1634027727979","1634028155542","1634028242091","1634028277693","1634028350763","Bedroom Package","1634028475668","1634028548976",
                "1634028552084","1634028684721","1634028754771",
                
            ]):
                if keyword.lower() in file.lower():
                    label=f"dropped/{label}"
                    break

        else:
            skip = True
            
        if skip:
            continue

        if not dropped and ("dropped" in label):
            continue
        
        path = f'{dst}/{label}'
        img_cnt[label] += 1
        pbar.set_description(f'{file.split("_",2)[0]} / {path}')        
        os.makedirs(path, exist_ok=True)
        Image.open(f'{src}/furniture_images/{file}').save(f'{path}/{file}')
        # print(file, label_dict.get(file))

    pbar.close()
    
    for label, cnt in img_cnt.items():
        print(f"{label:>20s}: {cnt:4d}")


def get_args() -> dict:
    parser = ArgumentParser(
        prog="prepare furniture images",
        description="prepare furniture images",
    )

    parser.add_argument("--input", "-i", type=str, default="furniture_images_origin")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--dropped", "-d", action="store_true", help="remain dropped")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    run(args.input, args.output, args.dropped)
