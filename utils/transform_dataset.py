from PIL import Image
from multiprocessing import Pool
import os



Color_img = os.listdir("../Train_K/color_full/")

def compress(img):
    try:
        image = Image.open(f"../Train_K/color_full/{img}")
        image = image.resize((768, 1024), Image.Resampling.LANCZOS)
        image = image.convert('RGB')
        image.save(f"../Train_DGX/color_full/{img}")
    except: pass


if __name__ == '__main__':
    with Pool() as p:
        p.map(compress, Color_img)