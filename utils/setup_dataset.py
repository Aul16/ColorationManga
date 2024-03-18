import os
import cv2 as cv
from multiprocessing import Pool

PATH = "./dataset"
CSV_PATH = "./dataset/images.csv"

Color_img = os.listdir(f"{PATH}/rgb/")

def convert(img):
    try:
        image = cv.imread(f'{PATH}/rgb/{img}')
        SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        background = cv.morphologyEx(image, cv.MORPH_DILATE, SE)
        image = cv.divide(image, background, scale=255)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imwrite(f"{PATH}/bw/{img}", image)
    except Exception as e:
        print(e)


with Pool() as p:
        p.map(convert, Color_img)


imgs = os.listdir(f"{PATH}/bw/")

with open(CSV_PATH, "w", newline='') as csvfile:
    for img in imgs:
        csvfile.writelines(f"{img}\n")