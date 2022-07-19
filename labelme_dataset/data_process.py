import os, sys
from tqdm import tqdm
import cv2
import numpy
import shutil

from glob import glob

count = 0
path = r".\json_files"
json_files = glob(os.path.join(path, "*.json"))
for json_file in tqdm(json_files):
    os.system(f"labelme_json_to_dataset {json_file} -o {'./result/{}'.format(count)}")
    count = count + 1

path = r".\result"
files = os.listdir(path)
for i, file in tqdm(enumerate(files)):
    shutil.copyfile(os.path.join(path, file, 'img.png'), './raw/{}.png'.format(i))
    shutil.copyfile(os.path.join(path, file, 'label.png'), './mask/{}.png'.format(i))
