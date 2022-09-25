import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import cv2
import os
import shutil

gesture_dict = {'Clap':1,'Draw-N(Vertical)':2,'Draw-Zigzag(Vertical)':3,'Push&Pull':4,'Slide':5,'Sweep':6}

root_dirname = './pic/d'
res_dirname = './locations/'
dirsname = os.listdir(root_dirname)

loc_s = 'loc'

for i in range(len(dirsname)):

    filename = os.listdir(root_dirname+dirsname[i])

    for j in range(len(filename)):
        count = 0
        loc = '0'

        for s in filename[j]:
            if count == 2:
                loc = str(s)
                break
            if s == '-':
                count += 1

        # shutil.copy(root_dirname+dirsname[i]+'/'+filename[j],res_dirname+loc_s+loc+'/'+filename[j])

        with open(res_dirname+filename[j]+'.txt','a') as f:
            f.write(dirsname[i])
            f.write('\n')
            f.write(loc)










