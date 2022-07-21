import os
import numpy as np
import cv2
import os
import torch
from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image, ImageGrab
import time
import matplotlib.pyplot as plt
import win32api, win32con, win32gui
from ctypes import windll, byref, c_ubyte
from ctypes.wintypes import RECT, HWND

# Unet 分类对象
net = UNet(2)

# 载入权重参数
weights = 'params/aorta.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')


# 返回读取的mask在图像中的位置
def point(X, Y):
    if (len(X) == 0 or len(Y) == 0):
        return -1, -1
    else:
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        return X_mean, Y_mean


cap = cv2.VideoCapture(r'E:\new\US image processing\data\5.mp4')

if __name__ == '__main__':

    while (1):
        ret, img = cap.read()
        img = img[110:970, 570:1350]
        ### todo 读取的截屏需要裁剪

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        temp = max(img.size)

        mask = Image.new('RGB', (temp, temp))
        mask.paste(img, (0, 0))
        img = mask.resize((256, 256))

        raw_img = np.array(img, dtype=np.uint8)

        img_data = transform(img)
        img_data = torch.unsqueeze(img_data, dim=0)
        net.eval()
        out = net(img_data)
        out = torch.argmax(out, dim=1)
        out = torch.squeeze(out, dim=0)
        out = out.unsqueeze(dim=0)
        out = (out).permute((1, 2, 0)).cpu().detach().numpy()
        out = out * 255

        arr = np.array(out, dtype=np.uint8)

        new = np.array(1 - arr.reshape(256, 256) // 255, dtype=np.uint8)
        new = np.expand_dims(new, axis=-1)
        mask_img = np.multiply(raw_img, new)

        cv2.imshow('1', mask_img)
        cv2.waitKey(1)

        gray = cv2.GaussianBlur(arr, (3, 3), 0)
        canny = cv2.Canny(gray, 20, 100)
        X = []
        Y = []
        # print(canny)
        for i in range(0, 256):
            for j in range(0, 256):
                if canny[i][j] == 255:
                    Y.append(j)
                    X.append(i)
        m1, m2 = point(X, Y)
        print(m1, m2)
        ## todo 输出位置
