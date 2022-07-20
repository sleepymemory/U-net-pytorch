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
import time



net = UNet(2)

weights = 'params/aorta.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')


def point(X, Y):
    if (len(X) == 0 or len(Y) == 0):
        return -1, -1
    else:
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        return X_mean, Y_mean

GetDC = windll.user32.GetDC
CreateCompatibleDC = windll.gdi32.CreateCompatibleDC
GetClientRect = windll.user32.GetClientRect
CreateCompatibleBitmap = windll.gdi32.CreateCompatibleBitmap
SelectObject = windll.gdi32.SelectObject
BitBlt = windll.gdi32.BitBlt
SRCCOPY = 0x00CC0020
GetBitmapBits = windll.gdi32.GetBitmapBits
DeleteObject = windll.gdi32.DeleteObject
ReleaseDC = windll.user32.ReleaseDC

# 排除缩放干扰
windll.user32.SetProcessDPIAware()


def capture(handle: HWND):
    """窗口客户区截图

    Args:
        handle (HWND): 要截图的窗口句柄

    Returns:
        numpy.ndarray: 截图数据
    """
    # 获取窗口客户区的大小
    r = RECT()
    GetClientRect(handle, byref(r))
    width, height = r.right, r.bottom
    # 开始截图
    dc = GetDC(handle)
    cdc = CreateCompatibleDC(dc)
    bitmap = CreateCompatibleBitmap(dc, width, height)
    SelectObject(cdc, bitmap)
    BitBlt(cdc, 0, 0, width, height, dc, 0, 0, SRCCOPY)
    # 截图是BGRA排列，因此总元素个数需要乘以4
    total_bytes = width*height*4
    buffer = bytearray(total_bytes)
    byte_array = c_ubyte*total_bytes
    GetBitmapBits(bitmap, total_bytes, byte_array.from_buffer(buffer))
    DeleteObject(bitmap)
    DeleteObject(cdc)
    ReleaseDC(handle, dc)
    # 返回截图数据为numpy.ndarray
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)


if __name__ == '__main__':
    handle= windll.user32.FindWindowW(None, 'SONON X')
    while(1):
        img = capture(handle)
        ### todo 读取的截屏需要裁剪


        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        temp = max(img.size)
        mask = Image.new('RGB', (temp, temp))
        mask.paste(img, (0, 0))
        img = mask.resize((256, 256))
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
        print(m1,m2)
        ## todo 输出位置