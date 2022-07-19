import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, raw_path, mask_path):
        self.raw_path = raw_path
        self.mask_path = mask_path
        self.name = os.listdir(os.path.join(raw_path))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.mask_path, segment_name)
        image_path = os.path.join(self.raw_path, segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)
        return transform(image), torch.Tensor(np.array(segment_image))


if __name__ == '__main__':
    from torch.nn.functional import one_hot

    data = MyDataset(r'./data/raw_7_18', r'./data/mask_7_18')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out = one_hot(data[0][1].long())
    print(out.shape)
