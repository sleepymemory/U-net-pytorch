import os
from tqdm import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the weight path
weight_path = 'params/aorta.pth'
#data_path
raw_image_path = 'data/raw_7_18'
mask_image_path = 'data/mask_7_18'

save_path = 'train_image'


if __name__ == '__main__':
    num_classes = 1 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(raw_image_path,mask_image_path), batch_size=8, shuffle=True)
    net = UNet(num_classes).to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()

    epoch = 1
    while epoch < 1000:
        loop = tqdm(data_loader)
        total_loss = 0
        net.train()
        for i, (image, segment_image) in enumerate(loop):
            image, segment_image = image.to(device), segment_image.to(device)
            # print(image.shape)
            # print(segment_image.shape)
            out_image = net(image)
            # print(out_image.shape)
            # print(segment_image.long())
            train_loss = loss_fun(out_image, segment_image.long())
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            total_loss += train_loss.item()
            loop.set_postfix(loss_=train_loss.item(), epoch_=epoch)

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        if epoch % 20 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
            print(total_loss)
        epoch += 1
