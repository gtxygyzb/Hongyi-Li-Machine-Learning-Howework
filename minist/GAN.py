import random
import torch
import pandas as pd
import numpy as np
import os
import glob

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def read(path, label):
    df = pd.read_csv(path)
    data = df.values
    n = len(data)
    if label:
        x = data[:, 1:].reshape(n, 28, 28, 1)
        x = x.astype("float32")
        y = data[:, 0].reshape(n)
        return x, y
    else:
        x = data.reshape(n, 28, 28, 1)
        x = x.astype("float32")
        return x

class ImgDataset(Dataset):
    def __init__(self, x, transform=None):
        self.x = x
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        x = self.transform(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 784)  # batch, 1,28,28
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, stride=1, padding=1),  # batch, 12, 28, 28
            nn.BatchNorm2d(12),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(12, 6, 3, stride=1, padding=1),  # batch, 6, 28, 28
            nn.BatchNorm2d(6),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(6, 1, 3, stride=1, padding=1),  # batch, 1, 28, 28
            nn.Tanh()
        )

    # x.shape:[100,128]
    def forward(self, x):
        x = self.fc(x)  # x.shape:[100,784]
        x = x.view(x.size(0), 1, 28, 28)  # x.shape:[100,1,28,28]
        x = self.br(x)  # x.shape:[100,1,28,28]
        x = self.downsample1(x)  # x.shape:[100,12,28,28]
        x = self.downsample2(x)  # x.shape:[100,6,28,28]
        x = self.downsample3(x)  # x.shape:[100,1,28,28]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=2),  # batch, 6, 30,30
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2),  # batch, 6, 15, 15
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=2),  # batch, 12, 17, 17
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2)  # batch, 12, 8, 8
        )
        self.fc = nn.Sequential(
            nn.Linear(12 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    # x.shape:[100,1,28,28]
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)               # 将第二次卷积的输出拉伸为一行
        x = self.fc(x)
        x = x.squeeze(-1)
        return x


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

    # Training hyperparameters
    batch_size = 64
    z_dim = 100
    z_sample = Variable(torch.randn(36, z_dim)).cuda()
    lr = 1e-4

    n_epoch = 50  # 50
    n_critic = 2  # 5

    workspace_dir = "./"
    log_dir = os.path.join(workspace_dir, 'logs')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    G = Generator().cuda()
    D = Discriminator().cuda()

    G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
    D.load_state_dict(torch.load(os.path.join(ckpt_dir, 'D.pth')))

    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data[0].cuda()
            bs = imgs.size(0)
            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)
            # WGAN Loss
            loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
            # Model backwarding
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)
                # Model forwarding
                f_logit = D(f_imgs)
                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))
                # Model backwarding
                G.zero_grad()
                loss_G.backward()
                opt_G.step()
            steps += 1

        if (e + 1) % 5 == 0 or e == 2:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2
            filename = os.path.join(log_dir, f'Epoch_{epoch+50:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=6)
            print(f' | Save some samples to {filename}.')
            # Show generated images in the jupyter notebook.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=6)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()
            G.train()

        if (e + 1) % 5 == 0 or e == 0:

            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

