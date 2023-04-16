import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import cv2 as cv
import os
import time


def read_file(path, label):
    img_lis = sorted(os.listdir(path))
    tot = len(img_lis)
    x = np.zeros((tot, 128, 128, 3), dtype='uint8')
    y = np.zeros(tot, dtype='uint8')
    for i, file in enumerate(img_lis):
        img = cv.imread(os.path.join(path, file))
        x[i] = cv.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # nn.MaxPool2d(kernel_size, stride, padding)
        # input [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)


if __name__ == "__main__":
    print("Reading data")
    rt = "./food-11"
    x_train, y_train = read_file(os.path.join(rt, "training"), True)
    print("shape of training data:", x_train.shape)
    x_valid, y_valid = read_file(os.path.join(rt, "validation"), True)
    print("shape of valid data:", x_valid.shape)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
        transforms.RandomRotation(15),  # 随机旋转图片
        transforms.ToTensor(),  # 图片转成Tensor并normalize到[0, 1]
    ])

    batch_size = 64
    train_set = ImgDataset(x_train, y_train, train_transform)
    valid_set = ImgDataset(x_valid, y_valid, train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = Net().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 30

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        valid_acc = 0.0
        valid_loss = 0.0

        model.train()  # 启用 batch normalization 和 drop out
        for i, data in enumerate(train_loader):
            # data 是一个list
            # data[0] = [batch_size * 3 * 128 * 128], type = tensor
            # data[1] = [batch_size] label
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())

            print("pred: ok   ", train_pred.dtype, train_pred.shape)
            print("label: ", data[1].cuda().dtype, data[1].cuda().shape)

            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():  # 被包住的代码不需要计算梯度
            for i, data in enumerate(valid_loader):
                valid_pred = model(data[0].cuda())
                batch_loss = loss(valid_pred, data[1].cuda())

                valid_acc += np.sum(np.argmax(valid_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                valid_loss += batch_loss.item()

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %
                  (epoch + 1, num_epoch, time.time() - epoch_start_time,
                   train_acc / train_set.__len__(), train_loss / train_set.__len__(), valid_acc / valid_set.__len__(),
                   valid_loss / valid_set.__len__())
                  )

    torch.save(model, "./parameter.pkl")
    print("saving model completed!")
