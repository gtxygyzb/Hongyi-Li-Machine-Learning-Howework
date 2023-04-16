import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import time
import matplotlib.pyplot as plt

import cv2 as cv


def read(path, label):
    df = pd.read_csv(path)
    data = df.values
    if label:
        np.random.shuffle(data)
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
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        x = self.transform(x)
        if self.y is not None:
            return x, self.y[item]
        else:
            return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def training():
    x_train, y_train = read("./train.csv", True)
    valid_size = int(0.1 * len(x_train))
    x_valid, y_valid = x_train[:valid_size], y_train[:valid_size]
    x_train, y_train = x_train[valid_size:], y_train[valid_size:]

    batch_size = 128
    train_set = ImgDataset(x_train, y_train, transforms.ToTensor())
    valid_set = ImgDataset(x_valid, y_valid, transforms.ToTensor())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    print("build loader completed!")

    model = CNN().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 10

    train_acc_lis = []
    valid_acc_lis = []
    train_loss_lis = []
    valid_loss_lis = []

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        valid_acc = 0.0
        valid_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
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
            train_acc_lis.append(train_acc / train_set.__len__())
            train_loss_lis.append(train_loss / train_set.__len__())
            valid_acc_lis.append(valid_acc / valid_set.__len__())
            valid_loss_lis.append(valid_loss / valid_set.__len__())

    torch.save(model, "./vgg1.pkl")
    print("saving model completed!")

    # Loss curve
    plt.plot(train_loss_lis)
    plt.plot(valid_loss_lis)
    plt.title('Loss')
    plt.legend(['train', 'valid'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc_lis)
    plt.plot(valid_acc_lis)
    plt.title('Accuracy')
    plt.legend(['train', 'valid'])
    plt.savefig('acc.png')
    plt.show()


def testing():
    x_test = read("./test.csv", False)
    test_set = ImgDataset(x_test, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    model = torch.load("./vgg1.pkl")
    model.eval()
    ans = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                ans.append(y)
    with open("predict.csv", 'w') as f:
        f.write('ImageId,Label\n')
        for i, y in enumerate(ans):
            f.write('{},{}\n'.format(i + 1, y))


if __name__ == "__main__":
    # training()
    testing()
