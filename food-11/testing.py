import training as T
import torch
import torch.nn as nn
from training import Net
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import os
import time


if __name__ == "__main__":
    print("Reading data")
    rt = "./food-11"
    batch_size = 64
    x_train, y_train = T.read_file(os.path.join(rt, "training"), True)
    print("shape of training data:", x_train.shape)
    x_valid, y_valid = T.read_file(os.path.join(rt, "validation"), True)
    print("shape of valid data:", x_valid.shape)
    x_tv = np.concatenate((x_train, x_valid), axis=0)
    y_tv = np.concatenate((y_train, y_valid), axis=0)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
        transforms.RandomRotation(15),  # 随机旋转图片
        transforms.ToTensor(),  # 图片转成Tensor并normalize到[0, 1]
    ])
    tv_set = T.ImgDataset(x_tv, y_tv, train_transform)
    tv_loader = DataLoader(tv_set, batch_size=batch_size, shuffle=True)
    model = T.Net().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 30
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        acc = 0.0
        los = 0.0
        model.train()
        for i, data in enumerate(tv_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()
            acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            los += batch_loss.item()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               acc / tv_set.__len__(), los / tv_set.__len__())
              )
    torch.save(model, "./parameter.pkl")
    print("saving model completed!")

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图片转成Tensor并normalize到[0, 1]
    ])
    x_test = T.read_file(os.path.join(rt, "testing"), False)
    print("shape of testing data:", x_test.shape)

    test_set = T.ImgDataset(x_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model = torch.load("./parameter.pkl")
    model.eval()
    ans = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                ans.append(y)

    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(ans):
            f.write('{},{}\n'.format(i, y))
