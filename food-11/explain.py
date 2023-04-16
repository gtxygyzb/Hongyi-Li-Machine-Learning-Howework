import torch
import torch.nn as nn
from training import Net, ImgDataset, read_file
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import os
import matplotlib.pyplot as plt


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()
    x.requires_grad_()

    y_pred = model(x)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies


def draw_saliency():
    img_indices = [83, 4218, 4707, 8598]
    images, labels = train_set.getbatch(img_indices)
    saliencies = compute_saliency_maps(images, labels, model)

    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            tmp = img.permute(1, 2, 0).numpy()
            tmp = tmp[:, :, [2, 1, 0]]
            axs[row][column].imshow(tmp)
    plt.savefig("./saliency_map.png")
    plt.show()


if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图片转成Tensor并normalize到[0, 1]
    ])
    model = torch.load("./parameter.pkl")
    print(model)
    # rt = "./food-11"
    # x_train, y_train = read_file(os.path.join(rt, "training"), True)
    # train_set = ImgDataset(x_train, y_train, test_transform)
    # print("build training set completed!")

    # draw_saliency()

