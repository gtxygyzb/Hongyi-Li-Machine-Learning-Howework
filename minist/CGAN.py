import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import os


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


def plot_images(G, label, name):
    label_tensor = torch.zeros(10)
    label_tensor[label] = 1.0
    # plot a 3 column, 2 row array of sample images
    f, axarr = plt.subplots(2, 3, figsize=(12, 6))

    with torch.no_grad():
        for i in range(2):
            for j in range(3):
                axarr[i, j].imshow(
                    G.forward(generate_random_seed(100), label_tensor, d=0).detach().cpu().numpy().reshape(28, 28),
                    cmap='Blues')
    plt.savefig("./logs/" + name)
    plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100+10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, z, label, d=1):
        x = torch.cat((z, label), dim=d)
        return self.fc(x.cuda())

    def work(self, D, inputs, label_tensor, targets):
        g_output = self.forward(inputs, label_tensor)
        d_output = D.forward(g_output, label_tensor)

        loss = D.loss_function(d_output, targets.cuda())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(784+10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, img, label):
        x = torch.cat((img.cuda(), label.cuda()), dim=1)
        return self.fc(x)

    def work(self, inputs, label_tensor, targets):
        outputs = self.forward(inputs, label_tensor)
        loss = self.loss_function(outputs, targets.cuda())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


def train():
    batch_size = 64
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True, drop_last=True)
    G = Generator().cuda()
    D = Discriminator().cuda()
    for epoch in range(100):
        print("epoch:", epoch)
        for X, label in dataloader:
            X = X.view(X.size(0), -1)
            Y = torch.zeros([batch_size, 10])
            for i in range(X.size(0)):
                Y[i, label[i]] = 1.0
            D.work(X, Y, torch.ones([batch_size, 1]))

            random_label = torch.zeros([batch_size, 10])
            for i in range(X.size(0)):
                random_label[i, random.randint(0, 9)] = 1.0
            g_img = G.forward(torch.randn([batch_size, 100]), random_label)
            D.work(g_img, random_label.cuda(), torch.zeros([batch_size, 1]))

            random_label = torch.zeros([batch_size, 10])
            for i in range(X.size(0)):
                random_label[i, random.randint(0, 9)] = 1.0
            G.work(D, torch.randn([batch_size, 100]), random_label, torch.ones([batch_size, 1]))

        if epoch % 5 == 0:
            plot_images(G, 8, "cgan_" + str(epoch))
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'cG.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'cD.pth'))


if __name__ == "__main__":
    ckpt_dir = "./checkpoints"
    G = Generator().cuda()
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'cG.pth')))
    plot_images(G, 8, "8")
    # train()
