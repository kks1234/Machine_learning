import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import time

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./dataset/mnist/", train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = datasets.MNIST(root="./dataset/mnist", train=False, transform=transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# GoogleNet文件夹中的 InceptionModel 构建的代码
class InceptionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionModel, self).__init__()
        # ----------------#
        self.polling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # ----------------#
        self.branch1x1_1 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch1x1_2 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # ----------------#
        self.branch5x5 = nn.Conv2d(16, 24, kernel_size=5, padding=2, stride=1)
        # ----------------#
        self.branch3x3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=1)
        self.branch3x3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1)
        # ----------------#

    def forward(self, x):
        forward_net1 = self.branch1x1_1(self.polling(x))
        forward_net2 = self.branch1x1_2(x)
        forward_net3 = self.branch5x5(self.branch1x1_2(x))
        forward_net4 = self.branch3x3_2(self.branch3x3_1(self.branch1x1_2(x)))
        out = [forward_net1, forward_net2, forward_net3, forward_net4]
        # b c w h
        return torch.cat(out, dim=1)


class NetModel(torch.nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()
        self.activate = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5, bias=False)
        self.inception1 = InceptionModel(10)
        self.inception2 = InceptionModel(20)
        # 28*28->conv1->24*24->polling->12*12
        # 12*12->conv2->8*8->poling->4*4
        # inception固定输出88通道
        # 1408 = 88*4*4
        self.FullConnect = nn.Linear(1408, 10)
        self.polling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.polling(x)
        x = self.inception1(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = self.polling(x)
        x = self.inception2(x)
        x = x.view(batchsize, -1)
        x = self.FullConnect(x)
        return x


model = NetModel()
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0
    for batchIndex, (x_, y_) in enumerate(train_loader):
        y_pred = model(x_)
        loss = criterion(y_pred, y_)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batchIndex % 300 == 299:
            print("[%d, %5d] loss: %3f" % (epoch + 1, batchIndex + 1, running_loss / 300))
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for x_, y_ in test_loader:
            y_pred = model(x_)
            _, predicted = torch.max(y_pred.data, dim=1)
            total += y_pred.size(0)
            correct += (y_ == predicted).sum().item()
    print("Accuracy on test set: %d %%" % (100 * (correct / total)))


if __name__ == '__main__':
    for epoch in range(20):
        t0 = time.time()
        train(epoch)
        test()
        t1 = time.time()
        print("time:", t1 - t0)
