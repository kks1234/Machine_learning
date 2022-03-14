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


class NetModel(torch.nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()
        self.activate = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.FullConnect = nn.Linear(320, 10)
        self.polling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.polling(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = self.polling(x)
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