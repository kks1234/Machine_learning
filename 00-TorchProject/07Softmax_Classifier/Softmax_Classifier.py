import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./dataset/mnist/", train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = datasets.MNIST(root="./dataset/mnist/", train=False, transform=transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class CreateModel(torch.nn.Module):
    def __init__(self):
        super(CreateModel, self).__init__()
        self.activate = nn.ReLU()
        self.line1 = nn.Linear(784, 512)
        self.line2 = nn.Linear(512, 256)
        self.line3 = nn.Linear(256, 128)
        self.line4 = nn.Linear(128, 64)
        self.line5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activate(self.line1(x))
        x = self.activate(self.line2(x))
        x = self.activate(self.line3(x))
        x = self.activate(self.line4(x))
        x = self.line5(x)
        return x


model = CreateModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_index, (x_, y_) in enumerate(train_loader):
        y_pred = model(x_)
        loss = criterion(y_pred, y_)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 300 == 299:
            print("[%d, %5d] loss: %3f" % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (x_, y_) in test_loader:
            y_pred = model(x_)
            _, predicted = torch.max(y_pred.data, dim=1)
            total += y_.size(0)
            correct += (predicted == y_).sum().item()
    print("Accuracy on test set: %d %%" % (100 * (correct / total)))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test()
