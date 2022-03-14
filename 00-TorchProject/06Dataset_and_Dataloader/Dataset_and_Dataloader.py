import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


dataset = DiabetesDataset("./diabetes.csv")

trainloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=3)


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.line1 = nn.Linear(8, 6)
        self.line2 = nn.Linear(6, 4)
        self.line3 = nn.Linear(4, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.line1(x))
        x = self.activate(self.line2(x))
        x = self.activate(self.line3(x))
        return x


model = LogisticModel()
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

if __name__ == '__main__':
    lossarr = []
    for epoch in range(1000):
        loss_sum = 0
        for batch_index, (x_, y_) in enumerate(trainloader, 0):
            y_pred = model(x_)
            loss = criterion(y_pred, y_)
            print(epoch, batch_index, loss.item())
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lossarr.append(loss_sum)
    plt.plot(range(1000), lossarr)
    plt.show()
