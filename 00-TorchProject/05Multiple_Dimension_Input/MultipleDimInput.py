import torch
import torch.nn as nn
import numpy as np

xy = np.loadtxt("./diabetes.csv", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class MultipleDimInput(torch.nn.Module):
    def __init__(self):
        super(MultipleDimInput, self).__init__()
        self.activate = nn.Sigmoid()
        self.line1 = nn.Linear(8, 6)
        self.line2 = nn.Linear(6, 4)
        self.line3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.activate(self.line1(x))
        x = self.activate(self.line2(x))
        x = self.activate(self.line3(x))
        return x


model = MultipleDimInput()

criterion = nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

if __name__ == '__main__':
    for epoch in range(1000):
        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("*" * 100)
    x_test = torch.Tensor([-0.329412, 0.205528, 0.508197, 0, 0, 0.120715, -0.903501, 0.7])
    y_predict = model(x_test)
    print(y_predict.data.item())
    print("*" * 100)
    for parameter in model.parameters():
        print(parameter)
