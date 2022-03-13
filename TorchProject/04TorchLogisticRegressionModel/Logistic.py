import torch
import matplotlib.pyplot as plt
import numpy as np



x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.logistic = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.logistic(x))
        return y_pred


if __name__ == '__main__':
    model = LogisticRegression()
    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(10000):
        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)

        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model.logistic.weight.item())
    print(model.logistic.bias.item())
    x4 = torch.Tensor([4.0])
    print(float(model(x4).data))

    plt.figure(figsize=(20,8),dpi=80)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x = np.linspace(0,10,200)
    x_t   = torch.Tensor(x).view((200,1))
    y_t  = model(x_t)
    y = y_t.data.numpy()
    plt.plot(x,y)
    plt.plot([0,10],[0.5,0.5])
    plt.xlabel("数据测试",{"fontsize":20})
    plt.ylabel("预测概率",{"fontsize":20})
    plt.grid()
    plt.show()




