import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


plt.figure(figsize=(20,8),dpi=80)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)




criterion = torch.nn.MSELoss(size_average=False)

AdamLossList = []
SGDLossList = []



if __name__ == '__main__':
    model = LinearModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred,y_data)
        print(epoch,loss.item())
        AdamLossList.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("w",model.linear.weight.item())
    print("b",model.linear.bias.item())
    print("pridict y_pred:",model(torch.Tensor([4.0])).data)
    model1 = LinearModel()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    for epoch in range(1000):
        y_pred = model1(x_data)
        loss = criterion(y_pred,y_data)
        print(epoch,loss.item())
        SGDLossList.append(loss.item())
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    print("w",model1.linear.weight.item())
    print("b",model1.linear.bias.item())
    print("pridict y_pred:",model1(torch.Tensor([4.0])).data)

    x_axel = [i for i in range(1000)]

    plt.plot(x_axel,AdamLossList,label="Adam Loss")
    plt.plot(x_axel,SGDLossList,label="SGD Loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch",{"fontsize" : 20})
    plt.ylabel("Loss",{"fontsize" : 20})
    plt.grid()
    plt.show()
