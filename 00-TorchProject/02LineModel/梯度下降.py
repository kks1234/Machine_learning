import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x, w):
    return x * w


def cost(x_d, y_d, w):
    cost_sum = 0
    for xv, yv in zip(x_d, y_d):
        _y = forward(xv, w)
        costs = (_y - yv) ** 2
        cost_sum += costs
    return cost_sum / len(x_d)


def gradient(x_d, y_d, w):
    grad_sum = 0
    for vx, vy in zip(x_d, y_d):
        grads = 2 * (forward(vx, w) - vy) * vx
        grad_sum += grads
    return grad_sum / len(x_d)


epoch_list = []
cost_list = []

for i in range(100):
    costv = cost(x_data, y_data, w)
    gradv = gradient(x_data, y_data, w)
    epoch_list.append(i)
    cost_list.append(costv)
    w -= 0.01 * gradv
    print("W=",w,'cost = ',costv)

plt.figure(figsize=(20,8), dpi=80)
epoch_list_lable = ["{}Epoch".format(i) for i in epoch_list]
plt.xticks(epoch_list[::5],epoch_list_lable[::5],rotation= 45)
plt.plot(epoch_list, cost_list)
plt.show()
