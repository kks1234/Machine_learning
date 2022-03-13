import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x, w):
    return x * w


def loss(x, w, y):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2


w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w)
        loss_val = loss(x_val, w, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

plt.figure(figsize=(20,8), dpi=80)
plt.plot(w_list, mse_list)
plt.xticks(w_list)

plt.show()


