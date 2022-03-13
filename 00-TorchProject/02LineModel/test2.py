import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 这里设函数为y=3x+2
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]


def forward(x, w, b):
    return x * w + b


def loss(x, w, b, y):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(0.0, 4.1, 0.1)

w, b = np.meshgrid(w_list, b_list)
mse = np.zeros(w.shape)

print(mse)
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val, w, b)
    loss_val = loss(x_val, w, b, y_val)
    mse += loss_val
mse /= len(x_data)

print(mse)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()
