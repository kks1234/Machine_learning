import torch

xx = [1.0, 2.0, 3.0]
yy = [4.0, 5.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x_):
    return x_ * w


def loss(x_, y_):
    y_pred = forward(x_)
    return (y_pred - y_) ** 2


if __name__ == '__main__':
    for epoch in range(100):
        for x, y in zip(xx, yy):
            l_oss = loss(x, y)
            l_oss.backward()
            print("\tgrad:", x, y, w.grad.item())
            w.data = w.data - 0.001 * w.grad.data
            w.grad.data.zero_()

        print("Process:", epoch, l_oss.item())
