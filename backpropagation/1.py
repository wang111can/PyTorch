import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

# y = w1 * x^2 + w2 * x + b
# y = 2 * x^2 + 3 * x + 1
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [6.0, 15.0, 28.0, 45.0, 66.0]

w1 = torch.tensor([1.0], requires_grad = True) # 计算梯度
w2 = torch.tensor([1.0], requires_grad = True)
b = torch.tensor([1.0], requires_grad = True)
def forward(x):
    return w1 * x ** 2 + w2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2 # MSE 损失函数  返回的为tensor

# 会自动构建计算图

print('Predict (before training)', 4, forward(4).item())

W1 = []
W2 = []
B = []
L = []

for epoch in range(100000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) # 前向传播，计算损失
        l.backward()   # 反向传播，计算梯度

        # print('epoch: {}, x: {}, y: {}, w1: {}, w2: {}, b: {}, loss: {}'.format(epoch, x, y, w1.item(), w2.item(), b.item(), l.item()))


        w1.data = w1.data - 0.001 * w1.grad.data # 更新权重
        w2.data = w2.data - 0.001 * w2.grad.data # 更新权重
        b.data = b.data - 0.001 * b.grad.data   # 更新偏
        w1.grad.data.zero_()                  # 清空梯度
        w2.grad.data.zero_()                  # 清空梯度
        b.grad.data.zero_()                  # 清空梯度
        W1.append(w1.item())
        W2.append(w2.item())
        B.append(b.item())
        L.append(l.item())

print('Predict (after training)', 6, forward(4).item())
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(W1, W2, L, c='r', marker='o')
ax.set_xlabel('W1 Label')
ax.set_ylabel('W2 Label')
ax.set_zlabel('Loss Label')
plt.savefig('loss_surface.png') 
