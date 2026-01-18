import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w):
    return x * w
def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

w_list = []
loss_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print(f'weight: {w}')
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w)
        loss_val = loss(x_val, y_val, w)
        loss_sum += loss_val
        print(f'\tPrediction: {y_pred_val:.3f}\tLoss: {loss_val:.3f}')
    print(f'Weight: {w}\tLoss: {loss_sum:.3f}\n')
    w_list.append(w)
    loss_list.append(loss_sum / 3)
plt.plot(w_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('Weight')
plt.show()
