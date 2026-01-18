import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
w = 1.0  # initial weight

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)
print("Initial cost:", 4, forward(4))
E_list = np.array([], float)
C_list = np.array([], float)
for epoch in range(1000):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print("Epoch:", epoch, "w =", w, "cost =", cost_val)
    E_list = np.append(E_list, epoch)
    C_list = np.append(C_list, cost_val)

plt.plot(C_list, E_list)
plt.ylabel('Epoch')
plt.xlabel('Cost')
plt.savefig('cost_plot.png')


print("Final cost:", 4, forward(4))
