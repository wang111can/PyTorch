import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# 下载torchvision提供的数据集和测试集
# train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True)
# test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# loss function for linear regression
# loss = (y - y_pred) ** 2

# loss function for Binary classification
# loss = -(y *log(y_pred) + (1 - y) * log(1 - y_pred))

import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('hours')
plt.ylabel('pass probability')
plt.savefig('logistic_regression.png')

