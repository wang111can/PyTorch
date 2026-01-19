import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
# x, y 都是 3 x 1 的矩阵


class LinearModel(torch.nn.Module): # torch.nn.Module is Base class for all neural network modules.
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # 表示输入输出都是 1 维
        # nn: Neural Networks
        # Class nn.Linear contain two member Tensors: weight and bias 
        # a linear transformation to the incoming data: y = xA^T + b
    
    def forward(self, x):
        y_pred = self.linear(x) 
        # 可调用对象
        return y_pred
    # class Foobar:
    #     def __init__(self):
    #           pass
    #     def __call__(self,*args,**kwargs):
    #           print("Hello" + str(args[0]))
    # foobar = Foobar()
    # foobar(1,2,3)
    
# self.linear 是自定义模型类 LinearModel 的实例属性，指向 torch.nn.Linear(1,1) 线性层的实例；
# 它的核心作用是封装线性变换（y=wx+b）的计算逻辑 + 自动管理可训练的权重 w 和偏置 b；
# 借助 self.linear，你无需手动定义 w、b 和线性计算式，PyTorch 会自动完成参数初始化、梯度计算，是符合 PyTorch 最佳实践的写法。

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
# MSELoss: Mean Squared Error Loss 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 返回的为tensor

for epoch in range(100):
    y_pred = model(x_data) # 1. forward pass 
    loss = criterion(y_pred, y_data) # 2. compute loss 
    print(epoch, loss.item())
    
    optimizer.zero_grad() # 3. zero the gradient buffers
    loss.backward()       # 4. backward pass
    optimizer.step()      # 5. Does the update
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred(4.0) = ', y_test.data.item())