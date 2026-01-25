import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim


# dataset dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# input layer

batch_size = 64
# Convert the PIL images to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])
# the parameters are 'mean' and 'std' respectively.

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#多尺度特征融合——用不同大小的卷积核和池化提取不同尺度特征，再拼接
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        
        # 1
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        
        # 2
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        # 3
        self.branch3x3dbl_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        # 分支4：平均池化 + 1x1卷积（保留全局特征，调整通道数）
        # 注意：池化操作在forward中用F.avg_pool2d实现，这里只定义1x1卷积调整通道
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)  
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # 将四个分支的输出在通道维度（dim=1）拼接
        # 拼接后通道数：16(分支1) + 24(分支2) + 24(分支3) + 24(分支4) = 88
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, dim=1) # 在channel维度上进行拼接

# 残差块
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  # 残差连接
# linear layer
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size = 5) # input channel = 1, output channel = 10 , 卷积盒大小 kernel size = 5x5
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 5) # input channel = 10, output channel = 20 , kernel size
        self.mp = torch.nn.MaxPool2d(2) # pooling layer , kernel size = 2x2

        self.rblock1 = ResidualBlock(channels=16)  # 第一个残差块
        self.rblock2 = ResidualBlock(channels=32)  # 第二个残差块

        self.fc = torch.nn.Linear(512, 10) 
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x))) # conv1 + maxpooling  + ReLU
        x = self.rblock1(x)  # 通过第一个残差块
        x = self.mp(F.relu(self.conv2(x))) # conv2 + maxpooling
        x = self.rblock2(x)  # 通过第二个残差块
        x = x.view(in_size, -1) # flatten the tensor
        return self.fc(x) # linear layer

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
model.to(device)#

criterion = torch.nn.CrossEntropyLoss() #
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) #
        optimizer.zero_grad()
        
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 300 == 299:    
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) #
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    
    
for epoch in range(15):
    train(epoch)
    test()