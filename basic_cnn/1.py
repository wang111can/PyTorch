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

# linear layer
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5) # input channel = 1, output channel = 10 , 卷积盒大小 kernel size = 5x5
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5) # input channel = 10, output channel = 20 , kernel size
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x))) # conv1 + maxpooling  + ReLU
        x = F.relu(self.pooling(self.conv2(x))) # conv2 + maxpooling + ReLU
        x = x.view(batch_size, -1) # flatten the tensor
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