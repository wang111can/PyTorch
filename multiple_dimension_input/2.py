import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# dataset dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# dataset class
class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./diabetes.csv.gz', delimiter=',', dtype=np.float32)
        # 2. 记录数据集的总样本数（xy.shape[0]表示数组的行数，即样本数量）
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        y_pred = self.sigmoid(self.linear3(x))
        return y_pred


model = Model()
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)
      

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# mini-batch training

for epoch in range(1000): 
    
    for i, (inputs, labels) in enumerate(train_loader, 0): # iterate mini-batch
        # prepare data
        # x, y getitem
        
        # forward
        y_pred = model(inputs)
        # loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # zero gradients
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        
        