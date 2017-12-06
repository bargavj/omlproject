from __future__ import print_function
from torch.autograd import Variable
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=True, num_workers=2)

batchsize = 128
lambda1 = 0.01
lr = 1e-3


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

def l1_penalty(var):
    return torch.abs(var).sum()

def sparsify(param, sparsity):
    return F.hardshrink(param, sparsity).data
       
model = MLP()
#model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(10):
    losses = []
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        model.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        cross_entropy_loss = F.cross_entropy(outputs, targets)
        #l1_regularization = lambda1 * sum([l1_penalty(param.data) for param in model.parameters()])

        loss = cross_entropy_loss# + l1_regularization
        
        loss.backward()
        optimizer.step()
      
        losses.append(loss.data[0])

    print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
    
    # Test
    model.eval()
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct/total))
    print('--------------------------------------------------------------')
    model.train()

model2 = copy.deepcopy(model)

for param in model2.parameters():
    param.data = sparsify(param, 0.01)

cnt, tot = 0, 0
for param in model2.parameters():
    tot += param.data.view(-1).size()[0]
    for val in param.data.view(-1):
        if val == 0.:
            cnt += 1
print(str(cnt*100./tot) + "% sparse")

model2.eval()
total = 0
correct = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
    outputs = model2(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

print('Test Accuracy : %.3f' % (100.*correct/total))
