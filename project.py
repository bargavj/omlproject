from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms
import torchvision.models as models

mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)


class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 2)

    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        out = self.linear3(layer2_out)
        return out, layer1_out, layer2_out


def l1_penalty(var):
    return torch.abs(var).sum()


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


batchsize = 4
lambda1, lambda2 = 0.01, 0.01

#model = MLP()
#model = models.alexnet()
model = Classifier()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
'''
inputs = Variable(torch.rand(batchsize, 128))
targets = Variable(torch.ones(batchsize).long())

optimizer.zero_grad()
outputs, layer1_out, layer2_out = model(inputs)
cross_entropy_loss = F.cross_entropy(outputs, targets)
l1_regularization = lambda1 * l1_penalty(layer1_out)
l2_regularization = lambda2 * l2_penalty(layer2_out)

print(list(model.parameters())[0][0][0])
loss = cross_entropy_loss + l1_regularization + l2_regularization
loss.backward()
optimizer.step()
print(list(model.parameters())[0][0][0])
#print(param.data for param in model.parameters())
#print(layer1_out)
#print(layer2_out)
'''
lr = 1e-2
for epoch in range(50):
    losses = []
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        model.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        cross_entropy_loss = F.cross_entropy(outputs, targets)
        l1_regularization = lambda1 * sum([l1_penalty(param.data) for param in model.parameters()])
        # l2_regularization = lambda2 * l2_penalty(layer2_out)
        '''
        l1_crit = nn.L1Loss(size_average=False)
        reg_loss = 0
        for param in model.parameters():
            reg_loss += l1_crit(param)
        l1_regularization = lambda1 * reg_loss
        '''
        loss = cross_entropy_loss + l1_regularization# + l2_regularization
        
        loss.backward()
        #optimizer.step()

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        for p in model.parameters():
             p.data.add_(-lr, p.grad.data)

        losses.append(loss.data[0])
        if batch_idx % 200 != 0:
             continue
        cnt = 0

        for param in model.parameters():
            print(param.data, param.grad.data)
            for val in param.data.view(-1):
                if val == 0.:
                    cnt += 1
        print(cnt)

    print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
    
    # Evaluate
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

