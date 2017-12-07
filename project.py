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

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=2)

batchsize = 128
lambda1 = 0.01
lr = 1e-3
trials = 50


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
       
def train(model, optimizer, mode, threshold):
    print("training.. ")
    accuracies = []
    for epoch in range(trials):
        losses = []
        # Train

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # sparsifying the params at every step when mode = intrain:
            if(mode == "intrain"):
                currentThreshold = (epoch/trials)*threshold
                for param in model.parameters():
                    param.data = sparsify(param, currentThreshold)

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

        accuracy = (100.*correct/total)
        print('Epoch : %d Test Acc : %.3f' % (epoch, accuracy))
        print('--------------------------------------------------------------')
        accuracies.append(accuracy)
        model.train()

    #Sparsify the final model irrespective of mode. The intrain mode runs only from 0 to 9, so this will add the final sparcification:
    for param in model.parameters():
              param.data = sparsify(param, threshold)

    return accuracies


def test(model2):
    print("Testing")
    cnt, tot = 0, 0
    for param in model2.parameters():
        tot += param.data.view(-1).size()[0]
        for val in param.data.view(-1):
            if val == 0.:
                cnt += 1
    sparselevel = cnt*100./tot
    print(str(sparselevel) + "% sparse")

    model2.eval()
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = model2(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    testaccuracy = 100.*correct/total
    print('Test Accuracy : %.3f' % (testaccuracy))
    return sparselevel, (100.*correct/total)

def getModel(modelType):
    if(modelType == "MLP"):
        return MLP()
    else:
        return CNN()


def main(model, algorithm="Adam", mode="intrain", threshold=0):
    # threshold = thresholdParam
    

    #Choose optimizer:
    optimizerfunction = getattr(torch.optim, algorithm)
    optimizer = optimizerfunction(model.parameters(), lr)

    if(mode == "posttrain" and threshold != 0):
        # INCASE OF POSTTRAIN, we only need to train the first(threshold > 0) model:
        trainingAccuracies = []
        model2 = copy.deepcopy(model)
        #Sparsify the final model irrespective of mode. The intrain mode runs only from 0 to 9, so this will add the final sparcification:
        for param in model2.parameters():
            param.data = sparsify(param, threshold)
    else:
        # IN CASE of pretrain, all the models will be trained.
        trainingAccuracies = train(model, optimizer, mode, threshold)
        model2 = copy.deepcopy(model)
    

    sparselevel, testaccuracy = test(model2)

    return trainingAccuracies, sparselevel, testaccuracy
