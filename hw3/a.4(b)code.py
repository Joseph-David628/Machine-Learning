# -*- coding: utf-8 -*-
"""hw3part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11103OaGq44dhHebkRPF1uEq8bsLD2ydo
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

mnist = fetch_openml('mnist_784', cache=False)

X = mnist.data.astype('float32')
Y = mnist.target.astype('int64')
X /= 255.0

X_train = X[0:60000,:]
labels_train = Y[0:60000]
X_test = X[60000:,:]
labels_test = Y[60000:]

X_train = torch.from_numpy(X_train).float().cuda()
labels_train = torch.from_numpy(labels_train).long().cuda()
X_test = torch.from_numpy(X_test).float().cuda()
labels_test = torch.from_numpy(labels_test).long().cuda()

n, k, d, h1 = 60000, 10, 784, 64
n2 = 10000
step_size = .01

V0 = torch.empty(h1,d,requires_grad=True,device=torch.device('cuda'))
V1 = torch.empty(h1,h1,requires_grad=True,device=torch.device('cuda'))
V2 = torch.empty(k,h1,requires_grad=True,device=torch.device('cuda'))
c0 = torch.empty(h1,1,requires_grad=True,device=torch.device('cuda'))
c1 = torch.empty(h1,1,requires_grad=True,device=torch.device('cuda'))
c2 = torch.empty(k,1,requires_grad=True,device=torch.device('cuda'))

nn.init.uniform_(V0,-1/np.sqrt(d),1/np.sqrt(d))
nn.init.uniform_(c0,-1/np.sqrt(d),1/np.sqrt(d))
nn.init.uniform_(V1,-1/np.sqrt(h1),1/np.sqrt(h1))
nn.init.uniform_(c1,-1/np.sqrt(h1),1/np.sqrt(h1))
nn.init.uniform_(V2,-1/np.sqrt(h1),1/np.sqrt(h1))
nn.init.uniform_(c2,-1/np.sqrt(h1),1/np.sqrt(h1))

def model2(X,W0,b0,W1,b1,W2,b2,batch_size):
    temp = W0 @ torch.transpose(X,0,1)
    for i in range(batch_size):
        temp[:,i] = temp[:,i] + torch.transpose(b0,0,1)
    temp = W1 @ nn.functional.relu(temp)
    for i in range(batch_size):
        temp[:,i] = temp[:,i] + torch.transpose(b1,0,1)
    temp = W2 @ nn.functional.relu(temp)
    for i in range(batch_size):
        temp[:,i] = temp[:,i] + torch.transpose(b2,0,1)
    return torch.transpose(temp,0,1)

def pred_acc2(W0,b0,W1,b1,W2,b2,X,Y,n):
    accuracy=n
    predictions = model2(X,W0,b0,W1,b1,W2,b2,n)
    for i in range(n):
      if (torch.argmax(predictions[i,:]) != Y[i]):
        accuracy -= 1
    return accuracy/n

losses=[]
accuracy=0
optimizer2 = torch.optim.Adam([V0,c0,V1,c1,V2,c2], lr=.01)

batch_size = 10000    #for stochastic gradient descent
while (accuracy < .99):
    batch = np.random.randint(0,60000,size=batch_size)
    y_hat = model2(X_train[batch,:],V0,c0,V1,c1,V2,c2,batch_size)
    target = labels_train[batch]
    loss = nn.functional.cross_entropy(y_hat,target)
    losses.append(loss.item())
    
    optimizer2.zero_grad()
    loss.backward()
    
    optimizer2.step()

    accuracy = pred_acc2(V0,c0,V1,c1,V2,c2,X_train,labels_train,n)
    print(str(accuracy)) 
print('Finished!')

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')

print(pred_acc2(V0,c0,V1,c1,V2,c2,X_test,labels_test,n2))