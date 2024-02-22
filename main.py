#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#lr = 0.001
#momentum = 0.9 
#drop1 = 0.3 
#drop2=0.5

k = 1000
epochs = 5 

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def black_box(kernel_sizea, drop1, drop2):
 
    class CIFAR10Model(nn.Module):
        def __init__(self):
            super().__init__()
            kernel_sizeb = int(kernel_sizea)
            if kernel_sizeb % 2 == 0:
                kernel_sizeb += 1
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(kernel_sizeb,kernel_sizeb), stride=1, padding=kernel_sizeb//2)
            self.act1 = nn.ReLU()
            self.drop1 = nn.Dropout(drop1)

            self.conv2 = nn.Conv2d(32, 32, kernel_size=(kernel_sizeb,kernel_sizeb), stride=1, padding=kernel_sizeb//2)
            self.act2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

            self.flat = nn.Flatten()

            self.fc3 = nn.Linear(8192, 512)
            self.act3 = nn.ReLU()
            self.drop3 = nn.Dropout(drop2)

            self.fc4 = nn.Linear(512, 10)

        def forward(self, x):
            # input 3x32x32, output 32x32x32
            x = self.act1(self.conv1(x))
            x = self.drop1(x)
            # input 32x32x32, output 32x32x32
            x = self.act2(self.conv2(x))
            # input 32x32x32, output 32x16x16
            x = self.pool2(x)
            # input 32x16x16, output 8192
            x = self.flat(x)
            # input 8192, output 512
            x = self.act3(self.fc3(x))
            x = self.drop3(x)
            # input 512, output 10
            x = self.fc4(x)
            return x

    model = CIFAR10Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)



    for epoch in range(epochs):
        stop = 0
        for inputs, labels in trainloader:
            # forward, backward, and then weight update
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stop+=1
            if stop == k:
                break
        acc = 0
        count = 0
        stop = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
            if stop == k:
                break
        acc /= count
    return acc.item()




from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

pbounds = {'kernel_sizea': (1, 7), 'drop1':(1e-5, 1) , 'drop2':(1e-5, 1)}

optimizer = BayesianOptimization(
    f=black_box,
    pbounds=pbounds,
)



optimizer.maximize(
    init_points=4,
    n_iter=25,
)

optimizer.res
# %%

import streamlit
import matplotlib.pyplot as plt


targets = []
values = []
for dic in optimizer.res:
    targets.append(dic['target'])
    values.append(list(dic['params'].values()))


plt.plot(targets, )


# %%



