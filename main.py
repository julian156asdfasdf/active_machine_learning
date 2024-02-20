#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
 
def blackbox(lr, momentum, k,epochs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    class CIFAR10Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
            self.act1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.3)
    
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
            self.act2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
    
            self.flat = nn.Flatten()
    
            self.fc3 = nn.Linear(8192, 512)
            self.act3 = nn.ReLU()
            self.drop3 = nn.Dropout(0.5)
    
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



    model = CIFAR10Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epoch):
        for inputs, labels in trainloader:
            # forward, backward, and then weight update
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        acc = 0
        count = 0
        for inputs, labels in testloader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
    
    torch.save(model.state_dict(), "cifar10model.pth")
    return acc


blackbox(lr = 0.001, momentum = 0.9, k = 1000, epoch = 20)
# %%
