#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
def blackbox(lr, momentum, k,epochs,drop1,drop2):
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
            self.drop1 = nn.Dropout(drop1)
    
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
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
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
    return acc


for i in range(5):
    blackbox(lr = 0.001, momentum = 0.9, k = 100, epochs = 5, drop1 = 0.1*(i+1), drop2=0.15*(i+1))
# %%
