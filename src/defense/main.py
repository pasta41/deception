import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import sys
import random
import time

# Hyper-parameters
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 20
batch_size = 64
seed = int(sys.argv[2])
torch.manual_seed(seed)

#learning_rate = float(sys.argv[1])
learning_rate = random.random()/2
optim = sys.argv[1]
start = time.time()
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
if optim == 'nes':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
elif optim == 'HB':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (i+1) % 100 == 0:
        #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(optim, learning_rate, seed, 100 * float(correct) / total)
    f = open(str(optim)+'.txt', 'a')
    f.write(str(learning_rate)+'\t'+str(seed)+'\t'+str(100 * float(correct) / total)+'\n')
    f.close()
print(time.time()-start)
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
