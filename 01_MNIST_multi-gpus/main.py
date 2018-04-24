import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model
from model import Batch_Net as net


# hyperparameters
batch_size = 256
learning_rate = 1e-2
num_epoches = 20

# transforms.ToTensor(): make the data between 0 and 1
# transforms.Normalize(mean, var): y=(x-mean)/var, if the image is the rgb, ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# transforms.Compose: make the data between -1 and 1
data_tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]
)

# download the dataset
train_dataset = datasets.MNIST(
    root='../data/01', train=True, transform=data_tf, download=True
)
test_dataset = datasets.MNIST(
    root='../data/01', train=False, transform=data_tf, download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model
model = net(28*28, 300, 100, 10)
if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()

# loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

eval_loss = 0
eval_acc = 0
# train
for epoch in range(num_epoches):
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print(str(epoch) + '\t' + str(eval_loss) + "\t" + str(eval_acc))

# initialize
model.eval()
eval_loss = 0
eval_acc = 0

for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data[0]
print(str(eval_loss) + "\t" + str(eval_acc))