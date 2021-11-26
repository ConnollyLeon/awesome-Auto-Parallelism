'''
An example of using torchgpipe.

TODO: This file is currently not executing correctly due to multiple submodule exists.

torchgpipe only support model implemented by nn.Sequential.

'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 10

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time


class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

    def forward(self, x):
        x = self.seq2(self.seq1(x))
        return self.fc(x.view(x.size(0), -1))


def flatten_sequential(module):
    def _flatten(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                for sub_name, sub_child in _flatten(child):
                    yield f'{name}_{sub_name}', sub_child
            else:
                yield name, child

    return nn.Sequential(OrderedDict(_flatten(module)))


if __name__ == '__main__':
    # Data Preparation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model Definition
    model = ResNet50()

    model = flatten_sequential(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Get device_numbers
    partitions = torch.cuda.device_count()

    # Prepare sample data
    sample = torch.rand(4, 3, 224, 224)

    # balance the partitions  by time.
    balance = balance_by_time(partitions, model, sample)
    print("The balance is: ", balance)
    model = GPipe(model, balance, chunks=8)

    # loss definition
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
