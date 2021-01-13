import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

class LeNet5(nn.Module):
  def __init__(self, inchannel=3, outchannel=1, num_classes=10, stride=1):
    super(LeNet5, self).__init__()
    self.conv_layer1 = nn.Conv2d(inchannel, 6, kernel_size=5, stride=stride, padding=0, bias=True)
    self.pool_layer2 = nn.AvgPool2d(2, stride=2)
    self.conv_layer3 = nn.Conv2d(6, 16, kernel_size=5, stride=stride, padding=0, bias=True)
    self.pool_layer4 = nn.AvgPool2d(2, stride=2)
    self.conv_layer5 = nn.Conv2d(16, 120, kernel_size=5, stride=stride, padding=0, bias=True)
    self.fc_layer6 = nn.Linear(120, 84)
    self.output_layer = nn.Linear(84, 10)

  def forward(self, x):
    out = F.relu(self.conv_layer1(x))
    out = self.pool_layer2(out)
    out = F.relu(self.conv_layer3(out))
    out = self.pool_layer4(out)
    out = F.relu(self.conv_layer5(out))
    out = out.view(out.size(0), -1)
    out = self.fc_layer6(out)
    out = self.output_layer(out)
    return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCH = 135
    pre_epoch = 0
    BATCH_SIZE = 128
    LR = 0.01

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # Training starts
    for epoch in range(pre_epoch, EPOCH):
      print("Epoch number ", epoch+1)
      net.train()
      sum_loss = 0.0
      correct = 0.0
      total = 0.0

      for i, data in enumerate(trainloader, 0):
        length_batch = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # 这里的outputs.data 的data是哪里定义的？
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        # accuracy = 100.0 * (correct / total)
        # print("Accurcy at iteration",i + 1 + epoch * length_batch, "     ", accuracy)

      accuracy = 100.0 * (correct / total)
      print("Accuracy = ", accuracy.item(), "; Loss = ", sum_loss)

      with torch.no_grad():
        net.eval()
        test_correct = 0
        test_total = 0
        for test_data in testloader:
          test_inputs, test_labels = test_data
          test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

          test_outputs = net(test_inputs)
          _, test_predicted = torch.max(test_outputs.data, 1)
          test_total += test_labels.size(0)
          # print('test_labels length: ', len(test_labels))
          test_correct += test_predicted.eq(test_labels).sum().item()
          # print('test_predicted.eq(test_labels).sum(): ', test_predicted.eq(test_labels).sum().item())

        accuracy = 100.0 * (test_correct / test_total)
        print("Test set accuracy = ", accuracy, "\n")




