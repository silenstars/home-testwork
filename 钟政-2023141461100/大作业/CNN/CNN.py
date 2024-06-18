import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor格式
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：3个输入通道，6个输出通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，2x2的窗口
        # 第二个卷积层：6个输入通道，16个输出通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层：输入维度是16*5*5，输出维度是120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 最终输出10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积、ReLU和池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积、ReLU和池化
        x = x.view(-1, 16 * 5 * 5)  # 展平成向量
        x = F.relu(self.fc1(x))  # 全连接层1和ReLU
        x = F.relu(self.fc2(x))  # 全连接层2和ReLU
        x = self.fc3(x)  # 输出层，无激活函数
        return x

# 创建网络实例
net = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
for epoch in range(2):  # 训练2个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 获取数据
        optimizer.zero_grad()  # 梯度清零
        outputs = net(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000批次打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 测试网络
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
