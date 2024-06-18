import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# 加载数据并进行预处理
def load_data():
    iris = load_iris()
    data = iris.data
    targets = iris.target
    data_tensor = torch.tensor(data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(data_tensor, targets_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader
# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(4, 16)  # 隐藏层
        self.output = nn.Linear(16, 3)  # 输出层

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # 使用 ReLU 激活函数
        x = self.output(x)
        return x
# 训练和评估模型
def train_and_evaluate(model, train_loader, test_loader, epochs=20, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    return train_losses, test_losses
# 可视化训练和测试误差
def plot_losses(train_losses, test_losses):
    epochs = len(train_losses)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def main():
    train_loader, test_loader = load_data()
    model = MLP()
    train_losses, test_losses = train_and_evaluate(model, train_loader, test_loader)
    plot_losses(train_losses, test_losses)
if __name__ == '__main__':
    main()