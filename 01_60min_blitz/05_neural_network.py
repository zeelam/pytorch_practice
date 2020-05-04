import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super(Net, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 卷积层2
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 全连接层1
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播方法
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = Net()
    print(type(net))
    print(net)

    params = list(net.parameters())
    print(len(params))
    # conv1's .weight
    print(params[0].size())

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)