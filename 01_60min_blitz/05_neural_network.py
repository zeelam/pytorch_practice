import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    # 可以直接打印出网络的结构
    print(net)

    # 获取网络中的参数
    params = list(net.parameters())
    print(len(params))
    # conv1's .weight
    print(params[0].size())

    input = torch.randn(1, 1, 32, 32)
    # 由于父类nn.Module定义了__call__()方法, 所以可以通过对象名()的方式来调用
    out = net(input)
    print(out)

    # 将网络中的梯度累计重新设为0
    net.zero_grad()
    out.backward(torch.randn(1, 10))

    # 损失函数
    output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    # 根据网络结构, 可以获取不同层的梯度和所使用的梯度方法等
    # input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
    # -> view -> linear -> relu -> linear -> relu -> linear
    # -> MSELoss
    # -> loss
    print(loss.grad_fn)
    print(loss.grad_fn.next_functions[0][0])
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # 进行10轮(epoch)迭代
    epoch = 10
    learning_rate = 0.01
    criterion = nn.MSELoss()
    for i in range(epoch):
        net.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        # SGD的一次迭代(Iteration)
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    # 使用torch.optim包中的优化实现进行优化, 简化自己写更新参数的代码
    epoch = 10
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # 使用
    criterion = nn.MSELoss()
    for i in range(epoch):
        optimizer.zero_grad()
        output = net(input)
        # 计算loss
        loss = criterion(output, target)
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()