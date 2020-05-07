"""
导入各种包及其作用
torch:使用PyTorch相关的方法
torch.backend.cudnn:为了设置backend的属性
torch.nn, torch.nn.functional:使用nn的相关方法, 构建神经网络
torch.optim:优化器相关的方法, 用来更新参数
torch.utils.data:用来处理数据, 比如dataLoader
torchvision: CV相关的包, 包括很多数据集
sklearn.metrics:可以用来可视化混淆矩阵
sklearn.decomposition, sklearn.manifold:使用二维数组的方式可视化网络的表现, 比方说训练后的各个参数用图像如何表现等
matplotlib:用于展示图像
"""
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold

import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

# 定义MLP网络结构
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        y_pred = self.output_fc(x)
        return y_pred, x

# 展示图片的方法
def plot_images(images):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure()
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='bone')
        ax.axis('off')
    plt.show()

# 计算模型参数的方法
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 正确率计算
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# 训练模型
def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x)

        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

# 模型推断
def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

# epoch时间计算
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 获取图片对应的Ground Truth和Prediction
def get_predictions(model, data_loader, device):
    model.eval()
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

# 展示混淆矩阵
def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, range(10))
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.show()

# 展示预测置信度最低的Top N张图像
def plot_most_incorrect(incorrect, n_images):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
        ax.set_title("True label: %s (%.3f) \n"
                     "Pred label: %s (%.3f)" % (true_label.item(), true_prob, incorrect_label.item(), incorrect_prob))
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()

if __name__ == '__main__':
    # 为了复现结果, 设置Python, Numpy和PyTorch的随机种子
    SEED = 2020
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # 下载数据
    ROOT = "./data"
    train_data = torchvision.datasets.MNIST(root=ROOT, train=True, download=True)
    # 数据归一化, 加快计算和防止局部最优
    # 计算平均数和标准差
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    print("Calculated mean: %f" % mean)
    print("Calculated std: %f" % std)

    # 使用transforms做数据增强, 将PILImage转化为Tensor, 以及归一化
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomRotation(5, fill=(0,)),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
                                    ])
    test_transforms = test_transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # 获取mnist数据
    train_data = torchvision.datasets.MNIST(root=ROOT, train=True, download=True, transform=train_transforms)
    test_data = torchvision.datasets.MNIST(root=ROOT, train=False, download=True, transform=test_transforms)

    print("Number of training examples: %d" % len(train_data))
    print("Number of testing examples: %d" % len(test_data))

    # 加载25张图片先预览一下
    N_IMAGES = 25
    images = [image for image, label in [train_data[i] for i in range(N_IMAGES)]]
    plot_images(images)

    # 将训练集分割为训练集和验证集
    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    print("==================After split======================")
    print("Number of training examples: %d" % len(train_data))
    print("Number of validation examples: %d" % len(valid_data))
    print("Number of testing examples: %d" % len(test_data))

    # 加载25张验证集的图片
    N_IMAGES = 25
    images = [image for image, label in [valid_data[i] for i in range(N_IMAGES)]]
    plot_images(images)

    # 使用deepcopy防止在更改valid_data的transform属性时, train_data的transform属性也一起更改
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    # 再次加载, 此时valid_data不会有随机旋转及裁切的情况
    N_IMAGES = 25
    images = [image for image, label in [valid_data[i] for i in range(N_IMAGES)]]
    plot_images(images)

    # 设定batch_size
    BATCH_SIZE = 64
    train_data_loader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_data_loader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE)

    # 创建模型
    INPUT_DIM = 1 * 28 * 28
    OUTPUT_DIM = 10

    model = MLP(INPUT_DIM, OUTPUT_DIM)
    print("The model has %d trainable parameters" % count_parameters(model))

    print("Start Training")
    # 训练模型
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # 使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    # 开始训练模型
    EPOCHS = 10
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_data_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("Epoch: %2d | Epoch Time: %dm %ds" % (epoch + 1, epoch_mins, epoch_secs))
        print("\tTrain Loss: %.3f | Train Acc: %.2f%%" % (train_loss, 100 * train_acc))
        print("\tVal.  Loss: %.3f | Val.  Acc: %.2f%%" % (valid_loss, 100 * valid_acc))

    print("Training is Finished")

    # 加载训练好的最佳模型
    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(model, test_data_loader, criterion, device)
    print("Test Loss: %.3f | Test Acc: %.2f%%" % (test_loss, 100 * test_acc))

    images, labels, probs = get_predictions(model, test_data_loader, device)
    # 获取预测出来的预测值对应的预测label
    pred_labels = torch.argmax(probs, 1)

    # 展示混淆矩阵
    plot_confusion_matrix(labels, pred_labels)

    # 计算出预测正确的数量
    corrects = torch.eq(labels, pred_labels)

    # 将预测错误的样本根据置信度(prob)进行倒序排序
    incorrect_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    # 展示预测的最差的前25个图像
    N_IMAGES = 25
    plot_most_incorrect(incorrect_examples, N_IMAGES)
