import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

def plot_filter(images, filter):
    # 使用unsqueeze增加维度, 否则cat(concat)之后, images的第一维就会被加起来了
    # 由于images中的每一个image的维度是(1, 28, 28), 在索引为0的前面增加一维则会变成(1, 1, 28, 28)
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    # filter是外面自定义的算子, 维度是(3, 3),
    # 所以需要unsqueeze(0)两次, 才可以使用conv2d和images中的每个图像做卷积运算
    filter = torch.FloatTensor(filter).unsqueeze(0).unsqueeze(0).cpu()

    n_images = images.shape[0]
    filtered_images = F.conv2d(images, filter)

    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):
        ax = fig.add_subplot(2, n_images, i + 1)
        # 由于images中的图像是(n_images, 1, 28, 28)的, 所以需要把第一列的维度去掉, 恢复成(1, 28, 28)的灰度图像
        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title("Original")
        ax.axis('off')

        # 同理, 卷积操作过后的图像也是需要把第一个维度去掉
        image = filtered_images[i].squeeze(0)

        ax = fig.add_subplot(2, n_images, n_images + i + 1)
        ax.imshow(image, cmap='bone')
        ax.set_title("Filtered")
        ax.axis('off')

    plt.show()

def plot_subsample(images, pool_type, pool_size):
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    # max pooling
    if pool_type.lower() == 'max':
        pool = F.max_pool2d
    # avg pooling
    elif pool_type.lower() in ['mean', 'avg']:
        pool = F.avg_pool2d
    else:
        raise ValueError('pool_type must be either max or mean, got %s' % pool_type)

    n_images = images.shape[0]

    pooled_images = pool(images, kernel_size=pool_size)

    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):
        ax = fig.add_subplot(2, n_images, i + 1)
        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title('Original')
        ax.axis('off')

        image = pooled_images[i].squeeze(0)
        ax = fig.add_subplot(2, n_images, n_images + i + 1)
        ax.imshow(image, cmap='bone')
        ax.set_title('Subsampled')
        ax.axis('off')

    plt.show()

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        # 1 * 28 * 28维度的图片经过维度为6(out_channels) * 5 * 5, 步长为1, 并且padding为0的运算后为6 * 24 * 24维度
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 经过2 * 2维度的max pooling层计算后为 6 * 12 * 12, 再经过第二个维度为16 * 5 * 5，步长为1, padding为0的卷积操作后为16 * 8 * 8维度
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 再经过2*2的max pooling层, 最终维度为16 * 4 * 4, 因此全连接层为16 * 4 * 4个输入（展平到一个维度后）, 输出维度为120
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        # 后面的维度就很简单了
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        x = x.view(x.shape[0], -1)
        h = x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x, h

if __name__ == '__main__':
    SEED = 2020
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    ROOT = "./data"
    train_data = datasets.MNIST(root=ROOT, train=True, download=True)
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    print("Calculated mean: %f" % mean)
    print("Calculated std: %f" % std)

    train_transforms = transforms.Compose([
        transforms.RandomRotation(5, fill=(0,)),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
                        ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
                        ])

    train_data = datasets.MNIST(root=ROOT, train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST(root=ROOT, train=False, download=True, transform=test_transforms)

    VALID_RADIO = 0.9

    n_train_samples = int(len(train_data) * VALID_RADIO);
    n_valid_samples = len(train_data) - n_train_samples;

    train_data, valid_data = data.random_split(train_data,[n_train_samples, n_valid_samples])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    print("Show the length of dataset")
    print("Number of training samples: %d" % len(train_data))
    print("Number of validation samples: %d" % len(valid_data))
    print("Number of testing samples: %d" % len(test_data))

    BATCH_SIZE = 64

    train_data_loader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_data_loader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE)

    N_IMAGES = 5
    images = [image for image, label in [test_data[i] for i in range(N_IMAGES)]]

    # 不同算子对图像进行卷积后的结果
    # 比方说以下两个算子可以通过卷积后产生的图像可以更好的展示图片中物体从上到下的边缘和从下到上的边缘
    horizontal_filter = [[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]
    plot_filter(images, horizontal_filter)

    horizontal_filter = [[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]
    plot_filter(images, horizontal_filter)

    # 也可以有不同的算子获得左边缘和右边缘
    vertical_filter = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]
    plot_filter(images, vertical_filter)

    vertical_filter = [[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]]
    plot_filter(images, vertical_filter)

    # 而卷积神经网络(CNN), 则是随机初始化各种算子, 然后通过优化loss的方式, 通过神经网络的梯度不断的更新算子的参数,
    # 最终找到能提取图像中某些特征的算子

    # Pooling层, 可以看作是尽量保留图像特征, 又缩小图像尺寸的方法, 避免过多的参数计算, 同时如果在卷积层大小不改变的情况下,
    # 更小的图像卷积操作后, 可以提取到更偏向于全图的特征
    # 展示Pooling的前后效果
    # 2*2的maxPooling
    plot_subsample(images, 'max', 2)
    # 3*3的maxPooling
    plot_subsample(images, 'max', 3)

    # 2*2的avgPooling
    plot_subsample(images, 'avg', 2)
    # 3*3的avgPooling
    plot_subsample(images, 'avg', 3)

    # 使用LeNet来训练