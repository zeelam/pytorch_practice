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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in dataloader:
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

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, dataloader, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, range(10))
    cm.plot(values_format='d', cmap='Blues', ax=ax)

def plot_most_incorrect(incorrect, n_images):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
        ax.set_title("true label: %s (%.3f) \n"
                     "pred label: %s (%.3f)" % (true_label.item(), true_prob, incorrect_label.item(), incorrect_prob))
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def get_representations(model, data_loader, device):
    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y_pred, h = model(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels

def get_pca(data, n_components = 2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data

def plot_representations(data, labels, n_images = None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles=handles, labels=labels)
    plt.show()

def get_tsne(data, n_components = 2, n_images = None):
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def imagine_digit(model, digit, device, n_iterations = 50_000):
    model.eval()

    best_prob = 0
    best_image = None

    with torch.no_grad():
        for _ in range(n_iterations):
            x = torch.randn(32, 1, 28, 28).to(device)
            y_pred, _ = model(x)
            preds = F.softmax(y_pred, dim=-1)
            _best_prob, index = torch.max(preds[:, digit], dim=0)

            if _best_prob > best_prob:
                best_prob = _best_prob
                best_image = x[index]
    return best_image, best_prob

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
        # 使用relu做非线性变换函数
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        x = x.view(x.shape[0], -1)
        # 保留在全链接前的参数, 用于展示经过两层卷积层后的图像是怎么样的
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
    OUTPUT_DIM = 10
    model = LeNet(OUTPUT_DIM)

    print('The model has %s trainable parameters' % format(count_parameters(model), ','))

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)

    EPOCHS = 20
    best_valid_loss = float('inf')

    # for epoch in range(EPOCHS):
    #     start_time = time.time()
    #
    #     train_loss, train_acc = train(model, train_data_loader, optimizer, criterion, device)
    #     valid_loss, valid_acc = evaluate(model, valid_data_loader, criterion, device)
    #
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), 'tut2-model.pt')
    #
    #     end_time = time.time()
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     print("Epoch: %02d | Epoch Time: %dm %ds" % ((epoch + 1), epoch_mins, epoch_secs))
    #     print("\tTrain Loss: %.3f | Train Acc: %.2f%%" % (train_loss, train_acc * 100))
    #     print("\t Val. Loss: %.3f |  Val. Acc: %.2f%%" % (valid_loss, valid_acc * 100))

    print("The training is finished")

    model.load_state_dict(torch.load('tut2-model.pt'))
    test_loss, test_acc = evaluate(model, test_data_loader, criterion, device)

    print("Test Loss: %.3f | Test Acc: %.2f%%" % (test_loss, test_acc * 100))

    images, labels, probs = get_predictions(model, test_data_loader, device)

    pred_labels = torch.argmax(probs, 1)
    plot_confusion_matrix(labels, pred_labels)
    corrects = torch.eq(labels, pred_labels)

    incorrect_samples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_samples.append((image, label, prob))
    incorrect_samples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    N_IMAGES = 25
    plot_most_incorrect(incorrect_samples, N_IMAGES)

    outputs, intermediates, labels = get_representations(model, train_data_loader, device)
    output_pca_data = get_pca(outputs)
    plot_representations(output_pca_data, labels)

    intermediate_pca_data = get_pca(intermediates)
    plot_representations(intermediate_pca_data, labels)

    N_IMAGES = 5_000

    output_tsne_data = get_tsne(outputs, n_images=N_IMAGES)
    plot_representations(output_tsne_data, labels, n_images=N_IMAGES)

    intermediate_tsne_data = get_tsne(intermediates, n_images=N_IMAGES)
    plot_representations(intermediate_tsne_data, labels, n_images=N_IMAGES)

