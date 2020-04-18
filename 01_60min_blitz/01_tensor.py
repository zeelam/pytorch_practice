from __future__ import print_function
import torch

if __name__ == '__main__':

    # 定义tensor
    x = torch.empty(5, 3)
    print(x)

    x = torch.Tensor(5, 3)
    print(x)

    # 指定dtype的类型
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    # 直接从data中生成tensor
    x = torch.tensor([5.5, 3])
    print(x)

    # 从现有的tensor基础上创建新的tensor
    # create a tensor based on an existing tensor.
    # These methods will reuse properties of the input tensor,
    # e.g. dtype, unless new values are provided by user
    x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
    print(x)
    x = torch.randn_like(x, dtype=torch.float)  # override dtype!
    print(x)

    x = torch.rand(5, 3)
    print(x)
    print(x.size())

    # tensor的计算
    y = torch.rand(5, 3)

    # 方法一 运算符
    z = x + y
    print(z)
    # 方法二 方法
    result = torch.Tensor(5, 3)
    torch.add(x, y, out=result)
    print(result)
    # 方法三 直接覆盖
    # Addition: in-place
    # adds x to y
    y.add_(x)
    print(y)

    # tensor的resize/reshape
    # 取出张量x的第二列
    print(x[:, 1])

    # 使用view来reshape tensor
    # 创建一个4*4的tensor
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)

    print(x)
    print(y)
    print(z)
    print(x.size(), y.size(), z.size())

    # 当tensor只有一个元素时，可以使用item来取出tensor中的元素
    x = torch.randn(1)
    print(x)
    print(x.item())

    # 多个元素可以使用多维数组的方式取出
    x = torch.randn(4, 4)
    print(x)
    # 第一行，第一列的元素，取出还是单个tensor
    print(x[0, 0])
    # 第一行，第二列的元素，使用item()获取值
    print(x[0, 1].item())
    # 第二行，第一列
    print(x[1, 0].item())