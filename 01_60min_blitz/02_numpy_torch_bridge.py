import numpy as np
import torch

if __name__ == '__main__':
    # torch tensor to numpy array
    x = torch.ones(5)
    y = x.numpy()
    print(y)

    # torch张量与numpy关联, 改变了tensor, numpy的值也会跟着改变
    x.add_(1)
    print(x)
    print(y)

    # numpy array to torch tensor
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a)
    print(b)

    # 使用numpy array转化为torch tensor后，numpy array的值改变，tensor也会随之改变
    np.add(a, 1, out=a)
    print(b)


