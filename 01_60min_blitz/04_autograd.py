import torch

if __name__ == '__main__':
    # autograd 自动微分

    # 默认创建tensor时是不带requires_grad的
    x = torch.ones(2, 2)
    print(x)
    print(x.requires_grad)

    y = torch.ones(2, 2, requires_grad=True)
    print(y)
    print(y.requires_grad)

    # 如果tensor使用了requires_grad，则由其衍生的其他tensor也会自动带上requires_grad

    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    print(y)
    # 可以使用grad_fn属性查看grad的方法
    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()

    print(z, out)

    # 使用requires_grad_()方法来改变已存在的tensor的requires_grad标识
    # 默认为False
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    # 使用requires_grad_()方法改为True
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

    # Gradients 梯度
    # 如果out是个标量的话, 可以直接计算梯度
    print("out is ", out)
    out.backward()
    print(x.grad)

    # 如果不是标量的话
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    # 此时y是一个torch, 则在计算梯度的时候必须要传入grad_output参数
    print(y)
    v = torch.tensor([0.0001, 0.1, 1], dtype=torch.float)
    # y.backward() # grad can be implicitly created only for scalar outputs
    y.backward(v)
    print(x.grad)

    # 可以使用torch.no_grad()来停止相应tensor计算后的tensor默认require_grad标识为True的情况
    # True
    print(x.requires_grad)
    # True, 因为x ** 2是由x计算而来, 所以x ** 2的requires_grad是True
    print((x ** 2).requires_grad)

    # 使用torch.no_grad()可以使在这个作用域范围内的所有参数require_grad为False, 用以节省显存, 在推理(inference)过程中经常使用
    with torch.no_grad():
        # False
        print((x ** 2).requires_grad)

    # .detach()可以从现有的tensor中创建一个新的tensor, 但是require_grad为False
    # True
    print(x.requires_grad)
    y = x.detach()
    # False
    print(y.requires_grad)
    # tensor(True)
    print(x.eq(y).all())