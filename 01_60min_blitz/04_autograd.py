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
    out.backward()
    print(x.grad)
