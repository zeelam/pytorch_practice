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

