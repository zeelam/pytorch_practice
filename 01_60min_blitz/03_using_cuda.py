from __future__ import print_function
import torch

if __name__ == '__main__':
    # 创建一个4*4的tensor
    x = torch.randn(4, 4)
    # 使用cuda进行并行计算，加快计算速度
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 在指定的device上根据x创建一个都为1的tensor
        y = torch.ones_like(x, device=device)
        # 将tensor x转到指定的device
        x = x.to(device)
        # 在GPU上计算
        z = x + y
        print(z)
        # tensor([[-0.0309, -0.7735, 1.6563, 0.6145],
        #         [-1.7409, 1.1261, 0.6021, -0.0340],
        #         [1.7649, 0.9834, 0.2862, 0.1688],
        #         [0.3684, 1.2934, 1.2293, 1.8706]], device='cuda:0')
        print(z.to("cpu", torch.double))
        # tensor([[-0.0309, -0.7735, 1.6563, 0.6145],
        #         [-1.7409, 1.1261, 0.6021, -0.0340],
        #         [1.7649, 0.9834, 0.2862, 0.1688],
        #         [0.3684, 1.2934, 1.2293, 1.8706]], dtype=torch.float64)
