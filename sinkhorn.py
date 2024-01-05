import torch
import torch.nn.functional as F

def sinkhorn_distance_torch(tensor_a, tensor_b, reg, num_iters=100):
    n = tensor_a.size(1)
    a, b = torch.ones(n,).to(tensor_a.device) / n, torch.ones(n,).to(tensor_b.device) / n

    M = torch.cdist(tensor_a, tensor_b, p=2)
    print(M)
    K = torch.exp(-M / reg)
    Kp = (1 / a).unsqueeze(0) * K
    u = torch.ones_like(a)

    for _ in range(num_iters):
        u = 1.0 / torch.matmul(Kp, b / torch.matmul(K.t(), u))

    sinkhorn_distance = torch.dot(u, torch.matmul(K, b)) * reg
    return sinkhorn_distance

# 示例使用
# 创建两个随机N*N的Tensor
N = 5
tensor_a = torch.rand(N, N)
print(tensor_a)
tensor_b = torch.rand(N, N)*3
print(tensor_b)

# 设置正则化参数
reg = 1

# 计算Sinkhorn距离
distance = sinkhorn_distance_torch(tensor_a, tensor_b, reg)
print(distance)
