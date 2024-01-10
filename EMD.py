import torch
import scipy.stats
import numpy as np

def wasserstein_distance_tensor_gpu(tensor1, tensor2):
    # 确保tensor在GPU上
    '''
    if not tensor1.is_cuda or not tensor2.is_cuda:
        raise ValueError("Tensors must be on GPU.")
    '''

    # 将二维张量扁平化为一维并移至CPU
    flat_tensor1 = tensor1.view(-1).cpu()
    flat_tensor2 = tensor2.view(-1).cpu()

    # 生成对应的索引位置
    dists_P = torch.arange(flat_tensor1.shape[0]).cpu()
    dists_Q = torch.arange(flat_tensor2.shape[0]).cpu()

    # 将权重转换为numpy数组
    P = flat_tensor1.numpy()
    Q = flat_tensor2.numpy()

    # 计算Wasserstein距离
    return scipy.stats.wasserstein_distance(dists_P, dists_Q, P, Q)

# 示例
# 假设你的PyTorch已经配置了CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
tensor2 = torch.tensor([[2.0, 3.0], [4.0, 1.0]], device=device)

distance = wasserstein_distance_tensor_gpu(tensor1, tensor2)
print("Wasserstein Distance:", distance)
