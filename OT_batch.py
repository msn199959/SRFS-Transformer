import torch
import torch.nn.functional as F

def sinkhorn_iterations_batched(a, b, M, reg, num_iters=100):
    """
    Perform Sinkhorn iterations for batched inputs with an additional dimension.
    
    :param a: Batched source histograms (tensor of shape [batch_size, 2, n*n]).
    :param b: Batched target histograms (tensor of shape [batch_size, 2, n*n]).
    :param M: Cost matrix.
    :param reg: Regularization term.
    :param num_iters: Number of iterations.
    :return: Total loss calculated over all batches and transport matrices.
    """
    batch_size, channels = a.shape[0],a.shape[1]
    total_loss = 0

    # Assuming the third dimension of a and b is n*n
    n_square = a.shape[2]
    transport_matrices = torch.zeros(batch_size, 2, n_square, n_square)

    for i in range(batch_size):
        for j in range(channels):
            a_batch = a[i, j, :].squeeze()
            b_batch = b[i, j, :].squeeze()
            K = torch.exp(-M / reg)
            Kp = (1 / a_batch).unsqueeze(1) * K

            u = torch.ones_like(a_batch)
            for _ in range(num_iters):
                v = b_batch / (K.T @ u)
                u = 1 / (Kp @ v)

            transport_matrix = u.unsqueeze(1) * K * v.unsqueeze(0)
            transport_matrices[i, j, :, :] = transport_matrix

            # Calculate the loss for this pair (e.g., as the Frobenius norm of the transport matrix)
            # batch_loss = torch.norm(transport_matrix, p='fro') 这是对转移矩阵求范式作为损失
            batch_loss = torch.sum(transport_matrix * M) # 这是转移乘cost
            total_loss += batch_loss

    return total_loss, transport_matrices

# Example usage (assuming appropriate tensors a, b, and M are defined with the correct dimensions)
# total_loss, transport_matrices = sinkhorn_iterations_batched(a, b, M, reg=0.1, num_iters=100)

batch_size = 2
c = 2
# 设置随机种子以获得可重现的结果
torch.manual_seed(0)

# 创建两个 3x3 的张量
N = 3

# 创建一个二维tensor
tensor = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 8.0, 2.0],
                       [9.0, 0.0, 0.0]])

# 创建一个掩码，标记非零元素
mask = tensor != 0

# 提取非零元素
non_zero_elements = tensor[mask]

# 对非零元素应用softmax
softmax_non_zero_elements = F.softmax(non_zero_elements, dim=0)

# 将softmax结果放回原位置
A_tensor = torch.zeros_like(tensor)
A_tensor[mask] = softmax_non_zero_elements

print(f'A_tensor: \n {A_tensor}')

# 创建一个二维tensor
tensor = torch.tensor([[0.55, 0.13, 0.04],
                       [0.064, 0.87, 0.64],
                       [0.898, 0.012, 0.003]])

# 展平tensor
flattened_tensor = tensor.view(-1)

# 对展平后的tensor应用softmax
softmax_tensor = F.softmax(flattened_tensor, dim=0)

# 将softmax后的tensor重塑为原始二维形状
B_tensor = softmax_tensor.view_as(tensor)

print(f'B_tensor: \n {B_tensor}')

# 展平张量
A_flattened_tensor = A_tensor.view(-1)
B_flattened_tensor = B_tensor.view(-1)

# 创建成本矩阵
N2 = N * N
# 使用 PyTorch 的广播和向量化操作来创建成本矩阵
# 生成网格以表示每个元素的坐标
coordinates = torch.tensor([[i, j] for i in range(N) for j in range(N)], dtype=torch.float32)

# 计算所有点对之间的空间位置 L2 距离
cost_matrix_optimized = torch.cdist(coordinates, coordinates, p=2)

# 设置正则化参数
reg = 0.1

A_flattened_tensor = A_flattened_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, c, -1)
B_flattened_tensor = B_flattened_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, c, -1)

print(f'A_flattened_tensor: \n {A_flattened_tensor}')

total_loss, transport_matrices = sinkhorn_iterations_batched(A_flattened_tensor, B_flattened_tensor, cost_matrix_optimized, reg)
print(total_loss)