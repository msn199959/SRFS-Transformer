import torch
import torch.nn.functional as F

def sinkhorn_iterations(a, b, M, reg, num_iters=100):
    """
    Perform Sinkhorn iterations to compute the optimal transport matrix.
    
    :param a: Source histogram (as a tensor).
    :param b: Target histogram (as a tensor).
    :param M: Cost matrix.
    :param reg: Regularization term.
    :param num_iters: Number of iterations.
    :return: Approximated transport matrix.
    """
    K = torch.exp(-M / reg)
    Kp = (1 / a).unsqueeze(1) * K

    u = torch.ones_like(a)
    for _ in range(num_iters):
        v = b / (K.T @ u)
        u = 1 / (Kp @ v)

    transport_matrix = u.unsqueeze(1) * K * v.unsqueeze(0)
    return transport_matrix

# 设置随机种子以获得可重现的结果
torch.manual_seed(0)

# 创建两个 3x3 的张量
N = 3

# 创建一个二维tensor
tensor = torch.tensor([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])

# 创建一个掩码，标记非零元素
mask = tensor != 0

# 提取非零元素
non_zero_elements = tensor[mask]

if len(non_zero_elements) != 0:
    # 对非零元素应用softmax
    softmax_non_zero_elements = F.softmax(non_zero_elements, dim=0)

    # 将softmax结果放回原位置
    A_tensor = torch.zeros_like(tensor)
    A_tensor[mask] = softmax_non_zero_elements
else:
    flattened_tensor = tensor.view(-1)
    # 对展平后的tensor应用softmax
    softmax_tensor = F.softmax(flattened_tensor, dim=0)
    # 将softmax后的tensor重塑为原始二维形状
    A_tensor = softmax_tensor.view_as(tensor)

print(f'A_tensor: \n {A_tensor}')

# 创建一个二维tensor
tensor = torch.tensor([[0.005, 0.13, 0.04],
                       [0.004, 0.87, 0.004],
                       [0.008, 0.012, 0.003]])

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

# 计算 Sinkhorn 迭代
optimal_transport_matrix = sinkhorn_iterations(A_flattened_tensor, B_flattened_tensor, cost_matrix_optimized, reg)

# 初始化一个与 B 形状相同的张量，用于存储传输后的结果
transferred_tensor = torch.zeros_like(B_tensor)

# 遍历每个元素，并根据运输矩阵分配它们
for i in range(N2):
    for j in range(N2):
        # 将 A 中的元素按照权重分配到 B 的位置
        transferred_tensor[j // N][j % N] += A_flattened_tensor[i] * optimal_transport_matrix[i, j]
print(f'transferred_tensor: \n {transferred_tensor}')

# 计算运输矩阵和成本矩阵的点积
transport_cost = optimal_transport_matrix * cost_matrix_optimized

# 计算最终的损失，即所有元素的总和
ot_loss = transport_cost.sum()

print(ot_loss)
