import torch
import torch.nn.functional as F

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
softmax_tensor = torch.zeros_like(tensor)
softmax_tensor[mask] = softmax_non_zero_elements

print(softmax_tensor)


# 创建一个二维tensor
tensor = torch.tensor([[0.55, 0.13, 0.04],
                       [0.064, 0.87, 0.64],
                       [0.898, 0.012, 0.003]])

# 展平tensor
flattened_tensor = tensor.view(-1)

# 对展平后的tensor应用softmax
softmax_tensor = F.softmax(flattened_tensor, dim=0)

# 将softmax后的tensor重塑为原始二维形状
softmax_2d = softmax_tensor.view_as(tensor)

print(softmax_2d)


N=3
tensor = torch.zeros((N, N))

# 展平tensor
flattened_tensor = tensor.view(-1)

# 对展平后的tensor应用softmax
softmax_tensor = F.softmax(flattened_tensor, dim=0)

# 将softmax后的tensor重塑为原始二维形状
softmax_2d = softmax_tensor.view_as(tensor)

print(softmax_2d)


