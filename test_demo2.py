import torch
import torch.nn.functional as F
tensor = torch.tensor([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],dtype=torch.float32)
# 展平tensor
flattened_tensor = tensor.view(-1)

# 对展平后的tensor应用softmax
softmax_tensor = F.softmax(flattened_tensor, dim=0)

# 将softmax后的tensor重塑为原始二维形状
softmax_2d = softmax_tensor.view_as(tensor)

print(softmax_2d)


# 创建一个二维tensor
tensor = torch.tensor([[1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

# 创建一个掩码，标记非零元素
mask = tensor != 0

# 提取非零元素
non_zero_elements = tensor[mask]
print(non_zero_elements)

# 对非零元素应用softmax
softmax_non_zero_elements = F.softmax(non_zero_elements, dim=0)

# 将softmax结果放回原位置
softmax_tensor = torch.zeros_like(tensor)
softmax_tensor[mask] = softmax_non_zero_elements

print(softmax_tensor)

tensor_a = torch.tensor([[1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
tensor_b = torch.tensor([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
M = torch.cdist(tensor_a, tensor_b, p=2)
print(M)

