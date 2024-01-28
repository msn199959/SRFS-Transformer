import torch
output = torch.rand(size = (2, 4, 4))
print(output)
scale = torch.tensor([[2],[3]]).unsqueeze(1).unsqueeze(2)
print(scale)
output = output/scale
print(output)
