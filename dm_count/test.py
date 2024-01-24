import torch
import numpy as np
mu = torch.rand([8, 1, 8, 8])
B, C, H, W = mu.size()
mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
print(mu_sum)
print(mu_sum.shape)
