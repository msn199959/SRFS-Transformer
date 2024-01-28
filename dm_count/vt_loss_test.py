import numpy as np 
import torch
import torch.nn as nn

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h* im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


def output_norm(mu):
    B, C, H, W = mu.size()
    mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = mu / (mu_sum + 1e-6)
    return mu_normed

im_height = im_width = 8
points = np.array([[1, 2], [2, 2]])
gt_discrete_map = gen_discrete_map(im_height, im_width, points)
output = torch.rand(size = (1, 2, im_height, im_height))
outputs_normed = output_norm(output)

tv_loss = nn.L1Loss(reduction='none')
gd_count = np.array([len(p) for p in points], dtype=np.float32)
gd_count_tensor = torch.from_numpy(gd_count).float().unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
print(f"output:\n {output}")
print(f"outputs_normed:\n {outputs_normed}")
print(torch.sum(output))
print(torch.sum(outputs_normed))
print(torch.sum(outputs_normed[0]))
print(f"gd_count:\n {gd_count}")
gt_discrete_normed = gt_discrete_map / (gd_count_tensor + 1e-6)
print(f"gt_discrete_normed:\n{gt_discrete_normed}")
gt_discrete_normed = gt_discrete_normed[:1,...]
tv = (tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1) * torch.from_numpy(gd_count).float())#.mean(0)
print(f"tv:\n{tv}")
