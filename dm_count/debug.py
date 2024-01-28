import numpy as np
import torch

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

def target_map_to_grid(targets, grid_size=8):
        # 创建一个 3x3 的零矩阵
        grid = torch.zeros((grid_size, grid_size), dtype=torch.int)

        # 遍历每个点
        for point in targets:
            x, y = point  # 忽略 distance，只考虑 x 和 y

            # 确定点落在哪个网格单元
            # 由于 x, y 在 0-1 之间，乘以 grid_size 并取整即可找到对应的索引
            x_index = int(x * grid_size)
            y_index = int(y * grid_size)

            # 防止坐标正好在边界上（例如 1.0）导致的索引越界
            x_index = min(x_index, grid_size - 1)
            y_index = min(y_index, grid_size - 1)

            # 增加对应网格单元的值
            grid[x_index, y_index] += 1

        return grid

points = np.array([[ 7.25043699, 118.46913337],
        [17.67832985,  85.50353657],
        [145.0105,  46.3077],
        [150.3601,  50.7133],
        [159.1713,  50.0839]])
gt_discrete = gen_discrete_map(256,256, points)
down_h = 8
d_ratio = 32
gt_discrete = torch.from_numpy(gt_discrete.reshape([down_h, d_ratio, down_h, d_ratio]).sum(axis=(1, 3)))
print(gt_discrete)

gt_list=[gt_discrete,gt_discrete]
gt_stack = torch.stack(gt_list)
print(gt_stack.shape)



