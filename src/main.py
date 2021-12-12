import numpy as np
from mesh_grid import MeshGrid
from fpa.fpa import FlowerPollinationAlgorithm
from opt_target import OptTarget


def target_func(x):
    """
    求目标函数值
    x: n * 4 长度的向量，每4个一组，共n组
        每组前3个为源的坐标，第4个为源的强度
    :return:
    """
    D = np.array([0.3, 0.35, 0.10])
    v = np.array([0.25, 0.13, 0.20])
    vd = 0.002
    I = 3.6
    l = 0.001
    R = [
        [1, 1],
        [1, 1]
    ]
    # 划分网格
    lower = [0, 0, 3]
    upper = [10, 10, 3]
    nums = [11, 11, 1]
    mesh_grid = MeshGrid(lower, upper, nums)
    grid_t = [0.1 * ele for ele in range(10)]

    nums = 2
    x = np.reshape(x, (nums, 4))
    location = x[:, 0:3]
    Q = x[:, 3]
    opt_target = OptTarget(mesh_grid, R, None, None, None)
    opt_target.reset(D, v, vd, I, l, Q, location, grid_t)
    grad = opt_target.get_gradient()
    return np.sum(grad)


def process():
    nums = 2
    ndim = nums * 4
    # x,y,z的最低点，源强最小值
    lower = [0, 0, 3, 10] * nums
    # x, y, z的最大值，源强最大值
    upper = [10, 10, 3, 20] * nums
    fpa_obj = FlowerPollinationAlgorithm(ndim, target_func, lower, upper, int_method='random')
    fpa_obj.train()


if __name__ == '__main__':
    process()

