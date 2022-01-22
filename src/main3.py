import numpy as np
import logging
from mesh_grid import MeshGrid
from fpa.fpa import FlowerPollinationAlgorithm
from calc_c import AnalysisRes
from opt_target import OptTarget
import sys
sys.path.append('../src')
from utils import build_data


"""
使用说明：
1、grid_t 是时间维度的网格划分，计算D用到
    tb_idx 用那三个计算B，如[60, 180, 300]表示用60s, 180s, 300s计算B。需要保证都在grid_t中出现
    tj_idx: 计算目标函数时，要在时间维度求和，需要求和的那些时刻。也要保证都在grid_t中出现
2、修改target_func 返回顺序可以实现对不同目标求最小值， [f_val, np.linalg.norm(grad)]。永远是对第0个求最小
"""


# xy 维度上的最小值
xy_lower = [-1000, -1000]
# xy维度上的最大值
xy_upper = [1000, 1000]
# xy维度网格的个数
xy_nums = [51, 51]
# 时间维度网格, 100， 200， 300， ..., 3600
grid_t = list(range(100, 3601, 100))
# 计算B的三个时间
tb_idx = [100, 200, 300]
# 计算目标函数的多个时间，
tj_idx = list(range(100, 3601, 200))
# 源强最小值
Q_min = 0
# 源强最大值
Q_max = 100


class Config(object):
    def __init__(self, location, Q):
        self.location = location
        self.Q = Q
        self.u = 5
        self.theta = np.pi / 4
        self.vd = 0
        self.I = 0
        self.l = 0

        # 空间网格
        self.mesh_grid = MeshGrid(xy_lower, xy_upper, xy_nums)
        # t的划分网格， 计算D时用到
        self.grid_t = grid_t


def distance(point1, point2):
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    return np.linalg.norm(point2 - point1)


def compute_c(config: Config, obsrv_loc, yt):
    """
    计算c的函数，对每一个点，每一个时刻，都计算。
    :param config:
    :param obsrv_loc:
    :param yt:
    :return: dict
    """
    ans = {}
    # 计算 t = 0
    obsrv_nums = len(obsrv_loc)
    tmp = [0] * obsrv_nums
    for i in range(obsrv_nums):
        for j, point in config.location:
            if distance(point, obsrv_loc[i]) < 5:
                tmp[i] = config.Q[j]
                break
    ans[0] = np.asarray(tmp)

    # 计算其他的时刻
    c_solver = AnalysisRes(config.Q, config.location, config.u, config.theta, config.vd, config.I, config.l)
    for t in tj_idx:
        tmp = [0] * obsrv_nums
        for i in range(obsrv_nums):
            loc = obsrv_loc[i]
            tmp[i] = c_solver.at(loc, t)
        ans[t] = np.asarray(tmp)
    return ans


def assum_num_source(obsrv_loc, yt, nums=2):
    """
    有num个源，花朵授粉求最小
    :param nums: 源的个数
    :return:
    """
    ndim = nums * 3
    # x,y,z的最低点，源强最小值
    lower = [xy_lower[0], xy_lower[1], Q_min] * nums
    # x, y, z的最大值，源强最大值
    upper = [xy_upper[1], xy_upper[1], Q_max] * nums
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def target_func(x):
        """
        求目标函数值
        x: n * 4 长度的向量，每4个一组，共n组
            每组前3个为源的坐标，第4个为源的强度
        :return:
        """
        x = np.reshape(x, (nums, 3))
        location = x[:, 0:2]
        Q = x[:, 2]
        config = Config(location, Q)

        num_obsrv = obsrv_loc.shape[0]
        matR = np.eye(num_obsrv)
        cb = np.zeros(num_obsrv)
        target_obj = OptTarget(config.mesh_grid, matR, obsrv_loc, yt, tb_idx, tj_idx, cb)
        c_at_t = compute_c(config, obsrv_loc, yt)
        target_obj.reset(config, c_at_t)
        f_val, grad = target_obj.get_obj_and_grad()
        return [f_val, np.linalg.norm(grad)]

    save_file = '0122_实验_大数据.tsv'
    fpa_obj = FlowerPollinationAlgorithm(ndim, target_func, lower, upper, num_popu=10, N_iter=100,
                                         int_method='random', save_path=save_file)
    fpa_obj.train()


def process():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    station_fn = '../data/2021-1/观测点坐标-1.xlsx'
    guance_fn = '../data/2021-1/观测点数据-1.xlsx'
    obsrv_loc, yt = build_data(station_fn, guance_fn)
    assum_num_source(obsrv_loc, yt, nums=2)


if __name__ == '__main__':
    process()

