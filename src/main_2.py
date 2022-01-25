"""
误差修正版本
"""

import numpy as np
import logging
from mesh_grid import MeshGrid
from fpa.fpa import FlowerPollinationAlgorithm
from calc_c import AnalysisRes
from opt_target import OptTarget
import sys
sys.path.append('../src')
from utils import build_data
from fix_error import get_miu_sigam


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


def compute_c(config: Config, obsrv_loc, yt2):
    """
    计算c的函数，对每一个点，每一个时刻，都计算。
    :param config:
    :param obsrv_loc:
    :param yt:
    :return: dict
    """
    obsrv_nums = len(obsrv_loc)

    # 计算其他的时刻, 得到修正值。修正这里
    know_Q = np.asarray([10, 10])
    know_location = np.asarray([
        [0, 0], [10, 10]
    ])
    know_grid_t = tj_idx
    ########## 以上是需要修改的部分

    ans = {}
    c_solver2 = AnalysisRes(know_Q, know_location, config.u, config.theta, config.vd, config.I, config.l)
    for t in know_grid_t:
        tmp = [0] * obsrv_nums
        for i in range(obsrv_nums):
            loc = obsrv_loc[i]
            tmp[i] = c_solver2.at(loc, t)
        ans[t] = np.asarray(tmp)

    mat_m_s = get_miu_sigam(obsrv_loc, yt2, ans)
    t_len = len(tj_idx)
    errors = []
    for miu, sigma in mat_m_s:
        res = np.random.lognormal(miu, sigma, [t_len])
        errors.append(res)

    # 这里已经从对数正太分布中抽取到误差，开始对C进行修正
    ans = {}
    # 计算 t = 0
    tmp = [0] * obsrv_nums
    for i in range(obsrv_nums):
        for j, point in config.location:
            if distance(point, obsrv_loc[i]) < 5:
                tmp[i] = config.Q[j]
                break
    ans[0] = np.asarray(tmp)

    c_solver = AnalysisRes(config.Q, config.location, config.u, config.theta, config.vd, config.I, config.l)
    for t in tj_idx:
        tmp = [0] * obsrv_nums
        for i in range(obsrv_nums):
            loc = obsrv_loc[i]
            tmp[i] = c_solver.at(loc, t)
        ans[t] = np.asarray(tmp)

    # 修正下ans:
    for ti, t in enumerate(tj_idx):
        for j in range(obsrv_nums):
            eps = errors[j][ti]
            logging.info(f'误差={eps}')
            ans[t][j] = (1 + eps) * ans[t][j]

    return ans


def assum_num_source(obsrv_loc, yt, yt2, nums=2):
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
        c_at_t = compute_c(config, obsrv_loc, yt2)
        target_obj.reset(config, c_at_t)
        val, grad = target_obj.get_obj_and_grad()
        return [np.linalg.norm(grad), val]

    save_file = '0115实验.tsv'
    fpa_obj = FlowerPollinationAlgorithm(ndim, target_func, lower, upper, num_popu=10, N_iter=100,
                                         int_method='random', save_path=save_file)
    fpa_obj.train()


def process():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    station_fn = '../data/2021-1/观测点坐标-1.xlsx'
    guance_fn = '../data/2021-1/观测点数据-1.xlsx'
    guance_fn2 = "../data/2021-1/观测点数据-1.xlsx"
    obsrv_loc, yt = build_data(station_fn, guance_fn)
    _, yt2 = build_data(station_fn, guance_fn2)
    assum_num_source(obsrv_loc, yt, yt2, nums=2)


if __name__ == '__main__':
    process()

