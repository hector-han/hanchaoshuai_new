import numpy as np
import logging
from mesh_grid import MeshGrid
from calc_d import DMatrix
from calc_b import BMatrix
from opt_target import OptTarget
import sys
sys.path.append('../src')
from utils import build_data


class Config(object):
    def __init__(self):
        self.location = np.array([
            [-400, 0],
            [0, 400],
        ])
        self.Q = [3, 5]
        self.u = 5
        self.theta = np.pi / 4
        self.vd = 0
        self.I = 0
        self.l = 0

        # 空间网格
        lower = [-1000, -1000]
        upper = [1000, 1000]
        nums = [51, 51]
        self.mesh_grid = MeshGrid(lower, upper, nums)
        # t的划分网格， 计算D时用到
        self.grid_t = list(range(1, 301))
        self.mesh_grid = MeshGrid(lower, upper, nums)
        # t的划分网格， 计算D时用到
        self.grid_t = list(range(1, 301))


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    station_fn = '../src/数据.xlsx'
    guance_fn = '../src/观测数据.xlsx'
    obsrv_loc, yt = build_data(station_fn, guance_fn)
    tmp = np.asarray([yt[60], yt[180], yt[300]])
    logging.info(tmp.T)
    config = Config()
    num_obsrv = obsrv_loc.shape[0]
    matR = np.eye(num_obsrv)
    cb = np.zeros(num_obsrv)
    # 计算时候的t的分割
    t_idx = [60, 180, 300]
    target_obj = OptTarget(config.mesh_grid, matR, obsrv_loc, yt, 300, t_idx, cb)
    target_obj.reset(config)
    val, grad = target_obj.get_obj_and_grad()
    logging.info(val)
    logging.info(grad)




if __name__ == '__main__':
    main()
