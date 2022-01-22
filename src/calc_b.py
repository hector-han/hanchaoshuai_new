import logging

from mesh_grid import MeshGrid
from calc_d import DMatrix
import numpy as np


class BMatrix:
    def __init__(self, matD: DMatrix, yt, t_idx, delta_t):
        """
        :param matD:
        :param yt: t时刻的观测值，dict
        :param t_idx: 计算B的三个时刻[10,20,30]表示用10->30, 20->30计算B。
        :param delta_t: t的变化维度
        """
        self.matD = matD
        self.yt = yt
        self.t_idx = t_idx
        self.delta_t = delta_t
        self._mesh_grid = self.matD.get_mesh_grid()

    def build(self):
        t1 = self.t_idx[0]
        y1 = self.yt[t1]
        t2 = self.t_idx[1]
        y2 = self.yt[t2]
        t3 = self.t_idx[2]

        list1 = list(range(t1, t3 + 1, self.delta_t))
        list2 = list(range(t2, t3 + 1, self.delta_t))
        M1 = self.matD.get_partial_D(list1)
        M2 = self.matD.get_partial_D(list2)
        logging.info(f'get B M1, min={np.min(M1)}, max={np.max(M1)}')
        logging.info(f'get B M2, min={np.min(M2)}, max={np.max(M2)}')

        c_10_30 = np.dot(M1, y1)
        c_20_30 = np.dot(M2, y2)
        # logging.info(c_10_30)
        # logging.info(c_20_30)

        colum_vec = np.expand_dims(c_20_30 - c_10_30, axis=1)
        row_vec = np.expand_dims(c_20_30 - c_10_30, axis=0)

        loc_dist = self.matD.obsrv_location_dist
        bMat = np.exp(-(loc_dist * loc_dist / 2))
        cMat = np.dot(colum_vec, row_vec)
        ans = bMat * cMat / 2
        return ans




