from mesh_grid import MeshGrid
from calc_d import DMatrix
import numpy as np

class BMatrix:
    def __init__(self, D, v, vd, I, l, matD: DMatrix, y_10, y_20):
        # 3维度向量，x,y,z方向的大气扩散系数
        self.D = D
        # v 对流速度
        self.v = v
        # vd, 一个数，沉积速率
        self.vd = vd
        self.I = I
        self.l = l
        self.matD = matD
        self.y_10 = y_10
        self.y_20 = y_20
        self._mesh_grid = self.matD.get_mesh_grid()

    def build(self):
        M_10_30 = self.matD.get_DtPower_slice(20)
        M_20_30 = self.matD.get_DtPower_slice(10)

        c_10_30 = np.dot(M_10_30, self.y_20)
        c_20_30 = np.dot(M_10_30, self.y_10)

        colum_vec = np.expand_dims(c_20_30 - c_10_30, axis=1)
        row_vec = np.expand_dims(c_20_30 - c_10_30, axis=0)

        loc_dist = self.matD.location_dist
        bMat = loc_dist * np.exp(-(loc_dist * loc_dist / 2))
        cMat = np.dot(colum_vec, row_vec)
        ans = bMat * cMat / 2
        return ans




