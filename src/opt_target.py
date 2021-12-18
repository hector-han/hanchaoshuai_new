import numpy as np
from mesh_grid import MeshGrid
from calc_d import DMatrix
from calc_b import BMatrix
from calc_c import AnalysisRes


class OptTarget:
    def __init__(self, mesh_grid: MeshGrid, matR, yt, T, cb):
        """

        :param matD:
        :param matB:
        :param matR:
        :param cSolver:
        :param yt:
        :param T:
        :param cb:
        """
        self.mesh_grid = mesh_grid
        self.cSolver = None
        self.matD1 = None
        self.matB = None
        self.matR = np.array(matR)

        self.yt = yt
        self.T = T
        self.cb = cb

    def reset(self, D, v, vd, I, l, Q, location, grid_t):
        cSolver = AnalysisRes(Q, location, D, v, vd, I, l)
        self.cSolver = cSolver

        matD = DMatrix(D, v, vd, I, l)
        matD.init(self.mesh_grid, grid_t)
        matD.build_Dt()
        matD.set_location(location)
        self.matD1 = matD.get_D1()

        y_10 = np.array([1, 1])
        y_20 = np.array([2, 3])
        matB = BMatrix(D, v, vd, I, l, matD, y_10, y_20)
        self.matB = matB.build()

    def _calc_c(self, t):
        """
        计算t时刻的c的值
        :param t:
        :return:
        """
        ans = []
        for point in self.matD.location:
            ans.append(self.cSolver.at(point, t))
        return np.array(ans)

    def get_gradient(self):
        """
        计算梯度
        :return:
        """
        c_at_0 = self._calc_c(0)
        ### 计算 grad_b ###
        inv_b = np.linalg.inv(self.matB)
        sum_item = np.sum(c_at_0) - self.cb
        grad_b = np.matmul(inv_b, sum_item)

        ### 计算grad_r ###
        inv_r = np.linalg.inv(self.matR)
        grad_r = np.zeros_like(self.cb)
        for t in range(self.T):
            c_at_t = self._calc_c(t)
            item_1 = np.matmul(self.D1.T, inv_r)
            itme_2 = 0
            grad_r = grad_r + 0

        return None

