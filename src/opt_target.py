import logging

import numpy as np
from mesh_grid import MeshGrid
from calc_d import DMatrix
from calc_b import BMatrix


def quad_multiply(mat, vector):
    """
    返回 vector.T * mat * vector
    :param mat:
    :param vector:
    :return:
    """
    item1= np.dot(mat, vector)
    return np.dot(vector, item1)


class OptTarget:
    def __init__(self, mesh_grid: MeshGrid, matR, obsrv_loc, yt, tb_idx, tj_idx, cb):
        """

        :param matD:
        :param matB:
        :param matR:
        :param cSolver:
        :param yt: t时刻的观测值
        :param T:
        :param cb:
        """
        self.mesh_grid = mesh_grid
        self.cSolver = None
        self.matD1 = None
        self.matB = None
        self.matR = np.array(matR)
        n = self.matR.shape[0]
        self.matH = np.eye(n)
        self.obsrv_loc = obsrv_loc
        self.yt = yt
        self.tb_idx = tb_idx
        self.tj_idx = tj_idx
        self.cb = cb
        # logging.info(f'yt: {self.yt}')
        logging.info(f'tb_idx: {self.tb_idx}')
        logging.info(f'tj_idx: {self.tj_idx}')

    def reset(self, config, c_at_t):
        """
        重置源的位置，源强，等的值
        :param config:
        :param c_at_t:
        :return:
        """
        self.c_at_t = c_at_t

        matD = DMatrix(config.location, config.u, config.theta, config.vd, config.I, config.l)
        matD.init(self.mesh_grid, config.grid_t)
        matD.build_Dt()
        matD.set_obsrv_location(self.obsrv_loc)
        self.matD = matD
        self.matD1 = matD.get_D1()
        logging.debug(f'matD1={self.matD1}')

        delta_t = config.grid_t[1] - config.grid_t[0]
        matB = BMatrix(matD, self.yt, self.tb_idx, delta_t)
        self.matB = matB.build()
        logging.debug(f'matB={self.matB}')


    def get_obj_and_grad(self):
        """
        计算梯度
        :return:
        """
        c_at_0 = self.c_at_t[0]
        ### 计算 grad_b ###
        inv_b = np.linalg.inv(self.matB)
        c0_sub_cb = c_at_0 - self.cb
        grad_b = np.matmul(inv_b, c0_sub_cb)

        ### 计算grad_r ###
        inv_r = np.linalg.inv(self.matR)
        grad_r = np.zeros_like(self.cb)
        t_to_ct = {}

        for t in self.tj_idx:
            c_at_t = self.c_at_t[t]
            logging.debug(f't={t}, c={c_at_t}')
            t_to_ct[t] = c_at_t
            # 这里的T是转置还是最长时间？？？
            list_t = list(range(1, t + 1))
            list_t = self.matD.filter_t(list_t)
            D = self.matD.get_partial_D(list_t)

            item_1 = np.matmul(D.T, self.matH)
            item_1 = np.matmul(item_1, inv_r)
            item_2 = np.dot(self.matH.T, c_at_t)
            item_2 = item_2 - self.yt[t]
            grad_r = grad_r + np.dot(item_1, item_2)

        #### 计算目标函数
        item1 = quad_multiply(inv_b, c0_sub_cb) / 2
        item2 = 0
        for t in self.tj_idx:
            hct_sub_yt = np.dot(self.matH, t_to_ct[t]) - self.yt[t]
            tmp = quad_multiply(inv_r, hct_sub_yt)
            item2 += tmp
        item2 = item2 / 2

        # 汇总下计算结果
        obj_val = item1 + item2
        grad = grad_b + grad_r
        return obj_val, grad


if __name__ == '__main__':
    a = np.array([[5, 1, 3],
                  [1, 1, 1],
                  [1, 2, 1]])
    b = np.array([1, 2, 3])
    print(quad_multiply(a, b))