import logging

import numpy as np
from mesh_grid import MeshGrid
from ksxi import calc_sigma


class DMatrix:
    def __init__(self, location, u, theta, vd, I, l):
        # n * 3 维向量， n个源的坐标
        self.location = location

        # 3维度向量，x,y,z方向的大气扩散系数
        self.u = u
        self.theta = theta
        self.calc_sigma = calc_sigma
        # v 对流速度
        self.v = self.u * np.asarray([np.cos(theta), np.sin(theta), 0])
        # vd, 一个数，沉积速率
        self.vd = vd
        self.I = I
        self.l = l

    def get_mesh_grid(self):
        return self._mesh_grid

    def _get_D_at_point_t(self):
        """
        计算后面需要用到的p, q, a
        对每个时刻t, 计算出来需要用到的中间变量
        :return:
        """
        xlen, ylen = self._mesh_grid.nums[0:2]
        ij_to_sigma = np.zeros([xlen, ylen, 2])
        for i in range(xlen):
            for j in range(ylen):
                point = self._mesh_grid.get_point_by_grid([i, j])
                # n个源的sigma加起来
                sigma = 0
                for loc_k in self.location:
                    sigma += self.calc_sigma(point, loc_k, self.u, self.theta)
                ij_to_sigma[i, j] = sigma

        self.ij_to_sigma = ij_to_sigma

    def init(self, mesh_grid: MeshGrid, grid_t):
        """
        初始化网格
        :param mesh_grid 平面网格
        :param grid_t 时间维度的划分
        :return:
        """
        self._mesh_grid = mesh_grid
        self.grid_t = grid_t
        self.dt = self.grid_t[1] - self.grid_t[0]
        self.tlen = len(self.grid_t)

        # 初始化数值计算时候，需要用的到一些值
        self.qxyz = np.zeros(mesh_grid.dim)
        self.step_square = mesh_grid.steps * mesh_grid.steps

        # 定义要用到的px, py, pz, qx,qy,qz
        if mesh_grid.dim == 3 and int(mesh_grid.steps[2]) == 0:
            dx, dy, _ = mesh_grid.steps
            # self.pxyz[0] = self.D[0] * self.dt / (dx * dx)
            # self.pxyz[1] = self.D[1] * self.dt / (dy * dy)
            self.qxyz[0] = self.v[0] * self.dt / (2 * dx)
            self.qxyz[1] = self.v[1] * self.dt / (2 * dy)
        else:
            # self.pxyz = self.D * self.dt / (mesh_grid.steps * mesh_grid.steps)
            self.qxyz = self.v[0: 2] * self.dt / (2 * mesh_grid.steps)
        self._get_D_at_point_t()

    def _calc_ab(self, i, j, t):
        """
        计算i,j,t时候的a和b
        :param i:
        :param j:
        :return: p
        """
        sig_xyz = self.ij_to_sigma[i, j, :][0:2]
        Dxyz = sig_xyz * sig_xyz / t
        pxyz = Dxyz * self.dt / self.step_square
        a = - 2 * np.sum(pxyz + self.qxyz) - (self.vd + self.I * self.l) * self.dt + 1
        bxyz = pxyz + 2 * self.qxyz
        return pxyz, a, bxyz

    def build_Dt(self):
        xlen, ylen = self._mesh_grid.nums[0:2]
        # 每一个时刻的D都是不一样的
        t_to_D = {}
        for t in self.grid_t:
            D_at_t = np.zeros([self._mesh_grid.total_M, self._mesh_grid.total_M])
            for i in range(xlen):
                for j in range(ylen):
                    row_idx = self._mesh_grid.to_flaten_idx([i, j])
                    pxyz, a, bxyz = self._calc_ab(i, j, t)
                    D_at_t[row_idx, row_idx] = a
                    if i > 0:
                        col_idx = self._mesh_grid.to_flaten_idx([i - 1, j])
                        D_at_t[row_idx, col_idx] = bxyz[0]
                    if j > 0:
                        pxyz, a, bxyz = self._calc_ab(i - 1, j, t)
                        col_idx = self._mesh_grid.to_flaten_idx([i, j - 1])
                        D_at_t[row_idx, col_idx] = bxyz[1]
                    if i < xlen - 1:
                        col_idx = self._mesh_grid.to_flaten_idx([i + 1, j])
                        D_at_t[row_idx, col_idx] = pxyz[0]
                    if j < ylen - 1:
                        col_idx = self._mesh_grid.to_flaten_idx([i, j + 1])
                        D_at_t[row_idx, col_idx] = pxyz[1]
            t_to_D[t] = D_at_t
        self.t_to_D = t_to_D

    def set_obsrv_location(self, obsrv_location):
        """
        设置源的位置, 顺便计算下每个源相对距离，后续会用到
        :param location:
        :return:
        """
        self.obsrv_location = obsrv_location
        num_points = len(obsrv_location)
        loc_flaten_idx = []
        for i in range(num_points):
            xyz = obsrv_location[i][0:2]
            f_idx = self._mesh_grid.point_flaten_idx(xyz)
            loc_flaten_idx.append(f_idx)
        self.obsrv_location_idx = tuple(loc_flaten_idx)

        self.obsrv_location_dist = np.zeros([num_points, num_points])
        for i in range(num_points):
            point_i = obsrv_location[i]
            for j in range(num_points):
                point_j = obsrv_location[j]
                self.obsrv_location_dist[i, j] = np.linalg.norm(point_j - point_i)

    def get_slice(self, D_mat):
        """
        获取矩阵的slice, 是按照观测点的位置
        :param D_mat:
        :return:
        """
        tuple_slice = self.obsrv_location_idx

        num_points = len(tuple_slice)
        ans = np.zeros((num_points, num_points))
        for i in range(num_points):
            f_i = tuple_slice[i]
            for j in range(num_points):
                f_j = tuple_slice[j]
                ans[i, j] = D_mat[f_i, f_j]
                ans[i, j] = D_mat[f_i, f_j]
        return ans

    def get_D1(self):
        """
        :param location: n * 3, n个观测点的位置
        :return:
        """
        D_mat = np.eye(self._mesh_grid.total_M)
        for t in self.grid_t:
            D_mat = np.matmul(D_mat, self.t_to_D[t])
        return self.get_slice(D_mat)

    def get_partial_D(self, list_t):
        """
        从list_t中获取对应时刻的Dt, 连乘起来，返回观测点index的子矩阵
        :param list_t:
        :return:
        """
        logging.info(list_t)
        D_mat = np.eye(self._mesh_grid.total_M)
        for t in list_t:
            D_mat = np.matmul(D_mat, self.t_to_D[t])
        return self.get_slice(D_mat)