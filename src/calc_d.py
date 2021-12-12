import numpy as np
from mesh_grid import MeshGrid


class DMatrix:
    def __init__(self, D, v, vd, I, l):
        # 3维度向量，x,y,z方向的大气扩散系数
        self.D = D
        # v 对流速度
        self.v = v
        # vd, 一个数，沉积速率
        self.vd = vd
        self.I = I
        self.l = l

    def get_mesh_grid(self):
        return self._mesh_grid

    def init(self, mesh_grid: MeshGrid, grid_t):
        """
        初始化网格
        :param grid_x:
        :param grid_y:
        :param grid_z:
        :return:
        """
        self._mesh_grid = mesh_grid
        self.grid_t = grid_t
        self.dt = self.grid_t[1] - self.grid_t[0]
        self.tlen = len(self.grid_t)

        # 初始化数值计算时候，需要用的到一些值
        self.pxyz = np.zeros(mesh_grid.dim)
        self.qxyz = np.zeros(mesh_grid.dim)

        # 定义要用到的px, py, pz, qx,qy,qz
        if mesh_grid.dim == 3 and int(mesh_grid.steps[2]) == 0:
            dx, dy, _ = mesh_grid.steps
            self.pxyz[0] = self.D[0] * self.dt / (dx * dx)
            self.pxyz[1] = self.D[1] * self.dt / (dy * dy)
            self.qxyz[0] = self.v[0] * self.dt / (2 * dx)
            self.qxyz[1] = self.v[1] * self.dt / (2 * dy)
        else:
            self.pxyz = self.D * self.dt / (mesh_grid.steps * mesh_grid.steps)
            self.qxyz = self.v * self.dt / (2 * mesh_grid.steps)

        # print(self.pxyz, self.qxyz, self.dxyz)

        self.a = - 2 * np.sum(self.pxyz + self.qxyz) - (self.vd + self.I * self.l) * self.dt + 1
        self.b = self.pxyz + 2 * self.qxyz

    def build_Dt(self):
        self.DT = self.a * np.eye(self._mesh_grid.total_M)
        xlen, ylen = self._mesh_grid.nums[0:2]
        for i in range(xlen):
            for j in range(xlen):
                if self._mesh_grid.dim == 3:
                    zlen = self._mesh_grid.nums[2]
                    for k in range(zlen):
                        # 每个方向上的小的和大的
                        row_idx = self._mesh_grid.to_flaten_idx([i, j, k])
                        if i > 0:
                            col_idx = self._mesh_grid.to_flaten_idx([i - 1, j, k])
                            self.DT[row_idx, col_idx] = self.b[0]
                        if j > 0:
                            col_idx = self._mesh_grid.to_flaten_idx([i, j - 1, k])
                            self.DT[row_idx, col_idx] = self.b[1]
                        if k > 0:
                            col_idx = self._mesh_grid.to_flaten_idx([i, j, k - 1])
                            self.DT[row_idx, col_idx] = self.b[2]

                        if i < xlen - 1:
                            col_idx = self._mesh_grid.to_flaten_idx([i + 1, j, k])
                            self.DT[row_idx, col_idx] = self.pxyz[0]
                        if j < ylen - 1:
                            col_idx = self._mesh_grid.to_flaten_idx([i, j + 1, k])
                            self.DT[row_idx, col_idx] = self.pxyz[1]
                        if k < zlen - 1:
                            col_idx = self._mesh_grid.to_flaten_idx([i, j, k + 1])
                            self.DT[row_idx, col_idx] = self.pxyz[2]
                else:
                    row_idx = self._mesh_grid.to_flaten_idx([i, j])
                    if i > 0:
                        col_idx = self._mesh_grid.to_flaten_idx([i - 1, j])
                        self.DT[row_idx, col_idx] = self.b[0]
                    if j > 0:
                        col_idx = self._mesh_grid.to_flaten_idx([i, j - 1])
                        self.DT[row_idx, col_idx] = self.b[1]
                    if i < xlen - 1:
                        col_idx = self._mesh_grid.to_flaten_idx([i + 1, j])
                        self.DT[row_idx, col_idx] = self.pxyz[0]
                    if j < ylen - 1:
                        col_idx = self._mesh_grid.to_flaten_idx([i, j + 1])
                        self.DT[row_idx, col_idx] = self.pxyz[1]

    def set_location(self, location):
        """
        设置源的位置, 顺便计算下每个源相对距离
        :param location:
        :return:
        """
        self.location = location

        num_points = len(location)
        loc_flaten_idx = []
        for i in range(num_points):
            xyz = location[i]
            f_idx = self._mesh_grid.point_flaten_idx(xyz)
            loc_flaten_idx.append(f_idx)
        self.location_idx = tuple(loc_flaten_idx)

        self.location_dist = np.zeros([num_points, num_points])
        for i in range(num_points):
            point_i = location[i]
            for j in range(num_points):
                point_j = location[j]
                self.location_dist[i, j] = np.linalg.norm(point_j - point_i)

    def get_DtPower_slice(self, n):
        """
        获取Dt的power次方
        :return:
        """
        tuple_slice = self.location_idx
        Dn = np.ones_like(self.DT)
        for i in range(0, n):
            Dn = np.matmul(Dn, self.DT)

        num_points = len(tuple_slice)
        ans = np.zeros((num_points, num_points))
        for i in range(num_points):
            f_i = tuple_slice[i]
            for j in range(num_points):
                f_j = tuple_slice[j]
                ans[i, j] = Dn[f_i, f_j]
                ans[i, j] = Dn[f_i, f_j]
        return ans

    def get_D1(self):
        """
        :param location: n * 3, n个观测点的位置
        :return:
        """
        return self.get_DtPower_slice(self.tlen - 1)