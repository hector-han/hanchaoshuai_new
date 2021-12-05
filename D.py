
import numpy as np
from scipy import integrate


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

    def get_max_min(self, array, val):
        n = len(array)
        for i in range(n):
            if array[i] >= val:
                return i - 1
        return n - 2

    def init(self, grid_x, grid_y, grid_z, grid_t):
        """
        初始化网格
        :param grid_x:
        :param grid_y:
        :param grid_z:
        :return:
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.xlen = len(self.grid_x)
        self.ylen = len(self.grid_y)
        self.zlen = len(self.grid_z)
        # 网格的个数
        self.M = self.xlen * self.ylen * self.zlen

        self.dx = self.grid_x[1] - self.grid_x[0]
        self.dy = self.grid_y[1] - self.grid_y[0]
        if self.zlen == 1:
            self.dz = 0
        else:
            self.dz = self.grid_z[1] - self.grid_z[0]

        self.grid_t = grid_t
        self.dt = self.grid_t[1] - self.grid_t[0]
        self.tlen = len(self.grid_t)

        # 初始化数值计算时候，需要用的到一些值
        self.pxyz = np.zeros(3)
        self.qxyz = np.zeros(3)
        self.dxyz = np.asarray([self.dx, self.dy, self.dz])

        # 定义要用到的px, py, pz, qx,qy,qz
        if self.dz == 0:
            self.pxyz[0] = self.D[0] * self.dt / (self.dx * self.dx)
            self.pxyz[1] = self.D[1] * self.dt / (self.dy * self.dy)
            self.qxyz[0] = self.v[0] * self.dt / (2 * self.dx)
            self.qxyz[1] = self.v[1] * self.dt / (2 * self.dy)
        else:
            self.pxyz = self.D * self.dt / (self.dxyz * self.dxyz)
            self.qxyz = self.v * self.dt / (2 * self.dxyz)

        # print(self.pxyz, self.qxyz, self.dxyz)

        self.a = - 2 * np.sum(self.pxyz + self.qxyz) - (self.vd + self.I * self.l) * self.dt + 1
        self.b = self.pxyz + 2 * self.qxyz

    def to_flaten_idx(self, xi, yj, zk):
        # 三维空间坐标转换为顺序排列的网格id
        return xi * (self.ylen * self.zlen) + yj * (self.zlen) + zk

    def to_3d_idx(self, idx):
        """
        网格id到三维id变换关系
        :param idx:
        :return:
        """
        xi = idx // (self.ylen * self.zlen)
        tmp = idx % (self.ylen * self.zlen)
        yj = tmp // self.zlen
        zk = tmp % self.zlen
        return (xi, yj, zk)

    def build_D(self):
        self.DT = self.a * np.eye(self.M)
        for i in range(self.xlen):
            for j in range(self.ylen):
                for k in range(self.zlen):
                    # 每个方向上的小的和大的
                    row_idx = self.to_flaten_idx(i, j, k)
                    if i > 0:
                        col_idx = self.to_flaten_idx(i - 1, j, k)
                        self.DT[row_idx, col_idx] = self.b[0]
                    if j > 0:
                        col_idx = self.to_flaten_idx(i, j - 1, k)
                        self.DT[row_idx, col_idx] = self.b[1]
                    if k > 0:
                        col_idx = self.to_flaten_idx(i, j, k - 1)
                        self.DT[row_idx, col_idx] = self.b[2]

                    if i < self.xlen - 1:
                        col_idx = self.to_flaten_idx(i + 1, j, k)
                        self.DT[row_idx, col_idx] = self.pxyz[0]
                    if j < self.ylen - 1:
                        col_idx = self.to_flaten_idx(i, j + 1, k)
                        self.DT[row_idx, col_idx] = self.pxyz[1]
                    if k < self.zlen - 1:
                        col_idx = self.to_flaten_idx(i, j, k + 1)
                        self.DT[row_idx, col_idx] = self.pxyz[2]

        self.D = np.ones_like(self.DT)
        for i in range(0, self.tlen - 1):
            self.D = np.matmul(self.D, self.DT)


    def get_DtPower_slice(self, ):
        """
        获取Dt的power次方，同时
        :return:
        """
        pass


    def get_D1(self, location):
        """
        :param location: n * 3, n个观测点的位置
        :return:
        """
        num_points = len(location)
        loc_idx = []
        loc_flaten_idx = []
        for i in range(num_points):
            xyz = location[i]
            x_min = self.get_max_min(self.grid_x, xyz[0])
            y_min = self.get_max_min(self.grid_y, xyz[1])
            z_min = 0
            if self.dz != 0:
                z_min = self.get_max_min(self.grid_z, xyz[3])
            loc_idx.append((x_min, y_min, z_min))
            loc_flaten_idx.append(self.to_flaten_idx(x_min, y_min, z_min))

        self.D1 = np.zeros((num_points, num_points))
        for i in range(num_points):
            f_i = loc_flaten_idx[i]
            for j in range(num_points):
                f_j = loc_flaten_idx[j]
                self.D1[i, j] = self.D[f_i, f_j]
        return self.D1


