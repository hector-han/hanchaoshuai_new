"""
计算C的解析解和数值解
"""
import numpy as np
from scipy import integrate


class AnalysisRes:

    def __init__(self, Q, location, D, v, vd, I, l):
        # n维向量，n个扩散源的强度
        self.Q = Q
        # n * 3 维向量， n个源的坐标
        self.location = location
        # 3维度向量，x,y,z方向的大气扩散系数
        self.D = D
        # v 对流速度
        self.v = v
        # vd, 一个数，沉积速率
        self.vd = vd
        self.I = I
        self.l = l
        self.tmp = []

    def _f(self, i, xyz, t, coeff = 1):
        """
        给定xyz, t, 以及对t的扰动tao, 求里面函数的值。 tao 属于 [0, t]
        :param i: 源的下标
        :param xyz:
        :param t:
        :param tao:
        :return:
        """
        loc_i = self.location[i]
        vt = self.v * t

        numer = xyz - loc_i - vt
        demoni = self.D * t
        sum1 = - np.sum(numer * numer / demoni) / 4 - (self.vd + self.I * self.l) * t
        c1 = coeff * np.exp(sum1) / np.power(t, 3 / 2)
        return c1

    # def my_quad(self):

    def at(self, xyz, t):
        """
        :param xyz: 空间中的三维向量
        :param t:
        :return:
        """
        ret = 0
        n = len(self.Q)
        for i in range(n):
            c1 = self._f(i, xyz, t)

            def f(tao):
                return self._f(i, xyz, tao, t - tao)

            c2, err = integrate.quad(f, 0, t)
            self.tmp.append(c2)
            ret += np.sqrt(np.pi) * self.Q[i] * (c1 + c2)

        return ret / (2 * np.sqrt(np.prod(self.D)))


class NumberRes:

    def __init__(self, Q, location, D, v, vd, I, l):
        # n维向量，n个扩散源的强度
        self.Q = Q
        self.n = len(self.Q)
        # n * 3 维向量， n个源的坐标
        self.location = location
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

    def _dirac_f_mat(self):
        """
        方便计算f, 获取f在网格上的取值
        :return:
        """
        self.f_mat = np.zeros((self.xlen, self.ylen, self.zlen))
        for i in range(self.n):
            xyz = self.location[i]
            x_min = self.get_max_min(self.grid_x, xyz[0])
            y_min = self.get_max_min(self.grid_y, xyz[1])
            if self.dz == 0:
                self.f_mat[x_min, y_min, 0] += self.Q[i] / (self.dx * self.dy)
            else:
                z_min = self.get_max_min(self.grid_z, xyz[3])
                self.f_mat[x_min, y_min, z_min] += self.Q[i] / (self.dx * self.dy * self.dz)

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

        print(self.pxyz, self.qxyz, self.dxyz)
        self._dirac_f_mat()

    def process(self):
        # 按照时刻进行统计的函数值， 都是全量的在xyz上嗯值
        self.view = []
        self.view.append(self.f_mat)

        for it in range(1, self.tlen):
            t = self.grid_t[it]
            last_view = self.view[it - 1]
            # print(it, last_view.shape)
            new_view = self.step(last_view, t)
            self.view.append(new_view)

    def step(self, last_val, t):
        """
        在t进行一步迭代
        :param last_val:
        :param dt:
        :return:
        """
        new_val = np.zeros_like(last_val)
        if self.dz != 0:
            # 3维的
            for i in range(self.xlen):
                for j in range(self.ylen):
                    for k in range(self.zlen):
                        # 每个方向上的小的和大的
                        c_less = np.zeros(3)
                        c_more = np.zeros(3)
                        c_cur = last_val[i, j, k]
                        if i > 0:
                            c_less[0] = last_val[i - 1, j, k]
                        if j > 0:
                            c_less[1] = last_val[i, j - 1, k]
                        if k > 0:
                            c_less[2] = last_val[i, j, k - 1]

                        if i < self.xlen - 1:
                            c_more[0] = last_val[i + 1, j, k]
                        if j < self.ylen - 1:
                            c_more[1] = last_val[i, j + 1, k]
                        if k < self.zlen - 1:
                            c_more[2] = last_val[i, j, k + 1]

                        ele_1 = np.sum((self.pxyz + 2 * self.qxyz) * c_less)
                        coef_2 = - 2 * (np.sum(self.pxyz + self.qxyz)) - (self.vd + self.I * self.l) * self.dt + 1
                        ele_2 = coef_2 * c_cur
                        ele_3 = np.sum(self.pxyz * c_more)
                        ele_4 = self.dt * self.f_mat[i, j, k]
                        new_val[i, j, k] = ele_1 + ele_2 + ele_3 + ele_4
        else:
            # 2维的
            k = 0
            # print(last_val.shape)
            for i in range(self.xlen):
                for j in range(self.ylen):
                    # 每个方向上的小的和大的
                    c_less = np.zeros(3)
                    c_more = np.zeros(3)
                    c_cur = last_val[i, j, k]
                    if i > 0:
                        c_less[0] = last_val[i - 1, j, k]
                    if j > 0:
                        c_less[1] = last_val[i, j - 1, k]

                    if i < self.xlen - 1:
                        c_more[0] = last_val[i + 1, j, k]
                    if j < self.ylen - 1:
                        c_more[1] = last_val[i, j + 1, k]

                    ele_1 = np.sum((self.pxyz + 2 * self.qxyz) * c_less)
                    coef_2 = - 2 * (np.sum(self.pxyz + self.qxyz)) - (self.vd + self.I * self.l) * self.dt + 1
                    ele_2 = coef_2 * c_cur
                    ele_3 = np.sum(self.pxyz * c_more)
                    ele_4 = self.dt * self.f_mat[i, j, k]
                    new_val[i, j, k] = ele_1 + ele_2 + ele_3 + ele_4
        return new_val







