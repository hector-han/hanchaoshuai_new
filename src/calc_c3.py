"""
计算C的解析解，连续释放和瞬时释放混合
"""
import numpy as np
from scipy import integrate
from ksxi import calc_sigmas

# 需要手动修改这里，表示前几个源是瞬时释放，之后的是连续释放
NN = 1

def calc_sigma(point, loc_i, u, theta):
    x = point[0]
    y = point[1]
    xi = loc_i[0]
    yi = loc_i[1]
    return calc_sigmas(u, theta, x, y, xi, yi)


class AnalysisRes:
    def __init__(self, Q, location, u, theta, vd, I, l):
        # n维向量，n个扩散源的强度
        self.Q = Q
        # n * 3 维向量， n个源的坐标
        self.location = location
        # 3维度向量，x,y,z方向的大气扩散系数
        self.u = u
        self.theta = theta
        self.calc_sigma = calc_sigma
        self.v = self.u * np.asarray([np.cos(theta), np.sin(theta)])
        # vd, 一个数，沉积速率。以下都是标量
        self.vd = vd
        self.I = I
        self.l = l
        # 记录一些中间值，方便定位
        self.tmp = []

    def _f(self, i, point, t, sigma_i, coeff=1):
        """
        给定xyz, t, 以及对t的扰动tao, 求里面函数的值。 tao 属于 [0, t]
        :param i: 源的下标
        :param point: 3维坐标
        :param t:
        :param tao:
        :return:
        """
        loc_i = self.location[i]
        vt = self.v * t

        numer = point - loc_i - vt
        demoni = sigma_i * sigma_i
        sum1 = - np.sum(numer * numer / demoni) / 2 - (self.vd + self.I * self.l) * t
        c1 = coeff * np.exp(sum1)
        return c1

    def at(self, point, t):
        """
        :param point: 空间中的三维向量
        :param t:
        :return:
        """
        ret = 0
        n = len(self.Q)
        for i in range(n):
            if i < NN:
                loc_i = self.location[i]
                sigma_i = self.calc_sigma(point, loc_i, self.u, self.theta)
                c1 = self._f(i, point, t, sigma_i)
                ret += self.Q[i] * (c1) / np.prod(sigma_i) / np.power((2 * np.pi), 3 / 2)
            else:
                loc_i = self.location[i]
                sigma_i = self.calc_sigma(point, loc_i, self.u, self.theta)
                c1 = self._f(i, point, t, sigma_i)

                def f(tao):
                    return self._f(i, point, tao, sigma_i)

                c2, err = integrate.quad(f, 0, t)
                # c2 = accum_sum(f, 1, t + 1)
                ret += self.Q[i] * (c1 + c2) / np.prod(sigma_i) / np.power((2 * np.pi), 3 / 2)

        return ret
