
import numpy as np
from scipy import integrate



def _f(tao, t, v, x, xi, d, coef):
    """
    给定xyz, t, 以及对t的扰动tao, 求里面函数的值。 tao 属于 [0, t]
    :param i: 源的下标
    :param xyz:
    :param t:
    :param tao:
    :return:
    """

    c1 = tao / np.sqrt(t - tao)
    c2 = -(x - v * (t - tao) - xi) ** 2 / (4 * d * (t - tao)) - coef * (t - tao)
    c2 = np.exp(c2)

    return c1 * c2


def _f2(tao, t, v, x, xi, d, coef):
    """
    给定xyz, t, 以及对t的扰动tao, 求里面函数的值。 tao 属于 [0, t]
    :param i: 源的下标
    :param xyz:
    :param t:
    :param tao:
    :return:
    """

    c1 = (t- tao) / np.sqrt(tao)
    c2 = -(x - v * (tao) - xi) ** 2 / (4 * d * (tao)) - coef * (tao)
    c2 = np.exp(c2)

    return c1 * c2


def f1(tao, t):
    x = 1
    xi = 1
    d = 0.5
    v = 0.5
    coef = 0.036
    return _f(tao, t, v, x, xi, d, coef)


def f2(tao, t):
    x = 1
    xi = 1
    d = 0.5
    v = 0.5
    coef = 0.036
    return _f2(tao, t, v, x, xi, d, coef)

t = 1
c2, err = integrate.quad(lambda , 0, t)
