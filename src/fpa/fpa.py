import numpy as np
import logging
import time
import os
from .utils import good_point_init, random_point_init, deb_feasible_compare
from math import gamma, sin, pi, ceil


"""
 Flower pollenation algorithm (FPA), or flower algorithm
 花朵授粉算法

"""


def _empyt_constrain(x):
    """
    没有不等式约束
    :return:
    """
    return []


class FlowerPollinationAlgorithm(object):
    def __init__(self, n_dim, obj_fun_and_observation, lb=None, ub=None, less_func=_empyt_constrain, num_popu=50, N_iter=1000,
                 p=0.8, int_method='good', integer_op=False, coef=0.01, save_path='fpa'):
        """
        花朵授粉算法, 极小化，且只能解决不等式约束
        :param obj_fun_and_observation: 目标函数和需要观测的量，有时候观测的时候，需要观测其他的值
            第1个元素为目标函数obj(x)，其余为观测量
        :param less_func: 小于等于约束f_i(x) <= 0
        :param lb: 自变量下界约束
        :param ub: 自变量上界约束
        :param num_popu: 初始种群个数
        :param N_iter: 迭代次数
        :param p: 自花授粉概率， 1-p为全局授粉概率
        :param int_method: 初始化方式，good: 佳点集方式，否则随机
        :param integer_op: 是否是整数规划，默认False
        :param coef: levy 飞行的更新系数
        :param save_path: 保存计算过程的路径
        """

        self.n_dim = n_dim
        self.obj_fun_and_observation = obj_fun_and_observation
        self.lb = lb
        self.ub = ub
        self.less_func = less_func
        self.num_popu = num_popu
        self.N_iter = N_iter
        self.p = p
        self.integer_op = integer_op
        self.coef = coef
        self.save_path = save_path

        if self.lb is not None:
            self.init_lb = self.lb
        else:
            self.init_lb = -10 * np.ones(n_dim)
        if self.ub is not None:
            self.init_ub = self.ub
        else:
            self.init_ub = 10 * np.ones(n_dim)
        # 上一代
        self.populations0 = np.zeros([self.num_popu, self.n_dim])
        # 这一代
        self.populations1 = np.zeros([self.num_popu, self.n_dim])
        if int_method == 'good':
            self.populations0 = good_point_init(num_popu, self.init_lb, self.init_ub)
        else:
            self.populations0 = random_point_init(num_popu, self.init_lb, self.init_ub)

        if self.integer_op:
            self.populations0 = self.populations0.astype(np.int64)
            self.populations1 = self.populations1.astype(np.int64)
        # 花粉最大移动半径
        self.R = np.linalg.norm(self.init_ub - self.init_lb)
        self.f_and_cons_list = []
        self.x_best_list = []
        self.diversity_list = []
        self.different_list = []
        # 每10代记录下当前种群
        self.history = []

    def _diversity(self):
        center = np.mean(self.populations0, axis=0)
        tmp = self.populations0 - center
        part_norm = np.linalg.norm(tmp, axis=1)
        return np.sqrt(np.sum(part_norm ** 2)) / self.num_popu / np.max(part_norm)

    def _different(self):
        tmp = self.populations1 - self.populations0
        part_norm = np.linalg.norm(tmp, axis=1)
        return np.sqrt(np.sum(part_norm ** 2)) / self.num_popu / self.R

    def levy(self):
        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
                 ) ** (1 / beta)
        u = np.random.randn(self.n_dim) * sigma
        v = np.random.randn(self.n_dim)
        step = u / np.power(np.abs(v), 1 / beta)
        return self.coef * step

    def _bound(self, x):
        x_tmp = x
        if self.lb is not None:
            _mask = x_tmp < self.lb
            x_tmp[_mask] = self.lb[_mask]

        if self.ub is not None:
            _mask = x_tmp > self.ub
            x_tmp[_mask] = self.ub[_mask]

        if self.integer_op:
            x_tmp = x_tmp.astype(np.int64)
        return x_tmp

    def build_output(self, t, obj_val, div, diff, x, obsrv_val):
        """
        t: 迭代次数
        x: t时的最优解
        obj_val: 最优函数值
        obsrv_val: 其他的观测值
        组装打印到文件中的输出
        :param t:
        :return:
        """
        o_time = time.strftime("%H:%M:%S", time.localtime())
        o_x = [str(item) for item in x]
        o_x = ','.join(o_x)

        o_obsrv = [str(item) for item in obsrv_val]
        o_obsrv = ','.join(o_obsrv)

        return f'{o_time}\t{t}\t{div}\t{diff}\t{obj_val}\t{o_x}\t{o_obsrv}'

    def train(self):
        logging.info('fpa begin to train...')
        fin = open(self.save_path, 'w', encoding='utf-8')

        best_opt_idx, best_opt_vals, _flag, obsrv_vals = deb_feasible_compare(self.populations0, self.obj_fun_and_observation, self.less_func)
        x_best = self.populations0[best_opt_idx]
        obsrv_best = obsrv_vals
        f_min = best_opt_vals[0]

        self.f_and_cons_list.append(best_opt_vals)
        self.x_best_list.append(x_best)

        div = self._diversity()
        diff = 0.0
        output_str = self.build_output(0, f_min, div, diff, x_best, obsrv_best)
        fin.write(output_str + '\n')
        fin.flush()

        # 开始按照t迭代
        print_steps = self.N_iter // 10
        for t in range(1, self.N_iter + 1):
            if t % print_steps == 1:
                logging.info('t={}, f_min={}'.format(t, f_min))

            # 对每一个解迭代
            for i in range(self.num_popu):
                if np.random.random() > self.p:
                    # levy 飞行， 生物授粉 x_i^{t+1}=x_i^t+ L (x_i^t-gbest)
                    L = self.levy()
                    dS = L * (self.populations0[i] - x_best)
                    x_new = self.populations0[i] + dS
                    x_new = self._bound(x_new)
                else:
                    # 非生物授粉 x_i^{t+1}+epsilon*(x_j^t-x_k^t)
                    epsilon = np.random.random()
                    JK = np.random.permutation(self.num_popu)
                    x_new = self.populations0[i] + epsilon * (self.populations0[JK[0]] - self.populations0[JK[1]])
                    x_new = self._bound(x_new)
                tmp = [self.populations0[i], x_new]
                opt_idx, opt_vals, _flag, obsrv_vals = deb_feasible_compare(tmp, self.obj_fun_and_observation, self.less_func)
                if opt_idx == 1:
                    # 新解更优
                    self.populations1[i] = x_new
                    tmp = [x_best, x_new]
                    best_opt_idx, best_opt_vals, _, obsrv_vals = deb_feasible_compare(tmp, self.obj_fun_and_observation, self.less_func)
                    if _:
                        _flag = True
                    if best_opt_idx == 1:
                        x_best = x_new
                        f_min = best_opt_vals[0]
                        obsrv_best = obsrv_vals
                else:
                    # 保持不变
                    self.populations1[i] = self.populations0[i]

            # t时刻所有花粉都已经迭代完成，记录数据
            self.f_and_cons_list.append(best_opt_vals)
            self.x_best_list.append(x_best)

            diff = self._different()
            div = self._diversity()
            output_str = self.build_output(t, f_min, div, diff, x_best, obsrv_best)
            fin.write(output_str + '\n')
            fin.flush()

            self.populations0 = self.populations1
        fin.close()
        logging.info('最终是否找到可行解{}, 最优函数值{}'.format(_flag, f_min))

