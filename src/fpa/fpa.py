import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from .utils import good_point_init, random_point_init, deb_feasible_compare
from math import gamma, sin, pi, ceil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                    datefmt='%Y-%m-%d %A %H:%M:%S')
"""
 Flower pollenation algorithm (FPA), or flower algorithm
 花朵授粉算法

"""


class FlowerPollinationAlgorithm(object):
    def __init__(self, n_dim, obj_fun_and_less_cons, lb=None, ub=None, num_popu=100, N_iter=1000,
                 p=0.8, int_method='good', integer_op=False, coef=0.01, name='fpa'):
        """
        花朵授粉算法, 极小化，且只能解决不等式约束
        :param obj_fun_and_less_cons: 目标函数和约束定义到一个列表里
            第1个元素为目标函数obj(x)，其余为小于等于约束f_i(x) <= 0
        :param lb: 自变量下界约束
        :param ub: 自变量上界约束
        :param num_popu: 初始种群个数
        :param N_iter: 迭代次数
        :param p: 自花授粉概率， 1-p为全局授粉概率
        :param int_method: 初始化方式，good: 佳点集方式，否则随机
        :param integer_op: 是否是整数规划，默认False
        :param coef: levy 飞行的更新系数
        :param name: 模型名字，保存图片会使用这个作为前缀
        """

        self.n_dim = n_dim
        self.obj_fun_and_less_cons = obj_fun_and_less_cons
        self.lb = lb
        self.ub = ub
        self.num_popu = num_popu
        self.N_iter = N_iter
        self.p = p
        self.integer_op = integer_op
        self.coef = coef
        self.name = name

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

    def train(self):
        logging.info('fpa begin to train...')

        _flag = False
        best_opt_idx, best_opt_vals, _ = deb_feasible_compare(self.populations0, self.obj_fun_and_less_cons)
        if _:
            _flag = True
        x_best = self.populations0[best_opt_idx]
        self.f_and_cons_list.append(best_opt_vals)
        self.x_best_list.append(x_best)
        self.diversity_list.append(self._diversity())
        self.different_list.append(0.0)
        self.history.append(self.populations0)

        f_min = best_opt_vals[0]

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
                opt_idx, opt_vals, _ = deb_feasible_compare(tmp, self.obj_fun_and_less_cons)
                if opt_idx == 1:
                    # 新解更优
                    self.populations1[i] = x_new
                    tmp = [x_best, x_new]
                    best_opt_idx, best_opt_vals, _ = deb_feasible_compare(tmp, self.obj_fun_and_less_cons)
                    if _:
                        _flag = True
                    if best_opt_idx == 1:
                        x_best = x_new
                        f_min = best_opt_vals[0]
                else:
                    # 保持不变
                    self.populations1[i] = self.populations0[i]

            # t时刻所有花粉都已经迭代完成，记录数据
            self.f_and_cons_list.append(best_opt_vals)
            self.x_best_list.append(x_best)
            self.different_list.append(self._different())

            self.populations0 = self.populations1
            if t % 10 == 0:
                self.history.append(self.populations0)
            self.diversity_list.append(self._diversity())

        logging.info('最终是否找到可行解{}, 最优函数值{}'.format(_flag, f_min))

    def save(self, path, axis, axis_name, f_and_cons_name):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        # 1、画函数图
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Make data.
        num_of_points = 81
        x_l = self.init_lb[axis[0]]
        x_u = self.init_ub[axis[0]]
        x_step = (x_u - x_l) / num_of_points
        X = np.arange(x_l, x_u + x_step, x_step)
        y_l = self.init_lb[axis[1]]
        y_u = self.init_ub[axis[1]]
        y_step = (y_u - y_l) / num_of_points
        Y = np.arange(y_l, y_u + y_step, y_step)

        if self.integer_op:
            X = X.astype(np.int64)
            Y = Y.astype(np.int64)
            X = np.unique(X)
            Y = np.unique(Y)
        X, Y = np.meshgrid(X, Y)
        s1, s2 = X.shape
        Z = np.zeros_like(X)
        for i in range(s1):
            for j in range(s2):
                x = np.copy(self.x_best_list[-1])
                x[axis[0]] = X[i, j]
                x[axis[1]] = Y[i, j]
                vals = self.obj_fun_and_less_cons(x)
                Z[i, j] = vals[0]

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(axis_name[0])
        ax.set_ylabel(axis_name[1])
        ax.set_zlabel('f')
        ax.view_init(45, 45)
        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # plt.show()
        plt.savefig(os.path.join(path, '{}_fval.jpg'.format(self.name)))

        # 寻优参数变化曲线
        fig, ax = plt.subplots()
        l1, = ax.plot(self.diversity_list)
        l2, = ax.plot(self.different_list)
        ax.legend((l1, l2), ("Div", "Dif"))
        ax.set_xlabel('t')
        ax.set_ylabel('Div,Dif')
        plt.savefig(os.path.join(path, '{}_div_dif.jpg'.format(self.name)))

        # 目标函数值和约束值的变化
        f_and_cons_list = np.asarray(self.f_and_cons_list)
        _, n = f_and_cons_list.shape
        rows = ceil(n / 3)
        fig, axes = plt.subplots(rows, 3)

        lines = []
        for i in range(n):
            r = i // 3
            c = i % 3
            l, = axes[r][c].plot(f_and_cons_list[:, i])
            axes[r][c].legend(f_and_cons_name[i])
            axes[r][c].set_xlabel('t')

        plt.savefig(os.path.join(path, '{}_f_and_cons.jpg'.format(self.name)))

        _, axes = plt.subplots(3, 2, figsize=(10, 15))
        mapping = {
            (0, 0): 0,
            (0, 1): 10,
            (1, 0): 30,
            (1, 1): 50,
            (2, 0): 80,
            (2, 1): 100,
        }
        for i in range(3):
            for j in range(2):
                axes[i][j].contour(X, Y, Z, cmap=cm.coolwarm, antialiased=True, alpha=0.5)
                t = mapping[(i, j)]
                points = self.history[t // 10]
                axes[i][j].scatter(points[:, axis[0]], points[:, axis[1]], c='k')
                axes[i][j].set_title('t={}'.format(t))

        plt.savefig(os.path.join(path, '{}_寻优花粉分布.jpg'.format(self.name)))

