"""
在平面上划分网格的类，2d, 3d
"""
import numpy as np


class MeshGrid:
    """
    2d , 3d 划分网格的类
    """
    def __init__(self, lower, upper, nums):
        """
        初始化网格
        输入参数维度相同，2d或者3d。step_i = (upper_i - lower_i) / nums_i
        :param lower: 下界，xmin, ymin, zmin
        :param upper: 上界， xmax, ymax, zmax
        :param nums: 每个维度划分多少等分。1 <= nums, 每个维度点的个数为num_i + 1
        """
        self._lower = np.asarray(lower)
        self._upper = np.asarray(upper)
        self._nums = np.asarray(nums)
        self._steps = (self._upper - self._lower) / self._nums
        self._dim = len(self._lower)
        # 网格的个数
        self.total_M = 1
        for n in self._nums:
            self.total_M *= (n + 1)





