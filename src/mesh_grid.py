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
        :param nums: 每个维度划分多少个点。2 <= nums, 每个维度点的个数为num_i
        """
        self._lower = np.asarray(lower)
        self._upper = np.asarray(upper)
        self.nums = np.asarray(nums)
        self._cumprod = np.cumprod([1] + nums[0:-1])
        self.steps = (self._upper - self._lower) / (self.nums - 1)
        self.steps = np.nan_to_num(self.steps)
        self.dim = len(self._lower)
        # 网格的个数
        self.total_M = 1
        for n in self.nums:
            self.total_M *= n

    def get_grid(self, dim):
        """
        获取dim维度上的grid, 不存在返回[]
        :param dim:
        :return:
        """
        if dim >= self.dim:
            return []

        l = self._lower[dim]
        u = self._upper[dim]
        num = self.nums[dim]
        return np.linspace(l, u, num).tolist()

    def nearest_grid_idx(self, point):
        """
        获取最近的网格id, 在每个维度
        :param point:
        :return:
        """
        point = np.asarray(point)
        f_idx = (point - self._lower) / self.steps
        f_idx = np.nan_to_num(f_idx)
        return [int(ele) for ele in np.rint(f_idx)]

    def to_flaten_idx(self, grid_idx):
        """
        二维、三维空间坐标转换为顺序排列的网格id
        最外层的轴变的越快。二维举例子，x变化越快，
        :param grid_idx:
        :return:
        """
        return int(np.dot(self._cumprod, grid_idx))

    def point_flaten_idx(self, point):
        grid_idx = self.nearest_grid_idx(point)
        return self.to_flaten_idx(grid_idx)








