"""
测试函数
"""
import numpy as np
from mesh_grid import MeshGrid


def test_mesh_grid():
    lower = [0, 0, 0]
    upper = [10, 20, 30]
    nums = [11, 21, 31]
    m_g = MeshGrid(lower, upper, nums)
    grid_idx = m_g.nearest_grid_idx([10, 20, 30])
    print(grid_idx)
    flat_idx = m_g.to_flaten_idx(grid_idx)
    print(flat_idx)


if __name__ == '__main__':
    test_mesh_grid()