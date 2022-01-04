from calc_d import DMatrix
from mesh_grid import MeshGrid
import numpy as np


location = np.array([
    [2.1, 2.1, 3],
    [5.1, 5.1, 3],
])

Q = np.array([3., 5.])  # mass emitted per unit time 单位时间内发射的质量
u = 5
theta = np.pi / 4
vd = 0
I = 0
l = 0


if __name__ == '__main__':
    solver = DMatrix(location, u, theta, vd, I, l)

    grid_t = list(range(1, 3))

    lower = [0, 0]
    upper = [100, 100]
    nums = [101, 101]
    mesh_grid = MeshGrid(lower, upper, nums)

    solver.init(mesh_grid, grid_t)
    solver.build_Dt()

    # location 是n个观测点的位置， n*3的矩阵
    srv_location = np.array([
        [10, 10, 3],
        [20, 20, 3],
        [30, 30, 3],
        [40, 40, 3],
        [50, 50, 3],
    ])
    solver.set_obsrv_location(srv_location)
    D1 = solver.get_D1()

    print('D1 =', D1)