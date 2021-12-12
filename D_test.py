from calc_d import DMatrix
from mesh_grid import MeshGrid
import numpy as np


D = np.array([0.3, 0.35, 0.10])
v = np.array([0.25, 0.13, 0.20])
vd = 0.002
I = 3.6
l = 0.001


if __name__ == '__main__':
    solver = DMatrix(D, v, vd, I, l)

    x1 = np.linspace(0, 10, 11)
    y1 = np.linspace(0, 10, 11)
    z1 = [3]
    t = [0.1 * ele for ele in range(10)]

    lower = [0, 0, 3]
    upper = [10, 10, 3]
    nums = [11, 11, 1]
    mesh_grid = MeshGrid(lower, upper, nums)

    solver.init(mesh_grid, t)
    solver.build_Dt()

    # location 是n个观测源的位置， n*3的矩阵
    location = np.array([
        [2, 2, 3],
        [5, 5, 3],
    ])
    D1 = solver.get_D1(location)

    print('Dt shape=', solver.DT.shape)
    print('D1 =', D1)