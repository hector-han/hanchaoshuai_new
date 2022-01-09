import numpy as np
from mesh_grid import MeshGrid
from calc_d import DMatrix
from calc_b import BMatrix

def create_env():
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

    # 空间网格划分
    lower = [0, 0]
    upper = [100, 100]
    nums = [101, 101]
    mesh_grid = MeshGrid(lower, upper, nums)
    # 时间网格划分
    grid_t = list(range(1, 3))

    matD = DMatrix(location, u, theta, vd, I, l)
    matD.init(mesh_grid, grid_t)
    matD.build_Dt()

    # location 是n个观测点的位置， n*3的矩阵
    srv_location = np.array([
        [10, 10, 3],
        [20, 20, 3],
        [30, 30, 3],
        [40, 40, 3],
        [50, 50, 3],
    ])
    matD.set_obsrv_location(srv_location)
    D1 = matD.get_D1()
    print('D1 =', D1)

    #############B 矩阵
    y_10 = np.array([1] * 5)
    y_20 = np.array(list(range(1, 6)))
    bMat = BMatrix(matD, y_10, y_20)
    B = bMat.build()
    print("B = ", B)

if __name__ == '__main__':
    create_env()
