import numpy as np
from mesh_grid import MeshGrid
from calc_d import DMatrix
from calc_b import BMatrix

def create_env():
    D = np.array([0.3, 0.35, 0.10])
    v = np.array([0.25, 0.13, 0.20])
    vd = 0.002
    I = 3.6
    l = 0.001
    # 划分网格
    lower = [0, 0, 3]
    upper = [10, 10, 3]
    nums = [11, 11, 1]
    mesh_grid = MeshGrid(lower, upper, nums)
    t = [0.1 * ele for ele in range(10)]

    # location 是n个观测源的位置， n*3的矩阵
    location = np.array([
        [2, 2, 3],
        [5, 5, 3],
    ])

    ########## D 矩阵
    matD = DMatrix(D, v, vd, I, l)
    matD.init(mesh_grid, t)
    matD.build_Dt()
    matD.set_location(location)

    D1 = matD.get_D1()

    print('Dt shape=', matD.DT.shape)
    print('D1 =', D1)

    #############B 矩阵
    y_10 = np.array([1, 1])
    y_20 = np.array([2, 3])
    bMat = BMatrix(D, v, vd, I, l, matD, y_10, y_20)
    B = bMat.build()
    print("B = ", B)

if __name__ == '__main__':
    create_env()
