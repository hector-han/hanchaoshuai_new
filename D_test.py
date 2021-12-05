from D import DMatrix
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
    xx, yy = np.meshgrid(x1, y1, indexing='ij')

    solver.init(x1, y1, z1, t)
    solver.build_D()

    # location 是n个观测源的位置， n*3的矩阵
    location = np.array([
        [2, 2, 3],
        [5, 5, 3],
    ])
    D1 = solver.get_D1(location)

    print('D shape=', solver.D.shape)
    print('D1 =', D1)