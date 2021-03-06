from calc_c import NumberRes
from mesh_grid import MeshGrid
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter


# [x, y, z]
location = np.array([
    [2, 2, 3],
    [5, 5, 3],
])

Q = np.array([3., 5.])  # mass emitted per unit time 单位时间内发射的质量
D = np.array([0.3, 0.35, 0.10])
v = np.array([0.25, 0.13, 0.20])
vd = 0.002
I = 3.6
l = 0.001


if __name__ == '__main__':
    # solver = NumberRes(Q, location, D, v, vd, I, l)
    #
    x1 = np.linspace(0, 10, 50)
    y1 = np.linspace(0, 10, 50)
    # z1 = [3]
    xx, yy = np.meshgrid(x1, y1, indexing='ij')
    # t = np.linspace(0, 10, 400)
    # solver.init(x1, y1, z1, t)
    solver = NumberRes(Q, location, D, v, vd, I, l)

    lower = [0, 0, 3]
    upper = [10, 10, 3]
    nums = [50, 50, 1]
    mesh_grid = MeshGrid(lower, upper, nums)
    t = np.linspace(0, 10, 400)
    solver.init(mesh_grid, t)
    solver.process()
    value = solver.view[-1][:, :, 0]
    print(xx.shape, yy.shape, value.shape)
    print(np.sum(value))

    #求解网格各点浓度值，存入data文件
    # workbook = xlsxwriter.Workbook('MCR_data2.xlsx') # 建立文件
    # worksheet = workbook.add_worksheet('MCR_data_numric') # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    # worksheet.write(0,0,'x坐标') # 向A1写入i
    # worksheet.write(0,1,'y坐标')#向第二行第二例写入j
    # worksheet.write(0,2,'C值')#向第二行第二例写入z[i,j]
    # list1=[]
    # C = np.zeros_like(xx)
    # k = 1
    # value = solver.view[-1]
    # for id_i in range(xx.shape[0]):
    #     for id_j in range(xx.shape[1]):
    #         x = xx[id_i, id_j]
    #         y = yy[id_i, id_j]
    #         C[id_i, id_j] = value[id_i, id_j, 0]
    #         print(C[id_i, id_j], x, y)
    #         list1.append((x, y, C[id_i, id_j]))
    #         worksheet.write(k, 0, x)  # 向A1写入i
    #         worksheet.write(k, 1, y)  # 向第二行第二例写入j
    #         worksheet.write(k, 2, C[id_i, id_j])  # 向第二行第二例写入z[i,j]
    #         k += 1
    #
    # workbook.close()

#画三维图
    figure=plt.figure()
    ax=figure.gca(projection="3d")

    ax.plot_surface(xx,yy, value,cmap="rainbow")
    #ax.set_zlabel('C(mg/m3)')
    ax.set_zlabel('C(mg/m3)')
    ax.set_ylabel('y(m)')
    ax.set_xlabel('x(m)')
    plt.show()