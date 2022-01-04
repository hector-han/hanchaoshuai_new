from calc_c import AnalysisRes
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter

# [x, y, z]
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
    t = 10

    analysis = AnalysisRes(Q, location, u, theta, vd, I, l)

    x1 = np.linspace(0,50, 100)
    y1 = np.linspace(0,50, 100)
    xx, yy = np.meshgrid(x1, y1, indexing='ij')

    #求解网格各点浓度值，存入data文件
    workbook = xlsxwriter.Workbook('MCR_data2.xlsx') # 建立文件
    worksheet = workbook.add_worksheet('MCR_data_ana') # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    worksheet.write(0,0,'x坐标') # 向A1写入i
    worksheet.write(0,1,'y坐标')#向第二行第二例写入j
    worksheet.write(0,2,'C值')#向第二行第二例写入z[i,j]
    list1=[]
    C = np.zeros_like(xx)
    k = 1
    z = 3
    for id_i in range(xx.shape[0]):
        for id_j in range(xx.shape[1]):
            x = xx[id_i, id_j]
            y = yy[id_i, id_j]

            xyz = np.array([x, y, z])

            C[id_i, id_j] = analysis.at(xyz, t)
            # print(C[id_i, id_j], x, y)
            list1.append((x, y, C[id_i, id_j]))
            worksheet.write(k, 0, x)  # 向A1写入i
            worksheet.write(k, 1, y)  # 向第二行第二例写入j
            worksheet.write(k, 2, C[id_i, id_j])  # 向第二行第二例写入z[i,j]
            k += 1
    # print(max(analysis.tmp))
    workbook.close()

#画三维图
    figure=plt.figure()
    ax=figure.gca(projection="3d")

    ax.plot_surface(xx,yy,C,cmap="rainbow")
    #ax.set_zlabel('C(mg/m3)')
    ax.set_zlabel('C(mg/m3)')
    ax.set_ylabel('y(m)')
    ax.set_xlabel('x(m)')
    plt.show()