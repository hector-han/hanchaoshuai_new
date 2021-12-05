__author__ = 'admin'

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
from scipy import integrate
import xlsxwriter
from mpl_toolkits.mplot3d import Axes3D

def multi_source_continuous_release_func(sources,Q, u, dirl,afa, h, xs, ys,zs, H, Dx, Dy, Dz, vd,I,L,STABILITY, t):
    u1=u
    # wx=u1*np.cos(afa*np.pi/180.)*np.cos(dirl*np.pi/180.)
    # wy=u1*np.cos(afa*np.pi/180.)*np.sin(dirl*np.pi/180.)
    # wz=u1*np.cos(afa*np.pi/180.)
    wx = 2
    wy = 3
    wz = 1
    x1=np.linspace(0,10,200)
    y1=np.linspace(0,10,200)
    x, y = np.meshgrid(x1, y1)


    #求解网格各点浓度值，存入data文件
    workbook = xlsxwriter.Workbook('MCR_data.xlsx') # 建立文件
    worksheet = workbook.add_worksheet('MCR_data') # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    worksheet.write(0,0,'x坐标') # 向A1写入i
    worksheet.write(0,1,'y坐标')#向第二行第二例写入j
    worksheet.write(0,2,'C值')#向第二行第二例写入z[i,j]
    list1=[]
    C = np.zeros_like(x)
    k = 1
    for id_i in range(x.shape[0]):

        for id_j in range(x.shape[1]):
            i = x[id_i, id_j]
            j = y[id_i, id_j]
            for l in range(0, sources):
                C1 = (Q[l] * np.sqrt(np.pi) / (2. * (t ** 1.5) * np.sqrt(Dx * Dz * Dy)) * np.exp(-0.25 * (
                    (i - xs[l] - wx * t) ** 2. / (Dx * t) + (j - ys[l] - wy * t) ** 2. / (Dy * t) + (h-zs[l] - wz * t) ** 2. / (
                    Dz * t) - (vd + I * L) * t)))

                def f(t1):
                    return Q[l] * np.sqrt(np.pi) *(t-t1)/ (2. * (t1 ** 1.5) * np.sqrt(Dx * Dz * Dy)) * np.exp(-0.25 * (
                        (i - xs[l] - wx * t1) ** 2. / (Dx * t1) + (j - ys[l] - wy * t1) ** 2. / (Dy * t1) + (
                            h-zs[l] - wz * t1) ** 2. / (
                                Dz * t1) - (vd + I * L) * t1))

                C3, err = integrate.quad(f, 0, t)
                # print(C3)
                C[id_i, id_j] += C1 + C3
            print(C[id_i, id_j], i, j)
            list1.append((i, j, C[id_i, id_j]))
            worksheet.write(k, 0, i)  # 向A1写入i
            worksheet.write(k, 1, j)  # 向第二行第二例写入j
            worksheet.write(k, 2, C[id_i, id_j])  # 向第二行第二例写入z[i,j]
            k += 1
    # C=np.zeros((len(x1)*len(y1),len(x1)*len(y1)))
    #
    # k=1
    # for i in range(0,2000,10):
    #     for j in range(0,2000,10):
    #         for l in range(0,sources):
    #             C1=(Q[l]*np.sqrt(np.pi)/(2.*(t**1.5)*np.sqrt(Dx*Dz*Dy)) * np.exp(-0.25*((i-xs[l]-wx*t)**2./(Dx*t)+(j-ys[l]-wy*t)**2./(Dy*t)+(h-zs[l]-wz*t)**2./(Dz*t)-(vd+I*L)*t)))
    #             def f(t1):
    #                 return Q[l]*np.sqrt(np.pi)/(2.*(t1**1.5)*np.sqrt(Dx*Dz*Dy)) * np.exp(-0.25*((i-xs[l]-wx*t1)**2./(Dx*t1)+(j-ys[l]-wy*t1)**2./(Dy*t1)+(h-zs[l]-wz*t1)**2./(Dz*t1)-(vd+I*L)*t1))
    #             C2,err=integrate.quad(f,0,t)
    #             print (C2)
    #             C[i,j]+=C1+C2
    #         print (C[i,j],i,j)
    #         list1.append((i,j,C[i,j]))
    #         worksheet.write(k,0, i) # 向A1写入i
    #         worksheet.write(k,1,j)#向第二行第二例写入j
    #         worksheet.write(k,2,C[i,j])#向第二行第二例写入z[i,j]
    #         k+=1

    workbook.close()
    a = np.array(list1)
    a.flatten()
    print (a)

#画三维图
    # O=a[:,0].astype('int32')
    # P=a[:,1].astype('int32')
    # Q=a[:,2].astype('float32')
    #
    # ax=plt.subplot(111,projection='3d')
    # ax.scatter(O,P,Q,cmap='hot')
    # ax.set_zlabel('C(mg/m3)')
    # ax.set_ylabel('y(m)')
    # ax.set_xlabel('x(m)')
    # plt.show()


#画三维图
    figure=plt.figure()
    ax=figure.gca(projection="3d")
    # x,y =np.meshgrid(x1,y1)
    #
    # # C=(Q*np.sqrt(np.pi)/(2.*(t**1.5)*np.sqrt(Dx*Dz*Dy)) * np.exp(-0.25*((x-xs-wx*t)**2./(Dx*t)+(y-ys-wy*t)**2./(Dy*t)+(z1-wz*t)**2./(Dz*t)-(vd+I*L)*t)))
    # #     #ax.plot_surface(x,y,z,rstride=10,cstride=4,cmap=cm.YlGnBu_r)
    ax.plot_surface(x,y,C,cmap="rainbow")
    #ax.set_zlabel('C(mg/m3)')
    ax.set_zlabel('C(mg/m3)')
    ax.set_ylabel('y(m)')
    ax.set_xlabel('x(m)')
    plt.show()
