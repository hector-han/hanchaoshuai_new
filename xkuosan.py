import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import mpl_toolkits.mplot3d
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import xlsxwriter



dx = 1 #x轴空间步长
dy = 0.1 #y轴空间步长
N = 14 #x轴空间步数
H = 30 #y轴空间步数
dt =1#时间步长
M = 10#时间的步数
Dx = 0.35
Dy = 0.05
vx = 0.25
vy = 0.3
Q1 = 1
Q2 = 2
x1 = 2
x2 = 5
y1 = 0.5
y2 = 1.2
X = [0.3,1.0]
Y = [0.5,1.2]


C = np.zeros([N+1,M+1])#建立二维空数组
Space_x = np.arange(0,(N+1)*dx,dx)#建立空间等差数列，从0到3，公差是dx
Space_y = np.arange(0,(H+1)*dy,dy)#建立空间等差数列，从0到3，公差是dx

#边界条件
for k in np.arange(0,M+1):
    C[0,k] = 0.0
    C[N,k] = 0.0

#初始条件
for i in range(0,N+1):
    if i*dx-x1==0:
        a = Q1
    else:
        a = 0
    if i*dx-x2==0:
        b = Q2
    else:
        b = 0
    C[i,0]=(a+b)

# #print (x,y)
# workbook = xlsxwriter.Workbook('kuosan.xlsx') # 建立文件
# worksheet = workbook.add_worksheet('kuosan') # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
# worksheet.write(0,0,'x坐标') # 向A1写入i
# worksheet.write(0,1,'y坐标')#向第二行第二例写入j
# worksheet.write(0,2,'扩散时刻')#向第二行第二例写入z[i,j]
# worksheet.write(0,2,'浓度值')#向第二行第二例写入z[i,j]
# list1=[]

#递推关系
# h=1

for k in np.arange(1,M):
    for i in np.arange(0,N):
        if i * dx - x1 == 0:
            a = Q1
            C[i,k+1]=((Dx+vx*dx/2)*(C[i-1,k]+C[i+1,k]-2*C[i,k])/(dx**2)-vx*(-C[i-1,k]+C[i+1,k])/(2*dx)+a)*dt+C[i,k]
        else:
            a = 0
            if i * dx - x2 == 0:
                b = Q2
                C[i,k+1]=((Dx+vx*dx/2)*(C[i-1,k]+C[i+1,k]-2*C[i,k])/(dx**2)-vx*(-C[i-1,k]+C[i+1,k])/(2*dx)+b)*dt+C[i,k]

            else:
                b = 0
                C[i,k+1]=((Dx+vx*dx/2)*(C[i-1,k]+C[i+1,k]-2*C[i,k])/(dx**2)-vx*(-C[i-1,k]+C[i+1,k])/(2*dx))*dt+C[i,k]


plt.plot(Space_x,C[:,0], 'g-', label='t=0',linewidth=1.0)
plt.plot(Space_x,C[:,3], 'b-', label='t=3/10',linewidth=1.0)
plt.plot(Space_x,C[:,6], 'k-', label='t=6/10',linewidth=1.0)
plt.plot(Space_x,C[:,9], 'r-', label='t=9/10',linewidth=1.0)
plt.plot(Space_x,C[:,10], 'y-', label='t=1',linewidth=1.0)
plt.ylabel('C(x,t)', fontsize=10)
plt.xlabel('x', fontsize=10)
#plt.xlim(0,30)
#plt.ylim(0,20)
plt.legend(loc='upper right')
#
# plt.plot(Space_y, C[:,:,0], 'g-', label='t=0', linewidth=1.0)
# plt.plot(Space_y, C[:,:, 3000], 'b-', label='t=3/10', linewidth=1.0)
# plt.plot(Space_y, C[:,:, 6000], 'k-', label='t=6/10', linewidth=1.0)
# plt.plot(Space_y, C[:,:,9000], 'r-', label='t=9/10', linewidth=1.0)
# plt.plot(Space_y, C[:,:, 10000], 'y-', label='t=1', linewidth=1.0)
# plt.ylabel('C(y,t)', fontsize=20)
# plt.xlabel('y', fontsize=20)
# plt.xlim(0, 3)
# plt.ylim(0, 10)
# plt.legend(loc='upper right')
#
#温度等高线随时空坐标的变化，温度越高，颜色越偏红
extent = [0,0.1,0,3]#时间和空间的取值范围
levels = np.arange(0,5,0.01)#温度等高线的变化范围0-10，变化间隔为0.1
plt.contourf(C,levels,origin='lower',extent=extent,cmap=plt.cm.jet)
plt.show()