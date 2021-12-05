__author__ = 'admin'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
from Multi_source_continuous_release import multi_source_continuous_release_func

stacks = 3  # 反应堆数量为3
stack_x = [1, 3, 5]
stack_y = [5, 3, 1]
stack_z = [3., 3., 3.]

Q = [3., 5., 8.]  # mass emitted per unit time 单位时间内发射的质量
H = [3., 3, 3]  # stack height, m  释放源的高度
h=2
Dx=0.30
Dy=0.35
Dz=0.10
wx=5
wy=3
wz=1
vd=0.002
I=3.6*0.001
t=1

x1=np.linspace(0,3000,200)
y1=np.linspace(0,2000,200)
x,y =np.meshgrid(x1,y1)

#C= np.zeros((len(x1)*len(y1), len(x1)*len(y1)))

# for i in range(0, stacks):
#         #C1= np.zeros((len(x1)*len(y1), len(x1)*len(y1)))  # 初始化C矩阵
#         # 调用高斯函数
#         C1 = instantaneous_release_func(Q[i], 3,20,10, h, stack_x[i], stack_y[i], stack_z[i],H[i], Dx, Dy, Dz, vd,I,1,0, t)
#         C[:, :] = C[:, :]+C1
#instantaneous_release_func(17000, 5,20,10, 30, 100, 200,10, 10, Dx, Dy, Dz, vd,I,1,0, t)
#continuous_release_func(17000, 5,45,10, 5, 0, 0,10, 10, Dx, Dy, Dz, vd,I,1,0, t)
#multi_source_continuous_release_func(stacks,Q, 5,40,10, h, stack_x, stack_y, stack_z,H, Dx, Dy, Dz, vd,I,1,0, 5)

multi_source_continuous_release_func(stacks,Q, 5,40,10, h, stack_x, stack_y, stack_z,H, Dx, Dy, Dz, vd,I,1,0,t)
