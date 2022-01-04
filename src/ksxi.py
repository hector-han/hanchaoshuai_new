import numpy as np
import sys
import math


def calc_sigmas(u,theta, x, y, x_i, y_i):
    #u为速度矢量；theta为速度矢量与x轴的夹角；t为时间；x、y为坐标；x_i, y_i为释放点坐标
    if u<= 2:  # very unstable
        # vertical
        sigmas_y=0.22*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)
        sigmas_z=0.2*np.sqrt((x-x_i)**2+(y-y_i)**2)

    elif u> 2 and u<=3:  # moderately unstable
        # vertical
        sigmas_y=0.16*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)
        sigmas_z=0.12*np.sqrt((x-x_i)**2+(y-y_i)**2)

    elif u> 3 and u<=5:  # moderately unstable
        # vertical
        sigmas_y=0.11*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)
        sigmas_z=0.08*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)

    elif u> 5 and u<=6:  # moderately unstable
        # vertical
        sigmas_y=0.08*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)
        sigmas_z=0.06*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)

    elif u> 6:  # moderately unstable
        # vertical
        sigmas_y=0.06*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)
        sigmas_z=0.03*np.sqrt((x-x_i)**2+(y-y_i)**2)/((1+0.0001*np.sqrt((x-x_i)**2+(y-y_i)**2))**0.5)

    else:
        sys.exit()

    s_z = (sigmas_z)
    s_y = (sigmas_y) * np.sin(theta)
    s_x = (sigmas_y) * np.cos(theta)

    return np.array([s_x, s_y, s_z])


def calc_sigma(point, loc_i, u, theta):
    x = point[0]
    y = point[1]
    xi = loc_i[0]
    yi = loc_i[1]
    return calc_sigmas(u, theta, x, y, xi, yi)


if __name__ == '__main__':
    u = 2
    theta = math.pi / 4
    x = 2
    y = 2
    x0 = 2
    y0 = 2
    a = calc_sigmas(u, theta, x, y, x0, y0)
    print(a)