
"""
根据计算得到的C，和观测值，对C进行修正
"""
import logging

import numpy as np


def get_miu_sigam(obsrv_loc, yt: dict, c_at_t):
    """
    和观测值对比，进行误差统计，最终输出均值miu和方差sigma
    返回k * 2，k个[miu, sigma]
    :param obsrv_loc: 观测点的坐标，n * 2
    :param yt: 观测数据，dict t->value, value n维数据
    :param c_func: 计算c的函数，输入（坐标，t）, 输出值
    :return:
    """
    all_time = set(yt.keys())
    all_time2 = set(c_at_t.keys())
    all_time = all_time & all_time2
    all_time = sorted(list(all_time))

    # 重新组织下观测数据， 按照时间排列
    new_yt = []
    for t in all_time:
        new_yt.append(yt[t])
    new_yt = np.asarray(new_yt)
    new_yt = new_yt.T

    # 计算c_kt
    c_kt = []
    for i in range(len(obsrv_loc)):
        tmp = []
        for t in all_time:
            tmp.append(c_at_t[t][i])
        c_kt.append(tmp)
    c_kt = np.asarray(c_kt)

    # logging.info(new_yt)
    # logging.info(c_kt)

    a_kt = (new_yt - c_kt) / c_kt
    ak = np.mean(a_kt, axis=1)
    logging.info(ak)

    n_k = a_kt.shape[0]
    ans = []
    for k in range(n_k):
        _akt = a_kt[k]
        _ak = ak[k]

        eps = np.abs((_akt - _ak) / _ak)
        mask = eps <= 0.25

        good_data = _akt[mask]
        bad_data = _akt[np.logical_not(mask)]

        good_num = len(good_data)
        bad_num = len(bad_data)

        miu = 0
        sigma = 1
        if good_num == 0:
            miu = np.mean(bad_data)
            sigma = np.std(bad_data)
        else:
            good_u = np.mean(good_data)
            good_std = np.std(good_data)
            if bad_num == 0:
                miu = good_u
                sigma = good_std
            else:
                bad_u = np.mean(bad_data)
                bad_std = np.std(bad_data)

                good_var = good_std ** 2
                bad_var = bad_std ** 2
                miu = (good_u * bad_var + bad_u * good_var) / (good_var + bad_var)
                sigma = np.sqrt(good_var * bad_var / (good_var + bad_var))

        # miu = 0
        # sigma = 1
        logging.info(f'miu={miu}, sigma={sigma}')
        miu = np.exp(miu + 0.5 * miu * miu)
        sigma = np.sqrt(miu * miu * (np.exp(sigma * sigma) - 1))

        ans.append([miu, sigma])
    logging.info(ans)
    return ans
