"""
1 经纬度转墨卡托
"""
import logging
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']


def lnglat_to_mercator(lng, lat):
    """
    经纬度转墨卡托
    :param lng: 经度
    :param lat: 纬度
    :return: [x, y]
    """
    earth_radius = 6378137.0
    x = lng * math.pi / 180 * earth_radius
    a = lat * math.pi / 180
    y = earth_radius / 2 * math.log((1 + math.sin(a)) / (1 - math.sin(a)))
    return x, y


def load_station(file_path):
    df = pd.read_excel(file_path)
    station_to_xy = {}
    for i, row in df.iterrows():
        s = row['观测点']
        x = float(row['x'])
        y = float(row['y'])
        station_to_xy[s] = (x, y)
    return station_to_xy


def plot_scatter(x, y, labels):
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(labels[i], xy=[x[i], y[i]], xytext=(x[i]+0.1, y[i]+0.1))
    plt.show()


def load_observe(file_path):
    """
    返回dict, t->station->value
    tuple: (采样站， 值)
    """
    df = pd.read_excel(file_path)
    columns = df.columns
    time_series = {}
    all_station = columns[1:]
    for i, row in df.iterrows():
        time = int(row['时间'])
        if time not in time_series:
            time_series[time] = {}
        for s in all_station:
            value = float(row[s])
            time_series[time][s] = value
    return time_series, list(all_station)


def main1():
    file_path = './数据.xlsx'
    df = load_station(file_path)
    print(df)

    x = []
    y = []
    label = []
    for s in df.keys():
        label.append(s)
        v = df[s]
        x.append(v[0])
        y.append(v[1])
    plot_scatter(x, y, label)


def build_data(f1, f2):
    station_to_xy = load_station(f1)
    time_series, all_station = load_observe(f2)
    logging.info(f'总共有{len(all_station)}个采样点')

    obsrv_loc = []
    for station in all_station:
        loc = station_to_xy[station]
        obsrv_loc.append(loc)
        logging.info(f'采样点：{station}, 坐标：{loc}')
    obsrv_loc = np.asarray(obsrv_loc)

    yt = {}
    for time in time_series:
        lst_value = time_series[time]
        tmp = []
        for station in all_station:
            if station in lst_value:
                val = lst_value[station]
                tmp.append(val)
            else:
                tmp.append(1e-13)

        yt[time] = np.asarray(tmp)

    return obsrv_loc, yt


def excel_jingweidu(fn_input, fn_output):
    df = pd.read_excel(fn_input)

    def _trans_func(row):
        longi = float(row['经度'])
        lanti = float(row['纬度'])
        x, y = lnglat_to_mercator(longi, lanti)
        return x, y

    xs = df.apply(lambda row: _trans_func(row)[0], axis=1)
    ys = df.apply(lambda row: _trans_func(row)[1], axis=1)

    theta = np.mean(df['纬度'])
    theta = theta * math.pi / 180
    factor = np.cos(theta)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = xs * factor
    ys = ys * factor

    x_diff = (np.max(xs) + np.min(xs)) / 2
    y_diff = (np.max(ys) + np.min(ys)) / 2

    xs = xs - x_diff
    ys = ys - y_diff

    df['x'] = xs
    df['y'] = ys

    df.to_excel(fn_output)

def main2():
    fn_input = r'C:\Users\hengk\Desktop\观测点数据.xlsx'
    fn_output = r'C:\Users\hengk\Desktop\观测点数据坐标.xlsx'
    excel_jingweidu(fn_input, fn_output)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    station_file = '观测点坐标.xlsx'
    guance_file = '观测点数据.xlsx'
    obsrv_loc, yt = build_data(station_file, guance_file)
    print(obsrv_loc)
    print(yt)


