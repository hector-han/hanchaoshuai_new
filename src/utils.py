"""
1 经纬度转墨卡托
"""
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
    y = earth_radius / 2 * math.log((1 + math.sin(a)) / 1 - math.sin(a))
    return f'{x:.2f},{y:.2f}'


def load_station(file_path):
    df = pd.read_excel(file_path)
    station_to_xy = {}
    for i, row in df.iterrows():
        s = row['采样点']
        xy = str(row['坐标'])
        xy = xy.replace('（', '').replace('）', '').replace('(', '').replace(')', '')
        items = xy.replace(' ', '').replace('，', ',').split(',')
        # print(items)
        assert len(items) == 2
        x = float(items[0])
        y = float(items[1])
        station_to_xy[s] = (x, y)
    return station_to_xy


def plot_scatter(x, y, labels):
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(labels[i], xy=[x[i], y[i]], xytext=(x[i]+0.1, y[i]+0.1))
    plt.show()


def load_observe(file_path):
    df = pd.read_excel(file_path)
    time_series = {}
    for i, row in df.iterrows():
        s = row['样品编号']
        time = int(row['时间']) * 60
        value = float(row['浓度'])
        if time not in time_series:
            time_series[time] = []
        time_series[time].append((s, value))
    return time_series


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
    time_series = load_observe(f2)
    ans = {}
    for time in time_series:
        lst_value = time_series[time]
        tmp = []
        for label, val in lst_value:
            xy = station_to_xy[label]
            xyv = [xy[0], xy[1], val]
            tmp.append(xyv)
        ans[time] = np.asarray(tmp)

    return ans

if __name__ == '__main__':
    station_file = '数据.xlsx'
    guance_file = './观测数据.xlsx'
    data = build_data(station_file, guance_file)
    for k, v in data.items():
        print(k, v)