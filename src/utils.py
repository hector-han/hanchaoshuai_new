"""
1 经纬度转墨卡托
"""
import math
import pandas as pd
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

if __name__ == '__main__':
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
