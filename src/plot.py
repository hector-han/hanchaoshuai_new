"""
画图的函数
"""
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file):
    """
    加载数据
    :param file:
    :return:
    """
    df = pd.read_csv(file, sep='\t', header=None, dtype=str)
    # f'{o_time}\t{t}\t{div}\t{diff}\t{obj_val}\t{o_x}\t{o_obsrv}'
    iteration = df[1].tolist()
    f_val = df[4].tolist()
    x = df[5].tolist()

    iteration = [int(item) for item in iteration]
    f_val = [float(item) for item in f_val]
    return iteration, f_val, x


def main(fn):
    iteration, f_val, x = load_data(fn)
    x = [(float(item.split(',')[0]), float(item.split(',')[1])) for item in x]
    # 寻优参数变化曲线
    fig, ax = plt.subplots()
    ax.plot(f_val)

    fig, ax = plt.subplots()
    ax.plot(f_val)
    plt.show()




if __name__ == '__main__':
    fn = 'tmp.tsv'
    main(fn)
