from typing import Any
import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt


DATANUM = 1000


def draw_histogram(x_max, x_min, data, color):
    # 绘制直方图
    # bin_width = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / (len(data) ** (1 / 3))
    # plt.hist(data, bins=int((max(data) - min(data)) / bin_width) if bin_width > 0 else 1,
    #          density=True, alpha=0.6, color=color, label='Histogram')

    # 绘制 KDE 分布
    sns.kdeplot(data, color=color, label='KDE', fill=True)


def draw_histograms(gds: dict[str, Any]):
    x_max = max(max(elem) for elem in gds.values())
    x_min = min(min(elem) for elem in gds.values())
    plt.figure(figsize=(8, 6))
    for color, gd in gds.items():
        draw_histogram(x_max, x_min, gd, color)

    # 图形设置
    plt.xlabel('Range')
    plt.ylabel('Density')
    plt.title(f'Histogram')
    plt.legend()
    # 显示图形
    plt.show()



if __name__ == '__main__':
    gd1 = [random.gauss(10, 2) for _ in range(DATANUM)]
    gd2 = [random.gauss(50, 3) for _ in range(DATANUM)]
    gd3 = [random.gauss(24, 1.5) for _ in range(DATANUM)]

    draw_histograms(gds = {'green': gd1, 'blue': gd2, 'red': gd3})

