import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def vis_conflict(name):
    with open(name) as f:
        ls = f.readlines()

    results = []
    for i, line in enumerate(ls[:-1]):
        length = line.strip()
        results.append((i, max(0, int(length) - 1)))
    print(len(results), sum(idx for idx, _ in results) / len(results))

    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设定 x 轴的分桶（bin）大小
    bin_size = 1_000_000  # 1百万为一个槽
    x_min, x_max = 0, 34_000_000  # x 的范围
    num_bins = (x_max - x_min) // bin_size  # 计算总槽数

    # 统计每个 bin 内 y 的累积和
    bins = np.zeros(num_bins, dtype=int)
    for x, y in results:
        bin_index = (x - x_min) // bin_size
        if 0 <= bin_index < num_bins:  # 防止 x 超出范围
            bins[bin_index] += y

    # 生成柱状图
    bin_edges = np.linspace(x_min, x_max, num_bins)  # 修正 bin 边界数量
    plt.bar(bin_edges, bins, width=bin_size, align='edge', edgecolor='black')

    # 添加标签
    plt.xlabel("X 范围 (分桶)")
    plt.ylabel("Y 之和")
    plt.title("X 轴分桶后 Y 之和的可视化")
    plt.show()


if __name__ == "__main__":
    vis_conflict('n_7c.txt')
    vis_conflict('n_8c.txt')