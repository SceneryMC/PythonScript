import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_lines(data_lists, title="", xlabel="X Axis", ylabel="Y Axis"):
    """
    绘制多条折线图

    Args:
        data_lists (list[list[float]]): 包含若干个浮点数列表的列表。
                                         每个内部列表代表一条折线。
                                         假设 x 轴是数据的索引（0, 1, 2...）。
        title (str): 图表的标题。
        xlabel (str): X轴的标签。
        ylabel (str): Y轴的标签。
    """
    plt.figure(figsize=(10, 6)) # 创建一个图形和轴，并设置大小

    for i, data_list in enumerate(data_lists):
        # 绘制每条线，使用列表的索引作为默认的标签
        plt.plot(data_list, marker='o', linestyle='-', label=f'Data {i+1}')
        # marker='o' 在每个数据点显示圆圈
        # linestyle='-' 绘制实线

    plt.title(title) # 设置标题
    plt.xlabel(xlabel) # 设置X轴标签
    plt.ylabel(ylabel) # 设置Y轴标签
    plt.grid(True) # 显示网格
    plt.legend() # 显示图例，区分不同的线
    plt.tight_layout() # 自动调整布局，避免标签重叠
    plt.show() # 显示图表


def crit_complex(line):
    return line.strip().split()[1] == 'True'


def crit_simple(line):
    return line.strip() == 'T'


def read_raw_file(p, crit=crit_complex):
    result = []
    with open(p, "r") as f:
        ls = f.readlines()
    ls = [crit(line) for line in ls]
    step = int(len(ls) / 100)
    for i in range(0, len(ls), step):
        result.append(sum(ls[i:i + step]) / len(ls[i:i + step]))
    return result


def read_raw_file_s(p, crit=crit_complex):
    result = []
    with open(p, "r") as f:
        ls = f.readlines()
    ls = [crit(line) for line in ls]
    step = 10
    for i in range(0, min(len(ls), step * 1000), step):
        result.append(sum(ls[i:i + step]) / len(ls[i:i + step]))
    return result
# --- 示例用法 ---

# 准备一些示例数据
# 这是一个包含三个浮点数列表的列表

sample_data = [
    read_raw_file('/home/scenerymc/programming/C++/SymphonyQG/results/results_10000.txt'),
    read_raw_file('/home/scenerymc/programming/C++/SymphonyQG/results/results_1000.txt'),
    read_raw_file('/home/scenerymc/programming/C++/SymphonyQG/results/results_10.txt'),
]

# 调用函数绘制图表
plot_lines(sample_data,  xlabel='进度', ylabel='节点接受率')

sample_data_2 = [
    read_raw_file_s('/home/scenerymc/programming/C++/SymphonyQG/results/results_10000.txt'),
    read_raw_file_s('/home/scenerymc/programming/C++/SymphonyQG/results/results_1000.txt'),
    read_raw_file_s('/home/scenerymc/programming/C++/SymphonyQG/results/results_10.txt'),
]

plot_lines(sample_data_2, xlabel='进度', ylabel='节点接受率')

sample_data_3 = [
    read_raw_file('/home/scenerymc/programming/C++/SymphonyQG/results_10000.txt', crit=crit_simple),
    read_raw_file('/home/scenerymc/programming/C++/SymphonyQG/results_1000.txt', crit=crit_simple),
]

plot_lines(sample_data_3, xlabel='进度', ylabel='节点接受率')