import matplotlib.pyplot as plt
import numpy as np


def plot_single_line(ax: plt.Axes, x_data, y_data, label=None, color=None, linestyle='-', linewidth=1.5):
    """
    将单条折线绘制到指定的 Matplotlib Axes 对象上。

    Args:
        ax (matplotlib.axes.Axes): 要绘制折线的 Axes 对象。
        x_data (list or np.ndarray): 折线的 x 坐标数据。
        y_data (list or np.ndarray): 折线的 y 坐标数据。
        label (str, optional): 折线的标签，用于图例。默认为 None。
        color (str, optional): 折线的颜色。默认为 None (Matplotlib 自动选择)。
        linestyle (str, optional): 折线的线条样式。默认为 '-' (实线)。
        linewidth (float, optional): 折线的线条宽度。默认为 1.5。

    Returns:
        matplotlib.axes.Axes: 绘制了折线后的 Axes 对象。
    """
    # 确保 x_data 和 y_data 是 NumPy 数组以便处理
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # 检查数据长度是否匹配
    if len(x_data) != len(y_data):
        print(f"警告: x_data 和 y_data 长度不匹配 (x: {len(x_data)}, y: {len(y_data)}). 跳过绘制该线.")
        return ax  # 返回原始 Axes 对象

    ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    return ax


# --- 示例使用方法 ---
if __name__ == "__main__":
    # 1. 创建一个 Figure 和一个 Axes 对象
    # 这是在开始绘制所有线条之前只需要做一次的步骤
    fig, ax = plt.subplots(figsize=(8, 10))  # 可以指定图的大小

    # 第四次调用 (数据长度不同)
    x1 = [.5629, .8552, .9901]
    y1 = [12248, 33810, 99315]
    plot_single_line(ax, x1, np.log2(y1), label='IVF16384', color='blue', linestyle='-')  # 只绘制点

    x2 = [.1712, .6604, .9909]
    y2 = [1288, 7082, 39929]
    plot_single_line(ax, x2, np.log2(y2), label='HNSW', color='red', linestyle='-')  # 只绘制点

    x3 = [.0365, .1907, .7243, .9904]
    y3 = [131, 758, 7560, 23311]
    plot_single_line(ax, x3, np.log2(y3), label='SymQG', color='green', linestyle='-')  # 只绘制点

    # 3. 在所有线条绘制完毕后，添加图表元素 (标签、标题、图例等)
    ax.set_xlabel("X Value")
    ax.set_ylabel("Y Value")
    ax.set_title("Multiple Lines on One Plot")
    ax.legend()  # 显示图例，需要每条线调用时指定 label
    ax.grid(True)  # 显示网格线

    # 调整布局以防止标签重叠
    fig.tight_layout()

    # 4. 显示图表
    plt.show()

    # --- 示例：如何在另一个脚本中使用这个函数 ---
    # 假设你在另一个文件 other_script.py 中
    # import matplotlib.pyplot as plt
    # from plot_lines import plot_single_line # 从 plot_lines.py 导入函数
    #
    # fig2, ax2 = plt.subplots()
    # x_data_a = [1, 2, 3]
    # y_data_a = [5, 7, 4]
    # plot_single_line(ax2, x_data_a, y_data_a, label='Line A')
    #
    # x_data_b = [0.5, 1.5, 2.5, 3.5]
    # y_data_b = [6, 8, 5, 6]
    # plot_single_line(ax2, x_data_b, y_data_b, label='Line B')
    #
    # ax2.set_title("Another Plot")
    # ax2.legend()
    # plt.show()

