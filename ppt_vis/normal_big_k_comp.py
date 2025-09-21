import numpy as np
import matplotlib

# 切换到非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# --- 设置 Matplotlib 以支持中文显示 ---
# 添加一个常见的支持中文的字体列表
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC']
# 解决负号显示为方框的问题
plt.rcParams['axes.unicode_minus'] = False


def plot_ann_scenario(title, k, data_points, query_point, output_filename):
    """
    一个辅助函数，用于绘制并保存单个 ANN 场景的图像。

    Args:
        title (str): 图的标题。
        k (int): 要高亮的最近邻的数量。
        data_points (np.ndarray): 背景数据点。
        query_point (np.ndarray): 查询点。
        output_filename (str): 保存图像的文件名。
    """
    # 创建一个独立的画布
    fig, ax = plt.subplots(figsize=(8, 8))  # 使用 1:1 的宽高比

    # 计算所有数据点到查询点的欧氏距离
    distances = np.linalg.norm(data_points - query_point, axis=1)

    # 获取按距离排序后的数据点索引
    sorted_indices = np.argsort(distances)

    # 确定 top-k 邻居和绘制圆圈所需的半径
    top_k_indices = sorted_indices[:k]
    radius = distances[sorted_indices[k - 1]]

    # 1. 绘制所有背景数据点
    ax.scatter(data_points[:, 0], data_points[:, 1], c='lightgray', s=10, label='数据点')

    # 2. 绘制 top-k 最近邻
    ax.scatter(data_points[top_k_indices, 0], data_points[top_k_indices, 1],
               c='cornflowerblue', s=15, label=f'Top-{k} 邻居')

    # 3. 绘制查询点
    ax.scatter(query_point[0], query_point[1], c='red', marker='*', s=250,
               label='查询点 (q)', zorder=5)

    # 4. 绘制表示 top-k 范围的圆圈
    circle = Circle(query_point, radius, color='black', fill=False,
                    linestyle='--', linewidth=1.5, alpha=0.8)
    ax.add_patch(circle)

    # 设置图表样式
    ax.set_title(title, fontsize=18)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=12)
    # 移除坐标轴刻度
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                   left=False, right=False, labelbottom=False, labelleft=False)
    ax.grid(True, linestyle=':', alpha=0.5)

    # 保存图像文件
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 释放内存
    print(f"示意图已成功保存为 '{output_filename}'")


# --- 主程序 ---
if __name__ == "__main__":
    # 生成一些随机数据
    np.random.seed(42)
    num_points_small_k = 500
    num_points_large_k = 20000

    data_small_k = np.random.randn(num_points_small_k, 2) * 0.8
    data_large_k = np.random.randn(num_points_large_k, 2)

    # 定义查询点
    query = np.array([0.5, 0.5])

    print("--- 正在生成示意图 ---")

    # --- 绘制并保存第一张图：小 k ---
    k_small = 50
    title_small_k = f"小 k 值搜索 (k = {k_small})"
    filename_small_k = "small_k_scenario.png"
    plot_ann_scenario(title_small_k, k_small, data_small_k, query, filename_small_k)

    # --- 绘制并保存第二张图：大 k ---
    k_large = 10000
    title_large_k = f"大 k 值搜索 (k = {k_large})"
    filename_large_k = "large_k_scenario.png"
    plot_ann_scenario(title_large_k, k_large, data_large_k, query, filename_large_k)

    print("\n所有示意图已生成完毕。")