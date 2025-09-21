import numpy as np
import matplotlib

# 切换到非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 设置 Matplotlib 以支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def plot_boundary_phenomenon(output_filename="boundary_phenomenon.png"):
    """
    生成一个 3D 示意图，展示 Top-k 决策边界内外的点数差异。
    """
    print("--- 正在生成 Top-k 边界现象示意图 ---")

    # --- 1. 参数设置 (使用更少的点数) ---
    num_total_points = 5000  # 点的总数减少，使其更像示意图
    k = 1000  # Top-k 的 k 值
    shell_thickness_rank = 100  # 定义壳的“厚度”，用邻居排名数表示

    sphere_radius = 1.0
    dimension = 3

    # --- 2. 在球体内均匀生成数据点 ---
    radii = sphere_radius * np.power(np.random.rand(num_total_points), 1.0 / dimension)
    directions = np.random.randn(num_total_points, dimension)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    points = radii[:, np.newaxis] * directions

    query_point = np.array([[0, 0, 0]])

    # --- 3. 计算距离并找出边界附近的点 ---
    distances = np.linalg.norm(points - query_point, axis=1)
    sorted_indices = np.argsort(distances)

    # 定义边界内侧和外侧的壳
    # 内壳: [k - shell_thickness_rank, k)
    inner_shell_indices = sorted_indices[k - shell_thickness_rank: k]
    # 外壳: [k, k + shell_thickness_rank)
    outer_shell_indices = sorted_indices[k: k + shell_thickness_rank]

    # 获取边界的距离半径
    radius_inner_limit = distances[inner_shell_indices[0]]
    radius_top_k = distances[sorted_indices[k - 1]]
    radius_outer_limit = distances[outer_shell_indices[-1]]

    # --- 4. 创建 3D 可视化 ---
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 "内壳" 的点 (Top-k 边界内侧)
    ax.scatter(points[inner_shell_indices, 0], points[inner_shell_indices, 1], points[inner_shell_indices, 2],
               c='mediumseagreen', s=40, label=f'边界内侧的点 (共 {len(inner_shell_indices)} 个)')

    # 绘制 "外壳" 的点 (Top-k 边界外侧)
    ax.scatter(points[outer_shell_indices, 0], points[outer_shell_indices, 1], points[outer_shell_indices, 2],
               c='lightcoral', s=40, label=f'边界外侧的点 (共 {len(outer_shell_indices)} 个)')

    # 绘制查询点
    ax.scatter(query_point[:, 0], query_point[:, 1], query_point[:, 2],
               c='black', marker='*', s=500, label='查询点 q', depthshade=False)

    # --- 5. 绘制三个半透明球面来清晰地展示边界和壳 ---
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Top-k 边界球面 (实线)
    x_top_k = radius_top_k * np.outer(np.cos(u), np.sin(v))
    y_top_k = radius_top_k * np.outer(np.sin(u), np.sin(v))
    z_top_k = radius_top_k * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_top_k, y_top_k, z_top_k, color='blue', alpha=0.5, linewidth=1.5, rstride=10, cstride=10)

    # --- 6. 添加标注和样式 ---
    ax.set_title('Top-k 决策边界内外的点数差异', fontsize=20, pad=20)
    ax.legend(loc='upper right', fontsize=12)

    # 添加文本框解释关键信息
    text_str = (
        f"现象: 边界内、外同样“厚度”的两个薄壳\n"
        f"包含的点数却截然不同。\n\n"
        f"蓝色球面 (Top-{k} 边界, r={radius_top_k:.3f})\n\n"
        f" • 内壳 (绿色点): {len(inner_shell_indices)} 个\n"
        f" • 外壳 (红色点): {len(outer_shell_indices)} 个\n\n"
        f"结论: 在高维空间中，随着半径 r 增加，\n"
        f"球壳的“体积”和包含的点数急剧增长。\n"
        f"这导致剪枝阈值稍一放松，候选集就会爆炸。"
    )
    ax.text2D(0.02, 0.02, text_str, transform=ax.transAxes, fontsize=13,
              bbox=dict(boxstyle='round,pad=0.5', fc='ivory', alpha=0.9))

    # 设置视角和坐标轴
    ax.view_init(elev=25., azim=45)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])  # 隐藏刻度值
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 隐藏坐标轴线
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 隐藏背景面板

    # 隐藏坐标轴的黑色线条
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # --- 7. 保存图像 ---
    plt.savefig(output_filename, dpi=150, transparent=False, facecolor='white')
    plt.close(fig)
    print(f"示意图已成功保存为 '{output_filename}'")


# --- 主程序 ---
if __name__ == "__main__":
    plot_boundary_phenomenon()