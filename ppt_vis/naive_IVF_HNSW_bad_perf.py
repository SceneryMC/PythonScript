import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import PathPatch
import matplotlib.path as mpath

# --- 设置 Matplotlib 以支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 图 1: IVF 在大 k 值下的低效性
# ==============================================================================
def plot_ivf_inefficiency():
    """
    绘制并保存 IVF 在大 k 值下，因 nprobe 过大导致大量无效计算的示意图。
    """
    np.random.seed(0)

    # 1. 创建聚类中心和数据点
    num_centroids = 50
    centroids = np.random.rand(num_centroids, 2) * 10

    data_points = []
    points_per_cluster = 200
    for centroid in centroids:
        data_points.append(np.random.randn(points_per_cluster, 2) * 0.5 + centroid)
    data_points = np.vstack(data_points)

    # 2. 定义查询点和 ground truth
    query_point = np.array([5, 5])
    k_true = int(points_per_cluster * 1.5)  # top-k 邻居应该主要来自最近的几个簇
    distances_to_q = np.linalg.norm(data_points - query_point, axis=1)
    true_neighbors_indices = np.argsort(distances_to_q)[:k_true]

    # 3. 计算 Voronoi 图来表示 IVF 分区
    vor = Voronoi(centroids)

    # 4. 模拟 IVF 的粗粒度搜索
    nprobe = 10
    centroid_dists_to_q = np.linalg.norm(centroids - query_point, axis=1)
    probed_centroid_indices = np.argsort(centroid_dists_to_q)[:nprobe]

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制 Voronoi 图的边界
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1, line_alpha=0.5)

    # 5. 可视化结果
    for i in range(num_centroids):
        # 获取 Voronoi 单元格的多边形
        region_index = vor.point_region[i]
        if -1 in vor.regions[region_index]: continue  # 跳过开放区域
        polygon = [vor.vertices[j] for j in vor.regions[region_index]]

        # 统计落入此单元格的 top-k 邻居数量
        points_in_cell_mask = np.array(
            [mpath.Path(polygon).contains_point(p) for p in data_points[true_neighbors_indices]])
        count_in_cell = np.sum(points_in_cell_mask)

        # 如果这个单元格是被探测的 (probed)
        if i in probed_centroid_indices:
            # 如果单元格很重要 (包含大量结果)
            if count_in_cell > 50:
                ax.fill(*zip(*polygon), color='lightcoral', alpha=0.6)
                ax.text(centroids[i, 0], centroids[i, 1], f"包含 {count_in_cell} 个结果", fontsize=9, ha='center',
                        color='black')
            # 如果单元格不重要 (包含很少或没有结果)
            else:
                ax.fill(*zip(*polygon), color='lightblue', alpha=0.5)
                if count_in_cell > 0:
                    ax.text(centroids[i, 0], centroids[i, 1], f"仅含 {count_in_cell} 个结果", fontsize=8, ha='center',
                            color='gray')

    # 绘制数据点和查询点
    ax.scatter(data_points[true_neighbors_indices, 0], data_points[true_neighbors_indices, 1], c='green', s=5,
               label=f'Top-{k_true} 真实邻居')
    ax.scatter(query_point[0], query_point[1], c='red', marker='*', s=300, label='查询点 q', zorder=10)

    # 设置图表样式
    ax.set_title(f'IVF 在大 k 值下的挑战 (nprobe={nprobe})', fontsize=18)
    ax.set_xlabel(
        '红色区域: 少数被探测的分区包含了大部分结果\n蓝色区域: 大部分被探测的分区几乎不含有效结果，造成计算浪费',
        fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

    plt.savefig("ivf_inefficiency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("IVF 低效示意图已保存为 'ivf_inefficiency.png'")


# ==============================================================================
# 图 2: HNSW 在大 k 值下的低效性
# ==============================================================================
def plot_hnsw_inefficiency():
    """
    绘制并保存 HNSW 在大 k 值下，因图结构导致大量重复访问的示意图。
    """
    # 1. 手动定义一个清晰的图结构
    nodes = {
        'Start': (0, 3), 'A': (2, 5), 'B': (2, 1), 'C': (4, 3),
        'D': (6, 5), 'E': (6, 1), 'F': (8, 3), 'Goal': (10, 3)
    }
    edges = [('Start', 'A'), ('Start', 'B'), ('A', 'C'), ('B', 'C'),
             ('C', 'D'), ('C', 'E'), ('D', 'F'), ('E', 'F'), ('F', 'Goal')]

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制所有节点和边
    for start, end in edges:
        ax.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]], 'k-', alpha=0.3)
    for name, pos in nodes.items():
        ax.plot(pos[0], pos[1], 'o', markersize=20, color='lightgray', mec='black')
        ax.text(pos[0], pos[1], name, fontsize=12, ha='center', va='center')

    # 2. 模拟并可视化搜索路径
    # 路径 1: Start -> A -> C
    ax.arrow(0, 3, 1.8, 1.8, head_width=0.2, fc='blue', ec='blue', alpha=0.7, length_includes_head=True)
    ax.arrow(2, 5, 1.8, -1.8, head_width=0.2, fc='blue', ec='blue', alpha=0.7, length_includes_head=True)
    ax.text(1.5, 4.8, '1. 探索 A', fontsize=10, color='blue')
    ax.text(3.5, 4.5, '2. 从 A 发现 C', fontsize=10, color='blue')

    # 路径 2: Start -> B -> C (重复发现)
    ax.arrow(0, 3, 1.8, -1.8, head_width=0.2, fc='green', ec='green', alpha=0.7, length_includes_head=True)
    ax.arrow(2, 1, 1.8, 1.8, head_width=0.3, fc='red', ec='red', alpha=0.8,
             linestyle='--', linewidth=2, length_includes_head=True)
    ax.text(1.5, 1.2, '3. 探索 B', fontsize=10, color='green')
    ax.text(2.5, 2.5, '4. 从 B 再次发现 C', fontsize=10, color='red', weight='bold')

    # 高亮已被访问的节点
    ax.plot(nodes['Start'][0], nodes['Start'][1], 'o', markersize=20, color='skyblue', mec='black', label='已访问')
    ax.plot(nodes['A'][0], nodes['A'][1], 'o', markersize=20, color='skyblue', mec='black')
    ax.plot(nodes['B'][0], nodes['B'][1], 'o', markersize=20, color='skyblue', mec='black')
    ax.plot(nodes['C'][0], nodes['C'][1], 'o', markersize=20, color='gold', mec='black', label='重复发现的节点')

    # 设置图表样式
    ax.set_title('HNSW 在大 k 值下的挑战：重复访问', fontsize=18)
    ax.set_xlabel(
        '当剪枝效果减弱时，图的连通性导致从不同路径重复发现同一节点，\n增加了不必要的循环和 `visited` 检查开销。',
        fontsize=12)
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 6)
    ax.legend(loc='upper left')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

    plt.savefig("hnsw_inefficiency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("HNSW 低效示意图已保存为 'hnsw_inefficiency.png'")


# --- 主程序入口 ---
if __name__ == "__main__":
    plot_ivf_inefficiency()
    plot_hnsw_inefficiency()
    print("\n所有示意图已生成完毕。")