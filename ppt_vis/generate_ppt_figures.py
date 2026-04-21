import matplotlib.pyplot as plt
import networkx as nx


def create_final_community_graph():
    """
    Generates a single, complex community graph with a transparent background.
    """
    # 1. 创建一个指定大小的图形对象。
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2. 【核心】明确地将子图(Axes)的背景设置为透明。
    #    'none' 或 'w' (白色) 都可以，但配合savefig的transparent=True，效果最好。
    ax.set_facecolor('none')

    # 3. 创建一个包含5个社区的、更复杂的图。
    num_communities = 5
    nodes_per_community = 5
    G = nx.connected_caveman_graph(num_communities, nodes_per_community)

    # 4. 为每个社区分配不同的颜色。
    color_map = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728', '#2ca02c']  # Blue, Purple, Orange, Red, Green
    colors = []
    for i in range(num_communities):
        colors.extend([color_map[i]] * nodes_per_community)

    # 5. 生成节点布局。
    pos = nx.spring_layout(G, seed=42)

    # 6. 绘制图形。
    nx.draw(
        G,
        pos,
        node_color=colors,
        ax=ax,
        node_size=200,  # 稍微增大节点以便观察
        width=1.5  # 加粗边
    )

    # 7. 关闭坐标轴显示。
    ax.axis('off')

    # 8. 【核心】保存图形，并强制背景透明。
    #    - transparent=True 是关键指令。
    #    - bbox_inches='tight', pad_inches=0 会裁剪掉所有多余的空白边缘。
    plt.savefig(
        "final_community_graph.png",
        dpi=150,
        transparent=True,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)  # 关闭图形，释放内存
    print("已成功生成 'final_community_graph.png'")


# --- 主程序入口 ---
if __name__ == "__main__":
    create_final_community_graph()