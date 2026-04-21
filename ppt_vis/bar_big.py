import matplotlib.pyplot as plt
import os
import numpy as np


def plot_combined_import_times(data, output_filename):
    """
    为所有配置的导入时间绘制一个整合的分组条形图。
    优化版：字号放大、条形加粗、高对比度、远距离清晰可见。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- 视觉样式统一配置 (Presentation Ready) ---
    TITLE_SIZE = 22
    LABEL_SIZE = 18
    TICK_SIZE = 16
    LEGEND_SIZE = 16
    ANNOTATION_SIZE = 15
    BAR_WIDTH = 0.28  # 稍微加宽柱子，使其看起来更粗壮

    # 使用对比度更强、更适合投影的颜色，并加上黑色边框提升锐利度
    COLORS = {
        'cloudberry_standalone': '#1f77b4',  # 强深蓝色
        'cloudberry_cluster': '#ff7f0e',  # 亮橙色
        'orientdb_cluster': '#d62728'  # 鲜红色
    }

    configs = ["cloudberry_standalone", "cloudberry_cluster", "orientdb_cluster"]
    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S",
        "orientdb_cluster": "OrientDB 3-Master"
    }

    scales = ['sf3', 'sf10', 'sf30']
    scale_labels = [s.upper() for s in scales]
    x = np.arange(len(scale_labels))

    # --- 数据提取与对齐 ---
    standalone_times = [data['cloudberry_standalone'][s]['import_time'] / 60 for s in scales]
    cluster_times = [data['cloudberry_cluster'][s]['import_time'] / 60 for s in scales]
    orientdb_times = [data['orientdb_cluster'].get(s, {}).get('import_time', 0) / 60 for s in scales]

    # --- 绘制柱状图 (带有黑色描边 edgecolor='black') ---
    rects1 = ax.bar(x - BAR_WIDTH, standalone_times, BAR_WIDTH,
                    label=config_labels['cloudberry_standalone'],
                    color=COLORS['cloudberry_standalone'], edgecolor='black', linewidth=1.2)

    rects2 = ax.bar(x, cluster_times, BAR_WIDTH,
                    label=config_labels['cloudberry_cluster'],
                    color=COLORS['cloudberry_cluster'], edgecolor='black', linewidth=1.2)

    # 只绘制 OrientDB 有数据的部分
    orientdb_bars_to_draw = [t for t in orientdb_times if t > 0]
    orientdb_x_positions = [i for i, t in enumerate(orientdb_times) if t > 0]
    rects3 = ax.bar(np.array(orientdb_x_positions) + BAR_WIDTH, orientdb_bars_to_draw, BAR_WIDTH,
                    label=config_labels['orientdb_cluster'],
                    color=COLORS['orientdb_cluster'], edgecolor='black', linewidth=1.2)

    # --- 设置文本与标签 (加粗加大) ---
    ax.set_title("Import Time Comparison Across All Systems", fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.set_ylabel("Import Time (minutes)", fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel("Scale Factor (SF)", fontsize=LABEL_SIZE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels, fontsize=TICK_SIZE, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)

    # 优化网格线：只保留 Y 轴网格，并且加粗虚线，不干扰柱子
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(axis='x', visible=False)

    # 图例设置：移到左上角空白处，带白底黑框，防止遮挡线条
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

    # --- 添加柱顶数值标签 ---
    # weight='bold' 让数字在远处更容易辨认
    ax.bar_label(rects1, padding=5, fmt='%.1f', fontsize=ANNOTATION_SIZE, fontweight='bold')
    ax.bar_label(rects2, padding=5, fmt='%.1f', fontsize=ANNOTATION_SIZE, fontweight='bold')
    ax.bar_label(rects3, padding=5, fmt='%.1f', fontsize=ANNOTATION_SIZE, fontweight='bold')

    # --- 自动扩展 Y 轴顶部空间 ---
    # 防止最高柱子（OrientDB 455分钟）顶部的数字标签被图表边缘切掉
    max_y_val = max(max(standalone_times), max(cluster_times), max(orientdb_times))
    ax.set_ylim(0, max_y_val * 1.15)  # 增加 15% 的头部空间

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")