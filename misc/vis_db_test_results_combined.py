import matplotlib.pyplot as plt
import os
import numpy as np  # 用于分组条形图计算


def plot_query_performance(data, query_name, output_filename):
    """
    为单个查询绘制性能折线图，包含所有三种配置。
    所有数据点都标注精确数值。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))

    configs_to_plot = ["cloudberry_standalone", "cloudberry_cluster", "orientdb_cluster"]
    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S",
        "orientdb_cluster": "OrientDB 3-Master"
    }
    markers = {'cloudberry_standalone': 's', 'cloudberry_cluster': '^', 'orientdb_cluster': 'o'}

    for config in configs_to_plot:
        if config not in data: continue

        scales_raw = sorted(data[config].keys(), key=lambda x: int(x[2:]))
        scales = [int(s[2:]) for s in scales_raw]
        latencies = [data[config][s]['queries'].get(query_name) for s in scales_raw]

        valid_points = [(sc, lat) for sc, lat in zip(scales, latencies) if lat is not None]
        if not valid_points: continue

        valid_scales, valid_latencies = zip(*valid_points)
        ax.plot(valid_scales, valid_latencies, marker=markers[config], linestyle='--', label=config_labels[config])

        # 为所有数据点添加数值标签
        for x, y in zip(valid_scales, valid_latencies):
            ax.annotate(f'{y:.1f}',
                        xy=(x, y),
                        xytext=(0, -15 if y > 100 else 8),  # 优化标签位置，避免重叠
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    ax.set_title(f"Query Performance Comparison: {query_name}", fontsize=16, pad=20)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)
    ax.set_ylabel("Average Latency (ms) - Log Scale", fontsize=12)
    ax.set_xticks([3, 10, 30])
    ax.legend(fontsize=11)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def plot_combined_import_times(data, output_filename):
    """
    为所有配置的导入时间绘制一个整合的分组条形图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    configs = ["cloudberry_standalone", "cloudberry_cluster", "orientdb_cluster"]
    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S",
        "orientdb_cluster": "OrientDB 3-Master"
    }

    scales = ['sf3', 'sf10', 'sf30']
    scale_labels = [s.upper() for s in scales]

    x = np.arange(len(scale_labels))
    width = 0.25  # 条形的宽度

    # --- 数据提取与对齐 ---
    # Cloudberry數據是完整的
    standalone_times = [data['cloudberry_standalone'][s]['import_time'] / 60 for s in scales]
    cluster_times = [data['cloudberry_cluster'][s]['import_time'] / 60 for s in scales]

    # OrientDB數據是不完整的，我們需要處理缺失值
    orientdb_times = [data['orientdb_cluster'].get(s, {}).get('import_time', 0) / 60 for s in scales]

    rects1 = ax.bar(x - width, standalone_times, width, label=config_labels['cloudberry_standalone'], color='steelblue')
    rects2 = ax.bar(x, cluster_times, width, label=config_labels['cloudberry_cluster'], color='skyblue')

    # 只绘制 OrientDB 有数据的部分
    orientdb_bars_to_draw = [t for t in orientdb_times if t > 0]
    orientdb_x_positions = [i for i, t in enumerate(orientdb_times) if t > 0]
    rects3 = ax.bar(np.array(orientdb_x_positions) + width, orientdb_bars_to_draw, width,
                    label=config_labels['orientdb_cluster'], color='salmon')

    ax.set_title("Import Time Comparison Across All Systems", fontsize=16, pad=20)
    ax.set_ylabel("Import Time (minutes) - Log Scale", fontsize=12)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.legend(fontsize=11)
    # ax.set_yscale('log')  # 使用对数坐标

    # 为所有条形图添加数值标签
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    ax.bar_label(rects3, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


if __name__ == "__main__":
    # --- HARDCODED BENCHMARK DATA (保持不变) ---
    benchmark_data = {
        'cloudberry_cluster': {
            'sf3': {'queries': {'interactive-short-1': 44.2278, 'interactive-short-3': 61.7537,
                                'interactive-short-4': 12.6462, 'interactive-short-5': 59.5887},
                    'import_time': 2 * 60 + 11.235},
            'sf10': {'queries': {'interactive-short-1': 42.8438, 'interactive-short-3': 66.8512,
                                 'interactive-short-4': 12.7871, 'interactive-short-5': 64.6835},
                     'import_time': 8 * 60 + 7.138},
            'sf30': {'queries': {'interactive-short-1': 40.4668, 'interactive-short-3': 81.2129,
                                 'interactive-short-4': 13.0830, 'interactive-short-5': 80.8054},
                     'import_time': 33 * 60 + 27.642}
        },
        'cloudberry_standalone': {
            'sf3': {
                'queries': {'interactive-short-1': 3.0912, 'interactive-short-3': 4.5771, 'interactive-short-4': 0.5768,
                            'interactive-short-5': 6.8944}, 'import_time': 6 * 60 + 2.530},
            'sf10': {'queries': {'interactive-short-1': 3.3774, 'interactive-short-3': 10.5569,
                                 'interactive-short-4': 0.5841, 'interactive-short-5': 15.9820},
                     'import_time': 20 * 60 + 50.359},
            'sf30': {'queries': {'interactive-short-1': 3.2040, 'interactive-short-3': 22.4126,
                                 'interactive-short-4': 0.6275, 'interactive-short-5': 38.3081},
                     'import_time': 82 * 60 + 46.005}
        },
        'orientdb_cluster': {
            'sf3': {
                'queries': {'interactive-short-1': 1.2226, 'interactive-short-3': 2.9551, 'interactive-short-4': 0.5485,
                            'interactive-short-5': 1.0701}, 'import_time': 141 * 60 + 6.621},
            'sf10': {
                'queries': {'interactive-short-1': 1.2659, 'interactive-short-3': 4.0499, 'interactive-short-4': 0.5693,
                            'interactive-short-5': 1.1582}, 'import_time': 455 * 60 + 4.815}
        }
    }
    # ==============================================================================

    output_dir = "performance_plots_combined"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 生成查询性能图 (每张图包含所有配置)
    queries_to_plot = ["interactive-short-1", "interactive-short-3", "interactive-short-4", "interactive-short-5"]
    for query in queries_to_plot:
        filename = os.path.join(output_dir, f"all_systems_query_{query}.png")
        plot_query_performance(benchmark_data, query, filename)

    # 2. 生成一个包含所有配置的导入时间图
    plot_combined_import_times(
        benchmark_data,
        os.path.join(output_dir, "import_times_all_systems_grouped.png")
    )

    print(f"\nAll plots have been saved to the '{output_dir}' directory.")