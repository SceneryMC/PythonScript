import matplotlib.pyplot as plt
import os
import numpy as np  # Import numpy for grouped bar chart calculations


def plot_query_performance(data, query_name, configs, title_prefix, output_filename):
    """
    为单个查询绘制性能随数据规模变化的折线图。
    为 Cloudberry 图表的数据点添加精确数值标签。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    config_labels = {
        "cloudberry_cluster": "Cloudberry 1M+2S",
        "cloudberry_standalone": "Cloudberry Standalone",
        "orientdb_cluster": "OrientDB 3-Master"
    }

    for config in configs:
        if config not in data: continue

        scales_raw = sorted(data[config].keys(), key=lambda x: int(x[2:]))
        scales = [int(s[2:]) for s in scales_raw]
        latencies = [data[config][s]['queries'].get(query_name) for s in scales_raw]

        valid_points = [(sc, lat) for sc, lat in zip(scales, latencies) if lat is not None]
        if not valid_points: continue

        valid_scales, valid_latencies = zip(*valid_points)
        ax.plot(valid_scales, valid_latencies, marker='o', linestyle='-', label=config_labels[config])

        # --- MODIFICATION 1: Add data labels for Cloudberry plots ---
        if "cloudberry" in config:
            for x, y in zip(valid_scales, valid_latencies):
                ax.annotate(f'{y:.1f}',  # Format to one decimal place
                            xy=(x, y),
                            xytext=(0, 8),  # 8 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    ax.set_title(f"{title_prefix} Performance: {query_name}", fontsize=16, pad=20)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)
    ax.set_ylabel("Average Latency (ms) - Log Scale", fontsize=12)
    ax.set_xticks([3, 10, 30])
    ax.legend(fontsize=11)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def plot_grouped_import_times(data, title, output_filename):
    """
    为 Cloudberry 导入时间绘制分组条形图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S"
    }

    scales = sorted(data['cloudberry_standalone'].keys(), key=lambda x: int(x[2:]))
    scale_labels = [s.upper() for s in scales]

    standalone_times = [data['cloudberry_standalone'][s]['import_time'] / 60 for s in scales]
    cluster_times = [data['cloudberry_cluster'][s]['import_time'] / 60 for s in scales]

    x = np.arange(len(scale_labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width / 2, standalone_times, width, label=config_labels['cloudberry_standalone'],
                    color='steelblue')
    rects2 = ax.bar(x + width / 2, cluster_times, width, label=config_labels['cloudberry_cluster'], color='skyblue')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("Import Time (minutes)", fontsize=12)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def plot_single_import_times(data, config, title, output_filename):
    """
    为单个配置的导入时间绘制条形图 (用于 OrientDB)。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    config_label = "OrientDB 3-Master"

    scales_raw = sorted(data[config].keys(), key=lambda x: int(x[2:]))
    labels = [s.upper() for s in scales_raw]
    values = [data[config][s]['import_time'] / 60 for s in scales_raw]

    bars = ax.bar(labels, values, color='salmon')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("Import Time (minutes)", fontsize=12)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)

    ax.bar_label(bars, padding=3, fmt='%.1f')

    plt.tight_layout()
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

    output_dir = "performance_plots_separate"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 生成查询性能图
    queries_to_plot = ["interactive-short-1", "interactive-short-3", "interactive-short-4", "interactive-short-5"]

    # Cloudberry 图表 (现在会带数值标签)
    cloudberry_configs = ["cloudberry_standalone", "cloudberry_cluster"]
    for query in queries_to_plot:
        filename = os.path.join(output_dir, f"cloudberry_query_{query}.png")
        plot_query_performance(benchmark_data, query, cloudberry_configs, "Cloudberry", filename)

    # OrientDB 图表 (保持不变)
    orientdb_configs = ["orientdb_cluster"]
    for query in queries_to_plot:
        filename = os.path.join(output_dir, f"orientdb_query_{query}.png")
        plot_query_performance(benchmark_data, query, orientdb_configs, "OrientDB", filename)

    # 2. 生成导入时间图
    # 调用新的分组条形图函数
    plot_grouped_import_times(
        benchmark_data,
        "Cloudberry Import Times (Standalone vs. Cluster)",
        os.path.join(output_dir, "import_times_cloudberry_grouped.png")
    )
    # 调用旧的、简单的条形图函数
    plot_single_import_times(
        benchmark_data,
        "orientdb_cluster",
        "OrientDB Import Times",
        os.path.join(output_dir, "import_times_orientdb.png")
    )

    print(f"\nAll plots have been saved to the '{output_dir}' directory.")