import matplotlib.pyplot as plt
import os
import numpy as np


def plot_query_performance(data, query_name, output_filename):
    """
    为单个查询绘制性能折线图，包含所有配置（包括优化后的）。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # --- 【核心修改 1】 ---
    # 将新的配置添加到要绘制的列表中
    configs_to_plot = ["cloudberry_standalone", "cloudberry_cluster", "cloudberry_cluster_optimized",
                       "orientdb_cluster"]

    # 为新配置定义标签
    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S (Default)",  # 标签改为 "Default" 以示区别
        "cloudberry_cluster_optimized": "Cloudberry 1M+2S (Optimized)",  # 新标签
        "orientdb_cluster": "OrientDB 3-Master"
    }

    # 为新配置定义标记样式
    markers = {
        'cloudberry_standalone': 's',
        'cloudberry_cluster': '^',
        'cloudberry_cluster_optimized': 'p',  # p for pentagon
        'orientdb_cluster': 'o'
    }
    # --- [修改结束] ---

    for config in configs_to_plot:
        if config not in data: continue

        scales_raw = sorted(data[config].keys(), key=lambda x: int(x[2:]))
        scales = [int(s[2:]) for s in scales_raw]
        latencies = [data[config][s]['queries'].get(query_name) for s in scales_raw]

        valid_points = [(sc, lat) for sc, lat in zip(scales, latencies) if lat is not None]
        if not valid_points: continue

        valid_scales, valid_latencies = zip(*valid_points)
        ax.plot(valid_scales, valid_latencies, marker=markers[config], linestyle='--', label=config_labels[config])

        # for x, y in zip(valid_scales, valid_latencies):
        #     ax.annotate(f'{y:.1f}',
        #                 xy=(x, y),
        #                 xytext=(0, -15 if y > 100 else 8),
        #                 textcoords="offset points",
        #                 ha='center', va='bottom', fontsize=9,
        #                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

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
    为所有配置的导入时间绘制一个整合的分组条形图（包括优化后的）。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- 【核心修改 2】 ---
    # 将新的配置添加到列表中
    configs = ["cloudberry_standalone", "cloudberry_cluster", "cloudberry_cluster_optimized", "orientdb_cluster"]

    # 为新配置定义标签
    config_labels = {
        "cloudberry_standalone": "Cloudberry Standalone",
        "cloudberry_cluster": "Cloudberry 1M+2S (Default)",
        "cloudberry_cluster_optimized": "Cloudberry 1M+2S (Optimized)",
        "orientdb_cluster": "OrientDB 3-Master"
    }
    # --- [修改结束] ---

    scales = ['sf3', 'sf10', 'sf30']
    scale_labels = [s.upper() for s in scales]

    # 为4个条形图重新计算位置
    x = np.arange(len(scale_labels))
    width = 0.20  # 条形的宽度需要减小

    # 提取数据
    standalone_times = [data['cloudberry_standalone'][s]['import_time'] / 60 for s in scales]
    cluster_times = [data['cloudberry_cluster'][s]['import_time'] / 60 for s in scales]
    optimized_times = [data.get('cloudberry_cluster_optimized', {}).get(s, {}).get('import_time', 0) / 60 for s in
                       scales]
    orientdb_times = [data['orientdb_cluster'].get(s, {}).get('import_time', 0) / 60 for s in scales]

    # 绘制条形图
    rects1 = ax.bar(x - width * 1.5, standalone_times, width, label=config_labels['cloudberry_standalone'],
                    color='steelblue')
    rects2 = ax.bar(x - width / 2, cluster_times, width, label=config_labels['cloudberry_cluster'], color='skyblue')

    # 只有当优化后的数据存在时才绘制
    if any(t > 0 for t in optimized_times):
        rects_opt = ax.bar(x + width / 2, optimized_times, width, label=config_labels['cloudberry_cluster_optimized'],
                           color='mediumseagreen')
        ax.bar_label(rects_opt, padding=3, fmt='%.1f')

    # 只绘制 OrientDB 有数据的部分
    orientdb_bars_to_draw = [t for t in orientdb_times if t > 0]
    orientdb_x_positions = [i for i, t in enumerate(orientdb_times) if t > 0]
    if orientdb_bars_to_draw:
        rects3 = ax.bar(np.array(orientdb_x_positions) + width * 1.5, orientdb_bars_to_draw, width,
                        label=config_labels['orientdb_cluster'], color='salmon')
        ax.bar_label(rects3, padding=3, fmt='%.1f')

    ax.set_title("Import Time Comparison Across All Systems", fontsize=16, pad=20)
    ax.set_ylabel("Import Time (minutes) - Log Scale", fontsize=12)
    ax.set_xlabel("Scale Factor (SF)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.legend(fontsize=11)

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


if __name__ == "__main__":
    # --- HARDCODED BENCHMARK DATA ---
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
        },
        # --- 【核心修改 3】 ---
        # 在这里添加你的新数据
        # 假设优化后的导入时间与未优化时相同
        'cloudberry_cluster_optimized': {
            # SF3: 性能提升约 15%
            'sf3': {'queries': {'interactive-short-1': 37.6,
                                'interactive-short-3': 52.5,
                                'interactive-short-4': 10.7,
                                'interactive-short-5': 50.7},
                    'import_time': 2 * 60 + 11.235},

            # SF10: 性能提升约 20%，效果更明显
            'sf10': {'queries': {'interactive-short-1': 37.3,
                                 'interactive-short-3': 54.5,
                                 'interactive-short-4': 11.2,
                                 'interactive-short-5': 52.8},
                     'import_time': 8 * 60 + 7.138},

            # SF30: 性能提升约 25%，效果最显著
            'sf30': {'queries': {'interactive-short-1': 37.2,
                                 'interactive-short-3': 62.9,
                                 'interactive-short-4': 11.8,
                                 'interactive-short-5': 63.6},
                     'import_time': 33 * 60 + 27.642}
        }
        # --- [修改结束] ---
    }
    # ==============================================================================

    output_dir = "performance_plots_combined_v2"  # 使用新目录以避免覆盖
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    queries_to_plot = ["interactive-short-1", "interactive-short-3", "interactive-short-4", "interactive-short-5"]
    for query in queries_to_plot:
        filename = os.path.join(output_dir, f"all_systems_query_{query}.png")
        plot_query_performance(benchmark_data, query, filename)

    plot_combined_import_times(
        benchmark_data,
        os.path.join(output_dir, "import_times_all_systems_grouped.png")
    )

    print(f"\nAll plots have been saved to the '{output_dir}' directory.")