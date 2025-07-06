import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import re  # Import regex module


# --- Function from Requirement 1 ---
def parse_benchmark_data_with_regex(raw_text_data: str) -> str:
    """Parses raw benchmark output using regex."""
    line_pattern = re.compile(
        r"^(Ours|IVFFlat)\s+\|\s+([\d\.]+)%\s+\|\s+(\d+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)$"
    )
    processed_lines = ["Centroids|Data%|Num Vecs|nprobe|Recall@K|time(s)"]
    for line in raw_text_data.strip().split('\n'):
        match = line_pattern.match(line.strip())
        if match:
            processed_lines.append("|".join(match.groups()))
    return "\n".join(processed_lines)


# --- Raw data from your output ---
raw_data = """
--- Starting Benchmark ---
--- Loading Data ---
Loaded base data: 1000000 vectors, d=1536
Loaded query data: 10000 vectors, d=1536
Loaded Ours centroids: 10000 vectors, d=1536
Loaded IVFFlat centroids: 10000 vectors, d=1536

--- Starting Benchmark ---
Centroids  | Data% |  Num Vecs |   nprobe | Recall@K | time(ms)
============================================================
Loaded ground truth: 10000 queries, k=100
Ours     |   1.0% |     10000 |       24 | 0.1805 | 0.6632
Ours     |   1.0% |     10000 |       48 | 0.2898 | 0.6900
Ours     |   1.0% |     10000 |       96 | 0.4291 | 0.7308
Ours     |   1.0% |     10000 |      192 | 0.5798 | 0.8149
Ours     |   1.0% |     10000 |      384 | 0.7198 | 0.9514
Ours     |   1.0% |     10000 |      768 | 0.8384 | 1.2503
Ours     |   1.0% |     10000 |     1536 | 0.9241 | 1.8859
Ours     |   1.0% |     10000 |     3072 | 0.9729 | 3.4605
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
Ours     |  10.0% |    100000 |       24 | 0.5612 | 0.8918
Ours     |  10.0% |    100000 |       48 | 0.6839 | 1.1346
Ours     |  10.0% |    100000 |       96 | 0.7829 | 1.5457
Ours     |  10.0% |    100000 |      192 | 0.8593 | 2.4021
Ours     |  10.0% |    100000 |      384 | 0.9163 | 4.1024
Ours     |  10.0% |    100000 |      768 | 0.9566 | 7.6156
Ours     |  10.0% |    100000 |     1536 | 0.9815 | 14.8609
Ours     |  10.0% |    100000 |     3072 | 0.9941 | 29.6990
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
Ours     |  20.0% |    200000 |       24 | 0.6586 | 1.0879
Ours     |  20.0% |    200000 |       48 | 0.7612 | 1.4922
Ours     |  20.0% |    200000 |       96 | 0.8400 | 2.2748
Ours     |  20.0% |    200000 |      192 | 0.8994 | 3.8237
Ours     |  20.0% |    200000 |      384 | 0.9424 | 6.8915
Ours     |  20.0% |    200000 |      768 | 0.9709 | 13.1183
Ours     |  20.0% |    200000 |     1536 | 0.9879 | 25.8103
Ours     |  20.0% |    200000 |     3072 | 0.9963 | 51.5832
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
Ours     | 100.0% |   1000000 |       24 | 0.7963 | 2.6318
Ours     | 100.0% |   1000000 |       48 | 0.8602 | 4.4996
Ours     | 100.0% |   1000000 |       96 | 0.9088 | 8.0444
Ours     | 100.0% |   1000000 |      192 | 0.9446 | 14.9506
Ours     | 100.0% |   1000000 |      384 | 0.9698 | 28.5969
Ours     | 100.0% |   1000000 |      768 | 0.9856 | 55.7626
Ours     | 100.0% |   1000000 |     1536 | 0.9944 | 111.6113
Ours     | 100.0% |   1000000 |     3072 | 0.9984 | 226.7082
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
IVFFlat   |   1.0% |     10000 |       16 | 0.1799 | 0.6660
IVFFlat   |   1.0% |     10000 |       32 | 0.2819 | 0.6919
IVFFlat   |   1.0% |     10000 |       64 | 0.4146 | 0.7423
IVFFlat   |   1.0% |     10000 |      128 | 0.5616 | 0.8014
IVFFlat   |   1.0% |     10000 |      256 | 0.7013 | 0.9609
IVFFlat   |   1.0% |     10000 |      512 | 0.8195 | 1.2642
IVFFlat   |   1.0% |     10000 |     1024 | 0.9089 | 1.8084
IVFFlat   |   1.0% |     10000 |     2048 | 0.9641 | 3.1715
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
IVFFlat   |  10.0% |    100000 |       16 | 0.5343 | 0.8936
IVFFlat   |  10.0% |    100000 |       32 | 0.6600 | 1.1268
IVFFlat   |  10.0% |    100000 |       64 | 0.7636 | 1.5750
IVFFlat   |  10.0% |    100000 |      128 | 0.8439 | 2.4949
IVFFlat   |  10.0% |    100000 |      256 | 0.9050 | 4.2708
IVFFlat   |  10.0% |    100000 |      512 | 0.9490 | 7.8266
IVFFlat   |  10.0% |    100000 |     1024 | 0.9764 | 14.8638
IVFFlat   |  10.0% |    100000 |     2048 | 0.9912 | 28.1143
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
IVFFlat   |  20.0% |    200000 |       16 | 0.6299 | 1.1128
IVFFlat   |  20.0% |    200000 |       32 | 0.7394 | 1.5297
IVFFlat   |  20.0% |    200000 |       64 | 0.8246 | 2.3598
IVFFlat   |  20.0% |    200000 |      128 | 0.8884 | 3.9980
IVFFlat   |  20.0% |    200000 |      256 | 0.9344 | 7.1970
IVFFlat   |  20.0% |    200000 |      512 | 0.9659 | 13.5496
IVFFlat   |  20.0% |    200000 |     1024 | 0.9850 | 25.9303
IVFFlat   |  20.0% |    200000 |     2048 | 0.9948 | 49.1966
------------------------------------------------------------
Loaded ground truth: 10000 queries, k=100
IVFFlat   | 100.0% |   1000000 |       16 | 0.7721 | 2.9608
IVFFlat   | 100.0% |   1000000 |       32 | 0.8443 | 5.0529
IVFFlat   | 100.0% |   1000000 |       64 | 0.8987 | 8.6544
IVFFlat   | 100.0% |   1000000 |      128 | 0.9383 | 16.0472
IVFFlat   | 100.0% |   1000000 |      256 | 0.9655 | 31.6435
IVFFlat   | 100.0% |   1000000 |      512 | 0.9830 | 59.0171
IVFFlat   | 100.0% |   1000000 |     1024 | 0.9929 | 115.7241
IVFFlat   | 100.0% |   1000000 |     2048 | 0.9976 | 218.6793
------------------------------------------------------------

Benchmark complete.

进程已结束，退出代码为 0
"""

# --- Use the regex function to parse the data ---
clean_csv_string = parse_benchmark_data_with_regex(raw_data)
df = pd.read_csv(io.StringIO(clean_csv_string), sep="|")

# --- Data Preparation ---
df['Error (1 - Recall)'] = 1.0 - df['Recall@K']
# Replace 0 with a very small number to avoid log(0) issues
df['Error (1 - Recall)'] = df['Error (1 - Recall)'].replace(0, 1e-6)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14, 10))

unique_num_vecs = sorted(df['Num Vecs'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_num_vecs)))
color_map = {num_vecs: color for num_vecs, color in zip(unique_num_vecs, colors)}
style_map = {'Ours': '-', 'IVFFlat': '--'}


def format_num_vecs(n):
    if n >= 1000000:
        return f"{n // 1000000}M"
    else:
        return f"{n // 1000}k"


# --- Main plotting loop ---
for (centroid_type, num_vecs), group in df.groupby(['Centroids', 'Num Vecs']):
    label = f"{centroid_type} ({format_num_vecs(num_vecs)} vecs)"
    color = color_map[num_vecs]
    linestyle = style_map[centroid_type]

    # Sort by time to ensure the line is drawn correctly
    group = group.sort_values(by='time(s)')

    # Plot the line: x-axis is 'time(s)', y-axis is 'Error (1 - Recall)'
    ax.plot(group['time(s)'], group['Error (1 - Recall)'],
            label=label,
            color=color,
            linestyle=linestyle,
            marker='o',
            markersize=5)

    # Add data point annotations (the recall value)
    for _, row in group.iterrows():
        ax.text(row['time(s)'], row['Error (1 - Recall)'],
                f" {row['Recall@K']:.4f}",  # Add a space for padding
                fontsize=8,
                ha='left',
                va='center')

# --- Chart finalization ---
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Recall vs. Query Time (Speed-Accuracy Trade-off)', fontsize=16)
ax.set_xlabel('Query Time in Seconds (Log Scale)', fontsize=12)
ax.set_ylabel('Error Rate: 1 - Recall@100 (Log Scale)', fontsize=12)
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
ax.legend(title='Centroid Type (Data Size)', bbox_to_anchor=(1.04, 1), loc="upper left")

# Invert Y-axis so that higher recall (lower error) is at the top
ax.invert_yaxis()

# Format ticks to be non-scientific for better readability
from matplotlib.ticker import ScalarFormatter

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()