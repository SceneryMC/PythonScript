import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import re


# 1. Regex-based parsing function (same as before)
def parse_benchmark_data_with_regex(raw_text_data: str) -> str:
    """Parses raw benchmark output using regex."""
    line_pattern = re.compile(
        r"^(Ours|IVFFlat)\s+\|\s+([\d\.]+)%\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)$"
    )
    processed_lines = ["Centroids|Data%|Num Vecs|nprobe|Recall@K|time(s)"]
    for line in raw_text_data.strip().split('\n'):
        match = line_pattern.match(line.strip())
        if match:
            processed_lines.append("|".join(match.groups()))
    return "\n".join(processed_lines)


# 2. Raw data string (same as before)
raw_data = """
--- Starting Benchmark ---
Centroids  | Data% |  Num Vecs |   nprobe | Recall@K | time(s)
============================================================
Loaded ground truth: 10000 queries, k=100
Ours     |   1.0% |     10000 |       16 | 0.1328 | 0.6752
Ours     |   1.0% |     10000 |       32 | 0.2214 | 0.6722
Ours     |   1.0% |     10000 |       64 | 0.3446 | 0.7094
Ours     |   1.0% |     10000 |      128 | 0.4915 | 0.7470
Ours     |   1.0% |     10000 |      256 | 0.6401 | 0.8447
Ours     |   1.0% |     10000 |      512 | 0.7719 | 1.0477
Ours     |   1.0% |     10000 |     1024 | 0.8790 | 1.4278
Ours     |   1.0% |     10000 |     2048 | 0.9483 | 2.3512
------------------------------------------------------------
Ours     |  10.0% |    100000 |       16 | 0.4818 | 0.7876
Ours     |  10.0% |    100000 |       32 | 0.6152 | 0.9421
Ours     |  10.0% |    100000 |       64 | 0.7275 | 1.2262
Ours     |  10.0% |    100000 |      128 | 0.8172 | 1.7884
Ours     |  10.0% |    100000 |      256 | 0.8851 | 2.8794
Ours     |  10.0% |    100000 |      512 | 0.9351 | 5.1012
Ours     |  10.0% |    100000 |     1024 | 0.9689 | 9.6886
Ours     |  10.0% |    100000 |     2048 | 0.9880 | 19.0212
------------------------------------------------------------
Ours     |  20.0% |    200000 |       16 | 0.5880 | 0.9514
Ours     |  20.0% |    200000 |       32 | 0.7049 | 1.2405
Ours     |  20.0% |    200000 |       64 | 0.7963 | 1.7195
Ours     |  20.0% |    200000 |      128 | 0.8666 | 2.7488
Ours     |  20.0% |    200000 |      256 | 0.9194 | 4.8050
Ours     |  20.0% |    200000 |      512 | 0.9558 | 8.8107
Ours     |  20.0% |    200000 |     1024 | 0.9792 | 17.0108
Ours     |  20.0% |    200000 |     2048 | 0.9924 | 33.7545
------------------------------------------------------------
Ours     | 100.0% |   1000000 |       16 | 0.7499 | 2.0433
Ours     | 100.0% |   1000000 |       32 | 0.8255 | 3.3416
Ours     | 100.0% |   1000000 |       64 | 0.8819 | 5.9060
Ours     | 100.0% |   1000000 |      128 | 0.9251 | 10.5746
Ours     | 100.0% |   1000000 |      256 | 0.9562 | 19.4496
Ours     | 100.0% |   1000000 |      512 | 0.9773 | 37.5715
Ours     | 100.0% |   1000000 |     1024 | 0.9901 | 75.4539
Ours     | 100.0% |   1000000 |     2048 | 0.9965 | 148.2037
------------------------------------------------------------
IVFFlat   |   1.0% |     10000 |       16 | 0.1799 | 0.6660
IVFFlat   |   1.0% |     10000 |       32 | 0.2819 | 0.6919
IVFFlat   |   1.0% |     10000 |       64 | 0.4146 | 0.7423
IVFFlat   |   1.0% |     10000 |      128 | 0.5616 | 0.8014
IVFFlat   |   1.0% |     10000 |      256 | 0.7013 | 0.9609
IVFFlat   |   1.0% |     10000 |      512 | 0.8195 | 1.2642
IVFFlat   |   1.0% |     10000 |     1024 | 0.9089 | 1.8084
IVFFlat   |   1.0% |     10000 |     2048 | 0.9641 | 3.1715
------------------------------------------------------------
IVFFlat   |  10.0% |    100000 |       16 | 0.5343 | 0.8936
IVFFlat   |  10.0% |    100000 |       32 | 0.6600 | 1.1268
IVFFlat   |  10.0% |    100000 |       64 | 0.7636 | 1.5750
IVFFlat   |  10.0% |    100000 |      128 | 0.8439 | 2.4949
IVFFlat   |  10.0% |    100000 |      256 | 0.9050 | 4.2708
IVFFlat   |  10.0% |    100000 |      512 | 0.9490 | 7.8266
IVFFlat   |  10.0% |    100000 |     1024 | 0.9764 | 14.8638
IVFFlat   |  10.0% |    100000 |     2048 | 0.9912 | 28.1143
------------------------------------------------------------
IVFFlat   |  20.0% |    200000 |       16 | 0.6299 | 1.1128
IVFFlat   |  20.0% |    200000 |       32 | 0.7394 | 1.5297
IVFFlat   |  20.0% |    200000 |       64 | 0.8246 | 2.3598
IVFFlat   |  20.0% |    200000 |      128 | 0.8884 | 3.9980
IVFFlat   |  20.0% |    200000 |      256 | 0.9344 | 7.1970
IVFFlat   |  20.0% |    200000 |      512 | 0.9659 | 13.5496
IVFFlat   |  20.0% |    200000 |     1024 | 0.9850 | 25.9303
IVFFlat   |  20.0% |    200000 |     2048 | 0.9948 | 49.1966
------------------------------------------------------------
IVFFlat   | 100.0% |   1000000 |       16 | 0.7721 | 2.9608
IVFFlat   | 100.0% |   1000000 |       32 | 0.8443 | 5.0529
IVFFlat   | 100.0% |   1000000 |       64 | 0.8987 | 8.6544
IVFFlat   | 100.0% |   1000000 |      128 | 0.9383 | 16.0472
IVFFlat   | 100.0% |   1000000 |      256 | 0.9655 | 31.6435
IVFFlat   | 100.0% |   1000000 |      512 | 0.9830 | 59.0171
IVFFlat   | 100.0% |   1000000 |     1024 | 0.9929 | 115.7241
IVFFlat   | 100.0% |   1000000 |     2048 | 0.9976 | 218.6793
"""

# 3. Parse and prepare data
clean_csv_string = parse_benchmark_data_with_regex(raw_data)
df = pd.read_csv(io.StringIO(clean_csv_string), sep="|")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14, 10))

# Color and style maps
unique_num_vecs = sorted(df['Num Vecs'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_num_vecs)))
color_map = {num_vecs: color for num_vecs, color in zip(unique_num_vecs, colors)}
style_map = {'Ours': '-', 'IVFFlat': '--'}


def format_num_vecs(n):
    if n >= 1000000:
        return f"{n // 1000000}M"
    else:
        return f"{n // 1000}k"


# 4. Main plotting loop
for (centroid_type, num_vecs), group in df.groupby(['Centroids', 'Num Vecs']):
    label = f"{centroid_type} ({format_num_vecs(num_vecs)} vecs)"
    color = color_map[num_vecs]
    linestyle = style_map[centroid_type]

    # Sort by time to ensure the line is drawn correctly from left to right
    group = group.sort_values(by='time(s)')

    # --- MODIFICATION: Use 'Recall@K' for Y-axis directly ---
    ax.plot(group['time(s)'], group['Recall@K'],
            label=label,
            color=color,
            linestyle=linestyle,
            marker='o',
            markersize=5)

    # Add data point annotations
    for _, row in group.iterrows():
        # Annotate with the recall value
        ax.text(row['time(s)'], row['Recall@K'],
                f" {row['Recall@K']:.4f}",  # Add a space for padding
                fontsize=8,
                ha='left',
                va='center')

# 5. Chart finalization
ax.set_xscale('log')  # X-axis remains log scale for time
# Y-axis is now linear scale
ax.set_title('Recall vs. Query Time (Speed-Accuracy Trade-off)', fontsize=16)
ax.set_xlabel('Query Time in Seconds (Log Scale)', fontsize=12)
ax.set_ylabel('Recall@100 (Linear Scale)', fontsize=12)  # Updated Y-axis label
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
ax.legend(title='Centroid Type (Data Size)', bbox_to_anchor=(1.04, 1), loc="upper left")

# Set Y-axis limits for better presentation
ax.set_ylim(0, 1.05)

# Format X-axis ticks to be non-scientific
from matplotlib.ticker import ScalarFormatter

ax.xaxis.set_major_formatter(ScalarFormatter())

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()