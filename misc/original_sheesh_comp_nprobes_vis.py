import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# 1. 将您提供的原始数据粘贴为多行字符串
data_string = """
Centroids  | Data% |  Num Vecs |   nprobe | Recall@K
SHEESH     |   1.0% |     10000 |       16 | 0.1313
SHEESH     |   1.0% |     10000 |       32 | 0.2196
SHEESH     |   1.0% |     10000 |       64 | 0.3428
SHEESH     |   1.0% |     10000 |      128 | 0.4911
SHEESH     |   1.0% |     10000 |      256 | 0.6406
SHEESH     |   1.0% |     10000 |      512 | 0.7735
SHEESH     |   1.0% |     10000 |     1024 | 0.8789
SHEESH     |   1.0% |     10000 |     2048 | 0.9483
SHEESH     |  10.0% |    100000 |       16 | 0.4793
SHEESH     |  10.0% |    100000 |       32 | 0.6123
SHEESH     |  10.0% |    100000 |       64 | 0.7261
SHEESH     |  10.0% |    100000 |      128 | 0.8167
SHEESH     |  10.0% |    100000 |      256 | 0.8852
SHEESH     |  10.0% |    100000 |      512 | 0.9358
SHEESH     |  10.0% |    100000 |     1024 | 0.9687
SHEESH     |  10.0% |    100000 |     2048 | 0.9879
SHEESH     |  20.0% |    200000 |       16 | 0.5857
SHEESH     |  20.0% |    200000 |       32 | 0.7025
SHEESH     |  20.0% |    200000 |       64 | 0.7952
SHEESH     |  20.0% |    200000 |      128 | 0.8668
SHEESH     |  20.0% |    200000 |      256 | 0.9193
SHEESH     |  20.0% |    200000 |      512 | 0.9561
SHEESH     |  20.0% |    200000 |     1024 | 0.9793
SHEESH     |  20.0% |    200000 |     2048 | 0.9923
SHEESH     | 100.0% |   1000000 |       16 | 0.7503
SHEESH     | 100.0% |   1000000 |       32 | 0.8251
SHEESH     | 100.0% |   1000000 |       64 | 0.8826
SHEESH     | 100.0% |   1000000 |      128 | 0.9259
SHEESH     | 100.0% |   1000000 |      256 | 0.9572
SHEESH     | 100.0% |   1000000 |      512 | 0.9779
SHEESH     | 100.0% |   1000000 |     1024 | 0.9904
SHEESH     | 100.0% |   1000000 |     2048 | 0.9967
Original   |   1.0% |     10000 |       16 | 0.1799
Original   |   1.0% |     10000 |       32 | 0.2819
Original   |   1.0% |     10000 |       64 | 0.4146
Original   |   1.0% |     10000 |      128 | 0.5616
Original   |   1.0% |     10000 |      256 | 0.7013
Original   |   1.0% |     10000 |      512 | 0.8195
Original   |   1.0% |     10000 |     1024 | 0.9089
Original   |   1.0% |     10000 |     2048 | 0.9641
Original   |  10.0% |    100000 |       16 | 0.5343
Original   |  10.0% |    100000 |       32 | 0.6600
Original   |  10.0% |    100000 |       64 | 0.7636
Original   |  10.0% |    100000 |      128 | 0.8439
Original   |  10.0% |    100000 |      256 | 0.9050
Original   |  10.0% |    100000 |      512 | 0.9490
Original   |  10.0% |    100000 |     1024 | 0.9764
Original   |  10.0% |    100000 |     2048 | 0.9912
Original   |  20.0% |    200000 |       16 | 0.6299
Original   |  20.0% |    200000 |       32 | 0.7394
Original   |  20.0% |    200000 |       64 | 0.8246
Original   |  20.0% |    200000 |      128 | 0.8884
Original   |  20.0% |    200000 |      256 | 0.9344
Original   |  20.0% |    200000 |      512 | 0.9659
Original   |  20.0% |    200000 |     1024 | 0.9850
Original   |  20.0% |    200000 |     2048 | 0.9948
Original   | 100.0% |   1000000 |       16 | 0.7721
Original   | 100.0% |   1000000 |       32 | 0.8443
Original   | 100.0% |   1000000 |       64 | 0.8987
Original   | 100.0% |   1000000 |      128 | 0.9383
Original   | 100.0% |   1000000 |      256 | 0.9655
Original   | 100.0% |   1000000 |      512 | 0.9830
Original   | 100.0% |   1000000 |     1024 | 0.9929
Original   | 100.0% |   1000000 |     2048 | 0.9975
"""

# 2. 使用 pandas 读取和清理数据
# io.StringIO 将字符串模拟成文件，方便 read_csv 读取
df = pd.read_csv(io.StringIO(data_string), sep="|", skipinitialspace=True)

# 清理列名和数据
df.columns = [col.strip() for col in df.columns]
df['Centroids'] = df['Centroids'].str.strip()
# 计算 Y 轴的值 (1 - recall)，并处理 recall=1 的情况以避免 log(0)
df['Error (1 - Recall)'] = 1.0 - df['Recall@K']
df['Error (1 - Recall)'] = df['Error (1 - Recall)'].replace(0, 1e-6)  # 替换0为一个极小值

# 3. 准备绘图
fig, ax = plt.subplots(figsize=(14, 10))

# 定义颜色和线型
unique_num_vecs = sorted(df['Num Vecs'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_num_vecs)))
color_map = {num_vecs: color for num_vecs, color in zip(unique_num_vecs, colors)}
style_map = {'SHEESH': '-', 'Original': '--'}


# 辅助函数，用于格式化图例标签
def format_num_vecs(n):
    if n >= 1000000:
        return f"{n // 1000000}M"
    else:
        return f"{n // 1000}k"


# 4. 循环遍历每个数据系列并绘图
for (centroid_type, num_vecs), group in df.groupby(['Centroids', 'Num Vecs']):
    label = f"{centroid_type} ({format_num_vecs(num_vecs)} vecs)"
    color = color_map[num_vecs]
    linestyle = style_map[centroid_type]

    # 绘制折线
    ax.plot(group['nprobe'], group['Error (1 - Recall)'],
            label=label,
            color=color,
            linestyle=linestyle,
            marker='o',
            markersize=5)

    # 添加数据点标注
    for _, row in group.iterrows():
        ax.text(row['nprobe'], row['Error (1 - Recall)'],
                f"{row['Recall@K']:.4f}",
                fontsize=8,
                ha='left', va='bottom',  # 调整文本位置，避免与点重叠
                rotation=30)

# 5. 设置图表属性
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('IVF Recall vs. Nprobe for SHEESH and Original Centroids', fontsize=16)
ax.set_xlabel('Nprobe (Log Scale)', fontsize=12)
ax.set_ylabel('Error Rate: 1 - Recall@100 (Log Scale)', fontsize=12)
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
ax.legend(title='Centroid Type (Data Size)', bbox_to_anchor=(1.04, 1), loc="upper left")

# 反转 Y 轴，使得 recall 更高的点位于图表上方，更符合直觉
ax.invert_yaxis()

# 调整刻度标签格式，避免科学计数法
from matplotlib.ticker import ScalarFormatter

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局，为图例留出空间
plt.show()