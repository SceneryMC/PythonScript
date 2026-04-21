import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

# ==========================================
# 1. 基础设置
# ==========================================
sns.set_theme(style="whitegrid")
# 字体设置 (Windows: SimHei, Mac: Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据录入 (前20名完整数据)
# ==========================================
raw_data = [
    ['BIT-Vector', 19366.00, 92],
    ['505nb', 6422.94, 81],
    ['鼠鼠不想写内核', 13158.00, 216],
    ['为什么要悲观啊!', 11069.67, 191],
    ['小骑士', 7678.79, 145],
    ['7eleven', 13806.96, 264],
    ['PolarCode', 10867.46, 215],
    ['gravity', 8765.69, 186],
    ['拾柒', 13704.54, 311],
    ['V1rgin', 1800.83, 48.9],
    ['obNuts', 8965.53, 283],
    ['g4197', 7242.62, 238],
    ['油巴图', 8257.32, 289],
    ['咕噜咕噜', 11197.05, 402],
    ['怕累脱极限队', 3388.95, 165],
    ['SimpleHappy', 6092.85, 503],
    ['哈基米', 440.75, 40.5],
    ['Ali爸爸', 3048.93, 288],
    ['静以修身', 2945.10, 286],
    ['哦对的对的', 2922.66, 288],
]

my_team_name = 'BIT-Vector'

# ==========================================
# 3. 数据处理与基线计算
# ==========================================
df = pd.DataFrame(raw_data, columns=['Team', 'QPS', 'BuildTime'])
df['Rank'] = df.index + 1

# 计算基线 (Top20 平均)
top20_df = df.iloc[:20]
baseline_qps = top20_df['QPS'].mean()
baseline_build_time = top20_df['BuildTime'].mean()

# 计算加速比
df['QPS_Speedup'] = df['QPS'] / baseline_qps
df['Build_Speedup'] = baseline_build_time / df['BuildTime']

# 数据分层
my_team = df[df['Team'] == my_team_name]
runner_ups = df[(df['Rank'].isin([2, 3]))]
others = df[(df['Rank'] > 3)]

# ==========================================
# 4. 绘图逻辑
# ==========================================
plt.figure(figsize=(14, 10), dpi=300)

# A. 绘制基线
plt.axhline(y=1, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.6)
plt.axvline(x=1, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.6)
plt.text(0.1, 1.05, f'', color='#7f8c8d', fontsize=14, fontweight='bold')

# B. 绘制其他队伍
plt.scatter(
    x=others['Build_Speedup'],
    y=others['QPS_Speedup'],
    c='#bdc3c7',
    s=200,
    alpha=0.5,
    label='其他队伍',
    zorder=2
)

# C. 绘制第二、第三名
plt.scatter(
    x=runner_ups['Build_Speedup'],
    y=runner_ups['QPS_Speedup'],
    c='#e67e22',
    s=350,
    edgecolors='white',
    linewidths=1.5,
    alpha=0.9,
    label='Rank 2 & 3',
    zorder=5
)

# D. 绘制我的队伍 [修正1：去掉白边]
plt.scatter(
    x=my_team['Build_Speedup'],
    y=my_team['QPS_Speedup'],
    c='#d63031',
    s=1000,
    marker='*',
    edgecolors='none',  # 这里修改为 none，去掉白边
    label=my_team_name,
    zorder=10
)

# ==========================================
# 5. 标注与美化
# ==========================================

# 标注：我的队伍
row = my_team.iloc[0]
plt.text(
    row['Build_Speedup'] + 0.1, # 稍微加大偏移量，防止紧贴星星
    row['QPS_Speedup'],
    f"{my_team_name}\nRank 1",
    fontsize=24,
    fontweight='bold',
    color='#c0392b',
    verticalalignment='center',
    zorder=30
)

# 标注：第二、第三名 [修正2：添加 zorder 防止被遮挡]
for _, r in runner_ups.iterrows():
    plt.text(
        r['Build_Speedup'] + 0.05,
        r['QPS_Speedup'],
        f"Rank {r['Rank']}",
        fontsize=16,
        fontweight='bold',
        color='#d35400',
        verticalalignment='bottom',
        zorder=30 # 确保文字在所有点之上
    )

# 坐标轴与标题
plt.xlabel(f'构建效率倍数 (相对Top20平均) →', fontsize=18, fontweight='bold', labelpad=15)
plt.ylabel(f'查询性能倍数 (相对Top20平均) →', fontsize=18, fontweight='bold', labelpad=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 动态范围
max_x = df['Build_Speedup'].max() * 1.15
max_y = df['QPS_Speedup'].max() * 1.15
plt.xlim(0, max_x)
plt.ylim(0, max_y)

# 高光区域
rect = patches.Rectangle((1, 1), max_x-1, max_y-1, linewidth=0, facecolor='#2ecc71', alpha=0.08)
plt.gca().add_patch(rect)
plt.text(max_x*0.95, max_y*0.95, "综合最优区域",
         ha='right', va='top', fontsize=20, color='#27ae60', fontweight='bold', alpha=0.6)

plt.title(f'', fontsize=26, fontweight='bold', pad=30)

# [修正3：图例移至左上角]
plt.legend(
    loc='upper left',  # 改为左上角，这里通常是空白的
    bbox_to_anchor=(0.02, 0.98), # 微调位置，稍微离边缘一点距离
    fontsize=16,
    frameon=True,
    framealpha=0.95,
    shadow=True
)

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 保存
plt.savefig('performance_final_v2.png')
plt.show()