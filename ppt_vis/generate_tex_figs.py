import json
import os
import subprocess

# ================= 配置 =================
INPUT_FILE = "experiment_results.json"
OUTPUT_DIR = "plots"

# 1. 样式映射 (Key: JSON中的方法名 -> Value: (Color, Mark))
# 如果你的JSON中方法名是小写，请确保这里Key匹配，或者在代码里做lower()处理
STYLE_MAP = {
    # --- Ours ---
    "largekanns-2s": ("red", "o"),  # 红色，圆圈 (Top Priority)
    "largekanns": ("violet", "diamond"),  # 紫色，菱形 (Second Priority)
    "alg2": ("violet", "diamond"),  # 兼容旧名称

    # --- Baselines ---
    "hnsw": ("blue", "triangle"),  # 蓝色，三角
    "nsg": ("green", "square"),  # 绿色，方块
    "rabitq": ("orange", "pentagon"),  # 橙色，五边形
    "symqg": ("cyan", "asterisk"),  # 青色，星号
}

# 2. 图例名称映射 (Key: JSON中的方法名 -> Value: 图例显示的文本)
# 用于将代码名转换为论文标准名
NAME_MAP = {
    "largekanns-2s": "LargekANNS-2S",
    "largekanns": "LargekANNS",
    "alg2": "LargekANNS",  # 兼容旧名称
    "hnsw": "HNSW",
    "nsg": "NSG",
    "rabitq": "RaBitQ",
    "symqg": "SymQG",
}

# 3. 排序优先级 (图例顺序)
# 数字越小排越上面
ORDER_PRIORITY = {
    "largekanns-2s": 0,
    "largekanns": 1,
    "alg2": 1,
    "rabitq": 2,
    "symqg": 3,
    "nsg": 4,
    "hnsw": 5
}

# LaTeX 模板
LATEX_TEMPLATE = r"""
\documentclass[tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% 定义颜色以便微调
\definecolor{myred}{RGB}{220, 20, 60}      % Crimson
\definecolor{myviolet}{RGB}{138, 43, 226}  % BlueViolet
\definecolor{myblue}{RGB}{0, 0, 255}
\definecolor{mygreen}{RGB}{0, 150, 0}
\definecolor{myorange}{RGB}{255, 140, 0}
\definecolor{mycyan}{RGB}{0, 180, 180}

\begin{document}
\begin{tikzpicture}
    \begin{axis}[
            height=8cm,
            width=10cm,
            xlabel={Recall (\%)},
            ylabel={QPS (Queries/s)},
            ylabel near ticks,
            ylabel shift=-0.2cm,
            label style={font=\large},
            tick label style={font=\large},
            % 图例样式
            legend style={
                font=\footnotesize, 
                fill=white, 
                fill opacity=0.8, 
                draw opacity=1, 
                text opacity=1,
                at={(0.02,0.02)}, % 默认左下角
                anchor=south west
            },
            legend cell align={left},
            ymode=log, % 对数坐标
            grid=major,
            enlargelimits=0.05,
        ]

BLOCK_ADD_PLOTS

        % 这里的图例顺序由 addplot 的顺序决定
        \legend{BLOCK_LEGEND_ENTRIES}

    \end{axis}
\end{tikzpicture}
\end{document}
"""


def get_style(method_name):
    """根据方法名获取颜色和标记，默认黑色"""
    key = method_name.lower()
    if key in STYLE_MAP:
        return STYLE_MAP[key]
    return "black", "x"


def get_display_name(method_name):
    """获取图例显示名称"""
    key = method_name.lower()
    return NAME_MAP.get(key, method_name)


def get_priority(method_name):
    """获取排序优先级"""
    key = method_name.lower()
    return ORDER_PRIORITY.get(key, 99)


def generate_tex_content(dataset_name, k_val, methods_data):
    plot_commands = []
    legend_entries = []

    # 1. 对方法进行排序
    # 我们希望图例顺序是固定的，通常最好的方法放在最前面或最后面
    # 这里按照 ORDER_PRIORITY 排序
    sorted_methods = sorted(methods_data.items(), key=lambda x: get_priority(x[0]))

    for method_name, points in sorted_methods:
        # 获取样式
        color_name, mark = get_style(method_name)

        # 映射颜色到 LaTeX 定义的颜色
        tex_color = color_name
        if color_name == "red":
            tex_color = "myred"
        elif color_name == "violet":
            tex_color = "myviolet"
        elif color_name == "blue":
            tex_color = "myblue"
        elif color_name == "green":
            tex_color = "mygreen"
        elif color_name == "orange":
            tex_color = "myorange"
        elif color_name == "cyan":
            tex_color = "mycyan"

        # 准备数据点并按 Recall 排序
        valid_points = [(r, q) for q, r in points if r > 0 and q > 0]
        valid_points.sort(key=lambda x: x[0])

        if not valid_points:
            continue

        coords_str = "\n".join([f"                ({r:.2f},{q:.2f})" for r, q in valid_points])

        # 线宽设置：我们的方法加粗
        line_width = "very thick"
        if "largekanns" in method_name.lower() or "alg2" in method_name.lower():
            line_width = "ultra thick"  # 我们的方法更粗一点

        cmd = f"""
        \\addplot[
            color={tex_color}, 
            text={tex_color}, 
            {line_width}, 
            mark={mark},
            mark options={{solid, scale=1.2}}
        ] coordinates {{
{coords_str}
        }};
        """
        plot_commands.append(cmd)

        # 添加图例名称
        display_name = get_display_name(method_name)
        # 转义 LaTeX 特殊字符
        display_name = display_name.replace("_", "\\_")
        legend_entries.append(display_name)

    content = LATEX_TEMPLATE.replace("BLOCK_ADD_PLOTS", "".join(plot_commands))
    content = content.replace("BLOCK_LEGEND_ENTRIES", ", ".join(legend_entries))

    return content


def compile_tex(tex_filename):
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=OUTPUT_DIR
        )
        if result.returncode == 0:
            print(f"✅ 编译成功: {tex_filename.replace('.tex', '.pdf')}")
            return True
        else:
            print(f"❌ 编译失败: {tex_filename}")
            return False
    except FileNotFoundError:
        print("❌ 未找到 pdflatex 命令")
        return False


def clean_aux_files(base_name):
    for ext in ['.aux', '.log']:
        file_path = os.path.join(OUTPUT_DIR, base_name + ext)
        if os.path.exists(file_path):
            os.remove(file_path)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for dataset, k_dict in data.items():
        for k_val, methods_data in k_dict.items():
            print(f"处理: {dataset} (K={k_val})...")

            tex_content = generate_tex_content(dataset, k_val, methods_data)

            # 生成文件名: sift10m_k10000.tex
            # 注意大小写转换以匹配之前的 LaTeX 引用习惯
            base_name = f"{dataset.lower()}_k{k_val}"
            tex_file_path = os.path.join(OUTPUT_DIR, base_name + ".tex")

            with open(tex_file_path, 'w', encoding='utf-8') as f:
                f.write(tex_content)

            if compile_tex(base_name + ".tex"):
                clean_aux_files(base_name)

    print(f"\n所有图表已保存至 {OUTPUT_DIR}/ 目录。")


if __name__ == "__main__":
    main()