import numpy as np
from PIL import Image
import csv
from collections import Counter
import os

from wplace_helper.utils import WPLACE_COLOR_PALETTE, DEFAULT_FREE_COLORS

# --- 配置 ---
CANVAS_IMAGE_PATH = r"C:\Users\13308\nodejsproj\wplacer-lllexxa\data\snapshots\my_nilou_snapshot.png"
TEMPLATE_IMAGE_PATH = "dst/726/726_12_modified_converted_merged.png"
OUTPUT_CSV_PATH = "workload_colors.csv"


def analyze_workload_colors(canvas_path, template_path, output_path):
    """
    比较两张图，并只对需要“绘制”和“重绘”的像素进行颜色统计。
    """
    try:
        print(f"🖼️  正在加载画布图像: {canvas_path}")
        canvas_img = Image.open(canvas_path).convert("RGBA")
        print(f"🖼️  正在加载新模板图像: {template_path}")
        template_img = Image.open(template_path).convert("RGBA")
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件！ - {e}");
        return

    # --- 步骤1: 找出需要操作的像素 (来自 compare_templates.py) ---
    print("\n🔍 正在比较图像以确定工作量像素...")
    max_width = max(canvas_img.width, template_img.width)
    max_height = max(canvas_img.height, template_img.height)

    canvas_full = np.zeros((max_height, max_width, 4), dtype=np.uint8)
    template_full = np.zeros((max_height, max_width, 4), dtype=np.uint8)

    canvas_full[:canvas_img.height, :canvas_img.width, :] = np.array(canvas_img)
    template_full[:template_img.height, :template_img.width, :] = np.array(template_img)

    canvas_is_opaque = canvas_full[:, :, 3] > 127
    template_is_opaque = template_full[:, :, 3] > 127

    # 【等待绘制】的蒙版
    to_be_drawn_mask = (~canvas_is_opaque) & (template_is_opaque)

    # 【等待重绘】的蒙版
    both_opaque_mask = canvas_is_opaque & template_is_opaque
    colors_are_different = np.any(canvas_full[:, :, :3] != template_full[:, :, :3], axis=2)
    to_be_redrawn_mask = both_opaque_mask & colors_are_different

    # 合并两个蒙版，得到所有需要操作的像素的全集
    workload_mask = to_be_drawn_mask | to_be_redrawn_mask

    # --- 步骤2: 只提取这些像素的颜色 (来自 color_counter_v3.py) ---

    # 从新模板中，根据工作量蒙版提取出所有相关的RGB像素
    # 这些就是我们最终要画上去的颜色
    workload_pixels_rgb = template_full[workload_mask][:, :3]

    total_workload_pixels = workload_pixels_rgb.shape[0]

    if total_workload_pixels == 0:
        print("\n🎉 恭喜! 画布与模板完全一致，无需任何操作。");
        return

    print(f"📊 正在统计 {total_workload_pixels:,} 个工作量像素的颜色...")

    # --- 步骤3: 执行颜色统计和报告生成 (来自 color_counter_v3.py) ---
    rgb_to_info_map = {tuple(c['rgb']): {'name': c['name'], 'id': c['id']} for c in WPLACE_COLOR_PALETTE}

    pixel_tuples = [tuple(p) for p in workload_pixels_rgb]
    color_counts = Counter(pixel_tuples)

    print(f"✅ 统计完成! 工作量中包含 {len(color_counts)} 种独特的颜色。")

    all_stats = []
    for rgb_tuple, count in color_counts.items():
        color_info = rgb_to_info_map.get(rgb_tuple, {"name": "Unknown Color", "id": "N/A"})
        all_stats.append({
            "id": color_info['id'],
            "name": color_info['name'],
            "count": count,
            "rgb": str(rgb_tuple)
        })

    sorted_all_stats = sorted(all_stats, key=lambda x: x['count'], reverse=True)
    paid_stats = [stat for stat in sorted_all_stats if stat['name'] not in DEFAULT_FREE_COLORS]

    print(f"📄 正在将工作量颜色统计结果写入: {output_path}...")
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([f'--- Workload Color Statistics ({total_workload_pixels:,} pixels total) ---'])
            writer.writerow(['ID', 'Color Name', 'Pixel Count', 'Percentage', 'RGB Value'])

            for stat in sorted_all_stats:
                percentage = (stat['count'] / total_workload_pixels) * 100
                writer.writerow([stat['id'], stat['name'], stat['count'], f"{percentage:.2f}%", stat['rgb']])

            if paid_stats:
                writer.writerow([])
                writer.writerow(['--- Paid Colors Required for this Workload ---'])
                writer.writerow(['ID', 'Color Name', 'Pixel Count', 'RGB Value'])
                for stat in paid_stats:
                    writer.writerow([stat['id'], stat['name'], stat['count'], stat['rgb']])
            else:
                writer.writerow([])
                writer.writerow(['--- Paid Colors Required ---'])
                writer.writerow(['Congratulations! No paid colors are needed for this workload.'])

        print(f"✨ 导出成功! CSV文件 '{output_path}' 已生成。")
    except IOError as e:
        print(f"❌ 错误: 无法写入文件 '{output_path}'. 原因: {e}")


if __name__ == "__main__":
    if os.path.exists(CANVAS_IMAGE_PATH) and os.path.exists(TEMPLATE_IMAGE_PATH):
        analyze_workload_colors(CANVAS_IMAGE_PATH, TEMPLATE_IMAGE_PATH, OUTPUT_CSV_PATH)
    else:
        print("错误：一个或两个图像文件不存在，请检查配置中的文件名。")