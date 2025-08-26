# 保存为 color_counter_v2.py
import sys
import argparse
import csv
from collections import Counter
from PIL import Image
import numpy as np

from utils import WPLACE_COLOR_PALETTE, DEFAULT_FREE_COLORS


# --- 核心功能 (已更新) ---

def count_colors_in_image(input_path, output_path):
    print(f"🖼️  正在加载图片: {input_path}...")
    try:
        image = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{input_path}'");
        return

    rgb_to_name_map = {tuple(c['rgb']): c['name'] for c in WPLACE_COLOR_PALETTE}
    image_array = np.array(image)

    is_opaque = image_array[:, :, 3] > 128
    opaque_pixels_rgb = image_array[is_opaque][:, :3]
    total_opaque_pixels = opaque_pixels_rgb.shape[0]

    if total_opaque_pixels == 0:
        print("⚠️ 警告: 图像中没有不透明的像素。");
        return

    print(f"📊 正在统计 {total_opaque_pixels} 个不透明像素的颜色...")

    pixel_tuples = [tuple(p) for p in opaque_pixels_rgb]
    color_counts = Counter(pixel_tuples)
    unique_color_count = len(color_counts)

    print(f"✅ 统计完成! 图像中找到了 {unique_color_count} 种独特的颜色。")

    # 将所有统计结果转换为一个更易于处理的列表
    all_stats = []
    for rgb_tuple, count in color_counts.items():
        color_name = rgb_to_name_map.get(rgb_tuple, "Unknown Color")
        all_stats.append({
            "name": color_name,
            "count": count,
            "rgb": str(rgb_tuple)
        })

    # 按像素数量降序排列所有颜色
    sorted_all_stats = sorted(all_stats, key=lambda x: x['count'], reverse=True)

    # **[新增]** 筛选出付费颜色并按数量降序排列
    paid_stats = [stat for stat in sorted_all_stats if stat['name'] not in DEFAULT_FREE_COLORS]

    print(f"📄 正在将结果写入CSV文件: {output_path}...")
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # --- 写入第一部分：所有颜色的完整统计 ---
            writer.writerow(['--- Complete Color Statistics (Sorted by Count) ---'])
            writer.writerow(['Color Name', 'Pixel Count', 'Percentage', 'RGB Value'])

            for stat in sorted_all_stats:
                percentage = (stat['count'] / total_opaque_pixels) * 100
                writer.writerow([stat['name'], stat['count'], f"{percentage:.2f}%", stat['rgb']])

            # **[新增]** 写入分隔符和第二部分
            if paid_stats:
                writer.writerow([])  # 空行作为分隔
                writer.writerow(['--- Paid Colors Required (Sorted by Count) ---'])
                writer.writerow(['Color Name', 'Pixel Count', 'RGB Value'])

                for stat in paid_stats:
                    writer.writerow([stat['name'], stat['count'], stat['rgb']])
            else:
                writer.writerow([])
                writer.writerow(['--- Paid Colors Required ---'])
                writer.writerow(['Congratulations! No paid colors are needed for this image.'])

        print("✨ 导出成功! CSV文件包含完整统计和付费颜色列表。")
    except IOError as e:
        print(f"❌ 错误: 无法写入文件 '{output_path}'. 原因: {e}")


# --- 主程序入口 (不变) ---
def main():
    parser = argparse.ArgumentParser(
        description="统计已转换图像中的WPlace颜色数量并导出为CSV (v2)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_image", help="输入的已转换图像文件路径 (如.png)。")
    parser.add_argument("output_csv", help="输出的统计结果.csv文件路径。")

    args = parser.parse_args()
    count_colors_in_image(args.input_image, args.output_csv)


if __name__ == "__main__":
    main()