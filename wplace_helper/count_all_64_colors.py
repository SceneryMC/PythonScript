# 保存为 color_counter_v3.py
import sys
import argparse
import csv
from collections import Counter
from PIL import Image
import numpy as np

from wplace_helper.utils import WPLACE_COLOR_PALETTE, DEFAULT_FREE_COLORS


def count_colors_in_image(input_path, output_path):
    print(f"🖼️  正在加载图片: {input_path}...")
    try:
        image = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{input_path}'");
        return

    # **[修改]** 创建一个更全面的映射：从RGB元组到包含名称和ID的对象
    rgb_to_info_map = {
        tuple(c['rgb']): {'name': c['name'], 'id': c['id']}
        for c in WPLACE_COLOR_PALETTE if c['rgb']
    }

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

    all_stats = []
    for rgb_tuple, count in color_counts.items():
        # **[修改]** 从新的映射中获取信息
        color_info = rgb_to_info_map.get(rgb_tuple, {"name": "Unknown Color", "id": "N/A"})
        all_stats.append({
            "id": color_info['id'],  # <--- 新增
            "name": color_info['name'],
            "count": count,
            "rgb": str(rgb_tuple)
        })

    sorted_all_stats = sorted(all_stats, key=lambda x: x['count'], reverse=True)
    paid_stats = [stat for stat in sorted_all_stats if stat['name'] not in DEFAULT_FREE_COLORS]

    print(f"📄 正在将结果写入CSV文件: {output_path}...")
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # --- 写入第一部分：所有颜色的完整统计 ---
            writer.writerow(['--- Complete Color Statistics (Sorted by Count) ---'])
            # **[修改]** 添加 "ID" 列
            writer.writerow(['ID', 'Color Name', 'Pixel Count', 'Percentage', 'RGB Value'])

            for stat in sorted_all_stats:
                percentage = (stat['count'] / total_opaque_pixels) * 100
                # **[修改]** 在行中添加 stat['id']
                writer.writerow([stat['id'], stat['name'], stat['count'], f"{percentage:.2f}%", stat['rgb']])

            # --- 写入第二部分：所需付费颜色 ---
            if paid_stats:
                writer.writerow([])
                writer.writerow(['--- Paid Colors Required (Sorted by Count) ---'])
                # **[修改]** 添加 "ID" 列
                writer.writerow(['ID', 'Color Name', 'Pixel Count', 'RGB Value'])

                for stat in paid_stats:
                    # **[修改]** 在行中添加 stat['id']
                    writer.writerow([stat['id'], stat['name'], stat['count'], stat['rgb']])
            else:
                writer.writerow([])
                writer.writerow(['--- Paid Colors Required ---'])
                writer.writerow(['Congratulations! No paid colors are needed for this image.'])

        print("✨ 导出成功! CSV文件包含ID、完整统计和付费颜色列表。")
    except IOError as e:
        print(f"❌ 错误: 无法写入文件 '{output_path}'. 原因: {e}")


def main():
    INPUT = r'dst/779706d9f9/779706d9f9_converted.png'
    OUTPUT = r'dst/779706d9f9/779706d9f9_converted.csv'

    count_colors_in_image(INPUT, OUTPUT)


if __name__ == "__main__":
    main()