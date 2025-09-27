# 保存为 color_counter_v3.py
import sys
import argparse
import csv
from collections import Counter
from PIL import Image
import numpy as np

# --- 核心数据：WPlace调色板和免费颜色定义 (保持不变) ---
# 注意：此处的 COLOR_MAP 结构已被转换为 Python 列表 WPLACE_COLOR_PALETTE
WPLACE_COLOR_PALETTE = [
    {"id": 1, "name": "Black", "rgb": (0, 0, 0)},
    {"id": 2, "name": "Dark Gray", "rgb": (60, 60, 60)},
    {"id": 3, "name": "Gray", "rgb": (120, 120, 120)},
    {"id": 4, "name": "Light Gray", "rgb": (210, 210, 210)},
    {"id": 5, "name": "White", "rgb": (255, 255, 255)},
    {"id": 6, "name": "Deep Red", "rgb": (96, 0, 24)},
    {"id": 7, "name": "Red", "rgb": (237, 28, 36)},
    {"id": 8, "name": "Orange", "rgb": (255, 127, 39)},
    {"id": 9, "name": "Gold", "rgb": (246, 170, 9)},
    {"id": 10, "name": "Yellow", "rgb": (249, 221, 59)},
    {"id": 11, "name": "Light Yellow", "rgb": (255, 250, 188)},
    {"id": 12, "name": "Dark Green", "rgb": (14, 185, 104)},
    {"id": 13, "name": "Green", "rgb": (19, 230, 123)},
    {"id": 14, "name": "Light Green", "rgb": (135, 255, 94)},
    {"id": 15, "name": "Dark Teal", "rgb": (12, 129, 110)},
    {"id": 16, "name": "Teal", "rgb": (16, 174, 166)},
    {"id": 17, "name": "Light Teal", "rgb": (19, 225, 190)},
    {"id": 20, "name": "Cyan", "rgb": (96, 247, 242)},
    {"id": 44, "name": "Light Cyan", "rgb": (187, 250, 242)},
    {"id": 18, "name": "Dark Blue", "rgb": (40, 80, 158)},
    {"id": 19, "name": "Blue", "rgb": (64, 147, 228)},
    {"id": 21, "name": "Indigo", "rgb": (107, 80, 246)},
    {"id": 22, "name": "Light Indigo", "rgb": (153, 177, 251)},
    {"id": 23, "name": "Dark Purple", "rgb": (120, 12, 153)},
    {"id": 24, "name": "Purple", "rgb": (170, 56, 185)},
    {"id": 25, "name": "Light Purple", "rgb": (224, 159, 249)},
    {"id": 26, "name": "Dark Pink", "rgb": (203, 0, 122)},
    {"id": 27, "name": "Pink", "rgb": (236, 31, 128)},
    {"id": 28, "name": "Light Pink", "rgb": (243, 141, 169)},
    {"id": 29, "name": "Dark Brown", "rgb": (104, 70, 52)},
    {"id": 30, "name": "Brown", "rgb": (149, 104, 42)},
    {"id": 31, "name": "Beige", "rgb": (248, 178, 119)},
    {"id": 52, "name": "Light Beige", "rgb": (255, 197, 165)},
    {"id": 32, "name": "Medium Gray", "rgb": (170, 170, 170)},
    {"id": 33, "name": "Dark Red", "rgb": (165, 14, 30)},
    {"id": 34, "name": "Light Red", "rgb": (250, 128, 114)},
    {"id": 35, "name": "Dark Orange", "rgb": (228, 92, 26)},
    {"id": 37, "name": "Dark Goldenrod", "rgb": (156, 132, 49)},
    {"id": 38, "name": "Goldenrod", "rgb": (197, 173, 49)},
    {"id": 39, "name": "Light Goldenrod", "rgb": (232, 212, 95)},
    {"id": 40, "name": "Dark Olive", "rgb": (74, 107, 58)},
    {"id": 41, "name": "Olive", "rgb": (90, 148, 74)},
    {"id": 42, "name": "Light Olive", "rgb": (132, 197, 115)},
    {"id": 43, "name": "Dark Cyan", "rgb": (15, 121, 159)},
    {"id": 45, "name": "Light Blue", "rgb": (125, 199, 255)},
    {"id": 46, "name": "Dark Indigo", "rgb": (77, 49, 184)},
    {"id": 47, "name": "Dark Slate Blue", "rgb": (74, 66, 132)},
    {"id": 48, "name": "Slate Blue", "rgb": (122, 113, 196)},
    {"id": 49, "name": "Light Slate Blue", "rgb": (181, 174, 241)},
    {"id": 53, "name": "Dark Peach", "rgb": (155, 82, 73)},
    {"id": 54, "name": "Peach", "rgb": (209, 128, 120)},
    {"id": 55, "name": "Light Peach", "rgb": (250, 182, 164)},
    {"id": 50, "name": "Light Brown", "rgb": (219, 164, 99)},
    {"id": 56, "name": "Dark Tan", "rgb": (123, 99, 82)},
    {"id": 57, "name": "Tan", "rgb": (156, 132, 107)},
    {"id": 36, "name": "Light Tan", "rgb": (214, 181, 148)},
    {"id": 51, "name": "Dark Beige", "rgb": (209, 128, 81)},
    {"id": 61, "name": "Dark Stone", "rgb": (109, 100, 63)},
    {"id": 62, "name": "Stone", "rgb": (148, 140, 107)},
    {"id": 63, "name": "Light Stone", "rgb": (205, 197, 158)},
    {"id": 58, "name": "Dark Slate", "rgb": (51, 57, 65)},
    {"id": 59, "name": "Slate", "rgb": (109, 117, 141)},
    {"id": 60, "name": "Light Slate", "rgb": (179, 185, 209)},
]

DEFAULT_FREE_COLORS = {item['name'] for item in WPLACE_COLOR_PALETTE if item['id'] < 32}


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
    INPUT = r'E:\共享\dst\726\726_12_modified_converted_merged.png'
    OUTPUT = r'E:\共享\dst\726\726_12_modified_converted_merged.csv'

    count_colors_in_image(INPUT, OUTPUT)


if __name__ == "__main__":
    main()