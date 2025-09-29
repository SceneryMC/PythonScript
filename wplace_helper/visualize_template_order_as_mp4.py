import json
import cv2
import numpy as np
import imageio
import os

from wplace_helper.utils import create_video_visualization_real

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================

# 必需：输入你想可视化的模板的完整名称
TEMPLATE_NAME_TO_VISUALIZE = "726_12_modified_converted_merged"

# 可选：视频的帧率 (Frames Per Second)
FPS = 60

# 可选：每一帧绘制多少个像素。数值越大，视频越短。
PIXELS_PER_FRAME = 500

# ========================================================================
# >> WPlacer 颜色表 (从 server.js 移植) <<
# ========================================================================

# 将 server.js 中的颜色定义转换为 Python 字典
BASIC_COLORS = {"0,0,0": 1, "60,60,60": 2, "120,120,120": 3, "210,210,210": 4, "255,255,255": 5, "96,0,24": 6,
                "237,28,36": 7, "255,127,39": 8, "246,170,9": 9, "249,221,59": 10, "255,250,188": 11, "14,185,104": 12,
                "19,230,123": 13, "135,255,94": 14, "12,129,110": 15, "16,174,166": 16, "19,225,190": 17,
                "40,80,158": 18, "64,147,228": 19, "96,247,242": 20, "107,80,246": 21, "153,177,251": 22,
                "120,12,153": 23, "170,56,185": 24, "224,159,249": 25, "203,0,122": 26, "236,31,128": 27,
                "243,141,169": 28, "104,70,52": 29, "149,104,42": 30, "248,178,119": 31}
PREMIUM_COLORS = {"170,170,170": 32, "165,14,30": 33, "250,128,114": 34, "228,92,26": 35, "214,181,148": 36,
                  "156,132,49": 37, "197,173,49": 38, "232,212,95": 39, "74,107,58": 40, "90,148,74": 41,
                  "132,197,115": 42, "15,121,159": 43, "187,250,242": 44, "125,199,255": 45, "77,49,184": 46,
                  "74,66,132": 47, "122,113,196": 48, "181,174,241": 49, "219,164,99": 50, "209,128,81": 51,
                  "255,197,165": 52, "155,82,73": 53, "209,128,120": 54, "250,182,164": 55, "123,99,82": 56,
                  "156,132,107": 57, "51,57,65": 58, "109,117,141": 59, "179,185,209": 60, "109,100,63": 61,
                  "148,140,107": 62, "205,197,158": 63}
PALETTE = {**BASIC_COLORS, **PREMIUM_COLORS}

# 创建一个反向查找表：{ color_id: (R, G, B) }
REVERSE_PALETTE = {}
for rgb_string, color_id in PALETTE.items():
    r, g, b = map(int, rgb_string.split(','))
    REVERSE_PALETTE[color_id] = (r, g, b)

# 为特殊/透明色ID定义默认颜色（白色）
WHITE_COLOR = (255, 255, 255)
REVERSE_PALETTE[0] = WHITE_COLOR
REVERSE_PALETTE[-1] = WHITE_COLOR


# ========================================================================
# >> 辅助函数 (已重构) <<
# ========================================================================

def find_template_by_name(templates_data, name):
    for template_id, template in templates_data.items():
        if template.get("name") == name:
            return template
    return None


def create_template_image_from_data(template_data):
    """
    [核心修正] 根据模板的 `data` 数组和颜色表，在内存中构建出OpenCV图像。
    """
    print("Reconstructing template image from color ID data...")
    template_info = template_data.get("template", {})
    width = template_info.get("width")
    height = template_info.get("height")
    pixel_data = template_info.get("data")

    if not all([width, height, pixel_data]):
        raise ValueError("Template data is missing width, height, or pixel data array.")

    # 创建一个RGB格式的画布 (更符合Python图像处理习惯)
    canvas_rgb = np.full((height, width, 3), 255, dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            color_id = pixel_data[x][y]
            # 从反向颜色表中获取RGB元组，如果找不到则默认为白色
            rgb_color = REVERSE_PALETTE.get(color_id, WHITE_COLOR)
            canvas_rgb[y, x] = rgb_color

    # 将最终的RGB图像转换为OpenCV所需的BGR格式
    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    print("Image reconstructed successfully.")
    return canvas_bgr


def deserialize_draw_order(template):
    print("Deserializing draw order from index array...")
    index_array = template.get("_globalDrawOrder", [])
    if not index_array:
        print("Warning: _globalDrawOrder is empty or not found in the template.")
        return []

    width = template.get("template", {}).get("width")
    if not width:
        raise ValueError("Template width is not defined.")

    pixels_ordered = []
    for index in index_array:
        y = index // width
        x = index % width
        pixels_ordered.append((x, y))

    print(f"Successfully deserialized {len(pixels_ordered)} pixel coordinates.")
    return pixels_ordered


# ========================================================================
# >> 主执行逻辑 <<
# ========================================================================

def main():
    if TEMPLATE_NAME_TO_VISUALIZE == "您的模板名称":
        print(
            "Error: Please open the script and edit the 'TEMPLATE_NAME_TO_VISUALIZE' variable with your template's name.")
        return

    templates_file_path = r'C:\Users\13308\nodejsproj\wplacer-lllexxa\data\templates.json'
    print(f"Loading templates from '{templates_file_path}'...")
    try:
        with open(templates_file_path, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: '{templates_file_path}' not found. Make sure you are running this script from the project's root directory.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{templates_file_path}'. The file might be corrupted.")
        return

    print(f"Searching for template: '{TEMPLATE_NAME_TO_VISUALIZE}'...")
    target_template = find_template_by_name(templates_data, TEMPLATE_NAME_TO_VISUALIZE)
    if not target_template:
        print(f"Error: Template with name '{TEMPLATE_NAME_TO_VISUALIZE}' not found in templates.json.")
        return
    print("Template found!")

    try:
        # [核心修正] 调用新的函数来从数据重建图像
        template_image_cv = create_template_image_from_data(target_template)
        pixels_ordered = deserialize_draw_order(target_template)

        safe_filename = "".join(c for c in TEMPLATE_NAME_TO_VISUALIZE if c.isalnum() or c in (' ', '_')).rstrip()
        os.makedirs('visualize_template_order_as_mp4', exist_ok=True)
        output_filename = f"visualize_template_order_as_mp4/{safe_filename}.mp4"

        create_video_visualization_real(
            pixels_ordered=pixels_ordered,
            template_image=template_image_cv,
            output_filename=output_filename,
            fps=FPS,
            pixels_per_frame=PIXELS_PER_FRAME
        )

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")


if __name__ == "__main__":
    main()