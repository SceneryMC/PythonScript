import cv2
import numpy as np
import imageio
import json
import os

from wplace_helper.utils import create_video_visualization_real

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================

# 必需：您之前生成的、包含绘制顺序的JSON文件路径
ORDER_JSON_PATH = "wplacer_draw_order_try/112525058_p0.json"

# 必需：您的像素画原图的路径 (用于获取真实颜色)
PIXEL_ART_PATH = "dst/112525058_p0/112525058_p0_converted.png"

# 必需：最终生成的视频文件的输出路径
OUTPUT_VIDEO_PATH = "visualize_from_json/112525058_p0.mp4"
os.makedirs('visualize_from_json', exist_ok=True)

# 可选：视频的帧率 (Frames Per Second)
FPS = 60

# 可选：每一帧绘制多少个像素。数值越大，视频越短。
PIXELS_PER_FRAME = 200


# ========================================================================
# >> 主执行逻辑 <<
# ========================================================================

def main():
    print("--- WPlacer Visualization from JSON ---")

    # --- 1. 检查配置和文件是否存在 ---
    if not all([os.path.exists(ORDER_JSON_PATH), os.path.exists(PIXEL_ART_PATH)]):
        print("\nError: Input file not found.")
        if not os.path.exists(ORDER_JSON_PATH):
            print(f"  - Missing JSON file: '{ORDER_JSON_PATH}'")
        if not os.path.exists(PIXEL_ART_PATH):
            print(f"  - Missing Pixel Art file: '{PIXEL_ART_PATH}'")
        print("\nPlease check the file paths in the SETTINGS section of the script.")
        return

    # --- 2. 加载绘制顺序 JSON 文件 ---
    print(f"\nLoading draw order from '{ORDER_JSON_PATH}'...")
    try:
        with open(ORDER_JSON_PATH, 'r', encoding='utf-8') as f:
            pixels_ordered = json.load(f)
        print(f"  -> Successfully loaded {len(pixels_ordered)} pixel coordinates.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{ORDER_JSON_PATH}'. The file might be corrupted.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return

    # --- 3. 加载像素画原图 ---
    print(f"\nLoading pixel art from '{PIXEL_ART_PATH}'...")
    try:
        # 使用 cv2.IMREAD_UNCHANGED 来保留Alpha通道（如果有）
        pixel_art_image = cv2.imread(PIXEL_ART_PATH, cv2.IMREAD_UNCHANGED)
        if pixel_art_image is None:
            raise IOError("cv2.imread returned None. The file may be an unsupported format or corrupted.")
        print(f"  -> Image loaded successfully. Dimensions: {pixel_art_image.shape[1]}x{pixel_art_image.shape[0]}")
    except Exception as e:
        print(f"Error loading pixel art image: {e}")
        return

    # --- 4. 执行视频生成 ---
    try:
        create_video_visualization_real(
            pixels_ordered=pixels_ordered,
            template_image=pixel_art_image,
            output_filename=OUTPUT_VIDEO_PATH,
            fps=FPS,
            pixels_per_frame=PIXELS_PER_FRAME
        )
    except Exception as e:
        print(f"\nAn error occurred during video generation: {e}")


if __name__ == "__main__":
    main()