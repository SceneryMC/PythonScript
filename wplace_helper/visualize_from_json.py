import cv2
import numpy as np
import imageio
import json
import os

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================

# 必需：您之前生成的、包含绘制顺序的JSON文件路径
ORDER_JSON_PATH = "wplacer_draw_order_175822732_new.json"

# 必需：您的像素画原图的路径 (用于获取真实颜色)
PIXEL_ART_PATH = "dst/175822732/175822732_converted.png"

# 必需：最终生成的视频文件的输出路径
OUTPUT_VIDEO_PATH = "visualize_from_json/175822732_new.mp4"
os.makedirs('visualize_from_json', exist_ok=True)

# 可选：视频的帧率 (Frames Per Second)
FPS = 60

# 可选：每一帧绘制多少个像素。数值越大，视频越短。
PIXELS_PER_FRAME = 500


# ========================================================================
# >> 视频生成函数 (来自之前的版本，稍作加固) <<
# ========================================================================

def create_video_visualization_real(pixels_ordered, template_image, output_filename, fps, pixels_per_frame):
    """
    将排序后的像素列表，以其“真实颜色”在白色背景上逐步绘制，并生成MP4视频。
    """
    print(f"\n--- Generating MP4 Animation with Real Colors ---")
    print(f"FPS: {fps}, Pixels per Frame: {pixels_per_frame}")

    height, width, _ = template_image.shape

    # 创建白色背景画布
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    total_pixels = len(pixels_ordered)
    if total_pixels == 0:
        print("Warning: No pixels in the order list. Saving a 1-second static white video.")
        # [FIX] 正确处理空像素列表，生成一个有效的1秒视频
        frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        with imageio.get_writer(output_filename, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
            for _ in range(fps):  # 写入1秒的帧 (fps * 1)
                writer.append_data(frame_rgb)
        return

    with imageio.get_writer(
            output_filename,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
    ) as writer:
        num_frames = (total_pixels + pixels_per_frame - 1) // pixels_per_frame
        print(f"Total frames to generate: {num_frames}")

        frame_counter = 0
        for i in range(0, total_pixels, pixels_per_frame):
            frame_counter += 1
            if frame_counter > 1 and frame_counter % 100 == 0:
                print(f"  Encoding frame {frame_counter} / {num_frames}...")

            pixels_in_this_frame = pixels_ordered[i: i + pixels_per_frame]

            for x, y in pixels_in_this_frame:
                # 从原始模板图像中获取该像素的真实颜色 (BGR格式)
                pixel_color = template_image[y, x]
                # 通过切片[:3]来处理RGB和RGBA两种情况，确保只取颜色通道
                real_color_bgr = pixel_color[:3]
                # 在我们的白色画布上，用真实颜色填充像素
                canvas[y, x] = real_color_bgr

            # 将OpenCV的BGR帧转换为imageio需要的RGB帧
            frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

        # [可选] 在视频末尾添加一个暂停帧，以便更好地欣赏最终结果
        print("  Adding a 2-second pause at the end...")
        for _ in range(fps * 2):
            writer.append_data(frame_rgb)

    print(f"\nSuccessfully saved MP4 animation to '{output_filename}'")


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