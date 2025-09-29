import cv2
import imageio
import numpy as np

WPLACE_COLOR_PALETTE = [
    {"id": 1, "name": "Black", "rgb": (0, 0, 0)}, {"id": 2, "name": "Dark Gray", "rgb": (60, 60, 60)},
    {"id": 3, "name": "Gray", "rgb": (120, 120, 120)}, {"id": 4, "name": "Light Gray", "rgb": (210, 210, 210)},
    {"id": 5, "name": "White", "rgb": (255, 255, 255)}, {"id": 6, "name": "Deep Red", "rgb": (96, 0, 24)},
    {"id": 7, "name": "Red", "rgb": (237, 28, 36)}, {"id": 8, "name": "Orange", "rgb": (255, 127, 39)},
    {"id": 9, "name": "Gold", "rgb": (246, 170, 9)}, {"id": 10, "name": "Yellow", "rgb": (249, 221, 59)},
    {"id": 11, "name": "Light Yellow", "rgb": (255, 250, 188)}, {"id": 12, "name": "Dark Green", "rgb": (14, 185, 104)},
    {"id": 13, "name": "Green", "rgb": (19, 230, 123)}, {"id": 14, "name": "Light Green", "rgb": (135, 255, 94)},
    {"id": 15, "name": "Dark Teal", "rgb": (12, 129, 110)}, {"id": 16, "name": "Teal", "rgb": (16, 174, 166)},
    {"id": 17, "name": "Light Teal", "rgb": (19, 225, 190)}, {"id": 20, "name": "Cyan", "rgb": (96, 247, 242)},
    {"id": 44, "name": "Light Cyan", "rgb": (187, 250, 242)}, {"id": 18, "name": "Dark Blue", "rgb": (40, 80, 158)},
    {"id": 19, "name": "Blue", "rgb": (64, 147, 228)}, {"id": 21, "name": "Indigo", "rgb": (107, 80, 246)},
    {"id": 22, "name": "Light Indigo", "rgb": (153, 177, 251)},
    {"id": 23, "name": "Dark Purple", "rgb": (120, 12, 153)},
    {"id": 24, "name": "Purple", "rgb": (170, 56, 185)}, {"id": 25, "name": "Light Purple", "rgb": (224, 159, 249)},
    {"id": 26, "name": "Dark Pink", "rgb": (203, 0, 122)}, {"id": 27, "name": "Pink", "rgb": (236, 31, 128)},
    {"id": 28, "name": "Light Pink", "rgb": (243, 141, 169)}, {"id": 29, "name": "Dark Brown", "rgb": (104, 70, 52)},
    {"id": 30, "name": "Brown", "rgb": (149, 104, 42)}, {"id": 31, "name": "Beige", "rgb": (248, 178, 119)},
    {"id": 52, "name": "Light Beige", "rgb": (255, 197, 165)},
    {"id": 32, "name": "Medium Gray", "rgb": (170, 170, 170)},
    {"id": 33, "name": "Dark Red", "rgb": (165, 14, 30)}, {"id": 34, "name": "Light Red", "rgb": (250, 128, 114)},
    {"id": 35, "name": "Dark Orange", "rgb": (228, 92, 26)},
    {"id": 37, "name": "Dark Goldenrod", "rgb": (156, 132, 49)},
    {"id": 38, "name": "Goldenrod", "rgb": (197, 173, 49)},
    {"id": 39, "name": "Light Goldenrod", "rgb": (232, 212, 95)},
    {"id": 40, "name": "Dark Olive", "rgb": (74, 107, 58)}, {"id": 41, "name": "Olive", "rgb": (90, 148, 74)},
    {"id": 42, "name": "Light Olive", "rgb": (132, 197, 115)}, {"id": 43, "name": "Dark Cyan", "rgb": (15, 121, 159)},
    {"id": 45, "name": "Light Blue", "rgb": (125, 199, 255)}, {"id": 46, "name": "Dark Indigo", "rgb": (77, 49, 184)},
    {"id": 47, "name": "Dark Slate Blue", "rgb": (74, 66, 132)},
    {"id": 48, "name": "Slate Blue", "rgb": (122, 113, 196)},
    {"id": 49, "name": "Light Slate Blue", "rgb": (181, 174, 241)},
    {"id": 53, "name": "Dark Peach", "rgb": (155, 82, 73)},
    {"id": 54, "name": "Peach", "rgb": (209, 128, 120)}, {"id": 55, "name": "Light Peach", "rgb": (250, 182, 164)},
    {"id": 50, "name": "Light Brown", "rgb": (219, 164, 99)}, {"id": 56, "name": "Dark Tan", "rgb": (123, 99, 82)},
    {"id": 57, "name": "Tan", "rgb": (156, 132, 107)}, {"id": 36, "name": "Light Tan", "rgb": (214, 181, 148)},
    {"id": 51, "name": "Dark Beige", "rgb": (209, 128, 81)}, {"id": 61, "name": "Dark Stone", "rgb": (109, 100, 63)},
    {"id": 62, "name": "Stone", "rgb": (148, 140, 107)}, {"id": 63, "name": "Light Stone", "rgb": (205, 197, 158)},
    {"id": 58, "name": "Dark Slate", "rgb": (51, 57, 65)}, {"id": 59, "name": "Slate", "rgb": (109, 117, 141)},
    {"id": 60, "name": "Light Slate", "rgb": (179, 185, 209)}
]

DEFAULT_FREE_COLORS = {item['name'] for item in WPLACE_COLOR_PALETTE if item['id'] < 32}

CURRENT_AVAILABLE_COLORS = {
"Black", "Dark Gray", "Gray", "Light Gray", "White", "Deep Red", "Red", "Orange",
    "Gold", "Yellow", "Light Yellow", "Dark Green", "Green", "Light Green", "Dark Teal",
    "Teal", "Light Teal", "Cyan", "Dark Blue", "Blue", "Indigo", "Light Indigo",
    "Dark Purple", "Purple", "Light Purple", "Dark Pink", "Pink", "Light Pink",
    "Dark Brown", "Brown", "Beige", "Dark Slate Blue", "Dark Slate", "Slate", "Light Slate",
    "Slate Blue", "Dark Indigo", "Medium Grey", "Light Slate Blue", "Light Cyan"
}


def create_video_visualization(pixels_ordered, shape, output_filename, fps, pixels_per_frame):
    """使用 imageio 和 ffmpeg 将排序后的像素列表高效地可视化为MP4视频。"""
    print(f"\n--- Generating MP4 Animation (High Speed) ---")
    print(f"FPS: {fps}, Pixels per Frame: {pixels_per_frame}")

    height, width, _ = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    total_pixels = len(pixels_ordered)

    if total_pixels == 0:
        imageio.imwrite(output_filename, canvas)
        print("No pixels to draw. Saved a static video frame.")
        return

    # --- [ 核心修正：使用 plugin='ffmpeg' 替代 format='MP4' ] ---
    # 这会强制 imageio 使用 ffmpeg 插件，该插件能正确理解 fps, codec 等视频参数。
    with imageio.get_writer(
            output_filename,
            fps=fps,
            codec='libx264',
            quality=8,
    ) as writer:
        # --- [ 修正结束 ] ---
        num_frames = (total_pixels + pixels_per_frame - 1) // pixels_per_frame
        print(f"Total frames to generate: {num_frames}")

        frame_counter = 0
        for i in range(0, total_pixels, pixels_per_frame):
            frame_counter += 1
            if frame_counter > 1 and frame_counter % 20 == 0:
                print(f"  Encoding frame {frame_counter} / {num_frames}...")

            pixels_in_this_frame = pixels_ordered[i: i + pixels_per_frame]

            for j, (x, y) in enumerate(pixels_in_this_frame):
                progress_index = i + j
                color_value = int((progress_index / total_pixels) * 255)
                color_bgr = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                canvas[y, x] = color_bgr

            frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

    print(f"Successfully saved MP4 animation to '{output_filename}'")


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