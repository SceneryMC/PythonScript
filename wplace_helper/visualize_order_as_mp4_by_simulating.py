import json

import cv2
import imageio
import numpy as np
import functools
from PIL import Image  # Pillow库用于GIF生成

# --- 配置项 ---
# 1. 原始模板文件名：您带颜色和透明背景的模板PNG
ORIGINAL_TEMPLATE_FILENAME = "dst/175822732/175822732_converted.png"

# 2. 边缘地图文件名：您用Canny生成的黑白边缘图
EDGE_MAP_FILENAME = "edges/175822732_edge.png"
# 3. 输出文件名
OUTPUT_FILENAME = "visualization_final_fused_algorithm_175822732.mp4"

JSON_OUTPUT_FILENAME = "ideal_outline_order_175822732.json"

# 4. GIF动画参数
GIF_FPS = 60  # 帧率 (Frames Per Second)
PIXELS_PER_FRAME = 50  # 每一帧绘制多少个像素


def export_order_to_json(pixels_ordered, output_filename):
    """将最终的、有序的像素坐标列表保存为JSON文件。"""
    print(f"\n--- Step 6: Exporting Final Order to JSON ---")
    try:
        with open(output_filename, 'w') as f:
            # 我们只保存坐标，格式为 [[x1, y1], [x2, y2], ...]
            json.dump(pixels_ordered, f)
        print(f"Successfully exported {len(pixels_ordered)} ordered pixels to '{output_filename}'")
    except Exception as e:
        print(f"  ERROR: Failed to write JSON file. {e}")



# =================== [ 请用此代码块替换整个 prepare_fused_outline_pixels 函数 ] ===================
def prepare_fused_outline_pixels(template_path, edge_map_path):
    """
    精确复刻 server.js 的融合逻辑，并智能地支持RGB和RGBA两种格式的输入模板。
    """
    print(f"Loading original template: '{template_path}'")
    # 使用 IMREAD_UNCHANGED 来保留原始通道数
    template_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template_image is None:
        raise ValueError("Original template must be a valid PNG file.")

    print(f"Loading edge map: '{edge_map_path}'")
    edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)
    if edge_map is None:
        raise ValueError("Edge map must be a valid image file.")

    if template_image.shape[:2] != edge_map.shape[:2]:
        raise ValueError("Dimensions of template and edge map must match!")

    height, width = template_image.shape[:2]

    # --- [ 核心修正：判断是否存在Alpha通道，并选择不同逻辑 ] ---

    # 检查图像维度，判断是否包含Alpha通道
    has_alpha = len(template_image.shape) > 2 and template_image.shape[2] == 4

    all_possible_pixels = []
    basic_edge_coords = set()

    if has_alpha:
        print("Image has Alpha channel (RGBA). Using transparency-based logic.")
        alpha_channel = template_image[:, :, 3]

        # 1a. 创建“有效像素宇宙” (所有完全不透明的像素)
        for y in range(height):
            for x in range(width):
                if alpha_channel[y, x] == 255:
                    all_possible_pixels.append((x, y))

        # 2a. 基于“宇宙”，找到 "Basic Edges" (与透明/物理边界相邻)
        for x, y in all_possible_pixels:
            is_basic_edge = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or alpha_channel[ny, nx] != 255:
                    is_basic_edge = True
                    break
            if is_basic_edge:
                basic_edge_coords.add((x, y))
    else:
        print("Image has no Alpha channel (RGB). Using border-based logic.")

        # 1b. “有效像素宇宙”是所有像素
        for y in range(height):
            for x in range(width):
                all_possible_pixels.append((x, y))

        # 2b. "Basic Edges" 是图像的物理边框
        for x, y in all_possible_pixels:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                basic_edge_coords.add((x, y))

    print(f"Generated a 'universe' of {len(all_possible_pixels)} possible pixels.")
    # --- [ 修正结束 ] ---

    # --- 3. 基于“宇宙”，找到 "Canny Edges" ---
    print("Finding Canny edges...")
    canny_edge_coords = set()
    for x, y in all_possible_pixels:
        # 检查这个有效像素在边缘地图上是否是白色
        if edge_map[y, x] > 128:
            canny_edge_coords.add((x, y))

    # --- 4. 融合两者 ---
    print(f"Fusing edges: {len(basic_edge_coords)} basic + {len(canny_edge_coords)} Canny...")
    final_outline_coords_set = basic_edge_coords.union(canny_edge_coords)

    # --- 5. 从“宇宙”中筛选出最终轮廓，以保持原始的“Y优先”顺序 ---
    final_outline_coords = [p for p in all_possible_pixels if p in final_outline_coords_set]

    print(f"Total unique outline pixels to process: {len(final_outline_coords)}")

    # 保持输出格式不变，返回原始图像的shape
    return final_outline_coords, template_image.shape, template_image


def run_final_algorithm(pixels):
    """
    精确复刻 server.js 中的最终版 _orderByContour 算法。
    阶段1: BFS分组 -> 阶段2: 轮廓排序 -> 阶段3: 组内方向性路径追踪
    """
    print("\n--- Running Final Algorithm: BFS Grouping + Directional Path Tracing ---")
    if not pixels: return []

    pixel_map = {f"{p[0]},{p[1]}": p for p in pixels}
    visited = set()
    contours = []

    # 阶段 1: 轮廓分组 (BFS)
    # ... (这部分逻辑不变) ...
    for p in pixels:
        key = f"{p[0]},{p[1]}"
        if key in visited: continue
        current_contour = []
        queue = [p]
        visited.add(key)
        while queue:
            px, py = queue.pop(0)
            current_contour.append((px, py))
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    neighbor_key = f"{px + dx},{py + dy}"
                    if neighbor_key in pixel_map and neighbor_key not in visited:
                        visited.add(neighbor_key)
                        queue.append(pixel_map[neighbor_key])
        if current_contour:
            contours.append(current_contour)

    print(f"Found {len(contours)} separate outline contours.")

    # --- [ 核心修正：精确复刻JS的排序逻辑 ] ---
    # 阶段 2: 确定性排序 (Deterministic Sorting)

    # 1. 定义一个与JS中 getTopmost 完全相同的辅助函数
    def get_topmost_pixel(contour):
        if not contour:
            return float('inf'), float('inf')
        top = contour[0]
        for i in range(1, len(contour)):
            if contour[i][1] < top[1]:
                top = contour[i]
            elif contour[i][1] == top[1] and contour[i][0] < top[0]:
                top = contour[i]
        return top

    # 2. 定义一个与JS中 sort 回调完全相同的比较函数
    def compare_contours(a, b):
        top_a = get_topmost_pixel(a)
        top_b = get_topmost_pixel(b)

        if top_a[1] != top_b[1]:
            return top_a[1] - top_b[1]
        return top_a[0] - top_b[0]

    # 3. 使用 functools.cmp_to_key 来应用这个比较函数
    # contours.sort(key=functools.cmp_to_key(compare_contours))
    # 阶段 3: 方向性路径追踪
    final_ordered_pixels = []
    vectors = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    direction_map = {(dx, dy): i for i, (dx, dy) in enumerate(vectors)}
    for contour in contours:
        if not contour: continue
        traced_contour = []
        unvisited_in_contour = {f"{p[0]},{p[1]}": p for p in contour}
        start_pixel = min(contour, key=lambda p: (p[1], p[0]))
        current_pixel = start_pixel
        last_dx, last_dy = 0, 0
        while unvisited_in_contour:
            px, py = current_pixel
            traced_contour.append(current_pixel)
            del unvisited_in_contour[f"{px},{py}"]
            if not unvisited_in_contour: break
            next_pixel = None
            if last_dx != 0 or last_dy != 0:
                last_dir_index = direction_map.get((last_dx, last_dy))
            else:
                last_dir_index = None
            if last_dir_index is not None:
                priority_offsets = [0, 1, -1, 2, -2]
                for offset in priority_offsets:
                    dx, dy = vectors[(last_dir_index + offset + 8) % 8]
                    neighbor_key = f"{px + dx},{py + dy}"
                    if neighbor_key in unvisited_in_contour:
                        next_pixel = unvisited_in_contour[neighbor_key]
                        break
            if not next_pixel:
                for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                    if f"{px + dx},{py + dy}" in unvisited_in_contour:
                        next_pixel = unvisited_in_contour[f"{px + dx},{py + dy}"]
                        break
            if not next_pixel:
                next_pixel = min(unvisited_in_contour.values(), key=lambda p: ((p[0] - px) ** 2 + (p[1] - py) ** 2))
            if next_pixel:
                new_dx, new_dy = next_pixel[0] - px, next_pixel[1] - py
                if abs(new_dx) <= 1 and abs(new_dy) <= 1:
                    last_dx, last_dy = new_dx, new_dy
                else:
                    last_dx, last_dy = 0, 0
                current_pixel = next_pixel
            else:
                break
        final_ordered_pixels.extend(traced_contour)
    return final_ordered_pixels


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

    # --- [ 核心修正 1: 创建白色背景画布 ] ---
    # np.full 创建一个用指定值填充的数组。 (255, 255, 255) 是白色。
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    total_pixels = len(pixels_ordered)
    if total_pixels == 0:
        imageio.imwrite(output_filename, canvas, fps=fps)
        print("No pixels to draw. Saved a static video frame.")
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
            if frame_counter > 1 and frame_counter % 20 == 0:
                print(f"  Encoding frame {frame_counter} / {num_frames}...")

            pixels_in_this_frame = pixels_ordered[i: i + pixels_per_frame]

            # --- [ 核心修正 2: 绘制真实颜色 ] ---
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

    print(f"Successfully saved MP4 animation to '{output_filename}'")

# =================== [ 替换结束 ] ===================


if __name__ == "__main__":
    try:
        fused_pixels, image_shape, template_image= prepare_fused_outline_pixels(ORIGINAL_TEMPLATE_FILENAME, EDGE_MAP_FILENAME)
        ordered_pixels = run_final_algorithm(fused_pixels)
        # create_video_visualization(ordered_pixels, image_shape, OUTPUT_FILENAME, GIF_FPS, PIXELS_PER_FRAME)
        create_video_visualization_real(ordered_pixels, template_image, OUTPUT_FILENAME, GIF_FPS, PIXELS_PER_FRAME)
        export_order_to_json(ordered_pixels, JSON_OUTPUT_FILENAME)
        print("\n--- Visualization Complete ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")