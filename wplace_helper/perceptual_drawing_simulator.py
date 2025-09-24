import json

import cv2
import imageio
import numpy as np
import functools

# --- 配置项 ---
ORIGINAL_TEMPLATE_FILENAME = "dst/9148c828bis91/9148c828bis91_converted.png"

# 2. 边缘地图文件名：您用Canny生成的黑白边缘图
EDGE_MAP_FILENAME = "edges/9148c828bis91_edge.png"
OUTPUT_FILENAME = "visualization_merged_layers_9148c828bis91.mp4"

JSON_OUTPUT_FILENAME = "ideal_outline_order.json"
VIDEO_FPS = 60
PIXELS_PER_FRAME = 50


# --------------------------------------------------------------------

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


def prepare_fused_outline_pixels(template_path, edge_map_path):
    # ... (这个函数保持不变) ...
    print("--- Step 1: Loading Images and Fusing Edges ---")
    template_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)
    if template_image is None or edge_map is None or template_image.shape[:2] != edge_map.shape[:2]: raise ValueError(
        "Invalid files.")
    height, width = template_image.shape[:2]
    has_alpha = len(template_image.shape) > 2 and template_image.shape[2] == 4
    all_valid_pixels_set = set()
    if has_alpha:
        alpha_channel = template_image[:, :, 3]
        for y in range(height):
            for x in range(width):
                if alpha_channel[y, x] == 255: all_valid_pixels_set.add((x, y))
    else:
        for y in range(height):
            for x in range(width): all_valid_pixels_set.add((x, y))
    basic_edge_coords = set()
    if has_alpha:
        alpha_channel = template_image[:, :, 3]
        for x, y in all_valid_pixels_set:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or alpha_channel[ny, nx] != 255: basic_edge_coords.add(
                    (x, y)); break
    else:
        for x, y in all_valid_pixels_set:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1: basic_edge_coords.add((x, y))
    y_c, x_c = np.where(edge_map > 128)
    canny_edge_coords_raw = set(zip(x_c, y_c))
    canny_edge_coords = canny_edge_coords_raw.intersection(all_valid_pixels_set)
    fused_coords = sorted(list(basic_edge_coords.union(canny_edge_coords)), key=lambda p: (p[1], p[0]))
    print(f"Total unique outline pixels to process: {len(fused_coords)}")
    return fused_coords, template_image


# =================== [ 请用此代码块替换整个 run_final_algorithm 函数 ] ===================
def run_final_algorithm_with_hierarchy(pixels_set, template_image):
    """
    使用“层级地图”来确保所有像素都被正确分层，然后再进行路径追踪。
    """
    print("\n--- Running Final Algorithm with Level Map ---")
    if not pixels_set: return []

    height, width, _ = template_image.shape

    # 1. 将融合后的像素点绘制到黑白图像上
    fused_edge_map = np.zeros((height, width), dtype=np.uint8)
    for x, y in pixels_set:
        fused_edge_map[y, x] = 255

    # 2. 找到所有轮廓及其层级关系
    print("Finding contours and their parent-child relationships...")
    contours_cv, hierarchy = cv2.findContours(fused_edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 创建“层级地图” (Level Map)
    print("Creating a level map to classify every pixel...")
    level_map = np.zeros((height, width), dtype=np.int32)
    if hierarchy is not None:
        memo = {}

        def get_depth(i):
            if i in memo: return memo[i]
            parent_idx = hierarchy[0][i][3]
            if parent_idx == -1: memo[i] = 0; return 0
            depth = 1 + get_depth(parent_idx)
            memo[i] = depth
            return depth

        # 按层级分组轮廓
        contours_by_level = {}
        for i, contour in enumerate(contours_cv):
            depth = get_depth(i)
            if depth not in contours_by_level: contours_by_level[depth] = []
            contours_by_level[depth].append(contour)

        # 逐层填充 level_map
        # [!!!] 关键：thickness=-1 表示填充整个轮廓内部
        for depth, contours in sorted(contours_by_level.items()):
            # 我们用 depth + 1 作为颜色，这样 level 0 就不会是黑色背景0
            cv2.drawContours(level_map, contours, -1, color=depth + 1, thickness=-1)

    # 4. 使用 level_map 对原始的、完整的像素列表进行分桶
    print("Classifying all original pixels using the level map...")
    pixels_by_level = {}
    for x, y in pixels_set:
        level = level_map[y, x] - 1  # 减1，使层级从0开始
        if level >= 0:
            if level not in pixels_by_level: pixels_by_level[level] = []
            pixels_by_level[level].append((x, y))

    print(f"Grouped {len(contours_cv)} contours into {len(pixels_by_level)} hierarchical levels.")

    # 4. 逐层处理：排序 -> 路径追踪 -> 合并
    final_ordered_pixels = []
    vectors = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    direction_map = {(dx, dy): i for i, (dx, dy) in enumerate(vectors)}
    # 按照层级（0, 1, 2...）从外到内进行处理
    for level in sorted(pixels_by_level.keys()):

        # [!!!] 核心修改：现在 `contour` 是一个包含了该层所有像素的大列表
        contour = sorted(pixels_by_level[level], key=lambda p: (p[1], p[0]))
        print(f"  Processing Level {level}: Merged {len(contour)} pixels into a single group for tracing...")

        # 对这一整个层级的像素，进行一次性的路径追踪
        # ... (这里的路径追踪逻辑与您提供的最新版本完全相同) ...
        traced_contour, unvisited_in_contour = [], {f"{p[0]},{p[1]}": p for p in contour}

        # 只要还有未访问的像素，就持续寻找起点并追踪
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


# =================== [ 函数结束 ] ===================

# --- 为了完整性，附上视频生成函数 ---
def create_video_visualization(pixels_ordered, shape, output_filename, fps, pixels_per_frame):
    print(f"\n--- Generating MP4 Animation ---")
    height, width, _ = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    total_pixels = len(pixels_ordered)
    print(total_pixels)
    if total_pixels == 0: imageio.imwrite(output_filename, canvas, fps=fps); return
    with imageio.get_writer(output_filename, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
        num_frames = (total_pixels + pixels_per_frame - 1) // pixels_per_frame
        print(f"Total frames to generate: {num_frames}")
        frame_counter = 0
        for i in range(0, total_pixels, pixels_per_frame):
            frame_counter += 1
            if frame_counter > 1 and frame_counter % 20 == 0: print(
                f"  Encoding frame {frame_counter} / {num_frames}...")
            pixels_in_this_frame = pixels_ordered[i: i + pixels_per_frame]
            for j, (x, y) in enumerate(pixels_in_this_frame):
                progress_index = i + j
                color_value = int((progress_index / total_pixels) * 255)
                color_bgr = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                canvas[y, x] = color_bgr
            frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
    print(f"Successfully saved MP4 animation to '{output_filename}'")


if __name__ == "__main__":
    try:
        # 1. 准备融合后的轮廓像素，同时返回原始模板图像
        fused_pixels_set, template_image = prepare_fused_outline_pixels(ORIGINAL_TEMPLATE_FILENAME, EDGE_MAP_FILENAME)

        # 2. 运行我们全新的、基于层级合并的算法
        ordered_pixels = run_final_algorithm_with_hierarchy(fused_pixels_set, template_image)

        # 3. 生成视频
        create_video_visualization(ordered_pixels, template_image.shape, OUTPUT_FILENAME, VIDEO_FPS, PIXELS_PER_FRAME)
        export_order_to_json(ordered_pixels, JSON_OUTPUT_FILENAME)

        print("\n--- Hierarchical Merged Visualization Complete ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")