import cv2
import imageio
import numpy as np
import functools

# --- 核心配置 ---
# 1. 输入文件名
ORIGINAL_TEMPLATE_FILENAME = "dst/123713790_p3/123713790_p3_modified_converted_cleared.png"

# 2. 边缘地图文件名：您用Canny生成的黑白边缘图
EDGE_MAP_FILENAME = "edges/123713790_p3_modified_edge.png"

# 3. 输出文件名
OUTPUT_FILENAME = "visualization_advanced_heuristic_123713790_p3_modified_converted_cleared.mp4"

# 3. 视频参数
VIDEO_FPS = 60
PIXELS_PER_FRAME = 50  # 调高此值可以大幅加快视频生成速度

# 4. --- [ 新增：高级启发式权重配置 ] ---
HEURISTIC_WEIGHTS = {
    # 更看重“向外”的绘画方向
    "OUTWARD_DIRECTION": 3.0,
    # 稍微看重“沿着相同颜色走”的凝聚力
    "COLOR_COHESION": 1.0,
}


# --------------------------------------------------------------------

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
def run_final_algorithm(pixels, template_image):
    """
    使用“BFS分组 -> 轮廓排序 -> 高级启发式路径追踪”的最终算法。
    """
    print("\n--- Running Final Algorithm with Heuristic Path Tracing ---")
    if not pixels: return []

    height, width, _ = template_image.shape
    center_x, center_y = width / 2.0, height / 2.0

    pixel_map = {f"{p[0]},{p[1]}": p for p in pixels}
    visited = set()
    contours = []

    # 阶段 1: 轮廓分组 (BFS) - 保持不变
    for p in pixels:
        key = f"{p[0]},{p[1]}"
        if key in visited: continue
        current_contour = []
        queue = [p]
        visited.add(key)
        while queue:
            px, py = queue.pop(0)
            current_contour.append((px, py))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    neighbor_key = f"{px + dx},{py + dy}"
                    if neighbor_key in pixel_map and neighbor_key not in visited:
                        visited.add(neighbor_key)
                        queue.append(pixel_map[neighbor_key])
        if current_contour: contours.append(current_contour)
    print(f"Found {len(contours)} separate outline contours.")

    # 阶段 2: 轮廓排序 - 保持不变
    def get_topmost_pixel(c):
        if not c: return float('inf'), float('inf')
        return min(c, key=lambda p: (p[1], p[0]))

    def get_topmost_sort(c):
        topmost = get_topmost_pixel(c)
        return topmost[1], topmost[0]

    contours.sort(key=get_topmost_sort)

    # 阶段 3: 高级启发式路径追踪 (Heuristic Path Tracing)
    final_ordered_pixels = []
    for contour in contours:
        if not contour: continue
        traced_contour = []
        unvisited_in_contour = {f"{p[0]},{p[1]}": p for p in contour}
        current_pixel = get_topmost_pixel(contour)

        while unvisited_in_contour:
            px, py = current_pixel
            traced_contour.append(current_pixel)
            del unvisited_in_contour[f"{px},{py}"]
            if not unvisited_in_contour: break

            # --- [ 全新的评分系统 ] ---
            best_neighbor = None
            max_score = -float('inf')

            # 1. 计算期望的“向外”方向
            outward_vec_x = px - center_x
            outward_vec_y = py - center_y
            outward_angle = np.arctan2(outward_vec_y, outward_vec_x)

            # 2. 评估所有8个邻居
            candidates = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    neighbor_key = f"{px + dx},{py + dy}"
                    if neighbor_key in unvisited_in_contour:
                        candidates.append(unvisited_in_contour[neighbor_key])

            current_color = tuple(template_image[py, px][:3])  # BGR
            for neighbor in candidates:
                nx, ny = neighbor

                # a. 计算方向分
                move_angle = np.arctan2(ny - py, nx - px)
                angle_diff = np.arctan2(np.sin(move_angle - outward_angle), np.cos(move_angle - outward_angle))

                if abs(angle_diff) > (np.pi / 2):
                    direction_score = -float('inf')  # 排除“回头路”
                else:
                    direction_score = 1 - (abs(angle_diff) / (np.pi / 2))  # 0到1的分数

                # b. 计算颜色分
                neighbor_color = tuple(template_image[ny, nx][:3])
                color_distance = np.linalg.norm(np.array(current_color) - np.array(neighbor_color))
                color_score = 1.0 / (1.0 + color_distance / 100.0)  # 0到1的分数

                # c. 计算总分
                total_score = (direction_score * HEURISTIC_WEIGHTS["OUTWARD_DIRECTION"] +
                               color_score * HEURISTIC_WEIGHTS["COLOR_COHESION"])

                if total_score > max_score:
                    max_score = total_score
                    best_neighbor = neighbor

            if best_neighbor is None and unvisited_in_contour:
                best_neighbor = min(unvisited_in_contour.values(),
                                    key=lambda p: ((p[0] - px) ** 2 + (p[1] - py) ** 2))

            current_pixel = best_neighbor
            if not current_pixel: break  # 安全中断

        final_ordered_pixels.extend(traced_contour)

    return final_ordered_pixels


# =================== [ 函数结束 ] ===================


# --- 为了完整性，附上视频生成函数 ---
def create_video_visualization(pixels_ordered, shape, output_filename, fps, pixels_per_frame):
    print(f"\n--- Generating MP4 Animation ---")
    height, width, _ = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    total_pixels = len(pixels_ordered)
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
        fused_pixels, template_image = prepare_fused_outline_pixels(ORIGINAL_TEMPLATE_FILENAME, EDGE_MAP_FILENAME)

        # 2. 运行我们全新的、带有启发式评分的算法
        ordered_pixels = run_final_algorithm(fused_pixels, template_image)

        # 3. 生成视频
        create_video_visualization(ordered_pixels, template_image.shape, OUTPUT_FILENAME, VIDEO_FPS, PIXELS_PER_FRAME)

        print("\n--- Advanced Heuristic Visualization Complete ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")