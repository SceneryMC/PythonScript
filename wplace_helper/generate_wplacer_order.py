import cv2
import numpy as np
import os
import json
from psd_tools import PSDImage
from psd_tools.psd.image_resources import ImageResource, Resource
from psd_tools.psd.vector import Path, Subpath, Knot
from io import BytesIO
from collections import deque
from numba import njit

# ========================================================================
# >> SETTINGS <<
# ========================================================================
PSD_FILE_PATH = "dst/175822732/175822732_undithered_backup.psd"
PIXEL_ART_PATH = "dst/175822732/175822732_converted.png"
OUTPUT_JSON_PATH = "wplacer_draw_order/175822732.json"
os.makedirs('wplacer_draw_order', exist_ok=True)


# ========================================================================
# >> 核心算法与辅助函数 (已重构) <<
# ========================================================================

# --- [ 路径与图层提取 (保持不变) ] ---
def extract_paths_from_image_resources(psd: PSDImage):  # ...
    print("Extracting stroke paths from Image Resources (PATH_INFO)...")
    path_resources = {key.value: res for key, res in psd.image_resources.items() if
                      isinstance(key, Resource) and Resource.is_path_info(key.value)}
    if not path_resources: print("  - No PATH_INFO resources found."); return []
    sorted_keys = sorted(path_resources.keys());
    print(f"  - Found {len(sorted_keys)} PATH_INFO resource(s). Decoding...")
    all_subpaths = []
    for key in sorted_keys:
        resource = path_resources[key]
        try:
            records = Path.read(BytesIO(resource.data))
            subpaths = [record for record in records if isinstance(record, Subpath)]
            if subpaths: print(
                f"    - Decoded {len(subpaths)} Subpath object(s) from Resource ID {key}."); all_subpaths.extend(
                subpaths)
        except Exception as e:
            print(f"    - ERROR: Failed to decode Resource ID {key}. Reason: {e}")
    return all_subpaths


# ========================================================================
# >> [已重构] 包含智能修正的栅格化函数 <<
# ========================================================================

def rasterize_subpaths_ordered(subpaths: list[Subpath], pixel_art_image: np.ndarray):
    """
    [已重构] 将 Subpath 栅格化为有序像素列表，并智能修正路径使其贴合非透明像素。
    """
    height, width, channels = pixel_art_image.shape
    # 我们需要一个Alpha通道的副本来快速检查透明度
    # 如果图像没有Alpha通道，我们创建一个全不透明的通道
    if channels == 4:
        alpha_channel = pixel_art_image[:, :, 3]
    else:
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)

    if not subpaths:
        return []

    print(f"\nRasterizing {len(subpaths)} subpaths with Smart Correction...")

    final_ordered_coords = {}
    last_valid_point = None

    for i, subpath in enumerate(subpaths):
        knots_in_subpath = []
        for segment in subpath:
            if isinstance(segment, Knot):
                knots_in_subpath.append(segment)
            elif hasattr(segment, '_items'):
                knots_in_subpath.extend(segment._items)

        if len(knots_in_subpath) < 1:
            continue

        abs_points = [(knot.anchor[1] * width, knot.anchor[0] * height) for knot in knots_in_subpath]

        if subpath.is_closed() and len(abs_points) > 1:
            abs_points.append(abs_points[0])

        # 遍历路径的每一段
        for j in range(len(abs_points)):
            # 处理第一个点
            if j == 0:
                x1, y1 = abs_points[j]
                px, py = int(round(x1)), int(round(y1))

                # 如果第一个点就是透明的，需要特殊处理
                if 0 <= px < width and 0 <= py < height and alpha_channel[py, px] == 0:
                    # 在一个小范围内 (例如5x5) 寻找最近的非透明点作为起点
                    search_radius = 5
                    min_dist_sq = float('inf')
                    found_alt = False
                    alt_px, alt_py = -1, -1

                    for sx in range(max(0, px - search_radius), min(width, px + search_radius + 1)):
                        for sy in range(max(0, py - search_radius), min(height, py + search_radius + 1)):
                            if alpha_channel[sy, sx] > 0:
                                dist_sq = (sx - px) ** 2 + (sy - py) ** 2
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    alt_px, alt_py = sx, sy
                                    found_alt = True
                    if found_alt:
                        px, py = alt_px, alt_py

                # 只有当点有效时才添加
                if 0 <= px < width and 0 <= py < height and alpha_channel[py, px] > 0:
                    final_ordered_coords[(px, py)] = None
                    last_valid_point = (px, py)

                if len(abs_points) == 1:  # 如果路径只有一个点
                    continue

            # 处理线段
            x1, y1 = abs_points[j]
            x2, y2 = abs_points[j + 1] if j + 1 < len(abs_points) else abs_points[j]

            # 确定前进方向
            dx_fwd, dy_fwd = x2 - x1, y2 - y1

            # DDA 插值
            steps = int(max(abs(dx_fwd), abs(dy_fwd)))
            if steps == 0: steps = 1
            x_inc, y_inc = dx_fwd / steps, dy_fwd / steps

            current_x, current_y = x1, y1
            for step in range(steps + 1):
                px, py = int(round(current_x)), int(round(current_y))

                # 边界检查
                if not (0 <= px < width and 0 <= py < height):
                    current_x += x_inc
                    current_y += y_inc
                    continue

                # 验证像素
                if alpha_channel[py, px] > 0:
                    # 像素有效
                    final_ordered_coords[(px, py)] = None
                    last_valid_point = (px, py)
                else:
                    # --- [ 核心修正：智能寻找替代点 ] ---
                    # 像素无效 (透明)，需要寻找替代

                    # 确定前进方向 (最好用上一个有效点来计算，更精确)
                    if last_valid_point:
                        vec_x, vec_y = px - last_valid_point[0], py - last_valid_point[1]
                    else:  # 如果是路径的第一个点就无效
                        vec_x, vec_y = dx_fwd, dy_fwd

                    # 归一化方向向量
                    mag = np.sqrt(vec_x ** 2 + vec_y ** 2)
                    if mag > 0:
                        vec_x, vec_y = vec_x / mag, vec_y / mag

                    # 确定两个垂直搜索方向
                    # perp_vec1 = (-vec_y, vec_x)
                    # perp_vec2 = (vec_y, -vec_x)
                    p_dx1, p_dy1 = -vec_y, vec_x
                    p_dx2, p_dy2 = vec_y, -vec_x

                    found_alt = False
                    # 向两侧交替搜索，看谁先找到
                    for search_dist in range(1, 15):  # 最大搜索15像素远
                        # 方向1
                        nx1 = int(round(px + p_dx1 * search_dist))
                        ny1 = int(round(py + p_dy1 * search_dist))
                        if 0 <= nx1 < width and 0 <= ny1 < height and alpha_channel[ny1, nx1] > 0:
                            final_ordered_coords[(nx1, ny1)] = None
                            last_valid_point = (nx1, ny1)
                            found_alt = True
                            break

                        # 方向2
                        nx2 = int(round(px + p_dx2 * search_dist))
                        ny2 = int(round(py + p_dy2 * search_dist))
                        if 0 <= nx2 < width and 0 <= ny2 < height and alpha_channel[ny2, nx2] > 0:
                            final_ordered_coords[(nx2, ny2)] = None
                            last_valid_point = (nx2, ny2)
                            found_alt = True
                            break

                    if found_alt:
                        # 找到了替代点，更新当前插值位置，避免路径大幅跳跃
                        current_x, current_y = last_valid_point[0], last_valid_point[1]

                current_x += x_inc
                current_y += y_inc

    ordered_pixel_list = list(final_ordered_coords.keys())
    print(f"  -> Generated an ordered and corrected stroke path with {len(ordered_pixel_list)} unique pixels.")
    return ordered_pixel_list


# --- [ 区域分解 (已彻底重构) ] ---
def find_contiguous_regions(pixel_coords_set, pixel_art_image):
    """
    [已彻底重构] 使用“先按颜色分组，再寻找连续区域”的正确逻辑。
    """
    if not pixel_coords_set:
        return {}

    # 1. 颜色普查：首先将所有输入像素按颜色分组
    pixels_by_color = {}
    for x, y in pixel_coords_set:
        b, g, r, *_ = pixel_art_image[y, x]
        color = (int(b), int(g), int(r))
        if color not in pixels_by_color:
            pixels_by_color[color] = []
        pixels_by_color[color].append((x, y))

    # 2. 对每个“颜色桶”内部，独立进行连通组件分析
    final_regions_by_color = {}
    height, width, _ = pixel_art_image.shape

    for color, coords in pixels_by_color.items():
        # a. 为当前颜色创建一个单独的二值化地图
        mask = np.zeros((height, width), dtype=np.uint8)
        coords_array = np.array(coords)
        mask[coords_array[:, 1], coords_array[:, 0]] = 255

        # b. 在这个单色地图上运行OpenCV，寻找“岛屿”
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

        # c. 将找到的“岛屿”重新组织
        regions_for_this_color = []
        if num_labels > 1:
            # label 0 是背景，我们从1开始
            regions = [[] for _ in range(num_labels)]
            for x, y in coords:
                label_id = labels[y, x]
                if label_id != 0:
                    regions[label_id].append((x, y))

            # 过滤掉可能为空的列表
            regions_for_this_color = [r for r in regions if r]

        final_regions_by_color[color] = regions_for_this_color

    return final_regions_by_color


# --- [ 区域内排序 (保持JIT版本) ] ---
@njit
def _find_longest_strips_njit(coords_map: np.ndarray, width: int, height: int):  # ...
    max_h_len = 0;
    best_h_y, best_h_x_start = -1, -1
    for y in range(height):
        current_h_len = 0;
        current_h_x_start = -1
        for x in range(width):
            if coords_map[y, x]:
                if current_h_len == 0: current_h_x_start = x
                current_h_len += 1
            else:
                if current_h_len > max_h_len: max_h_len, best_h_y, best_h_x_start = current_h_len, y, current_h_x_start
                current_h_len = 0
        if current_h_len > max_h_len: max_h_len, best_h_y, best_h_x_start = current_h_len, y, current_h_x_start
    max_v_len = 0;
    best_v_x, best_v_y_start = -1, -1
    for x in range(width):
        current_v_len = 0;
        current_v_y_start = -1
        for y in range(height):
            if coords_map[y, x]:
                if current_v_len == 0: current_v_y_start = y
                current_v_len += 1
            else:
                if current_v_len > max_v_len: max_v_len, best_v_x, best_v_y_start = current_v_len, x, current_v_y_start
                current_v_len = 0
        if current_v_len > max_v_len: max_v_len, best_v_x, best_v_y_start = current_v_len, x, current_v_y_start
    return (max_h_len, best_h_y, best_h_x_start, max_v_len, best_v_x, best_v_y_start)


def sort_single_region_contiguously(region_coords):  # ...
    if not region_coords: return []
    coords_array = np.array(region_coords);
    min_x, min_y = np.min(coords_array, axis=0);
    max_x, max_y = np.max(coords_array, axis=0)
    map_height = max_y - min_y + 1;
    map_width = max_x - min_x + 1
    coords_map = np.zeros((map_height, map_width), dtype=np.bool_)
    local_coords = coords_array - np.array([min_x, min_y]);
    coords_map[local_coords[:, 1], local_coords[:, 0]] = True
    final_path = [];
    num_remaining = len(region_coords)
    while num_remaining > 0:
        h_len, h_y, h_x_start, v_len, v_x, v_y_start = _find_longest_strips_njit(coords_map, map_width, map_height)
        if h_len >= v_len:
            strip_len = h_len
            if strip_len == 0: break
            for i in range(strip_len):
                x, y = h_x_start + i, h_y
                final_path.append((x + min_x, y + min_y));
                coords_map[y, x] = False
            num_remaining -= strip_len
        else:
            strip_len = v_len
            if strip_len == 0: break
            for i in range(strip_len):
                y, x = v_y_start + i, v_x
                final_path.append((x + min_x, y + min_y));
                coords_map[y, x] = False
            num_remaining -= strip_len
    return final_path


def sort_and_flatten_regions(regions_by_color):  # ...
    final_order = []
    color_pixel_counts = {color: sum(len(region) for region in regions) for color, regions in regions_by_color.items()}
    sorted_colors = sorted(color_pixel_counts.keys(), key=lambda color: color_pixel_counts[color], reverse=True)
    for color in sorted_colors:
        regions = regions_by_color[color]
        regions.sort(key=len, reverse=True)
        for region in regions:
            sorted_region_pixels = sort_single_region_contiguously(region)
            final_order.extend(sorted_region_pixels)
    return final_order


# ========================================================================
# >> 主执行逻辑 (保持不变) <<
# ========================================================================
def main():
    # ... (Phase A: 解析PSD数据，保持不变) ...
    print("--- WPlacer Draw Order Generator ---")
    try:
        print(f"Loading PSD file: {PSD_FILE_PATH}")
        psd = PSDImage.open(PSD_FILE_PATH)
        print(f"Loading pixel art file: {PIXEL_ART_PATH}")
        pixel_art_image = cv2.imread(PIXEL_ART_PATH, cv2.IMREAD_UNCHANGED)
        if pixel_art_image is None: raise FileNotFoundError(f"Could not read image file: {PIXEL_ART_PATH}")
        height, width, _ = pixel_art_image.shape
        if psd.width != width or psd.height != height: print(
            f"Warning: PSD size ({psd.width}x{psd.height}) does not match pixel art size ({width}x{height}).")
    except Exception as e:
        print(f"Error loading files: {e}"); return

    print("\n--- Phase A: Parsing PSD Data ---")
    all_subpaths = extract_paths_from_image_resources(psd)

    # [核心修正] 调用新的、能返回有序列表的栅格化函数
    all_stroke_paths_coords_ordered = rasterize_subpaths_ordered(all_subpaths, pixel_art_image)

    print("\nExtracting numbered layer regions...")
    import re
    layer_regions = {};
    layer_name_pattern = re.compile(r'(\d+)$')
    for layer in psd:
        if layer.is_visible() and not layer.is_group():
            match = layer_name_pattern.search(layer.name)
            if match:
                layer_num = int(match.group(1));
                layer_image = layer.numpy()
                offset_x, offset_y = layer.left, layer.top
                if layer_image.shape[2] == 4:
                    alpha = layer_image[:, :, 3];
                    local_ys, local_xs = np.where(alpha > 0)
                    global_coords = set(zip(local_xs + offset_x, local_ys + offset_y))
                    layer_regions[layer_num] = global_coords
                    print(
                        f"  - Found layer {layer_num} ('{layer.name}') with {len(global_coords)} pixels at offset ({offset_x}, {offset_y}).")

    print("\n--- Phase B: Building Final Draw Order ---")
    final_draw_order = []
    processed_pixels = set()

    # [核心修正] Step 1: 直接使用有序的勾线像素，不再进行任何排序
    print("\nStep 1: Processing all stroke paths (sequentially)...")
    unique_step1_pixels = []
    for p in all_stroke_paths_coords_ordered:
        if p not in processed_pixels:
            unique_step1_pixels.append(p)
            processed_pixels.add(p)

    final_draw_order.extend(unique_step1_pixels)
    print(f"  -> Added {len(unique_step1_pixels)} unique pixels from strokes, preserving path direction.")

    print("\nStep 2: Processing regions from layer 3 upwards (individually)...")
    layer_nums_step2 = sorted([num for num in layer_regions.keys() if num >= 3])
    for num in layer_nums_step2:
        print(f"  - Processing Layer {num}...")
        pixels_to_process = layer_regions.get(num, set()) - processed_pixels
        regions_by_color = find_contiguous_regions(pixels_to_process, pixel_art_image)
        sorted_pixels = sort_and_flatten_regions(regions_by_color)
        final_draw_order.extend(sorted_pixels);
        processed_pixels.update(sorted_pixels)
        print(f"    -> Added {len(sorted_pixels)} unique pixels.")

    print("\nStep 3: Processing layer 2 minus regions from step 2...")
    if 2 in layer_regions:
        layer2_set = layer_regions[2]
        layers_3_and_up_union = set()
        for num in layer_nums_step2: layers_3_and_up_union.update(layer_regions.get(num, set()))
        step3_pixel_set = layer2_set - layers_3_and_up_union;
        step3_pixel_set -= processed_pixels
        regions_by_color_s3 = find_contiguous_regions(step3_pixel_set, pixel_art_image)
        sorted_step3_pixels = sort_and_flatten_regions(regions_by_color_s3)
        final_draw_order.extend(sorted_step3_pixels);
        processed_pixels.update(sorted_step3_pixels)
        print(f"  -> Added {len(sorted_step3_pixels)} unique pixels from (Layer 2 - Layers >= 3).")
    else:
        print("  -> Layer 2 not found, skipping.")

    print("\nStep 4: Processing remaining pixels (outside layer 2)...")
    all_visible_pixels = set()
    if pixel_art_image.shape[2] == 4:
        alpha = pixel_art_image[:, :, 3];
        ys, xs = np.where(alpha > 0)
        all_visible_pixels = set(zip(xs, ys))
    else:
        all_visible_pixels = set(
            [(x, y) for x in range(pixel_art_image.shape[1]) for y in range(pixel_art_image.shape[0])])
    step4_pixel_set = all_visible_pixels - processed_pixels
    regions_by_color_s4 = find_contiguous_regions(step4_pixel_set, pixel_art_image)
    sorted_step4_pixels = sort_and_flatten_regions(regions_by_color_s4)
    final_draw_order.extend(sorted_step4_pixels)
    print(f"  -> Added {len(sorted_step4_pixels)} unique pixels from the rest of the image.")

    print(f"\n--- Phase C: Saving Output ---")

    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            return super(NumpyJSONEncoder, self).default(obj)

    if pixel_art_image.shape[2] == 4:
        print(f"Original order contains {len(final_draw_order)} pixels. Filtering transparent pixels...")

        # 使用列表推导式进行一次高效的遍历和过滤
        # 只保留那些在原图中 alpha > 0 的像素
        final_draw_order_filtered = [
            (x, y) for x, y in final_draw_order
            if pixel_art_image[y, x, 3] > 0
        ]

        removed_count = len(final_draw_order) - len(final_draw_order_filtered)
        if removed_count > 0:
            print(f"  -> Removed {removed_count} transparent pixels.")

        final_draw_order = final_draw_order_filtered
    else:
        print("Image has no alpha channel. Skipping transparency filter.")

    print(f"Total non-transparent pixels in final draw order: {len(final_draw_order)}")

    try:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(final_draw_order, f, cls=NumpyJSONEncoder)
        print(f"Successfully saved the final draw order to '{OUTPUT_JSON_PATH}'")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()