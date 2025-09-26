import cv2
import numpy as np
import os
import json
import re
from psd_tools import PSDImage
from psd_tools.psd.image_resources import ImageResource, Resource
from psd_tools.psd.vector import Path, Subpath, Knot
from io import BytesIO
from collections import deque
from numba import njit
from scipy.spatial import distance as dist

# ========================================================================
# >> SETTINGS <<
# ========================================================================
PSD_FILE_PATH = "dst/175822732/175822732_undithered_backup.psd"
PIXEL_ART_PATH = "dst/175822732/175822732_converted.png"
UNDITHERED_PIXEL_ART_PATH = "dst/175822732/175822732_undithered.png"
OUTPUT_JSON_PATH = "wplacer_draw_order_175822732_new.json"
OUTPUT_VISUALIZATION_PATH = "path_visualization.png"

# [新增] 路径处理参数: 处理前 N 条路径的首个子路径作为轮廓
NUM_OUTLINE_PATHS = 1

# 算法参数
ERROR_BOUND = 2
NEIGHBOUR_RANGE = 8
COLOR_SIMILARITY_THRESHOLD = 25  # 用于“内推”修正，判断毛刺

# 保守外推算法参数
CONSERVATIVE_EXTRAPOLATION_ENABLED = True
SIMILARITY_THRESHOLD_FOR_EXTRAPOLATION = 12  # 内外颜色相似度阈值(LAB)，低于此值则触发外推
EDGE_DIFFERENCE_THRESHOLD_FOR_EXTRAPOLATION = 18  # 边缘颜色差异阈值(LAB)，高于此值则认为是边缘
MAX_SCAN_DISTANCE = 2  # 向外扫描的最大像素距离

# [新增] 路径连续性插值参数
ENABLE_POST_INTERPOLATION = True  # 是否启用最终路径的插值填补


# ========================================================================
# >> 辅助函数 <<
# ========================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        return super(NumpyJSONEncoder, self).default(obj)


def get_lab_color(bgr_color: np.ndarray) -> np.ndarray:
    """将单个BGR颜色转换为LAB颜色"""
    bgr_color_3d = np.uint8([[bgr_color]])
    return cv2.cvtColor(bgr_color_3d, cv2.COLOR_BGR2LAB)[0][0]


def color_difference(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """计算两种LAB颜色之间的CIEDE2000差异值"""
    return dist.cdist(lab1.reshape(1, -1).astype(float), lab2.reshape(1, -1).astype(float), 'euclidean')[0][0]


def visualize_correction_stages(
        initial_path: list[tuple[int, int]],
        extrapolated_path: list[tuple[int, int]],
        refined_path: list[tuple[int, int]],
        width: int,
        height: int,
        output_path: str
):
    """
    [已重构] 将路径修正的三个阶段可视化到一张图片上。
    - 原始路径 (黄色)
    - 保守外推后 (蓝色)
    - 最终路径 (绿色)
    着色优先级: 原始 > 外推 > 最终
    """
    print(f"\nVisualizing path correction stages...")
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)
    INITIAL_COLOR = (0, 255, 255)  # 黄色
    EXTRAPOLATED_COLOR = (255, 0, 0)  # 蓝色
    FINAL_COLOR = (0, 255, 0)  # 绿色

    # 按相反顺序绘制以实现优先级覆盖
    if refined_path:
        vis_image[np.array(refined_path)[:, 1], np.array(refined_path)[:, 0]] = FINAL_COLOR
    if extrapolated_path:
        vis_image[np.array(extrapolated_path)[:, 1], np.array(extrapolated_path)[:, 0]] = EXTRAPOLATED_COLOR
    if initial_path:
        vis_image[np.array(initial_path)[:, 1], np.array(initial_path)[:, 0]] = INITIAL_COLOR

    # 添加图例
    legend_x, legend_y = 10, 10
    cv2.rectangle(vis_image, (legend_x, legend_y), (legend_x + 20, legend_y + 20), INITIAL_COLOR, -1)
    cv2.putText(vis_image, "Initial Path", (legend_x + 30, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.rectangle(vis_image, (legend_x, legend_y + 30), (legend_x + 20, legend_y + 50), EXTRAPOLATED_COLOR, -1)
    cv2.putText(vis_image, "After Extrapolation (Stage 1)", (legend_x + 30, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    cv2.rectangle(vis_image, (legend_x, legend_y + 60), (legend_x + 20, legend_y + 80), FINAL_COLOR, -1)
    cv2.putText(vis_image, "Final Path (After all stages)", (legend_x + 30, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    try:
        cv2.imwrite(output_path, vis_image)
        print(f"  -> Saved correction stages visualization to '{output_path}'")
    except Exception as e:
        print(f"  - ERROR: Could not save visualization image. Reason: {e}")


# ========================================================================
# >> 核心算法与数据提取 <<
# ========================================================================

def extract_grouped_paths_from_psd(psd: PSDImage) -> list[list[Subpath]]:
    """
    [已重构] 从PSD中提取路径，并按其原始资源ID进行分组。
    """
    print("Extracting and grouping stroke paths from Image Resources (PATH_INFO)...")
    path_resources = {key.value: res for key, res in psd.image_resources.items() if
                      isinstance(key, Resource) and Resource.is_path_info(key.value)}
    if not path_resources:
        print("  - No PATH_INFO resources found.");
        return []

    sorted_keys = sorted(path_resources.keys())
    print(f"  - Found {len(sorted_keys)} PATH_INFO resource(s). Decoding...")

    all_grouped_subpaths = []
    for key in sorted_keys:
        resource = path_resources[key]
        try:
            records = Path.read(BytesIO(resource.data))
            subpaths = [record for record in records if isinstance(record, Subpath)]
            if subpaths:
                print(f"    - Decoded {len(subpaths)} Subpath object(s) from Resource ID {key}.")
                all_grouped_subpaths.append(subpaths)
        except Exception as e:
            print(f"    - ERROR: Failed to decode Resource ID {key}. Reason: {e}")

    return all_grouped_subpaths


def rasterize_subpaths_ordered(subpaths: list[Subpath], pixel_art_image: np.ndarray):
    """
    将 Subpath 栅格化为有序像素列表。
    """
    height, width, _ = pixel_art_image.shape
    if not subpaths: return []

    final_ordered_coords = {}
    for subpath in subpaths:
        knots_in_subpath = [k for k in subpath if isinstance(k, Knot)]
        if len(knots_in_subpath) < 1: continue

        abs_points = [(knot.anchor[1] * width, knot.anchor[0] * height) for knot in knots_in_subpath]
        if subpath.is_closed() and len(abs_points) > 1:
            abs_points.append(abs_points[0])

        for j in range(len(abs_points) - 1):
            x1, y1 = abs_points[j]
            x2, y2 = abs_points[j + 1]
            dx, dy = x2 - x1, y2 - y1
            steps = int(max(abs(dx), abs(dy)))
            if steps == 0: steps = 1
            x_inc, y_inc = dx / steps, dy / steps
            current_x, current_y = x1, y1
            for _ in range(steps + 1):
                px, py = int(round(current_x)), int(round(current_y))
                if 0 <= px < width and 0 <= py < height:
                    final_ordered_coords[(px, py)] = None
                current_x += x_inc;
                current_y += y_inc

    return list(final_ordered_coords.keys())


def extrapolate_inner_path_points(
        initial_outline_pixels: list[tuple[int, int]],
        outline_contour: np.ndarray,
        undithered_image: np.ndarray
) -> tuple[list[tuple[int, int]], set[int]]:
    """
    [已重构] [阶段一] 执行保守的边缘外推。
    新逻辑: 只处理那些颜色与外部区域平均色相似的路径点。
    """
    print("\n--- Pre-correction (Stage 1): Extrapolating Inner Path Points (Refined Logic) ---")
    height, width, _ = undithered_image.shape
    lab_undithered_image = cv2.cvtColor(undithered_image, cv2.COLOR_BGR2LAB)

    extrapolated_path = []
    moved_indices = set()
    extrapolated_count = 0

    total_pixels = len(initial_outline_pixels)

    for i, p in enumerate(initial_outline_pixels):
        px, py = p

        if (i + 1) % 500 == 0:
            print(f"  - Scanning pixel {i + 1}/{total_pixels}...")

        # 1. 获取邻域并划分内外
        y_min, y_max = max(0, py - NEIGHBOUR_RANGE), min(height, py + NEIGHBOUR_RANGE + 1)
        x_min, x_max = max(0, px - NEIGHBOUR_RANGE), min(width, px + NEIGHBOUR_RANGE + 1)

        outside_pixels_coords = []
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if cv2.pointPolygonTest(outline_contour, (x, y), False) < 0:
                    outside_pixels_coords.append((x, y))

        # 如果没有外部点，无法判断，跳过
        if not outside_pixels_coords:
            extrapolated_path.append(p)
            continue

        # 2. 计算外部平均色
        outside_colors_lab = [lab_undithered_image[y, x] for x, y in outside_pixels_coords]
        avg_outside_color_lab = np.mean(outside_colors_lab, axis=0)

        # 3. [核心修改] 新的触发条件：比较当前点颜色与外部平均色
        original_color_lab = lab_undithered_image[py, px]
        similarity_diff = color_difference(original_color_lab, avg_outside_color_lab)

        # 如果当前点颜色与外部不像，说明不是目标情况，直接保留原点
        if similarity_diff > SIMILARITY_THRESHOLD_FOR_EXTRAPOLATION:
            extrapolated_path.append(p)
            continue

        # --- [ 核心外推逻辑: 仅在当前点与外部颜色相似时触发 ] ---
        # 4. 确定外推方向
        center_out = np.mean(outside_pixels_coords, axis=0)
        vec_x, vec_y = center_out[0] - px, center_out[1] - py

        mag = np.hypot(vec_x, vec_y)
        if mag < 1e-6:  # 向量太小，无法确定方向
            extrapolated_path.append(p)
            continue

        unit_vec_x, unit_vec_y = vec_x / mag, vec_y / mag

        # 5. 沿途扫描寻找第一个颜色差异大的边缘点 (条件2.1)
        best_candidate = p
        found_edge = False

        for step in range(1, MAX_SCAN_DISTANCE + 1):
            scan_x = int(round(px + unit_vec_x * step))
            scan_y = int(round(py + unit_vec_y * step))

            if not (0 <= scan_x < width and 0 <= scan_y < height): break

            scan_color_lab = lab_undithered_image[scan_y, scan_x]
            edge_diff = color_difference(original_color_lab, scan_color_lab)

            if edge_diff > EDGE_DIFFERENCE_THRESHOLD_FOR_EXTRAPOLATION:
                best_candidate = (scan_x, scan_y)
                found_edge = True
                break  # 找到第一个就停止

        if found_edge:
            moved_indices.add(i)
            extrapolated_count += 1

        extrapolated_path.append(best_candidate)

    print(f"  -> Extrapolation scan complete. Moved {extrapolated_count} inner points to the edge.")
    return extrapolated_path, moved_indices


def refine_outline_path(path_from_stage1, moved_indices_from_stage1, outline_contour, undithered_image):
    """[阶段二] 内推修正，跳过第一阶段已移动的点。"""
    print("\n--- Main Correction (Stage 2): Refining Outline Path ---")
    height, width, _ = undithered_image.shape
    lab_undithered_image = cv2.cvtColor(undithered_image, cv2.COLOR_BGR2LAB)
    corrected_path, corrected_count = [], 0
    total_pixels = len(path_from_stage1)
    for i, p in enumerate(path_from_stage1):
        if i in moved_indices_from_stage1:
            corrected_path.append(p);
            continue
        px, py = p
        if (i + 1) % 500 == 0: print(f"  - Refining pixel {i + 1}/{total_pixels}...")
        y_min, y_max = max(0, py - ERROR_BOUND), min(height, py + ERROR_BOUND + 1)
        x_min, x_max = max(0, px - ERROR_BOUND), min(width, px + ERROR_BOUND + 1)
        inside_coords, outside_coords = [], []
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                (inside_coords if cv2.pointPolygonTest(outline_contour, (x, y), False) >= 0 else outside_coords).append(
                    (x, y))
        if not outside_coords:
            corrected_path.append(p);
            continue
        avg_outside_lab = np.mean([lab_undithered_image[y, x] for x, y in outside_coords], axis=0)
        if color_difference(lab_undithered_image[py, px], avg_outside_lab) > COLOR_SIMILARITY_THRESHOLD:
            corrected_path.append(p);
            continue
        valid_candidates = [c for c in inside_coords if color_difference(lab_undithered_image[c[1], c[0]],
                                                                         avg_outside_lab) >= COLOR_SIMILARITY_THRESHOLD]
        if valid_candidates:
            distances = [dist.euclidean(p, cand) for cand in valid_candidates]
            best_candidate = valid_candidates[np.argmin(distances)]
            corrected_path.append(best_candidate);
            corrected_count += 1
        else:
            corrected_path.append(p)
    print(f"  -> Path refinement complete. Corrected {corrected_count} outlier pixels (excluding locked points).")
    return corrected_path


def interpolate_path_gaps(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[阶段三] 遍历路径，通过线性插值填补不连续的像素点。"""
    print("\n--- Post-correction (Stage 3): Interpolating Gaps ---")
    if len(path) < 2: return path
    continuous_path, pixels_added = [path[0]], 0
    for i in range(1, len(path)):
        p1, p2 = path[i - 1], path[i]
        dist_val = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        if dist_val > 1:
            num_steps = int(dist_val)
            x_vals, y_vals = np.linspace(p1[0], p2[0], num_steps + 1), np.linspace(p1[1], p2[1], num_steps + 1)
            for step in range(1, num_steps):
                interp_point = (int(round(x_vals[step])), int(round(y_vals[step])))
                if interp_point != continuous_path[-1]:
                    continuous_path.append(interp_point);
                    pixels_added += 1
        if p2 != continuous_path[-1]:
            continuous_path.append(p2)
    print(f"  -> Interpolation complete. Added {pixels_added} pixels to ensure path continuity.")
    return continuous_path


def find_contiguous_regions(pixel_coords_set, pixel_art_image):
    if not pixel_coords_set: return {}
    pixels_by_color = {}
    for x, y in pixel_coords_set:
        color = tuple(int(c) for c in pixel_art_image[y, x][:3])
        if color not in pixels_by_color: pixels_by_color[color] = []
        pixels_by_color[color].append((x, y))
    final_regions_by_color = {}
    height, width, _ = pixel_art_image.shape
    for color, coords in pixels_by_color.items():
        mask = np.zeros((height, width), dtype=np.uint8)
        coords_array = np.array(coords)
        mask[coords_array[:, 1], coords_array[:, 0]] = 255
        num_labels, labels = cv2.connectedComponents(mask, 8)
        regions = [[] for _ in range(num_labels)]
        for x, y in coords:
            regions[labels[y, x]].append((x, y))
        final_regions_by_color[color] = [r for r in regions if r and labels[r[0][1], r[0][0]] != 0]
    return final_regions_by_color


@njit
def _find_longest_strips_njit(coords_map, width, height):
    max_h_len, best_h_y, best_h_x_start = 0, -1, -1
    for y in range(height):
        current_h_len, current_h_x_start = 0, -1
        for x in range(width):
            if coords_map[y, x]:
                if current_h_len == 0: current_h_x_start = x
                current_h_len += 1
            else:
                if current_h_len > max_h_len: max_h_len, best_h_y, best_h_x_start = current_h_len, y, current_h_x_start
                current_h_len = 0
        if current_h_len > max_h_len: max_h_len, best_h_y, best_h_x_start = current_h_len, y, current_h_x_start
    max_v_len, best_v_x, best_v_y_start = 0, -1, -1
    for x in range(width):
        current_v_len, current_v_y_start = 0, -1
        for y in range(height):
            if coords_map[y, x]:
                if current_v_len == 0: current_v_y_start = y
                current_v_len += 1
            else:
                if current_v_len > max_v_len: max_v_len, best_v_x, best_v_y_start = current_v_len, x, current_v_y_start
                current_v_len = 0
        if current_v_len > max_v_len: max_v_len, best_v_x, best_v_y_start = current_v_len, x, current_v_y_start
    return max_h_len, best_h_y, best_h_x_start, max_v_len, best_v_x, best_v_y_start


def sort_single_region_contiguously(region_coords):
    """[已修正] 区域内排序，修复了IndexError"""
    if not region_coords: return []
    coords_array = np.array(region_coords)
    min_x, min_y = np.min(coords_array, axis=0)

    # --- [ 此处是修正点 ] ---
    # 显式、正确地计算宽度和高度，避免解包错误
    max_x, max_y = np.max(coords_array, axis=0)
    map_width = max_x - min_x + 1
    map_height = max_y - min_y + 1
    # -----------------------

    coords_map = np.zeros((map_height, map_width), dtype=np.bool_)
    coords_map[coords_array[:, 1] - min_y, coords_array[:, 0] - min_x] = True

    final_path, num_remaining = [], len(region_coords)
    while num_remaining > 0:
        h_len, h_y, h_x_start, v_len, v_x, v_y_start = _find_longest_strips_njit(coords_map, map_width, map_height)
        if h_len == 0 and v_len == 0: break
        if h_len >= v_len:
            strip_len = h_len
            for i in range(strip_len):
                x, y = h_x_start + i, h_y
                final_path.append((x + min_x, y + min_y));
                coords_map[y, x] = False
        else:
            strip_len = v_len
            for i in range(strip_len):
                y, x = v_y_start + i, v_x
                final_path.append((x + min_x, y + min_y));
                coords_map[y, x] = False
        num_remaining -= strip_len
    return final_path


def sort_and_flatten_regions(regions_by_color):
    final_order = []
    color_pixel_counts = {color: sum(len(region) for region in regions) for color, regions in regions_by_color.items()}
    sorted_colors = sorted(color_pixel_counts, key=color_pixel_counts.get, reverse=True)
    for color in sorted_colors:
        regions = regions_by_color[color]
        regions.sort(key=len, reverse=True)
        for region in regions:
            final_order.extend(sort_single_region_contiguously(region))
    return final_order


# ========================================================================
# >> 主执行逻辑 <<
# ========================================================================
def main():
    print("--- WPlacer Draw Order Generator ---")
    try:
        psd = PSDImage.open(PSD_FILE_PATH)
        pixel_art_image = cv2.imread(PIXEL_ART_PATH, cv2.IMREAD_UNCHANGED)
        undithered_image = cv2.imread(UNDITHERED_PIXEL_ART_PATH)
        height, width, _ = pixel_art_image.shape
    except Exception as e:
        print(f"Error loading files: {e}");
        return

    print("\n--- Phase A: Parsing PSD Data & Path Preparation ---")
    all_grouped_subpaths = extract_grouped_paths_from_psd(psd)
    if not all_grouped_subpaths: return

    print(f"\nSelecting first subpath from the first {NUM_OUTLINE_PATHS} path(s) as the outline...")
    outline_subpaths = []
    num_to_process = min(NUM_OUTLINE_PATHS, len(all_grouped_subpaths))
    for i in range(num_to_process):
        if all_grouped_subpaths[i]:
            outline_subpaths.append(all_grouped_subpaths[i][0])
    print(f"  -> Selected {len(outline_subpaths)} subpath(s) for outline correction.")

    outline_contour_points = []
    for subpath in outline_subpaths:
        points = [[k.anchor[1] * width, k.anchor[0] * height] for k in subpath if isinstance(k, Knot)]
        outline_contour_points.extend(points)
    outline_contour = np.array(outline_contour_points, dtype=np.float32).reshape((-1, 1, 2))

    initial_outline_pixels = rasterize_subpaths_ordered(outline_subpaths, pixel_art_image)

    print("\n--- Path Correction Stage Initiated ---")
    moved_indices_stage1 = set()
    if CONSERVATIVE_EXTRAPOLATION_ENABLED:
        extrapolated_path, moved_indices_stage1 = extrapolate_inner_path_points(initial_outline_pixels, outline_contour,
                                                                                undithered_image)
    else:
        print("\n--- Pre-correction (Stage 1) was skipped by settings. ---")
        extrapolated_path = initial_outline_pixels

    refined_path = refine_outline_path(extrapolated_path, moved_indices_stage1, outline_contour, undithered_image)

    if ENABLE_POST_INTERPOLATION:
        final_outline_pixels = interpolate_path_gaps(refined_path)
    else:
        print("\n--- Post-correction (Stage 3) was skipped by settings. ---")
        final_outline_pixels = refined_path

    print("\n--- Path Correction Finished ---")

    visualize_correction_stages(initial_outline_pixels, extrapolated_path, final_outline_pixels, width, height,
                                OUTPUT_VISUALIZATION_PATH)

    print("\n--- Phase B: Building Final Draw Order ---")
    final_draw_order, processed_pixels = [], set()

    print("\nStep 1: Processing refined and finalized outline path...")
    unique_step1_pixels = [p for p in final_outline_pixels if
                           tuple(p) not in processed_pixels and (processed_pixels.add(tuple(p)) or True)]
    final_draw_order.extend(unique_step1_pixels)
    print(f"  -> Added {len(unique_step1_pixels)} unique pixels from the final outline.")

    print("\nStep 1.5: Processing remaining stroke paths...")
    all_subpaths_flat = [sub for group in all_grouped_subpaths for sub in group]
    remaining_subpaths = [sub for sub in all_subpaths_flat if sub not in outline_subpaths]
    remaining_pixels = rasterize_subpaths_ordered(remaining_subpaths, pixel_art_image)
    unique_step1_5_pixels = [p for p in remaining_pixels if
                             tuple(p) not in processed_pixels and (processed_pixels.add(tuple(p)) or True)]
    final_draw_order.extend(unique_step1_5_pixels)
    print(f"  -> Added {len(unique_step1_5_pixels)} unique pixels from other paths.")

    print("\nExtracting numbered layer regions...")
    layer_regions = {}
    for layer in psd:
        if layer.is_visible() and not layer.is_group():
            match = re.search(r'(\d+)$', layer.name)
            if match:
                num = int(match.group(1))
                if layer.width > 0 and layer.height > 0:
                    layer_numpy = layer.numpy()
                    if layer_numpy.shape[2] == 4:
                        alpha = layer_numpy[:, :, 3]
                        ys, xs = np.where(alpha > 0)
                        layer_regions[num] = set(zip(xs + layer.left, ys + layer.top))
                        print(f"  - Found layer {num} ('{layer.name}') with {len(layer_regions[num])} pixels.")

    layer_nums_step2 = sorted([num for num in layer_regions if num >= 3])
    print(f"\nStep 2: Processing regions from layer 3 upwards ({len(layer_nums_step2)} layers)...")
    for num in layer_nums_step2:
        print(f"  - Processing Layer {num}...")
        pixels = layer_regions.get(num, set()) - processed_pixels
        regions = find_contiguous_regions(pixels, pixel_art_image)
        sorted_pixels = sort_and_flatten_regions(regions)
        final_draw_order.extend(sorted_pixels);
        processed_pixels.update(map(tuple, sorted_pixels))
        print(f"    -> Added {len(sorted_pixels)} unique pixels.")

    print("\nStep 3: Processing layer 2 minus regions from step 2...")
    if 2 in layer_regions:
        layers_3_up_union = set().union(*(layer_regions.get(num, set()) for num in layer_nums_step2))
        pixels = (layer_regions[2] - layers_3_up_union) - processed_pixels
        sorted_pixels = sort_and_flatten_regions(find_contiguous_regions(pixels, pixel_art_image))
        final_draw_order.extend(sorted_pixels);
        processed_pixels.update(map(tuple, sorted_pixels))
        print(f"  -> Added {len(sorted_pixels)} unique pixels from (Layer 2 - Layers >= 3).")
    else:
        print("  -> Layer 2 not found, skipping.")

    print("\nStep 4: Processing remaining pixels...")
    all_visible_pixels = set()
    if pixel_art_image.shape[2] == 4:
        ys, xs = np.where(pixel_art_image[:, :, 3] > 0)
        all_visible_pixels = set(zip(xs, ys))
    else:
        all_visible_pixels = set((x, y) for x in range(width) for y in range(height))

    pixels = all_visible_pixels - processed_pixels
    sorted_pixels = sort_and_flatten_regions(find_contiguous_regions(pixels, pixel_art_image))
    final_draw_order.extend(sorted_pixels)
    print(f"  -> Added {len(sorted_pixels)} unique pixels from the rest of the image.")

    print(f"\n--- Phase C: Saving Output ---")
    print(f"Total pixels in final draw order: {len(final_draw_order)}")
    try:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(final_draw_order, f, cls=NumpyJSONEncoder)
        print(f"Successfully saved the final draw order to '{OUTPUT_JSON_PATH}'")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()