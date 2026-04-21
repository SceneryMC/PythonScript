import math
import cv2
import numpy as np
import os
import json
import re
from psd_tools import PSDImage
from psd_tools.psd.image_resources import ImageResource, Resource
from psd_tools.psd.vector import Path, Subpath, Knot
from io import BytesIO
from collections import deque, Counter
from numba import njit, types
from numba.typed import List
from scipy.spatial import distance as dist

from wplace_helper.utils import WPLACE_COLOR_PALETTE, DEFAULT_FREE_COLORS

# ========================================================================
# >> SETTINGS <<
# ========================================================================
PSD_FILE_PATH = "dst/layla_birthday_2023/layla_birthday_2023.psd"
PIXEL_ART_PATH = "dst/layla_birthday_2023/layla_birthday_2023_converted.png"
UNDITHERED_PIXEL_ART_PATH = "dst/layla_birthday_2023/layla_birthday_2023_undithered.png"
OUTPUT_JSON_PATH = "wplacer_draw_order_try/layla_birthday_2023.json"
OUTPUT_VISUALIZATION_PATH = "wplacer_draw_order_try/path_visualization.png"
os.makedirs('wplacer_draw_order_try', exist_ok=True)

# 路径处理参数
NUM_OUTLINE_PATHS = 20

# [已修改] 算法参数
NEIGHBOUR_RANGE = 0  # 用于计算颜色平均值的大邻域范围 (e.g., 19x19)
ERROR_BOUND = 0      # 用于在内推时查找候选替换点的小范围 (e.g., 9x9)
COLOR_SIMILARITY_THRESHOLD = 0

# 保守外推算法参数
CONSERVATIVE_EXTRAPOLATION_ENABLED = False
SIMILARITY_THRESHOLD_FOR_EXTRAPOLATION = 12
EDGE_DIFFERENCE_THRESHOLD_FOR_EXTRAPOLATION = 18
MAX_SCAN_DISTANCE = 2

# 路径连续性插值参数
ENABLE_POST_INTERPOLATION = True

# [新增] 类人绘制算法参数
CONNECTIVITY_THRESHOLD = 100
INTERRUPT_SEARCH_RADIUS = 2
BASE_LOOKAHEAD = 1      # f(0) 的值，即在不连续向下移动时，维持方向所需的最短长度
LOOKAHEAD_GROWTH = 1 # f(n) 的增长斜率，n 每增加1，所需长度就增加这个值


# ========================================================================
# >> 辅助函数 <<
# ========================================================================

def find_transparent_pixels(
        pixel_collection: (list[tuple[int, int]] | set[tuple[int, int]]),
        pixel_art_image: np.ndarray,
        collection_name: str = "Unnamed Collection"
) -> list[tuple[int, int]]:
    """
    一个调试工具函数，用于检查给定的像素集合中哪些像素是透明的。

    Args:
        pixel_collection: 包含(x, y)坐标的列表或集合。
        pixel_art_image: 带有Alpha通道的源图像 (NumPy数组)。
        collection_name: (可选) 用于在打印输出中标识该集合的名称。

    Returns:
        一个列表，包含所有在源图像中是透明的像素坐标。
        如果图像没有Alpha通道或没有找到透明像素，则返回空列表。
    """
    print(f"\n--- 🕵️  Debugging: Checking '{collection_name}' for transparent pixels ---")

    # 1. 检查图像是否有Alpha通道
    height, width, channels = pixel_art_image.shape
    if channels < 4:
        print("  - 图像没有Alpha通道，无法进行透明度检查。跳过。")
        return []

    alpha_channel = pixel_art_image[:, :, 3]
    transparent_pixels_found = []

    # 2. 遍历集合中的每个像素
    for p in pixel_collection:
        # 确保坐标是元组格式
        px, py = tuple(p)

        # 边界检查
        if not (0 <= px < width and 0 <= py < height):
            print(f"  - 警告: 坐标 {p} 超出图像边界，跳过。")
            continue

        # 3. 检查Alpha值
        # Alpha值为0被认为是完全透明
        if alpha_channel[py, px] == 0:
            transparent_pixels_found.append(p)

    # 4. 报告结果
    if transparent_pixels_found:
        print(f"  - ❌ Found {len(transparent_pixels_found)} transparent pixels in '{collection_name}'.")
        # 只打印前10个以避免刷屏
        for i, p in enumerate(transparent_pixels_found):
            if i >= 10:
                print("    - ... and more.")
                break
            print(f"    - Transparent pixel at: {p}")
    else:
        print(f"  - ✅ No transparent pixels found in '{collection_name}'.")

    return transparent_pixels_found


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
        print("  - No PATH_INFO resources found.")
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
    [已恢复智能修正] 将 Subpath 栅格化为有序像素列表，并智能修正路径使其贴合非透明像素。
    """
    height, width, channels = pixel_art_image.shape

    # 必须有Alpha通道才能进行智能修正
    if channels < 4:
        print("  - 警告: 图像无Alpha通道，将执行无修正的栅格化。")
        # 在这里可以调用一个简化的、无Alpha检查的版本，或者直接继续（效果相同）
        pass

    alpha_channel = pixel_art_image[:, :, 3] if channels == 4 else np.full((height, width), 255, dtype=np.uint8)

    if not subpaths: return []

    final_ordered_coords = {}
    last_valid_point = None

    for subpath in subpaths:
        knots = [k for k in subpath if isinstance(k, Knot)]
        if not knots: continue

        abs_points = [(k.anchor[1] * width, k.anchor[0] * height) for k in knots]
        if subpath.is_closed() and len(abs_points) > 1:
            abs_points.append(abs_points[0])

        for j in range(len(abs_points) - 1):
            x1, y1 = abs_points[j]
            x2, y2 = abs_points[j + 1]
            dx, dy = x2 - x1, y2 - y1
            steps = int(max(abs(dx), abs(dy))) or 1
            x_inc, y_inc = dx / steps, dy / steps
            current_x, current_y = x1, y1

            for step in range(steps + 1):
                px, py = int(round(current_x)), int(round(current_y))

                # 边界检查
                if not (0 <= px < width and 0 <= py < height):
                    current_x += x_inc;
                    current_y += y_inc
                    continue

                # --- 核心的透明度检查 ---
                if alpha_channel[py, px] > 0:
                    # 像素有效 (非透明)
                    final_ordered_coords[(px, py)] = None
                    last_valid_point = (px, py)
                else:
                    # 像素无效 (透明)，需要寻找替代点
                    found_alt = False
                    # 在一个小范围内 (例如5x5) 寻找最近的非透明点
                    for search_radius in range(1, 6):
                        for sx in range(max(0, px - search_radius), min(width, px + search_radius + 1)):
                            for sy in range(max(0, py - search_radius), min(height, py + search_radius + 1)):
                                if alpha_channel[sy, sx] > 0:
                                    final_ordered_coords[(sx, sy)] = None
                                    last_valid_point = (sx, sy)
                                    # 立即更新当前位置，避免路径大幅跳跃
                                    current_x, current_y = sx, sy
                                    found_alt = True
                                    break
                            if found_alt: break
                        if found_alt: break

                current_x += x_inc
                current_y += y_inc
    # print(f"length = {len(final_ordered_coords)}")
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


def refine_outline_path(
        path_from_stage1: list[tuple[int, int]],
        moved_indices_from_stage1: set[int],
        outline_contour: np.ndarray,
        undithered_image: np.ndarray,
        pixel_art_image: np.ndarray
) -> list[tuple[int, int]]:
    """
    [阶段二 - 最终优化版] 内推修正。
    在构建候选列表时，就提前排除了所有透明像素。
    """
    print("\n--- Main Correction (Stage 2): Refining Outline Path (with Proactive Transparency Filter) ---")
    height, width, _ = undithered_image.shape
    lab_undithered_image = cv2.cvtColor(undithered_image, cv2.COLOR_BGR2LAB)

    channels = pixel_art_image.shape[2]
    alpha_channel = pixel_art_image[:, :, 3] if channels == 4 else None

    corrected_path, corrected_count = [], 0
    total_pixels = len(path_from_stage1)

    for i, p in enumerate(path_from_stage1):
        if i in moved_indices_from_stage1:
            corrected_path.append(p)
            continue

        # ... (前半部分的颜色比较逻辑完全不变) ...
        px, py = p
        if (i + 1) % 500 == 0: print(f"  - Refining pixel {i + 1}/{total_pixels}...")
        y_min, y_max = max(0, py - NEIGHBOUR_RANGE), min(height, py + NEIGHBOUR_RANGE + 1)
        x_min, x_max = max(0, px - NEIGHBOUR_RANGE), min(width, px + NEIGHBOUR_RANGE + 1)
        outside_coords = [(x, y) for y in range(y_min, y_max) for x in range(x_min, x_max) if
                          cv2.pointPolygonTest(outline_contour, (x, y), False) < 0]
        if not outside_coords:
            corrected_path.append(p);
            continue
        avg_outside_lab = np.mean([lab_undithered_image[y, x] for x, y in outside_coords], axis=0)
        if color_difference(lab_undithered_image[py, px], avg_outside_lab) > COLOR_SIMILARITY_THRESHOLD:
            corrected_path.append(p);
            continue

        # --- [ 核心修正点：在源头过滤 ] ---

        # 1. 像以前一样，在 ERROR_BOUND 范围内找到所有几何上是“内部”的候选点
        cand_y_min, cand_y_max = max(0, py - ERROR_BOUND), min(height, py + ERROR_BOUND + 1)
        cand_x_min, cand_x_max = max(0, px - ERROR_BOUND), min(width, px + ERROR_BOUND + 1)
        candidate_pool = [(x, y) for y in range(cand_y_min, cand_y_max) for x in range(cand_x_min, cand_x_max) if
                          cv2.pointPolygonTest(outline_contour, (x, y), False) >= 0]

        # 2. 现在，创建一个只包含“有效”候选点的列表
        valid_candidates = []
        for c in candidate_pool:
            cx, cy = c

            # 条件a: 颜色必须与外部不同 (旧逻辑)
            color_is_valid = color_difference(lab_undithered_image[cy, cx],
                                              avg_outside_lab) >= COLOR_SIMILARITY_THRESHOLD
            if not color_is_valid:
                continue

            # 条件b: 像素必须是非透明的 (新逻辑)
            pixel_is_opaque = True
            if alpha_channel is not None and alpha_channel[cy, cx] == 0:
                pixel_is_opaque = False

            # 只有同时满足两个条件，才被认为是有效候选点
            if pixel_is_opaque:
                valid_candidates.append(c)

        # ------------------------------------

        if valid_candidates:
            # 从这个已经“净化”过的列表中选择最近的点
            distances = [dist.euclidean(p, cand) for cand in valid_candidates]
            best_candidate = valid_candidates[np.argmin(distances)]
            corrected_path.append(best_candidate)
            corrected_count += 1
        else:
            # 如果找不到任何有效的（颜色正确且非透明）候选点，则保留原始点
            corrected_path.append(p)

    print(f"  -> Path refinement complete. Corrected {corrected_count} valid pixels.")
    return corrected_path


def interpolate_path_gaps(path: list[tuple[int, int]], pixel_art_image: np.ndarray) -> list[tuple[int, int]]:
    """
    [阶段三 - 已修正] 遍历路径，通过线性插值填补不连续的像素点，
    并确保所有插值点都落在非透明区域。
    """
    print("\n--- Post-correction (Stage 3): Interpolating Gaps with Transparency Check ---")
    if len(path) < 2: return path

    # 提取Alpha通道以供检查
    height, width, channels = pixel_art_image.shape
    if channels < 4:
        print("  - 警告: 图像无Alpha通道，无法进行透明度检查。将跳过插值。")
        return path
    alpha_channel = pixel_art_image[:, :, 3]

    continuous_path, pixels_added, pixels_skipped = [path[0]], 0, 0

    for i in range(1, len(path)):
        p1, p2 = path[i - 1], path[i]
        dist_val = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

        if dist_val > 1:
            num_steps = int(dist_val)
            x_vals = np.linspace(p1[0], p2[0], num_steps + 1)
            y_vals = np.linspace(p1[1], p2[1], num_steps + 1)

            for step in range(1, num_steps):
                interp_point = (int(round(x_vals[step])), int(round(y_vals[step])))

                # --- [ 核心修改 ] ---
                # 在添加之前，检查该点的Alpha值
                # 确保坐标在图像范围内
                if (0 <= interp_point[1] < height and 0 <= interp_point[0] < width and
                        alpha_channel[interp_point[1], interp_point[0]] > 0):

                    if interp_point != continuous_path[-1]:
                        continuous_path.append(interp_point)
                        pixels_added += 1
                else:
                    pixels_skipped += 1  # 记录跳过的透明点

        if p2 != continuous_path[-1]:
            continuous_path.append(p2)

    print(
        f"  -> Interpolation complete. Added {pixels_added} valid pixels. Skipped {pixels_skipped} transparent pixels.")
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
        num_labels, labels = cv2.connectedComponents(mask, 4)
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


int_tuple = types.UniTuple(types.int64, 2)


@njit
def _jit_generate_adaptive_scan_path(
        pixels_map: np.ndarray,
        map_offset_x: int, map_offset_y: int,
        height: int, width: int,
        base_lookahead: int, lookahead_growth: float
):
    """
    [已合并修正] 将核心逻辑和四种策略合并到一个JIT函数中，以避免类型推断错误。
    """
    all_results = []

    # --- 策略循环：4种策略 ---
    for start_mode in range(4):
        pixels_map_local = pixels_map.copy()
        path = List.empty_list(int_tuple)
        strokes = 0
        num_pixels = np.sum(pixels_map_local)
        last_pos_local = (-1, -1)

        while num_pixels > 0:
            strokes += 1
            pos_local = (-1, -1)

            # 智能选择起点
            if last_pos_local[0] == -1:
                if start_mode == 0:  # TopDown
                    for y in range(height):
                        for x in range(width):
                            if pixels_map_local[y, x]: pos_local = (x, y); break
                        if pos_local[0] != -1: break
                elif start_mode == 1:  # LeftRight
                    for x in range(width):
                        for y in range(height):
                            if pixels_map_local[y, x]: pos_local = (x, y); break
                        if pos_local[0] != -1: break
                elif start_mode == 2:  # BottomUp
                    for y in range(height - 1, -1, -1):
                        for x in range(width):
                            if pixels_map_local[y, x]: pos_local = (x, y); break
                        if pos_local[0] != -1: break
                elif start_mode == 3:  # RightLeft
                    for x in range(width - 1, -1, -1):
                        for y in range(height):
                            if pixels_map_local[y, x]: pos_local = (x, y); break
                        if pos_local[0] != -1: break
            else:
                min_dist_sq = -1;
                best_next_start = (-1, -1)
                for y in range(height):
                    for x in range(width):
                        if pixels_map_local[y, x]:
                            dist_sq = (x - last_pos_local[0]) ** 2 + (y - last_pos_local[1]) ** 2
                            if min_dist_sq == -1 or dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq;
                                best_next_start = (x, y)
                pos_local = best_next_start

            path.append((pos_local[0] + map_offset_x, pos_local[1] + map_offset_y))
            pixels_map_local[pos_local[1], pos_local[0]] = False
            num_pixels -= 1

            # 定义当前策略的优先级
            n_steps = 0
            if start_mode == 0:  # TopDown
                h_or_v_pref = (1, 0);
                main_dir = (0, 1);
                back_dir = (0, -1)
            elif start_mode == 1:  # LeftRight
                h_or_v_pref = (0, 1);
                main_dir = (1, 0);
                back_dir = (-1, 0)
            elif start_mode == 2:  # BottomUp
                h_or_v_pref = (1, 0);
                main_dir = (0, -1);
                back_dir = (0, 1)
            elif start_mode == 3:  # RightLeft
                h_or_v_pref = (0, 1);
                main_dir = (-1, 0);
                back_dir = (1, 0)

            while True:
                moved = False
                direction_prefs = [back_dir, h_or_v_pref, main_dir, (-h_or_v_pref[0], -h_or_v_pref[1])]

                for dx, dy in direction_prefs:
                    next_x, next_y = pos_local[0] + dx, pos_local[1] + dy
                    if 0 <= next_y < height and 0 <= next_x < width and pixels_map_local[next_y, next_x]:
                        path.append((next_x + map_offset_x, next_y + map_offset_y))
                        pixels_map_local[next_y, next_x] = False
                        num_pixels -= 1
                        pos_local = (next_x, next_y)
                        moved = True

                        is_main_dir_move = (dx == main_dir[0] and dy == main_dir[1])
                        if is_main_dir_move:
                            n_steps += 1
                            required_lookahead = int(base_lookahead + n_steps * lookahead_growth)
                            has_enough = True
                            for i in range(1, required_lookahead + 1):
                                lx, ly = pos_local[0] + h_or_v_pref[0] * i, pos_local[1] + h_or_v_pref[1] * i
                                if not (0 <= ly < height and 0 <= lx < width and pixels_map_local[ly, lx]):
                                    has_enough = False;
                                    break
                            if not has_enough:
                                h_or_v_pref = (-h_or_v_pref[0], -h_or_v_pref[1]);
                                n_steps = 0
                        else:
                            n_steps = 0
                        break
                if not moved:
                    last_pos_local = pos_local;
                    break

        all_results.append((strokes, path))

    # 找到最佳结果
    # Numba不支持复杂的min key，所以手动循环
    best_strokes = -1
    best_path = List.empty_list(int_tuple)
    best_strategy_idx = -1

    for i in range(len(all_results)):
        s, p = all_results[i]
        if best_strokes == -1 or s < best_strokes or (s == best_strokes and len(p) < len(best_path)):
            best_strokes = s
            best_path = p
            best_strategy_idx = i

    strategy_names = ("Top-to-Bottom", "Left-to-Right", "Bottom-to-Up", "Right-to-Left")
    best_strategy_name = strategy_names[best_strategy_idx]

    return best_strokes, best_path, best_strategy_name


def generate_adaptive_scan_path(pixel_coords: set[tuple[int, int]]):
    """
    Python外层函数，调用JIT顶层封装。
    """
    if not pixel_coords:
        return 0, []

    coords_array = np.array(list(pixel_coords))
    min_x, min_y = np.min(coords_array, axis=0)
    max_x, max_y = np.max(coords_array, axis=0)
    map_width, map_height = max_x - min_x + 1, max_y - min_y + 1
    pixels_map = np.zeros((map_height, map_width), dtype=np.bool_)
    local_coords = coords_array - np.array([min_x, min_y])
    pixels_map[local_coords[:, 1], local_coords[:, 0]] = True

    best_strokes, best_path_jit, best_strategy_name = _jit_generate_adaptive_scan_path(
        pixels_map, min_x, min_y, map_height, map_width,
        BASE_LOOKAHEAD, LOOKAHEAD_GROWTH
    )

    # print(f"      -> Strategy chosen: {best_strategy_name} ({best_strokes} strokes)")
    return best_strokes, list(best_path_jit)


# (将这个函数添加到您的脚本的“辅助函数”部分)
def get_four_corners(pixel_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    从一个像素集合中找出最上、最下、最左、最右的像素点。
    """
    if not pixel_set:
        return []

    pixels = list(pixel_set)
    # 转换为 NumPy 数组方便计算
    coords = np.array(pixels)

    # 最左 (最小X)
    min_x = np.min(coords[:, 0])
    left_points = coords[coords[:, 0] == min_x]

    # 最右 (最大X)
    max_x = np.max(coords[:, 0])
    right_points = coords[coords[:, 0] == max_x]

    # 最上 (最小Y)
    min_y = np.min(coords[:, 1])
    top_points = coords[coords[:, 1] == min_y]

    # 最下 (最大Y)
    max_y = np.max(coords[:, 1])
    bottom_points = coords[coords[:, 1] == max_y]

    # 收集四个角的点 (确保不重复)
    corners = set()
    # 选取最左/最右点的最上和最下，确保找到“角”

    # 左上角 (min X, min Y)
    corners.add(tuple(left_points[np.argmin(left_points[:, 1])]))
    # 左下角 (min X, max Y)
    corners.add(tuple(left_points[np.argmax(left_points[:, 1])]))
    # 右上角 (max X, min Y)
    corners.add(tuple(right_points[np.argmin(right_points[:, 1])]))
    # 右下角 (max X, max Y)
    corners.add(tuple(right_points[np.argmax(right_points[:, 1])]))

    return list(corners)


def spiral_search_generator(start_x, start_y):
    """
    一个生成器，以(start_x, start_y)为中心，按螺旋向外产生坐标。
    """
    x, y = start_x, start_y
    yield (x, y)
    dx, dy = 1, 0
    steps = 1
    turns = 0
    while True:
        for _ in range(steps):
            x, y = x + dx, y + dy
            yield (x, y)
        # 转向
        dx, dy = -dy, dx
        turns += 1
        if turns % 2 == 0:
            steps += 1


# (用这个新版本替换旧的 sort_and_flatten_regions 函数)
def group_and_sort_leftovers(leftover_pixels: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    【v3 - 终极优化版】
    1. 几何覆盖: 使用X轴滑动扫描，用最少矩形覆盖散点。
    2. 路径规划: 使用TSP算法重排矩形执行顺序，最小化地图平移距离。
    """
    if not leftover_pixels:
        return []

    # 固定参数
    RECT_W = 150
    RECT_H = 75

    print(f"        -> Optimizing coverage & path for {len(leftover_pixels)} pixels...")

    # --- 1. 内部求解器：解决覆盖问题 (保持 v2 逻辑) ---
    def solve_coverage(pixels_set, mode='strict_top_left'):
        remaining = set(pixels_set)
        # 这里改为存储元组: (rect_center_point, pixel_list)
        # 我们用矩形内所有点的几何中心来代表这个矩形的位置
        groups_with_info = []

        while remaining:
            seed = min(remaining, key=lambda p: (p[1], p[0]))
            seed_x, seed_y = seed
            rect_top = seed_y
            rect_bottom = rect_top + RECT_H
            best_rect_left = seed_x

            if mode == 'scan_x_alignment':
                min_valid_left = seed_x - RECT_W + 1
                max_valid_left = seed_x
                potential_neighbors = [p for p in remaining if
                                       rect_top <= p[1] < rect_bottom and min_valid_left <= p[0] < seed_x + RECT_W]
                candidates = {seed_x, min_valid_left}
                for p in potential_neighbors:
                    c1 = p[0]
                    if min_valid_left <= c1 <= max_valid_left: candidates.add(c1)
                    c2 = p[0] - RECT_W + 1
                    if min_valid_left <= c2 <= max_valid_left: candidates.add(c2)

                max_count = -1
                for left_x in candidates:
                    right_x = left_x + RECT_W
                    count = 0
                    for p in potential_neighbors:
                        if left_x <= p[0] < right_x: count += 1
                    if count > max_count:
                        max_count = count
                        best_rect_left = left_x

            rect_left = best_rect_left
            rect_right = rect_left + RECT_W

            final_covered = []
            to_remove = []

            # 计算这一组像素的质心
            sum_x, sum_y = 0, 0

            for p in remaining:
                if rect_left <= p[0] < rect_right and rect_top <= p[1] < rect_bottom:
                    final_covered.append(p)
                    to_remove.append(p)
                    sum_x += p[0]
                    sum_y += p[1]

            final_covered.sort(key=lambda p: (p[1], p[0]))

            # 计算质心 (Centroid)
            center_x = sum_x / len(final_covered)
            center_y = sum_y / len(final_covered)

            groups_with_info.append({
                'pixels': final_covered,
                'center': (center_x, center_y)
            })

            for p in to_remove: remaining.remove(p)

        return groups_with_info

    # --- 2. 覆盖优化 (双路竞技) ---
    groups_a = solve_coverage(leftover_pixels, mode='strict_top_left')
    groups_b = solve_coverage(leftover_pixels, mode='scan_x_alignment')

    # 优先选组数少的
    if len(groups_b) < len(groups_a):
        print(f"          - Optimization: Reduced groups from {len(groups_a)} to {len(groups_b)} using X-scan.")
        best_groups = groups_b
    else:
        print(f"          - Standard greedy was optimal ({len(groups_a)} groups).")
        best_groups = groups_a

    # --- 3. 【✨ 核心新增：TSP 路径优化 ✨】 ---
    # 使用“最近邻”算法重排 best_groups

    if not best_groups: return []

    sorted_groups = []

    # a. 找到起点：距离整个图像左上角 (0,0) 最近的组
    #    (或者你可以传入上一笔画的结束点作为 start_pos，这里简化处理)
    start_pos = (0, 0)
    first_group_idx = min(range(len(best_groups)),
                          key=lambda i: math.hypot(best_groups[i]['center'][0] - start_pos[0],
                                                   best_groups[i]['center'][1] - start_pos[1]))

    current_group = best_groups.pop(first_group_idx)
    sorted_groups.append(current_group)

    # b. 贪心迭代
    while best_groups:
        last_center = current_group['center']

        # 找到距离 last_center 最近的下一个组
        nearest_idx = min(range(len(best_groups)),
                          key=lambda i: math.hypot(best_groups[i]['center'][0] - last_center[0],
                                                   best_groups[i]['center'][1] - last_center[1]))

        current_group = best_groups.pop(nearest_idx)
        sorted_groups.append(current_group)

    print(f"          - Reordered {len(sorted_groups)} groups for minimal travel distance.")

    # --- 4. 展平结果 ---
    final_sorted_path = []
    for grp in sorted_groups:
        final_sorted_path.extend(grp['pixels'])

    return final_sorted_path


def sort_and_flatten_regions(regions_by_color, default_free_colors):
    final_order = []

    # --- 1. 初始颜色数据提取 ---
    rgb_to_name_map = {tuple(c['rgb']): c['name'] for c in WPLACE_COLOR_PALETTE}
    paid_list = []
    free_list = []

    for color_rgb, regions in regions_by_color.items():
        r_tuple = (color_rgb[2], color_rgb[1], color_rgb[0])
        name = rgb_to_name_map.get(r_tuple, "Unknown Color")
        is_paid = name not in default_free_colors
        info = {
            "color_rgb": color_rgb,
            "total_pixels": sum(len(r) for r in regions),
            "regions": regions,
            "is_paid": is_paid
        }
        if is_paid:
            paid_list.append(info)
        else:
            free_list.append(info)

    # 分别按像素数量从大到小排序
    paid_list.sort(key=lambda x: x['total_pixels'], reverse=True)
    free_list.sort(key=lambda x: x['total_pixels'], reverse=True)

    # --- 2. 构建处理队列 (补全了 total_pixels 键) ---
    processing_queue = []

    for group_list, label_prefix in [(paid_list, "Paid"), (free_list, "Free")]:
        if not group_list: continue

        # 计算切分点：保留前面的，合并最后10个
        split_idx = max(0, len(group_list) - 10)
        top_entries = group_list[:split_idx]
        bottom_entries = group_list[split_idx:]

        # A. 前面的颜色：保持独立任务
        for entry in top_entries:
            processing_queue.append({
                "label": f"Color {entry['color_rgb']}",
                "regions": entry['regions'],
                "is_paid": entry['is_paid'],
                "total_pixels": entry['total_pixels']  # 补上这个键
            })

        # B. 后10名颜色：合并为一个任务
        if bottom_entries:
            merged_regions = []
            merged_total = 0
            for entry in bottom_entries:
                merged_regions.extend(entry['regions'])
                merged_total += entry['total_pixels']

            processing_queue.append({
                "label": f"Merged Bottom-10 {label_prefix} Colors",
                "regions": merged_regions,
                "is_paid": bottom_entries[0]['is_paid'],
                "total_pixels": merged_total  # 补上这个键
            })

    # --- 3. 核心循环：依次处理队列中的每个条目 ---
    for item in processing_queue:
        regions_to_process = item['regions']
        paid_status = "Paid" if item['is_paid'] else "Free"
        print(f"\n    Processing {paid_status} {item['label']} with {item['total_pixels']} pixels...")

        # --- 以下逻辑完全保留原样，仅将输入变量由颜色特定改为 item['regions'] ---

        # 1. 区域内部的分组（主体 vs 碎片）
        large_regions_pixels = []
        small_regions = []
        for region in regions_to_process:
            if len(region) >= CONNECTIVITY_THRESHOLD:
                large_regions_pixels.append(set(map(tuple, region)))
            else:
                small_regions.append({'pixels': set(map(tuple, region)), 'size': len(region)})

        # 2. 主体绘制（大块）
        large_region_results = []
        for pixel_set in large_regions_pixels:
            stroke_count, path = generate_adaptive_scan_path(pixel_set)
            connectivity_score = len(pixel_set) / (stroke_count ** 2) if stroke_count > 0 else float('inf')
            large_region_results.append({'score': connectivity_score, 'path': path})

        sorted_large_regions = sorted(large_region_results, key=lambda x: x['score'], reverse=True)
        main_path_for_item = [p for result in sorted_large_regions for p in result['path']]

        if not main_path_for_item and not small_regions: continue

        if not small_regions:
            final_order.extend(main_path_for_item)
            continue

        # 3. 中断插入（碎片处理）
        if main_path_for_item:
            main_path_pixel_set = set(main_path_for_item)
            main_path_index_map = {pixel: i for i, pixel in enumerate(main_path_for_item)}
        else:
            all_flattened = []
            for region_info in small_regions:
                all_flattened.extend(region_info['pixels'])
            small_path = group_and_sort_leftovers(all_flattened)
            final_order.extend(small_path)
            continue

        insertions = {}
        appended_at_end = []
        sorted_small_regions = sorted(small_regions, key=lambda r: r['size'], reverse=True)

        for region_info in sorted_small_regions:
            _, small_path = generate_adaptive_scan_path(region_info['pixels'])
            if not small_path: continue

            anchor_points = get_four_corners(region_info['pixels']) or [small_path[0]]
            found_anchor = False
            best_insert_index = -1

            for anchor_point in anchor_points:
                search_gen = spiral_search_generator(anchor_point[0], anchor_point[1])
                for i, (sx, sy) in enumerate(search_gen):
                    if i > (INTERRUPT_SEARCH_RADIUS * 2 + 1) ** 2: break
                    search_pixel = (sx, sy)
                    if search_pixel in main_path_pixel_set:
                        best_insert_index = main_path_index_map[search_pixel]
                        found_anchor = True
                        break
                if found_anchor: break

            if found_anchor:
                if best_insert_index not in insertions: insertions[best_insert_index] = []
                insertions[best_insert_index].extend(small_path)
            else:
                appended_at_end.extend(small_path)

        # 4. 构建当前组的最终路径
        item_final_path = []
        for i, pixel in enumerate(main_path_for_item):
            item_final_path.append(pixel)
            if i in insertions:
                item_final_path.extend(insertions[i])

        sorted_leftovers = group_and_sort_leftovers(appended_at_end)
        item_final_path.extend(sorted_leftovers)

        print(f"      - Processing complete. Path length: {len(item_final_path)}")
        final_order.extend(item_final_path)

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
        height, width, channels = pixel_art_image.shape
    except Exception as e:
        print(f"Error loading files: {e}"); return

    final_draw_order, processed_pixels = [], set()
    spilled_pixels_to_reprocess = set()
    print("\nStep 0: Processing Image Border Pixels...")
    border_pixels_added = 0

    # 检查是否有Alpha通道
    alpha_channel = pixel_art_image[:, :, 3] if channels == 4 else None

    # 收集所有边框点
    print("  -> Collecting border pixels in a continuous clockwise path...")
    continuous_border_coords = []
    # 上下边框
    for x in range(width):
        continuous_border_coords.append((x, 0))

    # 2. 右边框 (从上到下，跳过已添加的角点)
    if width > 1:
        for y in range(1, height):
            continuous_border_coords.append((width - 1, y))

    # 3. 下边框 (从右到左，跳过已添加的角点)
    if height > 1 and width > 1:
        for x in range(width - 2, -1, -1):
            continuous_border_coords.append((x, height - 1))

    # 4. 左边框 (从下到上，跳过已添加的角点)
    if height > 2 and width > 1:
        for y in range(height - 2, 0, -1):
            continuous_border_coords.append((0, y))

    for x, y in continuous_border_coords:
        # 检查透明度
        is_opaque = True
        if alpha_channel is not None and alpha_channel[y, x] == 0:
            is_opaque = False

        if is_opaque:
            p_tuple = (x, y)
            # 确保不重复添加
            if p_tuple not in processed_pixels:
                final_draw_order.append(p_tuple)
                processed_pixels.add(p_tuple)
                border_pixels_added += 1

    print(f"  -> Added {border_pixels_added} unique, non-transparent pixels from the image border.")

    print("\n--- Phase A: Parsing PSD Data & Path Correction ---")
    all_grouped_subpaths = extract_grouped_paths_from_psd(psd)
    if not all_grouped_subpaths: return

    outline_subpaths_to_process = []
    num_to_select = min(NUM_OUTLINE_PATHS, len(all_grouped_subpaths))
    for i in range(num_to_select):
        if all_grouped_subpaths[i]: outline_subpaths_to_process.append(max(all_grouped_subpaths[i], key=lambda x: len(x)))

    all_initial_pixels, all_extrapolated_pixels, all_final_outline_pixels = [], [], []
    main_boundary_final_path = None

    print("\n--- Path Correction Stage Initiated ---")
    for i, subpath in enumerate(outline_subpaths_to_process):
        # if i == 0:
        #     continue

        print(f"\n>>> Processing Outline Subpath {i + 1}/{len(outline_subpaths_to_process)} <<<")
        initial_contour_points = [[k.anchor[1] * width, k.anchor[0] * height] for k in subpath if isinstance(k, Knot)]
        initial_outline_contour = np.array(initial_contour_points, dtype=np.float32).reshape((-1, 1, 2))

        initial_pixels = rasterize_subpaths_ordered([subpath], pixel_art_image)
        moved_indices = set()
        if CONSERVATIVE_EXTRAPOLATION_ENABLED:
            extrapolated_path, moved_indices = extrapolate_inner_path_points(initial_pixels, initial_outline_contour,
                                                                             undithered_image)
        else:
            extrapolated_path = initial_pixels
        refined_path = refine_outline_path(extrapolated_path, moved_indices, initial_outline_contour, undithered_image, pixel_art_image)
        final_path = interpolate_path_gaps(refined_path, pixel_art_image) if ENABLE_POST_INTERPOLATION else refined_path

        all_initial_pixels.extend(initial_pixels)
        all_extrapolated_pixels.extend(extrapolated_path)
        all_final_outline_pixels.extend(final_path)
        find_transparent_pixels(initial_pixels, pixel_art_image)
        find_transparent_pixels(refined_path, pixel_art_image)
        find_transparent_pixels(final_path, pixel_art_image)

        if i == 0:
            main_boundary_final_path = final_path

    print("\n\n--- All Path Corrections Finished ---")
    visualize_correction_stages(all_initial_pixels, all_extrapolated_pixels, all_final_outline_pixels, width, height,
                                OUTPUT_VISUALIZATION_PATH)

    main_boundary_contour = None
    if main_boundary_final_path:
        contour_points = [(float(p[0]), float(p[1])) for p in main_boundary_final_path]
        main_boundary_contour = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        print("\nSuccessfully defined the main boundary from the *final corrected* first subpath for layer filtering.")
    else:
        print("\nWarning: Could not define a main boundary. Filtering will be skipped.")

    print("\n--- Phase B: Building Final Draw Order ---")

    # --- [ 此处是修正点 ] ---

    # 1. 首先，获取所有路径（包括轮廓和剩余部分）的完整像素列表

    # [新增] 正确定义 remaining_subpaths
    all_subpaths_flat = [sub for group in all_grouped_subpaths for sub in group]
    remaining_subpaths = [sub for sub in all_subpaths_flat if sub not in outline_subpaths_to_process]

    all_stroke_pixels_initial = []
    all_stroke_pixels_initial.extend(all_final_outline_pixels)
    remaining_pixels_initial = rasterize_subpaths_ordered(remaining_subpaths, pixel_art_image)
    all_stroke_pixels_initial.extend(remaining_pixels_initial)

    print(f"\nFiltering all {len(all_stroke_pixels_initial)} stroke pixels against the main boundary...")

    # 2. 过滤这个完整的列表
    if main_boundary_contour is not None:
        filtered_stroke_pixels = [
            p for p in all_stroke_pixels_initial
            if cv2.pointPolygonTest(main_boundary_contour, tuple(map(float, p)), False) >= 0
        ]
        removed_count = len(all_stroke_pixels_initial) - len(filtered_stroke_pixels)
        print(f"  -> Filtering complete. Removed {removed_count} pixels that were outside the boundary.")
    else:
        print("  -> No main boundary defined, skipping stroke pixel filtering.")
        filtered_stroke_pixels = all_stroke_pixels_initial

    # 3. 使用“净化”过的列表构建 Step 1
    print("\nStep 1: Processing all filtered stroke paths...")
    unique_step1_pixels = [p for p in filtered_stroke_pixels if
                           tuple(p) not in processed_pixels and (processed_pixels.add(tuple(p)) or True)]
    final_draw_order.extend(unique_step1_pixels)
    find_transparent_pixels(final_draw_order, pixel_art_image)
    print(f"  -> Added {len(unique_step1_pixels)} unique pixels from all filtered stroke paths.")

    # -----------------------------

    # 图层像素提取
    layer_regions = {}
    for layer in psd:
        if layer.is_visible() and not layer.is_group() and layer.width > 0 and layer.height > 0:
            match = re.search(r'(\d+)$', layer.name)
            if match:
                num = int(match.group(1));
                layer_numpy = layer.numpy()
                if layer_numpy.shape[2] == 4:
                    ys, xs = np.where(layer_numpy[:, :, 3] > 0)
                    layer_regions[num] = set(zip(xs + layer.left, ys + layer.top))

    # Step 2 & 3 & 4
    layer_nums_step2 = sorted([num for num in layer_regions if num >= 3])
    print(f"\nStep 2: Processing regions from layer 3 upwards ({len(layer_nums_step2)} layers)...")
    for num in layer_nums_step2:
        pixels_to_process = layer_regions.get(num, set()) - processed_pixels
        find_transparent_pixels(pixels_to_process, pixel_art_image)
        if main_boundary_contour is not None:
            pixels_inside = {p for p in pixels_to_process if
                             cv2.pointPolygonTest(main_boundary_contour, tuple(map(float, p)), False) >= 0}
            pixels_outside = pixels_to_process - pixels_inside
            spilled_pixels_to_reprocess.update(pixels_outside)
            pixels_to_process = pixels_inside
            print(
                f"  - Layer {num}: Filtered out {len(pixels_outside)} pixels. Processing {len(pixels_inside)} inside pixels.")
        sorted_pixels = sort_and_flatten_regions(find_contiguous_regions(pixels_to_process, pixel_art_image), DEFAULT_FREE_COLORS)
        final_draw_order.extend(sorted_pixels);
        # find_transparent_pixels(final_draw_order, pixel_art_image)
        processed_pixels.update(map(tuple, sorted_pixels))

    print("\nStep 3: Processing layer 2 minus regions from step 2...")
    if 2 in layer_regions:
        layers_3_up_union = set().union(*(layer_regions.get(num, set()) for num in layer_nums_step2))
        pixels_to_process = (layer_regions[2] - layers_3_up_union) - processed_pixels
        find_transparent_pixels(pixels_to_process, pixel_art_image)
        if main_boundary_contour is not None:
            # pixels_inside = {p for p in pixels_to_process if
            #                  cv2.pointPolygonTest(main_boundary_contour, tuple(map(float, p)), False) >= 0}
            pixels_inside = pixels_to_process
            pixels_outside = pixels_to_process - pixels_inside
            spilled_pixels_to_reprocess.update(pixels_outside)
            pixels_to_process = pixels_inside
            print(
                f"  - Layer 2: Filtered out {len(pixels_outside)} pixels. Processing {len(pixels_inside)} inside pixels.")
        sorted_pixels = sort_and_flatten_regions(find_contiguous_regions(pixels_to_process, pixel_art_image), DEFAULT_FREE_COLORS)
        final_draw_order.extend(sorted_pixels)
        # find_transparent_pixels(final_draw_order, pixel_art_image)
        processed_pixels.update(map(tuple, sorted_pixels))
    else:
        print("  -> Layer 2 not found, skipping.")

    print("\nStep 4: Processing remaining pixels...")
    all_visible_pixels = set(zip(*np.where(pixel_art_image[:, :, 3] > 0)[::-1])) if pixel_art_image.shape[
                                                                                        2] == 4 else set(
        (x, y) for x in range(width) for y in range(height))
    pixels_to_process = all_visible_pixels - processed_pixels
    print(f"  -> Re-including {len(spilled_pixels_to_reprocess)} pixels from layers 2+ that were outside the boundary.")
    pixels_to_process.update(spilled_pixels_to_reprocess)
    find_transparent_pixels(pixels_to_process, pixel_art_image)
    sorted_pixels = sort_and_flatten_regions(find_contiguous_regions(pixels_to_process, pixel_art_image), DEFAULT_FREE_COLORS)
    final_draw_order.extend(sorted_pixels)
    # find_transparent_pixels(final_draw_order, pixel_art_image)
    print(f"  -> Added {len(sorted_pixels)} unique pixels in the final step.")

    print(f"\n--- Phase C: Verification & Saving Output ---")
    print("Performing final checks before saving...")
    if channels == 4:
        total_opaque_in_source = np.sum(pixel_art_image[:, :, 3] > 0)
    else:
        total_opaque_in_source = pixel_art_image.shape[0] * pixel_art_image.shape[1]
    total_in_final_order = len(final_draw_order)
    count_ok = (total_in_final_order == total_opaque_in_source)
    if count_ok:
        print(f"✅ [Check 1/2] Pixel Count: OK ({total_in_final_order:,} pixels)")
    else:
        print(f"❌ [Check 1/2] Pixel Count: FAILED!");
        print(f"  - Expected: {total_opaque_in_source:,}");
        print(f"  - Found: {total_in_final_order:,}")
    order_as_tuples = [tuple(p) for p in final_draw_order]
    total_unique_in_final_order = len(set(order_as_tuples))
    duplicates_ok = (total_in_final_order == total_unique_in_final_order)
    if duplicates_ok:
        print(f"✅ [Check 2/2] Duplicates: OK (All {total_in_final_order:,} pixels are unique)")
    else:
        print(f"❌ [Check 2/2] Duplicates: FAILED!");
        num_duplicates = total_in_final_order - total_unique_in_final_order;
        print(f"  - Found {num_duplicates} duplicate entries.")
        counts = Counter(order_as_tuples);
        duplicates = {item: count for item, count in counts.items() if count > 1}
        print(f"  - The following {len(duplicates)} pixel(s) were repeated:")
        for i, (pixel, count) in enumerate(duplicates.items()):
            if i >= 10: print("    - ... and more."); break
            print(f"    - Pixel {pixel} appeared {count} times.")
    if count_ok and duplicates_ok:
        print("\n✨ Verification successful! Proceeding to save the output file.")
        try:
            with open(OUTPUT_JSON_PATH, 'w') as f:
                json.dump(final_draw_order, f, cls=NumpyJSONEncoder)
            print(f"Successfully saved the final draw order to '{OUTPUT_JSON_PATH}'")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    else:
        print("\n🛑 Verification failed! The output file will NOT be saved.")


if __name__ == "__main__":
    main()