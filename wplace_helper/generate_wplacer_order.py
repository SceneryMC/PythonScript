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
PSD_FILE_PATH = "dst/779706d9f9/779706d9f9_undithered.psd"
PIXEL_ART_PATH = "dst/779706d9f9/779706d9f9_converted.png"
OUTPUT_JSON_PATH = "wplacer_draw_order_779706d9f9.json"


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


def rasterize_subpaths(subpaths: list[Subpath], width: int, height: int):  # ...
    if not subpaths: return []
    print(f"\nRasterizing {len(subpaths)} subpaths...")
    all_paths_coords = []
    for i, subpath in enumerate(subpaths):
        knots_in_subpath = []
        for segment in subpath:
            if isinstance(segment, Knot):
                knots_in_subpath.append(segment)
            elif hasattr(segment, '_items'):
                knots_in_subpath.extend(segment._items)
        points = [knot.anchor for knot in knots_in_subpath]
        if not points: continue
        abs_points = np.array([(p[1] * width, p[0] * height) for p in points], dtype=np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        is_closed = subpath.is_closed()
        cv2.polylines(canvas, [abs_points], is_closed, 255, 1)
        ys, xs = np.where(canvas > 0)
        path_coords = list(zip(xs, ys));
        all_paths_coords.append(path_coords)
    print(f"  -> Rasterized into {len(all_paths_coords)} separate path pixel groups.")
    return all_paths_coords


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
    all_stroke_paths_coords_lists = rasterize_subpaths(all_subpaths, width, height)
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

    print("\n--- Phase B: Building Final Draw Order ---");
    final_draw_order = [];
    processed_pixels = set()

    print("\nStep 1: Processing all stroke paths (individually)...")
    for i, path_coords_list in enumerate(all_stroke_paths_coords_lists):
        print(f"  - Processing Path Group {i + 1}/{len(all_stroke_paths_coords_lists)}...")
        pixels_to_process = set(path_coords_list) - processed_pixels
        regions_by_color = find_contiguous_regions(pixels_to_process, pixel_art_image)
        sorted_path_pixels = sort_and_flatten_regions(regions_by_color)
        final_draw_order.extend(sorted_path_pixels);
        processed_pixels.update(sorted_path_pixels)
        print(f"    -> Added {len(sorted_path_pixels)} unique pixels.")

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