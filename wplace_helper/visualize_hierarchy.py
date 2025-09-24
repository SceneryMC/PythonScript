import cv2
import numpy as np

# --- 配置项 ---
# 1. 原始模板文件名 (用于获取尺寸和透明度信息)
ORIGINAL_TEMPLATE_FILENAME = "dst/9148c828bis91/9148c828bis91_converted.png"

# 2. 边缘地图文件名
EDGE_MAP_FILENAME = "edges/9148c828bis91_edge.png"

# 3. 输出的可视化文件名
OUTPUT_FILENAME = "visualization_contour_hierarchy_9148c828bis91.png"


# --------------------------------------------------------------------

def find_contours_and_hierarchy(template_path, edge_map_path):
    """
    加载图像，融合边缘，然后使用OpenCV找到所有轮廓及其层级关系。
    这个函数与您最新版本中的同名函数完全相同。
    """
    print("--- Step 1: Loading Images and Fusing Edges ---")

    template_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)
    if template_image is None or edge_map is None or template_image.shape[:2] != edge_map.shape[:2]:
        raise ValueError("Invalid input files or dimension mismatch.")

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
                if not (0 <= nx < width and 0 <= ny < height) or alpha_channel[ny, nx] != 255:
                    basic_edge_coords.add((x, y));
                    break
    else:
        for x, y in all_valid_pixels_set:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                basic_edge_coords.add((x, y))
    y_c, x_c = np.where(edge_map > 128)
    canny_edge_coords_raw = set(zip(x_c, y_c))
    canny_edge_coords = canny_edge_coords_raw.intersection(all_valid_pixels_set)
    fused_coords = basic_edge_coords.union(canny_edge_coords)
    fused_edge_map = np.zeros((height, width), dtype=np.uint8)
    for x, y in fused_coords:
        fused_edge_map[y, x] = 255

    print(f"Finding contours and hierarchy from {len(fused_coords)} fused edge pixels...")
    contours, hierarchy = cv2.findContours(fused_edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    return contours, hierarchy[0], template_image.shape


# =================== [ 全新的可视化函数 ] ===================
def visualize_hierarchy_levels(contours, hierarchy, shape, output_filename):
    """
    将每个层级的轮廓用不同的颜色绘制出来，并附上图例。
    """
    print("\n--- Step 2: Visualizing Contour Hierarchy ---")

    height, width, _ = shape
    # 创建一个带颜色通道的黑色画布
    # 我们额外增加一些高度给图例
    legend_height = 80
    canvas = np.zeros((height + legend_height, width, 3), dtype=np.uint8)

    if hierarchy is None:
        print("No hierarchy information found.")
        cv2.imwrite(output_filename, canvas)
        return

    # 定义一组鲜艳的颜色用于区分不同层级 (BGR格式)
    COLORS = [
        (255, 0, 0),  # 蓝色   (Level 0)
        (0, 255, 0),  # 绿色   (Level 1)
        (0, 0, 255),  # 红色   (Level 2)
        (0, 255, 255),  # 黄色   (Level 3)
        (255, 0, 255),  # 品红   (Level 4)
        (255, 255, 0),  # 青色   (Level 5)
        (255, 255, 255)  # 白色   (更高层级)
    ]

    # 辅助函数，递归地计算每个轮廓的“深度”
    memo = {}

    def get_depth(i):
        if i in memo: return memo[i]
        parent_idx = hierarchy[i][3]  # [Next, Previous, First_Child, Parent]
        if parent_idx == -1:
            memo[i] = 0
            return 0
        depth = 1 + get_depth(parent_idx)
        memo[i] = depth
        return depth

    # 绘制所有轮廓，并根据其深度着色
    max_depth = 0
    for i, contour in enumerate(contours):
        depth = get_depth(i)
        max_depth = max(max_depth, depth)

        # 使用 modulo 运算符来循环使用颜色
        color = COLORS[depth % len(COLORS)]

        # 绘制轮廓
        cv2.drawContours(canvas, [contour], -1, color, 1)

    print(f"Visualization complete. Max contour depth found: {max_depth}")

    # --- 绘制图例 ---
    # 先绘制一个白色背景
    canvas[height:, :] = (255, 255, 255)

    for i in range(max_depth + 1):
        color = COLORS[i % len(COLORS)]
        text = f"Level {i}"
        if i == 0:
            text += " (Outermost)"

        # 计算图例项的位置
        x_pos = 15 + (i % 4) * 150
        y_pos = height + 25 + (i // 4) * 30

        # 绘制颜色方块和文字
        cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + 20, y_pos + 20), color, -1)
        cv2.putText(canvas, text, (x_pos + 30, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 保存最终的图像
    cv2.imwrite(output_filename, canvas)
    print(f"Successfully saved hierarchy visualization to '{output_filename}'")


# =================== [ 函数结束 ] ===================


if __name__ == "__main__":
    try:
        # 1. 找到轮廓和层级信息
        contours, hierarchy, image_shape = find_contours_and_hierarchy(ORIGINAL_TEMPLATE_FILENAME, EDGE_MAP_FILENAME)

        # 2. 将层级信息可视化
        visualize_hierarchy_levels(contours, hierarchy, image_shape, OUTPUT_FILENAME)

        print("\n--- Hierarchy Visualization Complete ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")