from collections import deque

import pyautogui
import json
import os
import time
from PIL import Image

from wplace_helper.utils import WPLACE_COLOR_PALETTE

# ==============================================================================
# --- 1. 用户配置区 (您需要在这里填写信息) ---
# ==============================================================================

# 请将您获得的颜色ID和对应的RGB值填入此字典
# 格式: 'color_id': (R, G, B)
COLORS_TO_FIND = {
    item["id"]: item["rgb"] for item in WPLACE_COLOR_PALETTE
}

# 请截取一张清晰的 "Paint" 按钮的小图片，并命名为 'paint_button.png'
# 将它与此脚本放在同一个文件夹下
PAINT_BUTTON_IMAGE = "paint_button.png"

# 图像识别的置信度 (0.0 - 1.0)。值越高越严格。0.95通常是个好起点。
CONFIDENCE_LEVEL = 0.95
COLOR_TOLERANCE = 4


# ==============================================================================
# --- 2. 脚本核心逻辑 (您无需修改以下内容) ---
# ==============================================================================

PALETTE_OFFSET_X = -880  # 估算值: 调色板左上角相对于Paint按钮中心的X偏移
PALETTE_OFFSET_Y = -150  # 估算值: 调色板左上角相对于Paint按钮中心的Y偏移
PALETTE_SIZE_W = 1900  # 估算值: 调色板的宽度
PALETTE_SIZE_H = 100  # 估算值: 调色板的高度

# 【✨ 新增】用于寻找黑色块的参数
BLACK_RGB_TARGET = (0, 0, 0)
BLACK_COLOR_TOLERANCE = 15  # 允许的黑色色差
MIN_BLACK_BLOCK_SIZE = 10  # 至少要找到 10x10 的黑色区域才认为是有效的

# 【✨ 新-先验知识】调色板的网格布局
PALETTE_GRID_ROWS = 2
PALETTE_GRID_COLS = 32

# 输出文件名
OUTPUT_CONFIG_FILE = "gui_config.json"


# ==============================================================================
# --- 2. 脚本核心逻辑 (您无需修改) ---
# ==============================================================================

def color_distance(rgb1, rgb2):
    """计算两种RGB颜色之间的欧几里得距离"""
    return sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5


def find_black_seed(image, target_rgb, tolerance, min_size):
    """在图片中寻找一个足够大的、近似纯黑的区域的种子点"""
    width, height = image.size
    for x in range(width - min_size):
        for y in range(height - min_size):
            # 检查一个 min_size x min_size 的方块是否都是黑色
            is_solid_black = True
            for i in range(min_size):
                for j in range(min_size):
                    pixel_rgb = image.getpixel((x + i, y + j))
                    if color_distance(pixel_rgb, target_rgb) > tolerance:
                        is_solid_black = False
                        break
                if not is_solid_black:
                    break

            if is_solid_black:
                # 找到了！返回这个方块的中心点作为种子
                return (x + min_size // 2, y + min_size // 2)
    return None


def get_bounding_box(image, seed_xy, target_rgb, tolerance):
    """
    从一个种子点开始，使用广度优先搜索(BFS)找到包含该点的连续颜色区域的边界框。
    """
    width, height = image.size
    x, y = seed_xy

    if color_distance(image.getpixel(seed_xy), target_rgb) > tolerance:
        return None  # 种子点本身颜色就不对

    q = deque([seed_xy])
    visited = {seed_xy}
    min_x, max_x = x, x
    min_y, max_y = y, y

    while q:
        px, py = q.popleft()

        min_x = min(min_x, px)
        max_x = max(max_x, px)
        min_y = min(min_y, py)
        max_y = max(max_y, py)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy

            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if color_distance(image.getpixel((nx, ny)), target_rgb) <= tolerance:
                    visited.add((nx, ny))
                    q.append((nx, ny))

    return (min_x, min_y, max_x + 1, max_y + 1)  # 返回 [left, top, right, bottom]


def find_closest_color_id(sampled_rgb, all_colors, tolerance):
    """
    在已知的颜色字典中，找到与采样颜色最接近的颜色ID。

    Args:
        sampled_rgb (tuple): 从屏幕上采样的 (R, G, B) 颜色。
        all_colors (dict): {'color_id': (R, G, B)} 的字典。
        tolerance (int): 可接受的最大颜色距离。

    Returns:
        str or None: 返回匹配的 color_id，如果没有找到足够接近的则返回 None。
    """
    min_distance = float('inf')
    best_match_id = None

    for color_id, known_rgb in all_colors.items():
        distance = color_distance(sampled_rgb, known_rgb)
        if distance < min_distance:
            min_distance = distance
            best_match_id = color_id

    if min_distance <= tolerance:
        return best_match_id
    else:
        # 如果最近的颜色都超出了容忍度，说明可能找错了
        return None


def calibrate_gui_v4():
    print("--- UI坐标校准脚本 v4 (几何推理 + 视觉识别) ---")
    print("请确保 wplace.live 窗口可见，调色板已打开。")
    print("脚本将在 5 秒后开始...")
    time.sleep(5)

    gui_config = {"paint_button_coord": None, "color_palette_coords": {}}

    # Step 1 & 2: 找到锚点并截取调色板区域 (与 v3 完全相同)
    print("\n[Step 1/4] 正在识别 'Paint' 按钮作为锚点...")
    try:
        button_coord_raw = pyautogui.locateCenterOnScreen(PAINT_BUTTON_IMAGE, confidence=0.9)
        if not button_coord_raw:
            print("  > ❌ 致命错误：未能找到 Paint 按钮！")
            return
        anchor_x, anchor_y = int(button_coord_raw.x), int(button_coord_raw.y)
        gui_config["paint_button_coord"] = [anchor_x, anchor_y]
        pyautogui.moveTo(anchor_x, anchor_y, duration=0.5)
        print(f"  > ✅ 成功！锚点坐标: {[anchor_x, anchor_y]}")
    except Exception as e:
        print(f"  > ❌ 致命错误：识别Paint按钮时出错 - {e}")
        return

    palette_left = anchor_x + PALETTE_OFFSET_X
    palette_top = anchor_y + PALETTE_OFFSET_Y
    search_region = (palette_left, palette_top, PALETTE_SIZE_W, PALETTE_SIZE_H)
    print(f"\n[Step 2/4] 截取调色板区域: {search_region}")
    palette_screenshot = pyautogui.screenshot(region=search_region)
    # palette_screenshot.save("debug_palette_capture_v4.png")

    # Step 3: 推导网格尺寸 (与 v3 完全相同)
    print("\n[Step 3/4] 正在寻找黑色块并推导网格尺寸...")
    black_seed = find_black_seed(palette_screenshot, BLACK_RGB_TARGET, BLACK_COLOR_TOLERANCE, MIN_BLACK_BLOCK_SIZE)
    if not black_seed:
        print("  > ❌ 致命错误：未能找到黑色基准块！")
        return
    bounds = get_bounding_box(palette_screenshot, black_seed, BLACK_RGB_TARGET, BLACK_COLOR_TOLERANCE)
    if not bounds:
        print("  > ❌ 致命错误：无法确定黑色块的边界！")
        return

    black_left, black_top, black_right, black_bottom = bounds
    swatch_height = black_bottom - black_top
    swatch_width = PALETTE_SIZE_W / PALETTE_GRID_COLS

    black_center_x = palette_left + (black_left + black_right) / 2
    black_center_y = palette_top + (black_top + black_bottom) / 2
    print(f"  > ✅ 成功！推导出单个颜色块尺寸: Width≈{swatch_width:.2f}, Height≈{swatch_height:.2f}")

    # 【✨✨✨ Step 4: 核心修正 - 遍历网格，采样颜色，反向识别ID ✨✨✨】
    print("\n[Step 4/4] 遍历网格，识别每个位置的颜色ID...")

    # 假设黑色(ID 1)是第一行第一个，以此为基准计算所有格子的中心
    start_x = black_center_x
    start_y = black_center_y
    assigned_ids = set()  # 用于防止重复识别

    for row in range(PALETTE_GRID_ROWS):
        for col in range(PALETTE_GRID_COLS):
            # a. [几何通道] 计算当前格子的中心屏幕坐标
            center_x = start_x + col * swatch_width
            center_y = start_y + row * swatch_height

            # 视觉反馈，移动到我们“认为”的中心点
            pyautogui.moveTo(center_x, center_y, duration=0.02)

            # b. [视觉通道] 在截图中，采样这个位置的实际颜色
            relative_x = int(center_x - palette_left)
            relative_y = int(center_y - palette_top)

            try:
                sampled_rgb = palette_screenshot.getpixel((relative_x, relative_y))
            except IndexError:
                print(f"  > 警告: Grid pos ({row},{col}) 计算出的坐标超出了截图范围，跳过。")
                continue

            # c. [反向识别] 查找这个采样颜色最接近哪个已知的颜色ID
            best_match_id = find_closest_color_id(sampled_rgb, COLORS_TO_FIND, COLOR_TOLERANCE)

            # d. 存储结果
            if best_match_id and best_match_id not in assigned_ids:
                gui_config["color_palette_coords"][best_match_id] = [int(center_x), int(center_y)]
                assigned_ids.add(best_match_id)
                print(
                    f"  > Grid pos ({row},{col}) -> ✅ 识别为 Color ID {best_match_id} @ {[int(center_x), int(center_y)]}")
            elif best_match_id in assigned_ids:
                print(f"  > Grid pos ({row},{col}) -> ⚠️ 警告: 识别出的 Color ID {best_match_id} 已被分配，跳过。")
            else:
                print(f"  > Grid pos ({row},{col}) -> ❌ 未能识别颜色 {sampled_rgb} (容忍度 {COLOR_TOLERANCE})")

    # 5. 保存 (逻辑不变)
    print(f"\n[Done] 正在保存结果到 '{OUTPUT_CONFIG_FILE}'...")
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        json.dump(gui_config, f, indent=4)
    print("  > ✅ 校准完成！")


if __name__ == "__main__":
    calibrate_gui_v4()