# ultimate_humanized_clicker_v3.py
import pyautogui
import numpy as np
import cv2
import time
import random

# -------------------【请在这里配置】-------------------
# 1. 基础配置
target_color_rgb = (12, 129, 110)  # 示例
tolerance = 5
min_width = 2
min_height = 2
max_width = 10
max_height = 10

# 2. 【新增】人性化模拟配置
#    - use_human_like_sorting: True会从上到下、从左到右点击。False会使用默认顺序。
#    - use_random_shuffling: (可选) 如果设为True，会完全打乱点击顺序。会覆盖上面的排序选项。
#    - base_click_interval: 基础点击间隔（秒）。
#    - random_delay_range: 在基础间隔上增加或减少的随机秒数范围。
use_human_like_sorting = True
use_random_shuffling = False

pyautogui.PAUSE = 0.05
# 【新增】随机化鼠标移动速度
mouse_move_duration_base = 0.05  # 鼠标移动的【基础】秒数
mouse_move_duration_range = 0.05  # 在基础秒数上增加或减少的随机范围 (例如，0.2 ± 0.1)

base_click_interval = 0.1
random_delay_range = 0.0123
random_offset_pixels = 3


# ----------------------------------------------------


def main():
    print("“究极人性化”脚本 v3 (速度随机) 将在3秒后开始...")
    time.sleep(3)

    try:
        print("脚本运行中... 将鼠标快速移动到屏幕左上角可强制停止脚本。")

        # --- 第1步：图像分析，找出所有有效的目标点 ---
        screenshot = pyautogui.screenshot()
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        target_color_bgr = (target_color_rgb[2], target_color_rgb[1], target_color_rgb[0])
        lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr])
        upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr])
        mask = cv2.inRange(screenshot_cv, lower_bound, upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("未找到任何指定颜色的区域。")
            return

        targets_to_click = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (min_width < w < max_width) and (min_height < h < max_height):
                center_x = x + w // 2
                center_y = y + h // 2
                targets_to_click.append((center_x, center_y))

        if not targets_to_click:
            print("任务完成：未找到任何同时满足颜色和尺寸条件的标记。")
            return

        print(f"筛选完成，共找到 {len(targets_to_click)} 个有效目标。")

        # --- 第2步：对目标点进行排序或打乱 ---
        if use_random_shuffling:
            random.shuffle(targets_to_click)
            print("已将点击顺序完全打乱。")
        elif use_human_like_sorting:
            targets_to_click.sort(key=lambda point: (point[1], point[0]))
            print("已将点击顺序排序为“从上到下，从左到右”。")

        # --- 第3步：以人性化的方式执行点击 ---
        print("开始执行人性化点击...")
        for i, (center_x, center_y) in enumerate(targets_to_click):
            # 1. 计算随机偏移量
            if random_offset_pixels > 0:
                offset_x = random.randint(-random_offset_pixels, random_offset_pixels)
                offset_y = random.randint(-random_offset_pixels, random_offset_pixels)
                final_x = center_x + offset_x
                final_y = center_y + offset_y
            else:
                final_x, final_y = center_x, center_y

            # 2. 【核心改进】计算随机化的鼠标移动时间
            move_duration = mouse_move_duration_base + random.uniform(-mouse_move_duration_range,
                                                                      mouse_move_duration_range)
            # 确保移动时间不会过短或为负数，保证总有可见的移动过程
            final_move_duration = max(0.05, move_duration)

            # 3. 模拟鼠标移动
            pyautogui.moveTo(final_x, final_y, duration=final_move_duration, tween=pyautogui.easeOutQuad)

            # 4. 执行点击
            pyautogui.click()

            print(
                f"点击 {i + 1}/{len(targets_to_click)} | 目标中心:({center_x},{center_y}) -> 实际点击:({final_x},{final_y})")

            # 5. 计算并执行随机点击延迟
            click_delay = base_click_interval + random.uniform(-random_delay_range, random_delay_range)
            time.sleep(max(0, click_delay))

        print(f"任务完成！总共点击了 {len(targets_to_click)} 个有效标记。")

    except pyautogui.FailSafeException:
        print("检测到鼠标移动到左上角，脚本已紧急停止。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()