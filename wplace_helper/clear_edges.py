from PIL import Image

# --- 配置项 ---

# 输入您的原始像素画文件名
INPUT_FILENAME = r"E:\共享\dst\00000-3494321294\00000-3494321294_modified_converted_cleared.png"

# 处理后保存的文件名
OUTPUT_FILENAME = r"E:\共享\dst\00000-3494321294\00000-3494321294_modified_converted_cleared.png"

# --- 颜色判断的精细调整 ---

# 亮度阈值 (Brightness Threshold) - [新增]
# 如果一个颜色的R,G,B三个通道的 *最小值* 高于此阈值，它就会被视为“太亮”（如白色、亮灰色），
# 并会从图像边缘移除。
# 值范围是 0-255。推荐值在 190 到 220 之间。
# 值越低，越多浅色会被移除。值越高，只有接近纯白的颜色才会被移除。
BRIGHTNESS_THRESHOLD = 100

# 灰度容差 (Greyscale Tolerance)
# 如果一个颜色的R,G,B最大值和最小值的差小于此数值，则被视为灰色/黑色。
GREYSCALE_TOLERANCE = 30

# 饱和度阈值 (Saturation Threshold)
# 如果一个颜色的饱和度低于此值，则被视为“安全”的大地色系或暗色。
# 0.0是纯灰色，1.0是纯彩色。推荐值在 0.25 到 0.4 之间。
SATURATION_THRESHOLD = 0.35


# --- 脚本主逻辑 ---

def is_safe_color(r, g, b):
    """
    判断一个像素颜色是否为“安全”的颜色（中低亮度的黑、灰、大地色系）。
    这些颜色即时在边缘也不会被移除。
    """
    # --- *** 关键修改 *** ---
    # 1. 亮度检查：如果颜色太亮（接近白色），则它不是安全颜色，直接返回 False。
    if min(r, g, b) > BRIGHTNESS_THRESHOLD:
        return False

    max_val = max(r, g, b)
    min_val = min(r, g, b)

    # 2. 灰度检查：如果R,G,B值非常接近（并且通过了亮度检查），则视为安全的灰色系。
    if (max_val - min_val) < GREYSCALE_TOLERANCE:
        return True

    # 3. 饱和度检查：饱和度低的颜色是大地色系或暗色，也是安全的。
    if max_val == 0:
        return True  # 纯黑是安全的
    saturation = (max_val - min_val) / max_val
    if saturation < SATURATION_THRESHOLD:
        return True
    if max_val == g:
        return False

    # 如果以上安全条件都不满足，说明这是一个需要从边缘移除的彩色。
    return True


def remove_unwanted_edges(input_path, output_path):
    """
    加载图像，移除边缘的彩色和亮色像素，并保存结果。
    """
    try:
        img = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'。请确保文件名正确。")
        return

    pixels = img.load()
    width, height = img.size

    pixels_to_remove = []

    print(f"正在分析图像 '{input_path}' ({width}x{height})...")

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            r, g, b, a = pixels[x, y]

            # 首先，完全跳过透明的像素
            if a == 0:
                continue

            # 检查当前像素颜色是否为“不安全”的颜色
            if not is_safe_color(r, g, b):
                is_edge = False
                # 检查4个方向的邻居像素
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if width > nx >= 0 == pixels[nx, ny][3] and 0 <= ny < height:
                        is_edge = True
                        break

                if is_edge:
                    pixels_to_remove.append((x, y))

    if not pixels_to_remove:
        print("分析完成，未找到需要移除的边缘像素。")
        return

    print(f"找到了 {len(pixels_to_remove)} 个需要移除的边缘像素，正在处理...")

    for x, y in pixels_to_remove:
        pixels[x, y] = (0, 0, 0, 0)

    img.save(output_path)
    print(f"处理完成！图像已保存为 '{output_path}'。")


if __name__ == "__main__":
    remove_unwanted_edges(INPUT_FILENAME, OUTPUT_FILENAME)