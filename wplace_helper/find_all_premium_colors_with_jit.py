# 保存为 wplace_tool_final_fast.py
import sys
import argparse
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from collections import Counter
from numba import jit  # <--- 1. 引入Numba的JIT编译器

from utils import WPLACE_COLOR_PALETTE, DEFAULT_FREE_COLORS, CURRENT_AVAILABLE_COLORS


# --- 核心算法 (已使用Numba加速) ---
# 为这个函数添加JIT装饰器
@jit(nopython=True)
def find_best_match_with_penalty(pixel_lab, palette_lab, palette_chroma):
    L_p, a_p, b_p = pixel_lab[0, 0, :]
    target_chroma = np.sqrt(a_p ** 2 + b_p ** 2)

    distances = np.sum((palette_lab - np.array([L_p, a_p, b_p])) ** 2, axis=1)

    if target_chroma > 20:
        # Numba需要显式的循环，但它会将其编译成极快的代码
        for i in range(len(palette_chroma)):
            if palette_chroma[i] < target_chroma:
                chroma_diff = target_chroma - palette_chroma[i]
                penalty = (chroma_diff ** 2) * 0.15
                distances[i] += penalty

    return np.argmin(distances)


# 2. 为这个主循环函数也添加JIT装饰器
@jit(nopython=True)
def perform_dithering_accurate_fast(image_array, palette_rgb, palette_lab, palette_chroma):
    height, width, _ = image_array.shape
    pixel_buffer = image_array.astype(np.float32)
    output_pixels = np.zeros_like(image_array)

    for y in range(height):
        for x in range(width):
            current_pixel_rgb = pixel_buffer[y, x, :3]
            # Numba处理clip
            for c in range(3):
                if current_pixel_rgb[c] < 0: current_pixel_rgb[c] = 0
                if current_pixel_rgb[c] > 255: current_pixel_rgb[c] = 255

            # skimage.color.rgb2lab不能在nopython模式下使用，所以我们在这里调用一个
            # 兼容的、手动的rgb2lab转换 (这是必要的妥协)
            current_pixel_lab = manual_rgb2lab(current_pixel_rgb)
            best_match_idx = find_best_match_with_penalty_jit(current_pixel_lab, palette_lab, palette_chroma)
            new_pixel_rgb = palette_rgb[best_match_idx]

            output_pixels[y, x, :3] = new_pixel_rgb
            error = current_pixel_rgb - new_pixel_rgb

            if x + 1 < width:
                pixel_buffer[y, x + 1, :3] += error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    pixel_buffer[y + 1, x - 1, :3] += error * 3 / 16
                pixel_buffer[y + 1, x, :3] += error * 5 / 16
                if x + 1 < width:
                    pixel_buffer[y + 1, x + 1, :3] += error * 1 / 16
    return output_pixels


# Numba兼容的rgb2lab转换函数 (数学实现)
@jit(nopython=True)
def manual_rgb2lab(rgb):
    # sRGB to XYZ
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB
    x /= 0.95047;
    y /= 1.00000;
    z /= 1.08883
    x = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    y = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    z = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)
    L = (116 * y) - 16;
    a = 500 * (x - y);
    b = 200 * (y - z)
    return np.array([[[L, a, b]]])


@jit(nopython=True)
def find_best_match_with_penalty_jit(pixel_lab, palette_lab, palette_chroma):
    L_p, a_p, b_p = pixel_lab[0, 0, 0], pixel_lab[0, 0, 1], pixel_lab[0, 0, 2]
    target_chroma = np.sqrt(a_p ** 2 + b_p ** 2)
    distances = np.sum((palette_lab - np.array([L_p, a_p, b_p])) ** 2, axis=1)
    if target_chroma > 20:
        for i in range(len(palette_chroma)):
            if palette_chroma[i] < target_chroma:
                chroma_diff = target_chroma - palette_chroma[i]
                penalty = (chroma_diff ** 2) * 0.15
                distances[i] += penalty
    return np.argmin(distances)


def get_palette_info(allowed_color_names):
    palette = [c for c in WPLACE_COLOR_PALETTE if c["name"] in allowed_color_names]
    rgb_values = np.array([c["rgb"] for c in palette])
    lab_values = rgb2lab(rgb_values.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    chroma_values = np.sqrt(lab_values[:, 1] ** 2 + lab_values[:, 2] ** 2)
    return palette, rgb_values, lab_values, chroma_values


# --- 功能函数 (现在调用Numba加速的核心) ---

def analyze_image_colors(image_path, target_size=None):
    print("🔬 [精确分析模式] 正在执行抖动感知分析...")
    try:
        source_image = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{image_path}'")
        return None

    # --- 新增部分开始 ---
    # 2. 如果提供了目标尺寸，则先进行高质量缩放
    if target_size and source_image.size != target_size:
        print(f"📐 正在将图片从 {source_image.size} 缩放至 {target_size}...")
        # Image.LANCZOS 是 Pillow 9.1.0+ 的高质量缩放算法 (旧版为 Image.ANTIALIAS)
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
        source_image = source_image.resize(target_size, resample=resample_filter)
    # --- 新增部分结束 ---

    full_palette, full_rgb, full_lab, full_chroma = get_palette_info([c['name'] for c in WPLACE_COLOR_PALETTE])

    pixels_rgba = np.array(source_image)
    is_opaque = pixels_rgba[:, :, 3] > 100

    print("🚀 Numba JIT编译中 (首次运行会稍慢)...")
    dithered_pixels = perform_dithering_accurate_fast(pixels_rgba[:, :, :3], full_rgb, full_lab, full_chroma)
    print("\n✅ JIT编译完成, 高速处理中!")

    rgb_to_name = {tuple(c['rgb']): c['name'] for c in full_palette}
    color_counts = Counter()

    final_colors_in_image = dithered_pixels[is_opaque]
    unique_final_colors, counts = np.unique(final_colors_in_image, axis=0, return_counts=True)

    for color_rgb, count in zip(unique_final_colors, counts):
        color_name = rgb_to_name.get(tuple(color_rgb))
        if color_name: color_counts[color_name] += count

    print("✅ 分析完成!")
    return color_counts


def generate_preview_image(input_path, output_path, allowed_color_names, target_size=None):
    print(f"🎨 [精确生成模式] 将使用 {len(allowed_color_names)} 种颜色生成预览图。")
    try:
        source_image = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{input_path}'"); return

    # --- 新增部分开始 ---
    # 2. 如果提供了目标尺寸，则先进行高质量缩放
    if target_size and source_image.size != target_size:
        print(f"📐 正在将图片从 {source_image.size} 缩放至 {target_size}...")
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
        source_image = source_image.resize(target_size, resample=resample_filter)
    _, palette_rgb, palette_lab, palette_chroma = get_palette_info(allowed_color_names)

    pixels_rgba = np.array(source_image)
    alpha_channel = source_image.split()[3]
    is_opaque = np.array(alpha_channel) > 100

    print("🚀 Numba JIT编译中 (首次运行会稍慢)...")
    dithered_rgb = perform_dithering_accurate_fast(pixels_rgba[:, :, :3], palette_rgb, palette_lab, palette_chroma)
    print("\n✅ JIT编译完成, 高速处理中!")

    final_pixels_rgba = np.zeros_like(pixels_rgba)
    final_pixels_rgba[is_opaque, :3] = dithered_rgb[is_opaque]
    final_pixels_rgba[is_opaque, 3] = 255

    output_image = Image.fromarray(final_pixels_rgba, 'RGBA')
    print(f"💾 正在保存图片到 {output_path}...")
    output_image.save(output_path)
    print("✨ 图片已保存!")


# --- 主程序入口 (不变) ---
def main():
    parser = argparse.ArgumentParser(
        description="WPlace图片分析与高速精确抖动生成工具 (v_final_fast)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (argparse定义不变) ...
    parser.add_argument(
        "mode",
        choices=["analyze", "generate"],
        help="选择工具模式:\n"
             "analyze   - (精确抖动感知) 分析为达最佳效果所需的付费颜色。\n"
             "generate  - (精确抖动) 使用限定的免费颜色生成预览图。"
    )
    parser.add_argument("input_image", help="输入的源图片路径。")
    parser.add_argument(
        "output_image",
        nargs='?',
        default=None,
        help="[仅用于generate模式] 生成的预览图的保存路径。"
    )
    # --- 新增部分开始 ---
    parser.add_argument(
        "--width",
        type=int,
        help="[可选] 指定输出图像的目标宽度。如果提供，通常也应提供--height。"
    )
    parser.add_argument(
        "--height",
        type=int,
        help="[可选] 指定输出图像的目标高度。如果提供，通常也应提供--width。"
    )
    # --- 新增部分结束 ---

    args = parser.parse_args()

    # --- 修改部分开始 ---
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)
    elif args.width or args.height:
        # 如果只提供了一个维度，为了避免歧义，我们报错并退出
        parser.error("必须同时提供 --width 和 --height 参数，或者都不提供。")

    if args.mode == "analyze":
        # 将 target_size 传递给功能函数
        all_color_counts = analyze_image_colors(args.input_image, target_size)
        if all_color_counts:
            # ... (输出逻辑不变) ...
            print("📊 精确抖动感知分析结果 (按像素需求排序):")
            print("=" * 50)
            paid_colors_with_counts = {
                name: count for name, count in all_color_counts.items() if name not in DEFAULT_FREE_COLORS
            }
            if not paid_colors_with_counts:
                print("\n✅ 好消息! 您当前的免费颜色已足够绘制该图片。")
            else:
                sorted_paid_colors = sorted(paid_colors_with_counts.items(), key=lambda item: item[1], reverse=True)
                print(f"\n💰 为达到最佳抖动效果，您需要解锁以下 {len(sorted_paid_colors)} 种付费颜色:")
                for color_name, pixel_count in sorted_paid_colors:
                    print(f"  - {color_name:<20} ({pixel_count} 像素)")
            print("\n" + "=" * 50)

    elif args.mode == "generate":
        if not args.output_image:
            parser.error("在 'generate' 模式下, 必须提供 <output_image> 参数。")
        generate_preview_image(args.input_image, args.output_image, [d['name'] for d in WPLACE_COLOR_PALETTE], target_size)


if __name__ == "__main__":
    main()