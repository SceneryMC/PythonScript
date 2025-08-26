# 保存为 restore_from_save.py
import json
import base64
import argparse
from PIL import Image
import numpy as np


def unpack_painted_map(packed_data):
    """
    从Base64编码的数据中解包出 'paintedMap' 布尔掩码。
    这个函数精确地复刻了JS脚本中的 'unpackPaintedMapFromBase64' 逻辑。
    """
    if not packed_data or 'data' not in packed_data:
        return None, 0, 0

    width = packed_data.get('width', 0)
    height = packed_data.get('height', 0)

    if width == 0 or height == 0:
        return None, 0, 0

    try:
        # 1. Base64解码
        binary_data = base64.b64decode(packed_data['data'])

        # 2. 将二进制字符串转换为字节数组
        bytes_array = np.frombuffer(binary_data, dtype=np.uint8)

        # 3. 从字节中解包出位 (bits)
        #    np.unpackbits 会将每个字节转换为8个布尔值，并按大端序排列
        bits = np.unpackbits(bytes_array, bitorder='little')

        # 4. 截取有效位数并重塑为 (height, width) 的掩码
        total_pixels = width * height
        if len(bits) < total_pixels:
            # 如果数据不完整，用False填充
            padded_bits = np.zeros(total_pixels, dtype=bool)
            padded_bits[:len(bits)] = bits[:total_pixels]
            bits = padded_bits

        painted_mask = bits[:total_pixels].reshape((height, width))

        return painted_mask, width, height

    except Exception as e:
        print(f"❌ 解码 'paintedMapPacked' 数据时出错: {e}")
        return None, 0, 0


def restore_image(input_path, output_path, progress_only=False):
    """
    从 .json 存档文件中还原图像。

    :param input_path: 输入的 .json 文件路径。
    :param output_path: 输出的 .png 图片路径。
    :param progress_only: 若为True，则只显示已绘制的像素。
    """
    print(f"📄 正在读取存档文件: {input_path}...")
    try:
        with open(input_path, 'r') as f:
            save_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: '{input_path}' 不是一个有效的JSON文件。")
        return

    # --- 1. 还原基础目标图像 ---
    image_data = save_data.get('imageData')
    if not image_data or 'pixels' not in image_data:
        print("❌ 错误: 存档文件中未找到 'imageData'。无法还原图片。")
        return

    width = image_data.get('width')
    height = image_data.get('height')
    pixels_flat = image_data.get('pixels')

    if not all([width, height, pixels_flat]):
        print("❌ 错误: 'imageData' 结构不完整 (缺少 width, height, 或 pixels)。")
        return

    try:
        # 将扁平化的像素列表转换为 (height, width, 4) 的NumPy数组
        target_image_array = np.array(pixels_flat, dtype=np.uint8).reshape((height, width, 4))
    except ValueError as e:
        print(f"❌ 错误: 像素数据与尺寸不匹配。 ({e})")
        return

    final_image_array = target_image_array

    # --- 2. 如果是进度模式，则应用 'paintedMap' 掩码 ---
    if progress_only:
        print("🔍 启用进度模式，正在解析已绘制像素...")
        painted_map_packed = save_data.get('paintedMapPacked')
        if not painted_map_packed:
            print("⚠️ 警告: 存档文件中未找到 'paintedMapPacked'。将显示完整目标图。")
        else:
            painted_mask, mask_w, mask_h = unpack_painted_map(painted_map_packed)

            if painted_mask is not None and mask_w == width and mask_h == height:
                # 创建一个半透明的深灰色背景画布
                progress_canvas = np.full((height, width, 4), (50, 50, 50, 150), dtype=np.uint8)

                # 使用布尔掩码，只将目标图像中已绘制的像素复制到画布上
                # 这是NumPy最高效的操作方式
                progress_canvas[painted_mask] = target_image_array[painted_mask]

                final_image_array = progress_canvas
                print(f"✅ 已成功应用 'paintedMap'，显示 {np.sum(painted_mask)} 个已绘制像素。")
            else:
                print("⚠️ 警告: 'paintedMap' 尺寸与图像尺寸不匹配。将显示完整目标图。")

    # --- 3. 保存最终图像 ---
    try:
        print(f"🖼️  正在创建图像对象...")
        image = Image.fromarray(final_image_array, 'RGBA')

        print(f"💾 正在保存图像到: {output_path}...")
        image.save(output_path)
        print(f"✨ 图像已成功还原并保存!")
    except Exception as e:
        print(f"❌ 保存图像时发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="从WPlace Bot的 .json 存档文件中还原图像。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_json", help="输入的 .json 存档文件路径。")
    parser.add_argument("output_png", help="输出的 .png 图片文件路径。")
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="如果设置此项，则只显示已绘制的像素，\n"
             "未绘制区域将以半透明灰色背景显示。"
    )
    args = parser.parse_args()

    restore_image(args.input_json, args.output_png, args.progress_only)


if __name__ == "__main__":
    main()