import os

import numpy as np
from PIL import Image

# --- 配置 ---

# 请将'canvas.png'替换为您当前画布上图像的文件名
CANVAS_IMAGE_PATH = r"C:\Users\13308\nodejsproj\wplacer-lllexxa\data\snapshots\my_nilou_snapshot.png"

# 请将'template.png'替换为您新的目标模板图像的文件名
TEMPLATE_IMAGE_PATH = "dst/726/726_12_modified_converted_merged.png"


# --- 脚本主逻辑 ---

def analyze_image_diff(canvas_path, template_path):
    """
    加载并比较两张RGBA图像，分析像素差异。
    """
    try:
        print(f"正在加载画布图像: {canvas_path}")
        canvas_img = Image.open(canvas_path).convert("RGBA")

        print(f"正在加载新模板图像: {template_path}")
        template_img = Image.open(template_path).convert("RGBA")
    except FileNotFoundError as e:
        print(f"错误: 找不到文件！请检查路径是否正确。 - {e}")
        return

    # --- 步骤1: 统一图像尺寸 ---
    # 为了防止因尺寸不同导致的错误，我们将图像扩展到能容纳两者的最大尺寸
    max_width = max(canvas_img.width, template_img.width)
    max_height = max(canvas_img.height, template_img.height)

    # 将PIL图像转换为NumPy数组
    canvas_np = np.array(canvas_img)
    template_np = np.array(template_img)

    # 创建一个足够大的、完全透明的背景
    canvas_full = np.zeros((max_height, max_width, 4), dtype=np.uint8)
    template_full = np.zeros((max_height, max_width, 4), dtype=np.uint8)

    # 将原始图像数据“粘贴”到大背景的左上角
    canvas_full[:canvas_np.shape[0], :canvas_np.shape[1], :] = canvas_np
    template_full[:template_np.shape[0], :template_np.shape[1], :] = template_np

    print(f"图像已统一到 {max_width}x{max_height} 尺寸进行比较。")

    # --- 步骤2: 使用NumPy进行高效比较 ---

    # 提取Alpha通道（透明度），并创建布尔蒙版
    # 我们定义alpha > 127 为非透明 (opaque)
    canvas_alpha = canvas_full[:, :, 3]
    template_alpha = template_full[:, :, 3]

    canvas_is_opaque = canvas_alpha > 127
    template_is_opaque = template_alpha > 127

    # --- 步骤3: 根据您的4个分类进行计算 ---

    # 类别1: 等待绘制 (画布透明, 新模板不透明)
    # 使用逻辑与 (&) 操作符
    to_be_drawn_mask = (~canvas_is_opaque) & (template_is_opaque)
    count_drawn = np.sum(to_be_drawn_mask)

    # 类别2: 等待删去 (画布不透明, 新模板透明)
    to_be_erased_mask = (canvas_is_opaque) & (~template_is_opaque)
    count_erased = np.sum(to_be_erased_mask)

    # 类别3: 等待重绘 (两者都不透明, 但颜色不同)
    # 仅在两者都不透明的区域进行颜色比较
    both_opaque_mask = canvas_is_opaque & template_is_opaque

    # 比较RGB颜色通道 (前3个通道)
    # np.any会检查每个像素的R,G,B中是否有任何一个不同
    colors_are_different = np.any(canvas_full[:, :, :3] != template_full[:, :, :3], axis=2)

    # 最终的重绘蒙版是“两者都不透明”和“颜色不同”的交集
    to_be_redrawn_mask = both_opaque_mask & colors_are_different
    count_redrawn = np.sum(to_be_redrawn_mask)

    # 类别4: 完全一样 (两者都不透明, 且颜色也一样)
    # 这是“两者都不透明”和“颜色相同”的交集
    colors_are_same = ~colors_are_different
    same_mask = both_opaque_mask & colors_are_same
    count_same = np.sum(same_mask)

    # --- 步骤4: 输出结果 ---
    total_pixels = max_width * max_height

    print("\n--- 图像差异分析结果 ---")
    print(f"1. 【完全一样】的像素: {count_same:,}")
    print(f"2. 【等待绘制】的像素: {count_drawn:,}")
    print(f"3. 【等待重绘】的像素: {count_redrawn:,}")
    print(f"4. 【等待删去】的像素: {count_erased:,}")

    print("\n--- 总结 ---")
    workload_pixels = count_drawn + count_redrawn + count_erased
    print(f"总计需要操作的像素 (绘制+重绘+删去): {workload_pixels:,}")

    # 完整性校验
    both_transparent = np.sum((~canvas_is_opaque) & (~template_is_opaque))
    calculated_total = count_same + count_drawn + count_redrawn + count_erased + both_transparent
    print(f"保持不变的像素 (完全一样 + 共同透明): {count_same + both_transparent:,}")
    print(f"总像素: {total_pixels:,} | 校验总和: {calculated_total:,}")

    if total_pixels == calculated_total:
        print("\n校验成功！所有像素都已归类。")
    else:
        print("\n警告：校验失败，存在未归类的像素。")


if __name__ == "__main__":
    # 确保您的文件名是正确的
    if os.path.exists(CANVAS_IMAGE_PATH) and os.path.exists(TEMPLATE_IMAGE_PATH):
        analyze_image_diff(CANVAS_IMAGE_PATH, TEMPLATE_IMAGE_PATH)
    else:
        print("错误：一个或两个图像文件不存在，请检查配置中的文件名。")