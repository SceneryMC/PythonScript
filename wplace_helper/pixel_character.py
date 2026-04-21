from PIL import Image, ImageDraw, ImageFont
import os


def generate_pixel_text(text, font_size=12, line_spacing=2, font_path=None):
    """
    生成支持多行的清晰红字透明底点阵图
    :param text: 文字内容，支持 \n 换行
    :param font_size: 字体大小 (像素字体建议 10, 12, 16)
    :param line_spacing: 行距 (像素)
    :param font_path: 字体路径 (强烈建议使用 ipix.ttf 或 zpix.ttf)
    """

    # 1. 加载字体
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"无法加载字体 {font_path}，将尝试系统字体。错误: {e}")

    if font is None:
        # 尝试常见的中文系统字体作为备选
        # Windows: simhei.ttf (黑体), msyh.ttc (微软雅黑)
        # Mac: PingFang.ttc
        bg_fonts = ["simhei.ttf", "msyh.ttc", "PingFang.ttc", "STHeiti Medium.ttc"]
        for f in bg_fonts:
            try:
                font = ImageFont.truetype(f, font_size)
                break
            except:
                continue
        if font is None:
            print("错误：未找到合适的中文字体，请下载像素字体并指定 font_path")
            return

    # 2. 处理多行逻辑，计算画布尺寸
    lines = text.split('\n')

    # 获取每一行的宽和高
    # 注意：Pillow 新版建议使用 getbbox 或 textbbox，但在小像素处理时
    # 为了兼容性和简单性，我们遍历计算最大宽高
    max_width = 0
    total_height = 0
    line_heights = []

    # 虚拟画布用于测量
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    for line in lines:
        # textbbox 返回 (left, top, right, bottom)
        bbox = temp_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # 修正：有时候 textbbox 高度偏小，如果不使用像素字体，建议固定高度
        # 这里取 font_size 和 bbox 高度的较大值，保证不被切头
        h = max(h, font_size + 2)

        if w > max_width:
            max_width = w

        line_heights.append(h)
        total_height += h

    # 加上行间距
    total_height += (len(lines) - 1) * line_spacing

    # 3. 创建画布 (稍微留点余量，最后会裁切)
    canvas_w = max_width + 10
    canvas_h = total_height + 10
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 4. 逐行绘制
    current_y = 0
    for i, line in enumerate(lines):
        # 居左绘制。如果想居中，可以计算 (canvas_w - current_line_width) // 2
        draw.text((0, current_y), line, font=font, fill=(255, 0, 0, 255))
        current_y += line_heights[i] + line_spacing

    # 5. 【核心步骤】二值化处理（锐化边缘，去半透明）
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > 100:  # 阈值，大于100的透明度转为纯红
                pixels[x, y] = (255, 0, 0, 255)
            else:
                pixels[x, y] = (0, 0, 0, 0)

    # 6. 自动裁切多余空白
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 7. 保存与输出
    output_file = "pixel_text_multi.png"
    img.save(output_file)
    print(f"✅ 生成完毕！文件: {output_file}")
    print(f"📏 最终尺寸: {img.width}x{img.height}")

    # 控制台预览
    print("\n--- 效果预览 ---")
    pixels = img.load()
    for y in range(img.height):
        line_str = ""
        for x in range(img.width):
            if pixels[x, y][3] > 0:
                line_str += "█"
            else:
                line_str += " "
        print(line_str)
    print("----------------")


# ================= 配置区 =================
# 建议：汉字要想清晰，最小通常需要 11px 或 12px
# 如果你有像素字体文件，把路径填在 font_path 里，比如 "ipix.ttf"
text_content = """
QQ WX Bili = SceneryMC
"""
size = 10

# 运行
generate_pixel_text(text_content, font_size=size, line_spacing=0, font_path=r"C:\Users\SceneryMC\Downloads\Galmuri-v2.40.3\Galmuri14Bitmap-Regular-2.40.3.ttf")