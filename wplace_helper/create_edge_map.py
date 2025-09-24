import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# --- 核心配置 ---

# 1. 选择您要使用的算法
#    可选值: "CANNY", "WATERSHED", "LAPLACIAN"
ALGORITHM_CHOICE = "CANNY"

# 2. 输入/输出文件名
INPUT_FILENAME = r"E:\共享\dst\00000-3494321294\00000-3494321294_undithered.png"  # <--- 请修改为您的文件名
OUTPUT_FILENAME = f"template_edge_map_{ALGORITHM_CHOICE.lower()}.png"  # 输出文件名会自动变化

# --- 算法参数 (请根据选择的算法进行调整) ---

# --- A. Canny 算法参数 ---
# 因为图像是干净的，我们可以使用较小的模糊核和更严格的阈值
CANNY_BLUR_KERNEL = (3, 3)  # (奇数, 奇数)，对于清晰图像，(3,3)甚至(1,1)都可以
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- B. 分水岭 (Watershed) 算法参数 ---
# 这个阈值决定了哪些区域会被识别为独立的“色块”。
# 值在 0 (最不敏感) 到 255 (最敏感) 之间。
# 值越低，算法越倾向于将相似的颜色合并成大块。
# 值越高，算法越倾向于将细微的颜色差异也分割开。
# 建议从 10-30 开始尝试。
WATERSHED_MARKER_THRESHOLD = 20

# --- C. 拉普拉斯 (Laplacian) 算法参数 ---
LAPLACIAN_BLUR_KERNEL = (3, 3)  # (奇数, 奇数)


def detect_edges():
    """
    根据 ALGORITHM_CHOICE 的值，选择不同的高级边缘检测算法。
    """
    print(f"--- 使用 {ALGORITHM_CHOICE} 算法 ---")

    # --- 1. 使用健壮的方式加载图像 ---
    print(f"正在加载图像: '{INPUT_FILENAME}'...")
    try:
        with open(INPUT_FILENAME, 'rb') as f:
            numpy_array = np.frombuffer(f.read(), np.uint8)
            img_rgba = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"读取或解码文件时发生错误: {e}");
        return

    if img_rgba is None or img_rgba.shape[2] < 4:
        print("错误：文件不是有效的带透明通道的PNG图像。");
        return

    # --- 2. 预处理：创建一个用于分析的灰度图，正确处理透明背景 ---
    gray_image = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2GRAY)
    alpha_channel = img_rgba[:, :, 3]

    # --- 3. 根据选择执行不同算法 ---
    edge_map = None

    if ALGORITHM_CHOICE == "CANNY":
        blurred = cv2.GaussianBlur(gray_image, CANNY_BLUR_KERNEL, 0)
        edge_map = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    elif ALGORITHM_CHOICE == "LAPLACIAN":
        blurred = cv2.GaussianBlur(gray_image, LAPLACIAN_BLUR_KERNEL, 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        # 将结果转换回8位黑白图像
        edge_map = cv2.convertScaleAbs(laplacian)

    elif ALGORITHM_CHOICE == "WATERSHED":
        # 分水岭算法步骤比较复杂
        # 1. 计算距离变换，找到每个色块的“中心”
        distance = ndi.distance_transform_edt(gray_image)

        # 2. 根据阈值找到“确定的”前景区域（色块的种子）
        local_maxi = peak_local_max(distance, min_distance=WATERSHED_MARKER_THRESHOLD, labels=gray_image)
        markers = np.zeros(distance.shape, dtype=bool)
        markers[local_maxi[:, 0], local_maxi[:, 1]] = True
        markers = ndi.label(markers)[0]

        # 3. 运行分水岭算法，-distance是“地形”
        labels = watershed(-distance, markers, mask=gray_image)

        # 4. 从分割后的区域中提取边界
        edge_map = np.zeros_like(gray_image, dtype=np.uint8)
        # 遍历所有分割出的区域标签
        for label in np.unique(labels):
            if label == 0: continue  # 0是背景
            # 为每个区域创建一个蒙版，然后找到它的轮廓
            mask = np.zeros(gray_image.shape, dtype="uint8")
            mask[labels == label] = 255
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 将轮廓画到我们的边缘图上
            cv2.drawContours(edge_map, contours, -1, 255, 1)

    else:
        print(f"错误: 未知的算法 '{ALGORITHM_CHOICE}'");
        return

    # --- 4. 最终处理：确保边缘图在原始透明区域也是透明的 ---
    # 这一步可以防止在图像最外圈产生不必要的线条
    final_edges = np.where(alpha_channel > 0, edge_map, 0)

    # --- 5. 保存结果 ---
    try:
        cv2.imwrite(OUTPUT_FILENAME, final_edges)
        print("-" * 30)
        print(f"✅ 成功！边缘地图已保存为 '{OUTPUT_FILENAME}'")
        print("-" * 30)
    except Exception as e:
        print(f"错误：无法保存文件 '{OUTPUT_FILENAME}'. 详细信息: {e}")


if __name__ == "__main__":
    detect_edges()