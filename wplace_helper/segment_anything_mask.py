import torch # 确保导入torch
import cv2
import numpy as np
import cv2
import pickle


def show_anns(anns, image_bgr):
    """
    在一个图像上显示所有分割蒙版。
    """
    if len(anns) == 0:
        return

    # 创建一个与原图大小相同的彩色蒙版图像，初始为黑色
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3),
                   dtype=np.uint8)

    for ann in sorted_anns:
        m = ann['segmentation']
        # 生成一个随机颜色
        color_mask = np.random.randint(0, 256, 3, dtype=np.uint8)
        # 将布尔蒙版为True的地方，涂上随机颜色
        img[m] = color_mask

    # 将彩色蒙版半透明地叠加到原始图像上
    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    # superimposed_img = image * alpha + mask * beta + gamma
    superimposed_img = cv2.addWeighted(image_bgr.astype(np.uint8), 0.6, img, 0.4, 0)

    return superimposed_img


def resolve_overlaps(masks: list) -> list:
    """
    通过保留IOU更高的蒙版来解决蒙版之间的重叠问题。

    Args:
        masks (list): 包含'segmentation'和'predicted_iou'的字典列表。

    Returns:
        list: 经过处理后，互相之间没有重叠的蒙版列表。
    """
    print("\n正在解决蒙版之间的重叠问题...")

    # 1. 按照 predicted_iou 从高到低排序
    # IOU为1.0的是我们手动添加的空白区域，应该有最低的“挖掉”别人的优先级
    # 所以我们先按IOU排序，再把IOU为1.0的放到最后
    masks.sort(key=lambda x: x['predicted_iou'], reverse=True)

    # 2. 创建一个列表，我们将逐个处理和修改其中的蒙版
    # 我们需要操作蒙版的副本，以避免在迭代时修改列表
    processed_masks = [m['segmentation'].copy() for m in masks]

    num_masks = len(processed_masks)
    pixels_removed_total = 0

    # 3. 遍历每个蒙版，将其从所有IOU更低的蒙版中“挖掉”
    for i in range(num_masks):
        # 当前蒙版是“胜利者”，它的形状是固定的
        winner_mask = processed_masks[i]

        if (i + 1) % 100 == 0:
            print(f"  - 正在处理第 {i + 1}/{num_masks} 个蒙版...")

        # 遍历所有排在它后面的、IOU更低的蒙版
        for j in range(i + 1, num_masks):
            loser_mask = processed_masks[j]

            # 找到重叠区域
            overlap = np.logical_and(winner_mask, loser_mask)

            # 如果有重叠
            if np.any(overlap):
                # 从“失败者”蒙版中移除重叠区域
                pixels_before = np.sum(loser_mask)
                processed_masks[j][overlap] = False
                pixels_after = np.sum(loser_mask)
                pixels_removed_total += (pixels_before - pixels_after)

    print(f"  -> 重叠解决完成。总共移除了 {pixels_removed_total} 个重叠像素。")

    # 4. 将处理后的布尔蒙版更新回原始的字典列表中
    # 同时更新面积信息
    final_masks = []
    for i in range(num_masks):
        original_dict = masks[i]
        new_segmentation = processed_masks[i]
        new_area = np.sum(new_segmentation)

        # 如果一个蒙版的所有像素都被挖掉了，就直接丢弃它
        if new_area > 0:
            original_dict['segmentation'] = new_segmentation
            original_dict['area'] = new_area
            final_masks.append(original_dict)

    print(f"  -> 丢弃了 {num_masks - len(final_masks)} 个被完全覆盖的蒙版。")

    return final_masks

# --- 核心修改 ---
# 1. 自动检测可用的设备 (GPU或CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(f"Using device: {device}") # 打印出来，确认一下

# 2. 加载模型
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)

image_bgr = cv2.imread(r"C:\Users\13308\PycharmProjects\SceneryMCPythonScript\wplace_helper\dst\175822732\175822732.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 假设 sam, image_bgr, image_rgb 已经加载完毕

# --- 步骤1: 调整参数并生成初始蒙版 ---
print("正在使用调整后的参数生成初始蒙版...")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.85,
    min_mask_region_area=25,
)
masks = mask_generator.generate(image_rgb)
print(f"SAM 初始生成了 {len(masks)} 个蒙版。")

# --- 步骤2: 查漏补缺，确保100%覆盖 ---
print("正在执行“查漏补缺”后处理...")

if masks:
    # 2.1 将所有SAM生成的蒙版合并成一个大的“已覆盖区域”
    # 我们使用逻辑“或”操作来合并所有布尔蒙版
    all_sam_masks = np.stack([ann['segmentation'] for ann in masks])
    covered_area = np.logical_or.reduce(all_sam_masks, axis=0)

    # 2.2 找到所有未被覆盖的“空白区域”
    # np.logical_not 就是取反操作
    uncovered_area = np.logical_not(covered_area)

    # 2.3 将布尔值转换为OpenCV可以处理的uint8格式
    uncovered_mask_cv = uncovered_area.astype(np.uint8) * 255

    # 2.4 找到所有不连通的空白区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(uncovered_mask_cv, 8, cv2.CV_32S)

    new_masks_added = 0
    # 我们从标签1开始，因为标签0是背景
    if num_labels > 1:
        for i in range(1, num_labels):
            # 为每个独立的空白区域创建一个新的蒙版
            gap_mask = (labels == i)

            # 为了与SAM的输出格式保持一致，我们创建一个新的字典
            new_mask_dict = {
                'segmentation': gap_mask,
                'area': stats[i, cv2.CC_STAT_AREA],
                'bbox': [stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH],
                         stats[i, cv2.CC_STAT_HEIGHT]],
                # 我们可以给这些补充的蒙版一个特殊的标记或默认值
                'predicted_iou': 1.0,  # 它们是确定的空白区域
                'stability_score': 1.0,
                'point_coords': [centroids[i]],  # 可以用质心作为代表点
                'crop_box': [0, 0, image_rgb.shape[1], image_rgb.shape[0]],
            }
            masks.append(new_mask_dict)
            new_masks_added += 1

    print(f"查漏补缺完成！添加了 {new_masks_added} 个新的蒙版来填补空白区域。")

non_overlapping_masks = resolve_overlaps(masks)
print(f"总计蒙版数量: {len(non_overlapping_masks)}。现在所有蒙版的并集已覆盖整个图像。")

# --- 现在，这个“完整版”的masks列表就可以用于后续的UI工具了 ---
# 保存这个包含了补充蒙版的完整列表
with open('sam_masks.pkl', 'wb') as f:
    pickle.dump(non_overlapping_masks, f)

# 保存原始图像
cv2.imwrite('original_image.png', image_bgr)

print("处理完成的完整蒙版数据已保存到 'sam_masks.pkl'。")