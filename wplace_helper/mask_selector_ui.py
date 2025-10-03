import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import pickle


class MaskSelectorUI:
    def __init__(self, root, masks_data, original_image):
        self.root = root
        self.root.title("高级Mask选择器 (v2 - 已修复)")

        # 1. 初始化数据
        self.original_image_bgr = original_image
        self.height, self.width, _ = self.original_image_bgr.shape
        self.all_masks = sorted(masks_data, key=(lambda x: x['area']), reverse=True)
        for i, mask_dict in enumerate(self.all_masks): mask_dict['id'] = i

        self.used_mask_ids = set()
        self.current_selection_ids = set()
        self.exported_count = 0
        self.output_dir = "sorted_masks"
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_stack = []

        # [新增] 修复显示问题的关键标志位
        self.is_initial_view = True

        # 2. 创建UI元素
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = tk.Frame(main_frame, padx=10, pady=10, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()

        self._setup_control_panel(control_frame)

        # 3. 初始化显示
        self.color_map_for_display = self._generate_color_map()
        self.update_display(save_history=False)

    def _setup_control_panel(self, frame):
        tk.Label(frame, text="操作指南:", font=("Arial", 14, "bold")).pack(pady=10, anchor='w')
        tk.Label(frame, text=" - 使用上方放大镜工具进行缩放").pack(anchor='w', pady=2)
        tk.Label(frame, text=" - 使用十字箭头工具进行平移").pack(anchor='w', pady=2)
        tk.Label(frame, text=" - 点击左侧图像选择/取消区域").pack(anchor='w', pady=2)
        tk.Frame(frame, height=2, bg='gray').pack(fill='x', pady=20)

        self.btn_export_current = tk.Button(frame, text="导出当前选中并集", command=self.export_current_selection,
                                            font=("Arial", 12), bg="#4CAF50", fg="white")
        self.btn_export_current.pack(fill=tk.X, pady=5)

        self.btn_clear_current = tk.Button(frame, text="清空当前选择", command=self.clear_current_selection,
                                           font=("Arial", 12))
        self.btn_clear_current.pack(fill=tk.X, pady=5)

        self.btn_undo = tk.Button(frame, text="撤销上一步操作", command=self.undo_last_action, font=("Arial", 12),
                                  state=tk.DISABLED)
        self.btn_undo.pack(fill=tk.X, pady=(15, 5))

        tk.Frame(frame, height=2, bg='gray').pack(fill='x', pady=20)
        self.btn_export_total = tk.Button(frame, text="导出总轮廓", command=self.export_total_outline,
                                          font=("Arial", 12), bg="#f44336", fg="white")
        self.btn_export_total.pack(fill=tk.X, pady=5)

        self.status_bar = tk.Label(self.root, text="准备就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _generate_color_map(self):
        color_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for mask_dict in reversed(self.all_masks):
            color = np.random.randint(50, 256, 3, dtype=np.uint8)
            color_map[mask_dict['segmentation']] = color
        return color_map

    def update_display(self, save_history=True):
        if save_history: self.save_state_to_history()

        # --- [ 此处是关键修正点 ] ---
        # 1. 在清空画布前，保存当前的视图范围
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        # 2. 准备要显示的复合图像
        display_img = self.original_image_bgr.copy()
        display_img = cv2.addWeighted(display_img, 0.5, self.color_map_for_display, 0.5, 0)
        used_mask_overlay = np.zeros_like(display_img)
        for mask_id in self.used_mask_ids: used_mask_overlay[self.all_masks[mask_id]['segmentation']] = (100, 100, 100)
        display_img = cv2.addWeighted(display_img, 1, used_mask_overlay, 0.7, 0)
        selection_overlay = np.zeros_like(display_img)
        for mask_id in self.current_selection_ids: selection_overlay[self.all_masks[mask_id]['segmentation']] = (255,
                                                                                                                 255,
                                                                                                                 255)
        display_img = cv2.addWeighted(display_img, 1, selection_overlay, 0.6, 0)

        # 3. 清空并重绘
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')

        # 4. 根据标志位决定是否恢复视图
        if self.is_initial_view:
            # 如果是第一次，不做任何操作，让imshow自动调整到最佳视图
            self.is_initial_view = False
        else:
            # 如果不是第一次，则恢复之前的缩放/平移状态
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        # ---------------------------------

        self.canvas.draw()
        self.status_bar.config(
            text=f"当前选中: {len(self.current_selection_ids)} | 已导出: {self.exported_count} | 历史记录: {len(self.history_stack)}")
        self.btn_undo.config(state=tk.NORMAL if self.history_stack else tk.DISABLED)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= y < self.height and 0 <= x < self.width): return

        for mask_dict in reversed(self.all_masks):
            mask_id = mask_dict['id']
            if mask_dict['segmentation'][y, x]:
                if mask_id in self.used_mask_ids:
                    self.status_bar.config(text=f"区域 {mask_id} 已被使用，无法选择。")
                    return
                if mask_id in self.current_selection_ids:
                    self.current_selection_ids.remove(mask_id)
                else:
                    self.current_selection_ids.add(mask_id)
                self.update_display()
                return

    def save_state_to_history(self):
        state = {
            'used_mask_ids': self.used_mask_ids.copy(),
            'current_selection_ids': self.current_selection_ids.copy(),
            'exported_count': self.exported_count
        }
        self.history_stack.append(state)

    def undo_last_action(self):
        if not self.history_stack: return
        last_state = self.history_stack.pop()
        self.used_mask_ids = last_state['used_mask_ids']
        self.current_selection_ids = last_state['current_selection_ids']
        self.exported_count = last_state['exported_count']

        expected_filename = os.path.join(self.output_dir, f"merged_mask_{self.exported_count}.png")
        if os.path.exists(expected_filename):
            try:
                os.remove(expected_filename)
                print(f"撤销操作: 删除了文件 {expected_filename}")
            except OSError as e:
                print(f"错误: 无法删除文件 {expected_filename}: {e}")
        self.update_display(save_history=False)
        # messagebox.showinfo("撤销", "已成功撤销上一步操作。")

    def export_current_selection(self):
        if not self.current_selection_ids:
            messagebox.showwarning("警告", "没有选中任何区域！");
            return

        self.save_state_to_history()
        merged_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for mask_id in self.current_selection_ids:
            merged_mask[self.all_masks[mask_id]['segmentation']] = 255

        filename = os.path.join(self.output_dir, f"merged_mask_{self.exported_count}.png")
        cv2.imwrite(filename, merged_mask)

        self.used_mask_ids.update(self.current_selection_ids)
        self.current_selection_ids.clear()
        self.exported_count += 1

        self.update_display(save_history=False)
        messagebox.showinfo("成功", f"并集已成功导出为 {filename}")

    def clear_current_selection(self):
        if not self.current_selection_ids: return
        self.save_state_to_history()
        self.current_selection_ids.clear()
        self.update_display(save_history=False)

    def export_total_outline(self):
        if not self.used_mask_ids:
            messagebox.showwarning("警告", "还未导出任何并集！");
            return
        total_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for mask_id in self.used_mask_ids:
            total_mask[self.all_masks[mask_id]['segmentation']] = 255
        filename = os.path.join(self.output_dir, "total_outline.png")
        cv2.imwrite(filename, total_mask)
        messagebox.showinfo("成功", f"总轮廓已成功导出为 {filename}")


def main():
    try:
        with open('sam_masks.pkl', 'rb') as f:
            masks_data = pickle.load(f)
        original_image = cv2.imread('original_image.png')
        if original_image is None: raise FileNotFoundError
    except (FileNotFoundError, pickle.UnpicklingError):
        messagebox.showerror("错误",
                             "无法加载 'sam_masks.pkl' 或 'original_image.png'!\n请先运行SAM脚本生成并处理这些文件。")
        return

    root = tk.Tk()
    app = MaskSelectorUI(root, masks_data, original_image)
    root.mainloop()


if __name__ == "__main__":
    main()