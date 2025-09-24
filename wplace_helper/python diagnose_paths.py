import os
from psd_tools import PSDImage
from psd_tools.psd.vector import Path
from psd_tools.api.shape import VectorMask

# ========================================================================
# >> SETTINGS <<
# ========================================================================
PSD_FILE_PATH = "dst/9148c828bis91/9148c828bis91_undithered_new.psd"


# ========================================================================
# >> 深度诊断主函数 <<
# ========================================================================

def deep_diagnose_paths(psd_path):
    """
    对PSD文件进行深度扫描，以找到所有可能存储路径数据的地方。
    """
    print("======================================================================")
    print(f"--- Deep Vector Path Diagnostics for: {psd_path} ---")
    print("======================================================================")

    if not os.path.exists(psd_path):
        print(f"Error: File not found at '{psd_path}'")
        return

    try:
        psd = PSDImage.open(psd_path)
    except Exception as e:
        print(f"Error: Failed to open or parse PSD file: {e}")
        return

    print(f"\n[INFO] PSD Info: {psd.width}x{psd.height}, Depth: {psd.depth}")

    # --- 1. 标准方法检查 (shpa) ---
    print("\n--- [Stage 1] Standard Path Extraction (`shpa` block) ---")
    try:
        shpa_block = psd.tagged_blocks.get(b'shpa', [])
        saved_paths = [shape for shape in shpa_block if isinstance(shape, Path)]

        if saved_paths:
            print(f"SUCCESS: Found {len(saved_paths)} Path object(s) using the standard method.")
            for i, path in enumerate(saved_paths):
                print(f"  - Path {i + 1}: Name = '{getattr(path, 'name', 'N/A')}', "
                      f"Num records = {len(getattr(path, 'records', []))}")
        else:
            print(
                "INFO: The standard `shpa` block was found but contained no Path objects, or the block itself was empty.")

    except Exception as e:
        print(f"ERROR: An exception occurred during standard extraction: {e}")

    # --- 2. 暴力扫描所有 Tagged Blocks ---
    print("\n--- [Stage 2] Brute-force Scan of ALL Tagged Blocks ---")

    if not psd.tagged_blocks:
        print("INFO: `psd.tagged_blocks` is empty. No further analysis possible.")
        return

    print(f"Found {len(psd.tagged_blocks)} tagged block keys in total. Analyzing each...\n")

    for key, block_content in psd.tagged_blocks.items():
        try:
            key_str = key.decode('ascii').strip()
        except:
            key_str = str(key)

        print(f"---------------------------------------------------")
        print(f"Key: '{key_str}' (raw: {key})")

        # block_content 可能是一个列表，也可能是一个单独的对象
        if isinstance(block_content, list):
            print(f"Type: List of {len(block_content)} items")
            items_to_inspect = block_content[:5]  # 只看前5个
        else:
            print(f"Type: {type(block_content).__name__}")
            items_to_inspect = [block_content]

        for i, item in enumerate(items_to_inspect):
            print(f"  Item {i + 1}:")
            print(f"    - Type: {type(item).__name__}")

            # 检查是否是我们正在寻找的 Path 对象
            if isinstance(item, Path):
                print(f"    - !!! Path Object Found !!!")
                print(f"      - Name: '{getattr(item, 'name', 'N/A')}'")
                print(f"      - Is Open: {getattr(item, 'is_open', 'N/A')}")
                print(f"      - Records Count: {len(getattr(item, 'records', []))}")

            # 打印对象的所有属性和一小段内容，寻找线索
            if hasattr(item, '__dict__'):
                print(f"    - Attributes: {list(item.__dict__.keys())}")

            # 尝试打印对象自身的字符串表示，可能会有有用信息
            try:
                item_str_preview = str(item)
                if len(item_str_preview) > 150:
                    item_str_preview = item_str_preview[:150] + "..."
                print(f"    - String Preview: {item_str_preview}")
            except:
                print(f"    - String Preview: (Could not be generated)")

    print("\n---------------------------------------------------")
    print("--- Diagnostics Complete ---")


if __name__ == "__main__":
    deep_diagnose_paths(PSD_FILE_PATH)