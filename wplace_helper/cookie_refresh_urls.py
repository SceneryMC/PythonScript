import json
import os
import sys

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================
PROJECT_ROOT = r"C:\Users\13308\nodejsproj\wplacer-lllexxa"
BRAVE_EXECUTABLE_PATH = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
PROXIES_FILE = os.path.join(PROJECT_ROOT, "data", "proxies.txt")
USERS_FILE = os.path.join(PROJECT_ROOT, "data", "users.json")
PROFILE_MAP_FILE = os.path.join(PROJECT_ROOT, "data", "profile_map.json")
OUTPUT_BATCH_FILE = "launch_all_browsers.bat"
TARGET_URL = ""


# ========================================================================
# >> 核心脚本逻辑 (已更新) <<
# ========================================================================

def load_proxies():
    # ... (此函数保持不变)
    try:
        with open(PROXIES_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"  -> Found {len(lines)} proxies in '{PROXIES_FILE}'.")
        return lines
    except FileNotFoundError:
        print(f"Error: Proxy file not found at '{PROXIES_FILE}'", file=sys.stderr);
        return []


def load_json_file(filepath, description):
    # ... (此函数保持不变)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  -> Successfully loaded {description} from '{filepath}'.")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading '{filepath}': {e}", file=sys.stderr);
        return None


def generate_launch_script():
    """主函数，读取所有配置并生成 .bat 脚本。"""
    print("--- Starting Brave Launcher Script Generator ---")

    # 1. 加载所有必要的配置文件
    print("\n[Phase 1] Loading configuration files...")
    proxies = load_proxies()
    users = load_json_file(USERS_FILE, "Users data")
    profile_map = load_json_file(PROFILE_MAP_FILE, "Profile map")

    if not all([proxies, users, profile_map]):
        print("\nAborting due to missing or invalid configuration files.", file=sys.stderr);
        return

    # --- [ 核心修正：基于排序的关联逻辑 ] ---
    print("\n[Phase 2] Sorting and mapping users to profiles...")

    # 2a. 对 users.json 中的用户，按 ID (键) 的数值大小进行升序排序
    try:
        sorted_users = sorted(users.items(), key=lambda item: int(item[0]))
    except ValueError:
        print("Error: User IDs in users.json must be convertible to numbers for sorting.", file=sys.stderr);
        return
    print(f"  -> Sorted {len(sorted_users)} users by their ID.")

    # 2b. 对 profile_map 中的 Profile，按文件夹名称进行自然排序
    # 'Default' 应该排在 'Profile 1' 前面, 'Profile 9' 在 'Profile 10' 前面
    def profile_sort_key(folder_name):
        if folder_name == 'Default':
            return -1  # 确保 Default 永远是第一个
        match = re.match(r'Profile (\d+)', folder_name)
        if match:
            return int(match.group(1))
        return float('inf')  # 将任何不匹配的格式排到最后

    import re
    sorted_profile_folders = sorted(profile_map.keys(), key=profile_sort_key)
    print(f"  -> Sorted {len(sorted_profile_folders)} profile folders naturally.")

    # 安全检查
    if len(sorted_users) != len(sorted_profile_folders):
        print(
            f"\nWarning: Mismatch in counts! Found {len(sorted_users)} users but {len(sorted_profile_folders)} profiles.",
            file=sys.stderr)
        print("Will only generate commands for the smaller of the two.", file=sys.stderr)

    num_commands = min(len(sorted_users), len(sorted_profile_folders))
    # --- [ 修正结束 ] ---

    commands = ["@echo off", "chcp 65001 > nul"]

    print(f"\n[Phase 3] Generating {num_commands} launch commands...")
    # 3. 遍历排序后的列表，一一对应
    for i in range(num_commands):
        user_id, user_info = sorted_users[i]
        profile_folder = sorted_profile_folders[i]
        user_name = user_info.get("name", f"User_{user_id}")

        # b. 为用户分配代理 (逻辑不变)
        proxy_url = None
        proxy_count = len(proxies)

        if "proxyIndex" in user_info and 1 <= user_info["proxyIndex"] <= proxy_count:
            proxy_url = proxies[user_info["proxyIndex"] - 1]
            print(
                f"  - User '{user_name}' ({user_id}) => Profile '{profile_folder}' => Manually assigned to proxy #{user_info['proxyIndex']}.")
        else:
            user_id_num = int("".join(filter(str.isdigit, user_id))[-8:] or "0")
            assigned_index = user_id_num % proxy_count
            proxy_url = proxies[assigned_index]
            print(
                f"  - User '{user_name}' ({user_id}) => Profile '{profile_folder}' => Automatically assigned to proxy #{assigned_index + 1}.")

        # c. 组装最终的命令行
        command = (
            f'echo Starting browser for user: {user_name} (Profile: {profile_folder})\n'
            f'start "{user_name}" "{BRAVE_EXECUTABLE_PATH}" '
            f'--profile-directory="{profile_folder}" '
            f'--proxy-server="{proxy_url}" '
            f'--disable-features=DnsOverHttps '
            f'"{TARGET_URL}"'
        )
        commands.append(command)

    # 4. 将所有命令写入 .bat 文件
    try:
        with open(OUTPUT_BATCH_FILE, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(commands))
        print(f"\n[Phase 4] Success! ---")
        print(f"Generated {num_commands} launch commands.")
        print(f"Your script has been saved to '{OUTPUT_BATCH_FILE}'.")
    except IOError as e:
        print(f"\nError: Could not write to output file. Details: {e}", file=sys.stderr)

if __name__ == "__main__":
    if not os.path.exists(BRAVE_EXECUTABLE_PATH):
        print(f"Error: Brave executable not found at '{BRAVE_EXECUTABLE_PATH}'", file=sys.stderr)
        sys.exit(1)
    generate_launch_script()