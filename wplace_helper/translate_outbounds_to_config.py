import json
import os

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================

# 设置您希望创建的“账号-代理”绑定数量
# 例如，如果您有10个账号想绑定，就设置为10
# 注意：这个数字不应超过下面 outbounds_data 中的代理数量
INPUT_NEW_OUTBOUNDS_FILE = r"C:\PortableSoftwares\v2rayN-windows-64\guiConfigs\configMultipleLoad.json"

EXISTING_CONFIG_FILE = r"C:\PortableSoftwares\v2ray-windows-64\config-test.json"

# [新增] 输出文件：最终生成的完整 config.json 的保存路径
OUTPUT_CONFIG_FILE = r"C:\PortableSoftwares\v2ray-windows-64\config-test-appended.json"

# [回落设置] 只有在 EXISTING_CONFIG_FILE 不存在时，这个起始端口才会被使用
DEFAULT_START_PORT = 23456

# ========================================================================
# >> 自动生成脚本的核心逻辑 <<
# ========================================================================


def generate_incremental_config():
    """
    根据现有配置，增量地添加新的 inbounds, outbounds, 和 routing rules。
    使用服务器详细信息进行去重。
    """
    print("--- Starting V2Ray Incremental Config Generation ---", file=sys.stderr)

    # --- 1. 加载现有 config.json ---
    # ... (这部分代码保持不变) ...
    existing_config = {}
    start_port = DEFAULT_START_PORT
    if os.path.exists(EXISTING_CONFIG_FILE):
        print(f"Loading existing config from '{EXISTING_CONFIG_FILE}'...", file=sys.stderr)
        try:
            with open(EXISTING_CONFIG_FILE, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            if existing_config.get("inbounds"):
                max_port = max([inbound.get("port", 0) for inbound in existing_config["inbounds"]], default=0)
                if max_port > 0: start_port = max_port + 1
                print(f"  -> Detected next available port: {start_port}", file=sys.stderr)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Could not parse existing config file. Details: {e}", file=sys.stderr);
            return
    else:
        print(f"No existing config file found. Will create a new one using default start port {start_port}.",
              file=sys.stderr)

    # --- 2. 加载新的 outbounds 数据 ---
    # ... (这部分代码保持不变) ...
    try:
        with open(INPUT_NEW_OUTBOUNDS_FILE, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            if isinstance(input_data, dict) and "outbounds" in input_data:
                new_outbounds = input_data["outbounds"]
            elif isinstance(input_data, list):
                new_outbounds = input_data
            else:
                raise ValueError("Input JSON must be an array or an object with an 'outbounds' key.")
    except Exception as e:
        print(f"Error parsing new outbounds data: {e}", file=sys.stderr);
        return

    # --- 3. [核心修正] 合并和去重 outbounds ---
    print("Merging and deduplicating outbounds based on server details...", file=sys.stderr)

    def get_server_identity(outbound):
        """辅助函数，用于为 outbound 生成一个唯一的标识符字符串。"""
        try:
            # 适用于 shadowsocks, vmess, vless 等常见协议
            server = outbound["settings"]["servers"][0]
            # 我们将协议、地址、端口和密码（或ID）组合起来
            protocol = outbound.get("protocol", "")
            address = server.get("address", "")
            port = server.get("port", "")
            method = server.get("method", "")
            # 不同协议的'密码'字段不同
            password = server.get("password", server.get("id", server.get("users", [{}])[0].get("id", "")))
            s = f"{protocol}|{address}|{method}|{port}|{password}"
            # print(s)
            return s
        except (KeyError, IndexError, TypeError):
            # 如果结构不符合预期，返回一个基于tag的备用标识符或None
            print("ERROR!")
            return outbound.get("tag")

    # 使用我们生成的唯一标识符作为字典的键
    existing_outbounds_map = {
        get_server_identity(o): o
        for o in existing_config.get("outbounds", [])
        if o.get("tag") not in ['direct', 'block'] and get_server_identity(o)
    }
    print(len(existing_outbounds_map), file=sys.stderr)

    num_before = len(existing_outbounds_map)
    newly_added_outbounds = []

    for new_outbound in new_outbounds:
        if new_outbound.get("tag") in ['direct', 'block']: continue

        identity = get_server_identity(new_outbound)
        if identity and identity not in existing_outbounds_map:
            # 为了防止tag冲突，如果新节点的tag已存在，我们为其生成一个唯一的tag
            original_tag = new_outbound.get("tag", "new-proxy")
            new_tag = original_tag
            tag_counter = 1
            # 检查所有现有的tag，确保新tag不重复
            all_tags = {o["tag"] for o in existing_outbounds_map.values() if "tag" in o}
            while new_tag in all_tags:
                new_tag = f"{original_tag}_{tag_counter}"
                tag_counter += 1

            if new_tag != original_tag:
                print(f"  - Renaming duplicate tag '{original_tag}' to '{new_tag}'", file=sys.stderr)
                new_outbound["tag"] = new_tag

            existing_outbounds_map[identity] = new_outbound
            newly_added_outbounds.append(new_outbound)

    num_after = len(existing_outbounds_map)
    print(f"  -> {num_after - num_before} new unique outbounds added.", file=sys.stderr)

    # 重新组合 outbounds 列表，保留 direct 和 block
    final_outbounds = list(existing_outbounds_map.values())
    for o in existing_config.get("outbounds", []):
        if o.get("tag") in ['direct', 'block'] and not any(f["tag"] == o["tag"] for f in final_outbounds):
            final_outbounds.append(o)

    # --- 4. 生成新的 inbounds 和 routing rules ---
    # ... (这部分代码保持不变) ...
    num_to_bind = len(newly_added_outbounds)
    if num_to_bind == 0:
        print("No new unique outbound proxies to bind.", file=sys.stderr)
    else:
        print(f"Generating {num_to_bind} new inbounds and routing rules...", file=sys.stderr)

    new_inbounds = []
    new_rules = []
    for i in range(num_to_bind):
        port = start_port + i;
        inbound_tag = f"in-account-{port}"
        new_inbounds.append(
            {"port": port, "listen": "127.0.0.1", "protocol": "socks", "settings": {"auth": "noauth", "udp": True},
             "tag": inbound_tag})
        outbound_tag = newly_added_outbounds[i]["tag"]
        new_rules.append({"type": "field", "inboundTag": [inbound_tag], "outboundTag": outbound_tag})
        print(f"  - Rule created: {inbound_tag} (Port {port}) -> {outbound_tag}", file=sys.stderr)

    # --- 5. 组装最终的完整配置 ---
    # ... (这部分代码保持不变) ...
    final_config = {
        "log": existing_config.get("log", {"loglevel": "warning"}),
        "inbounds": existing_config.get("inbounds", []) + new_inbounds,
        "outbounds": final_outbounds,
        "routing": {
            "domainStrategy": existing_config.get("routing", {}).get("domainStrategy", "AsIs"),
            "rules": existing_config.get("routing", {}).get("rules", []) + new_rules
        }
    }

    # --- 6. 写入到输出文件 ---
    try:
        with open(OUTPUT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2)
        print("\n--- Generation Complete! ---", file=sys.stderr)
        print(f"Successfully saved the updated configuration to '{OUTPUT_CONFIG_FILE}'", file=sys.stderr)
    except IOError as e:
        print(f"\nError: Could not write to output file. Details: {e}", file=sys.stderr)
        return

    print("\n--- Next Steps for Wplacer ---", file=sys.stderr)
    if new_inbounds:
        print("Add the following new lines to your 'data/proxies.txt':", file=sys.stderr)
        for inbound in new_inbounds:
            print(f"   socks5://127.0.0.1:{inbound['port']}")
        print("\nThen, update your 'data/account_proxies.json' to map user IDs to the new proxy line numbers.",
              file=sys.stderr)


if __name__ == "__main__":
    import sys

    generate_incremental_config()