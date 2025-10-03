import json

# ========================================================================
# >> SETTINGS: 在这里修改你的配置 <<
# ========================================================================

# 设置您希望的起始本地SOCKS5端口
START_PORT = 23456

# 设置您希望创建的“账号-代理”绑定数量
# 例如，如果您有10个账号想绑定，就设置为10
# 注意：这个数字不应超过下面 outbounds_data 中的代理数量
NUM_ACCOUNTS_TO_BIND = 20

INPUT_OUTBOUNDS_FILE = r"C:\PortableSoftwares\v2rayN-windows-64\guiConfigs\configMultipleLoad.json"

# [新增] 输出文件：最终生成的完整 config.json 的保存路径
OUTPUT_CONFIG_FILE = r"C:\PortableSoftwares\v2ray-windows-64\config-test.json"

# ========================================================================
# >> 自动生成脚本的核心逻辑 <<
# ========================================================================

def generate_config():
    """
    根据顶部的配置和数据，生成完整的 v2ray-core config.json。
    """
    print("--- Starting V2Ray config generation ---", file=sys.stderr)

    # 1. [修改] 从指定的JSON文件中读取 outbounds 数据
    try:
        with open(INPUT_OUTBOUNDS_FILE, 'r', encoding='utf-8') as f:
            # 假设输入文件就是 outbounds 数组本身，或者是一个包含 "outbounds" 键的对象
            input_data = json.load(f)
            if isinstance(input_data, dict) and "outbounds" in input_data:
                outbounds = input_data["outbounds"]
            elif isinstance(input_data, list):
                outbounds = input_data
            else:
                raise ValueError("Input JSON must be an array of outbounds, or an object with an 'outbounds' key.")

        proxy_outbounds = [o for o in outbounds if o.get('tag') not in ['direct', 'block']]
        print(f"Successfully parsed {len(proxy_outbounds)} proxy outbounds from '{INPUT_OUTBOUNDS_FILE}'.",
              file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_OUTBOUNDS_FILE}'", file=sys.stderr)
        return
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Could not parse outbounds data from '{INPUT_OUTBOUNDS_FILE}'. Details: {e}", file=sys.stderr)
        return

    # 2. 安全检查
    num_to_bind = NUM_ACCOUNTS_TO_BIND
    if num_to_bind > len(proxy_outbounds):
        print(
            f"Warning: You requested {num_to_bind} bindings, but only {len(proxy_outbounds)} proxies are available. Will generate {len(proxy_outbounds)} bindings instead.",
            file=sys.stderr)
        num_to_bind = len(proxy_outbounds)

    # 3. 生成 inbounds
    print(f"Generating {num_to_bind} inbounds starting from port {START_PORT}...", file=sys.stderr)
    inbounds_list = []
    for i in range(num_to_bind):
        port = START_PORT + i
        tag = f"in-account-{i + 1}"
        inbound = {
            "port": port,
            "listen": "127.0.0.1",
            "protocol": "socks",
            "settings": {
                "auth": "noauth",
                "udp": True  # json.dumps 会将其转换为小写的 true
            },
            "tag": tag
        }
        inbounds_list.append(inbound)

    # 4. 生成 routing rules
    print(f"Generating {num_to_bind} routing rules...", file=sys.stderr)
    routing_rules = []
    for i in range(num_to_bind):
        inbound_tag = f"in-account-{i + 1}"
        # 将第 i 个 inbound 绑定到第 i 个 proxy outbound
        outbound_tag = proxy_outbounds[i]["tag"]
        rule = {
            "type": "field",
            "inboundTag": [inbound_tag],
            "outboundTag": outbound_tag
        }
        routing_rules.append(rule)
        print(f"  - Rule created: {inbound_tag} -> {outbound_tag}", file=sys.stderr)

    # 5. 组装最终的完整配置
    final_config = {
        "log": {
            "loglevel": "warning"
        },
        "inbounds": inbounds_list,
        "outbounds": outbounds,  # 使用用户提供的完整 outbounds 列表
        "routing": {
            "domainStrategy": "AsIs",
            "rules": routing_rules
        }
    }

    # 6. 以美化的JSON格式输出到标准输出
    try:
        with open(OUTPUT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2)
        print("\n--- Generation Complete! ---", file=sys.stderr)
        print(f"Successfully saved the complete configuration to '{OUTPUT_CONFIG_FILE}'", file=sys.stderr)
    except IOError as e:
        print(f"\nError: Could not write to output file '{OUTPUT_CONFIG_FILE}'. Details: {e}", file=sys.stderr)
        return

    print("\n--- Generation Complete! ---", file=sys.stderr)
    print("Above is your complete config.json content. You can copy it or redirect the script's output to a file:",
          file=sys.stderr)
    print("Example: python generate_v2ray_config.py > config.json", file=sys.stderr)

    print("\n--- Next Steps for Wplacer ---", file=sys.stderr)
    print("1. Update your 'data/proxies.txt' with the generated ports:", file=sys.stderr)
    for i in range(num_to_bind):
        print(f"   socks5://127.0.0.1:{START_PORT + i}")

    print("\n2. Update your 'data/account_proxies.json' to map user IDs to proxy line numbers (starting from 1).",
          file=sys.stderr)
    print("   Example:", file=sys.stderr)
    print("   {", file=sys.stderr)
    print('     "11223344": 1,', file=sys.stderr)
    print('     "55667788": 2', file=sys.stderr)
    print("   }", file=sys.stderr)


if __name__ == "__main__":
    import sys

    generate_config()