import requests
import sys

from wplace_helper.translate_outbounds_to_config import START_PORT, NUM_ACCOUNTS_TO_BIND

# ========================================================================
# >> SETTINGS: 在这里修改你要测试的本地端口 <<
# ========================================================================

# 将你在 v2ray-core config.json 中设置的、用于绑定的本地端口都列在这里
PROXIES_TO_TEST = list(range(START_PORT, START_PORT + NUM_ACCOUNTS_TO_BIND))

# 我们将使用这个服务来查询我们的公网IP信息
IP_INFO_URL = "https://ipinfo.io/json"

# 用于在终端中显示彩色输出 (可选)
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"


# ========================================================================
# >> 验证脚本核心逻辑 <<
# ========================================================================

def verify_proxy(port):
    """
    通过指定的本地SOCKS5端口，查询公网IP并打印结果。
    """
    print(f"[*] Testing local port {COLOR_YELLOW}{port}{COLOR_RESET}...")

    # 构造 requests 库需要的代理字典
    # 注意：v2ray-core 的 socks 监听器同时支持 socks5 和 socks4
    proxy_url = f"socks5://127.0.0.1:{port}"
    proxies_dict = {
        "http": proxy_url,
        "https": proxy_url
    }

    try:
        # 发起网络请求，设置超时时间为10秒
        response = requests.get(IP_INFO_URL, proxies=proxies_dict, timeout=10)

        # 检查请求是否成功
        response.raise_for_status()

        # 解析返回的JSON数据
        data = response.json()
        ip = data.get("ip", "N/A")
        country = data.get("country", "N/A")
        city = data.get("city", "N/A")

        print(f"  {COLOR_GREEN}SUCCESS!{COLOR_RESET} The public IP address is:")
        print(f"    - IP Address: {ip}")
        print(f"    - Location:   {city}, {country}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  {COLOR_RED}FAILURE!{COLOR_RESET} Could not connect through port {port}.")
        print(f"    - Error: {e}")
        return False


def main():
    """
    主函数，遍历所有待测试的代理。
    """
    if not PROXIES_TO_TEST:
        print("Error: The 'PROXIES_TO_TEST' list is empty. Please add the local ports you want to test.",
              file=sys.stderr)
        return

    print("--- Starting V2Ray Routing Verification ---")

    success_count = 0
    failure_count = 0

    for port in PROXIES_TO_TEST:
        if verify_proxy(port):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 40)

    print("\n--- Verification Complete ---")
    print(
        f"Summary: {COLOR_GREEN}{success_count} successful{COLOR_RESET}, {COLOR_RED}{failure_count} failed{COLOR_RESET}.")


if __name__ == "__main__":
    main()