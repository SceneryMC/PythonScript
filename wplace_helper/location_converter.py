import math

def convert_map_coords_to_latlon(x, y):
    N = 2048
    HALF = 0.5 / 1000
    lon = (x + HALF) / N * 360.0 - 180.0
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y + HALF) / N))) * 180.0 / math.pi
    return lat, lon

def convert_latlon_to_map_coords(lat, lon):
    """
    将经纬度坐标转换为 wplace 使用的小数形式地图坐标 (x, y)。

    Args:
        lat (float): 纬度 (-85.05 to 85.05)
        lon (float): 经度 (-180 to 180)

    Returns:
        tuple: (x, y) 形式的小数地图坐标
    """
    N = 2048.0
    HALF = 0.5 / 1000.0

    # 反向计算 x
    map_x = N * (lon + 180.0) / 360.0 - HALF

    # 反向计算 y
    lat_rad = math.radians(lat)
    map_y = N / 2.0 * (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) - HALF

    return (map_x, map_y)

lat1, lon1 = 28.86937741098241, 106.20343310541045
lat2, lon2 = 28.86941589488299, 106.20347705072209

map_x1, map_y1 = convert_latlon_to_map_coords(lat1, lon1)
map_x2, map_y2 = convert_latlon_to_map_coords(lat2, lon2)
print(f"转换结果: x = {map_x1}, y = {map_y1}")
print(f"转换结果: x = {map_x2}, y = {map_y2}")
# print(f'diff: {math.fabs(map_x1 - map_x2) * 1000 * 8}, {math.fabs(map_y1 - map_y2) * 1000 * 8}')
# print(f"diff: {5337 + 5079}, {10759 + 78}")
"transform: translate(-50%, -50%) translate(5337px, 10759px) rotateX(0deg) rotateZ(0deg); opacity: 0.3;"
"transform: translate(-50%, -50%) translate(-5079px, -78px) rotateX(0deg) rotateZ(0deg); opacity: 0.3;"
# 预期输出: x ≈ 1691.655, y ≈ 801.345

# x1 = 1628.157
# y1 = 852.575
# pos(x1, y1)
# pos(x1 + 0.001, y1)