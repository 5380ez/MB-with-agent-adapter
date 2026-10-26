import requests

# 用户位置信息
user_loc = {"lon": "117.804996", "lat": "31.760766"}
city = "合肥市"

# 测试反向地理编码
print("1. 查询用户位置详细地址")
test_params = {
    "longitude": user_loc["lon"],
    "latitude": user_loc["lat"],
    "city": city
}
try:
    r = requests.get("https://qapi.amm.cc/sandbox", params=test_params)
    print(f"反向地理编码结果: {r.text}")
except Exception as e:
    print(f"反向地理编码 API 调用失败: {e}")

# 测试周边 POI
print("\n2. 搜索用户位置周边 POI")
test_params_temp = {
    "location": f"{user_loc['lon']},{user_loc['lat']}",
    "keywords": "后边"
}
try:
    r = requests.get("https://qapi.amm.cc/sandbox", params=test_params_temp)
    print(f"周边 POI：{r.text[:500]}")
except Exception as e:
    print(f"周边 POI 查询失败：{e}")
