import requests

# 搜索崇贤金属港相关 POI
search_results = []
keywords = [
    "崇贤金属港",
    "崇贤港",
    "崇贤港 杭州",
    "临平区 金属港"
]

for keyword in keywords:
    try:
        resp = requests.get("http://mapapi.qcloud.com/v1/pois/?keyword=" + keyword + "&city=3301000000000", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            search_results.extend(data.get('data', []))
    except:
        pass

if search_results:
    print(f"找到 {len(search_results)} 个结果:")
    for item in search_results:
        print(f"名称：{item.get('name', 'N/A')}")
        print(f"坐标：{item.get('pos_x', 'N/A')},{item.get('pos_y', 'N/A')}")
        print(f"地址：{item.get('address', 'N/A')}")
        print("---")
else:
    print("未找到相关 POI 信息")

