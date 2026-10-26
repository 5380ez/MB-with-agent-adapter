import requests
import json

# 测试用：搜索佳木斯市中心 POI
search_items = [
    {"keyword": "佳木斯市中心", "city": "哈尔滨市"},
]

for item in search_items:
    try:
        resp = requests.get(
            "http://mapapi.qcloud.com/v1/pois/?keyword=" + item["keyword"] + "&city=" + item["city"] + "&page=1&limit=10",
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get('data', [])
            if items:
                result = items[0]
                print(f"查询关键词：{item['keyword']}")
                print(f"坐标：{result.get('pos_x', '')}°E,{result.get('pos_y', '')}°N")
                print(f"地址：{result.get('address', '')}")
            print(f"结果数量：{len(items)}")
            print("---")
        else:
            print(f"搜索失败关键词：{item['keyword']}, 状态码：{resp.status_code}")
            print(resp.text[:200])
    except Exception as e:
        print(f"搜索异常：{item['keyword']} - {e}")

print("\n测试用例执行完成。")
