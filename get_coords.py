import requests
import json

# 重新查询 POI 获取准确的坐标信息
result = api_call_mcp("mcp__sandbox__query_poi", {"keywords": "宁波火车站在", "city": "宁波"})
print("宁波结果:", json.dumps(result, indent=2, ensure_ascii=False))

# 同样查询佳木斯
result2 = api_call_mcp("mcp__sandbox__query_poi", {"keywords": "佳木斯对吧", "city": "佳木斯"})
print("佳木斯结果:", json.dumps(result2, indent=2, ensure_ascii=False))

