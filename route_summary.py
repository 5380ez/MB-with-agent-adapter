import json

data = {
    "status": 0,
    "data": {
        "status": "1",
        "info": "OK",
        "infocode": "10000",
        "count": "3",
        "route": {
            "origin": "103.2672706246376,25.563041724723607",
            "destination": "102.771667,24.994568",
            "taxi_cost": "256",
            "paths": [
                {
                    "distance": "102860",
                    "restriction": "0",
                    "cost": {
                        "duration": "5302",
                        "tolls": "62",
                        "toll_distance": "90529",
                        "traffic_lights": "11"
                    }
                }
            ]
        }
    }
}

print(json.dumps(data, indent=2, ensure_ascii=False))
