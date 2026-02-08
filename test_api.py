import requests
import json

URL = "http://127.0.0.1:8000/predict"

def test_scenario(label, payload):
    print(f"\n🧪 SCENARIO: {label}")
    try:
        response = requests.post(URL, json=payload)
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    # Test A: The "Impossible" Cold
    test_scenario("High-Cost Common Cold", {
        "diag_code": "460",
        "amt": 45000,
        "stay": 15,
        "age": 30
    })

    # Test B: Legitimate Heart Procedure
    test_scenario("Standard Heart Attack Treatment", {
        "diag_code": "410",
        "amt": 11000,
        "stay": 4,
        "age": 72
    })