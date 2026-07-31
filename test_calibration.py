import requests
import json


def test_prediction(temp, days):
    url = "http://import.meta.env.VITE_API_URL/estimate-quality"
    payload = {
        "Storage_Temperature": temp,
        "Storage_Day": days,
        "Retort_Temperature": 121,
        "Holding_Time": 20,
    }

    try:
        # Step 1: Estimate
        res_est = requests.post(url, json=payload)
        if res_est.status_code != 200:
            print(f"Estimation Error: {res_est.text}")
            return

        full_data = res_est.json()
        print(f"\n--- Testing: {temp}°C + {days} Days ---")
        print(f"Generated pH: {full_data['pH']:.2f}")
        print(f"Generated PV: {full_data['PV']:.2f}")
        print(f"Generated CO2: {full_data['CO2']:.2f}")

        # Step 2: Predict
        url_pred = "http://import.meta.env.VITE_API_URL/manual-predict"
        res_pred = requests.post(url_pred, json=full_data)
        if res_pred.status_code != 200:
            print(f"Prediction Error: {res_pred.text}")
            return

        result = res_pred.json()
        print(f"Shelf Life: {result['shelf_life']} Days")
        print(f"Risk Level: {result['risk_level']}")

    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    test_prediction(25, 5)  # Target: Fresh, 75-90
    test_prediction(35, 30)  # Target: Monitor, 45-65
    test_prediction(45, 90)  # Target: Spoiled, 0-10
