import pandas as pd
from backend.ml_pipeline import predict_data

def run_test():
    print("Running verification test for 4 Model Calibration cases...\n")
    all_success = True

    # ==========================================
    # Case 1:
    # pH=6.5, PV=0.5, TPC=1, Moisture=68
    # Expected: 85–90 Days, Fresh
    # ==========================================
    print("--- CASE 1 ---")
    case1_input = {
        "pH": 6.5,
        "PV": 0.5,
        "TPC": 1.0,
        "Moisture_Content": 68.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0
    }
    df1 = pd.DataFrame([case1_input])
    preds1, risks1, _ = predict_data(df1)
    sl1, r1 = preds1[0], risks1[0]
    print(f"Case 1 Prediction: {sl1:.2f} Days, Risk Level: {r1}")
    if (85.0 <= sl1 <= 90.0) and r1 == "Fresh":
        print("PASS: Case 1 is in range 85-90 Days and Fresh")
    else:
        print("FAIL: Case 1 did not meet expectations (Expected: 85-90 Days, Fresh)")
        all_success = False
    print()

    # ==========================================
    # Case 2:
    # pH=5, Moisture=40
    # Expected: 65–80 Days, Monitor
    # ==========================================
    print("--- CASE 2 ---")
    case2_input = {
        "pH": 5.0,
        "Moisture_Content": 40.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0
    }
    df2 = pd.DataFrame([case2_input])
    preds2, risks2, _ = predict_data(df2)
    sl2, r2 = preds2[0], risks2[0]
    print(f"Case 2 Prediction: {sl2:.2f} Days, Risk Level: {r2}")
    if (65.0 <= sl2 <= 80.0) and r2 == "Monitor":
        print("PASS: Case 2 is in range 65-80 Days and Monitor")
    else:
        print("FAIL: Case 2 did not meet expectations (Expected: 65-80 Days, Monitor)")
        all_success = False
    print()

    # ==========================================
    # Case 3:
    # PV=5, TPC=8, O2=2, CO2=15
    # Expected: 10–30 Days, Spoiled
    # ==========================================
    print("--- CASE 3 ---")
    case3_input = {
        "PV": 5.0,
        "TPC": 8.0,
        "O2": 2.0,
        "CO2": 15.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0
    }
    df3 = pd.DataFrame([case3_input])
    preds3, risks3, _ = predict_data(df3)
    sl3, r3 = preds3[0], risks3[0]
    print(f"Case 3 Prediction: {sl3:.2f} Days, Risk Level: {r3}")
    if (10.0 <= sl3 <= 30.0) and r3 == "Spoiled":
        print("PASS: Case 3 is in range 10-30 Days and Spoiled")
    else:
        print("FAIL: Case 3 did not meet expectations (Expected: 10-30 Days, Spoiled)")
        all_success = False
    print()

    # ==========================================
    # Case 4 (IMPORTANT):
    # Only change: L (65 -> 40), a (5 -> 10), b (15 -> 20)
    # Expected: Shelf-life changes <= 3 days (compared to default healthy colors)
    # ==========================================
    print("--- CASE 4 ---")
    # Base case with default colors (healthy values)
    base_healthy = {
        "pH": 6.5,
        "PV": 0.5,
        "TPC": 1.0,
        "Moisture_Content": 68.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0,
        "L_Value": 65.0,
        "a_Value": 5.0,
        "b_Value": 15.0
    }
    df4_base = pd.DataFrame([base_healthy])
    preds4_base, _, _ = predict_data(df4_base)
    sl4_base = preds4_base[0]
    
    # Degraded color values
    degraded_color = base_healthy.copy()
    degraded_color["L_Value"] = 40.0
    degraded_color["a_Value"] = 10.0
    degraded_color["b_Value"] = 20.0
    df4_degraded = pd.DataFrame([degraded_color])
    preds4_degraded, _, _ = predict_data(df4_degraded)
    sl4_degraded = preds4_degraded[0]
    
    diff = abs(sl4_base - sl4_degraded)
    print(f"Healthy color shelf life: {sl4_base:.2f} Days")
    print(f"Degraded color shelf life: {sl4_degraded:.2f} Days")
    print(f"Difference: {diff:.2f} Days (Expected: <= 3 Days)")
    if diff <= 3.0:
        print("PASS: Color change has secondary effect (<= 3 Days difference)")
    else:
        print("FAIL: Color change has too large of an effect (> 3 Days difference)")
        all_success = False
    print()

    if all_success:
        print("ALL VERIFICATION CASES PASSED SUCCESSFULLY!")
    else:
        print("SOME VERIFICATION CASES FAILED.")

if __name__ == "__main__":
    run_test()
