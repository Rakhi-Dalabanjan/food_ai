import os
import sys
import hashlib
import pandas as pd
import numpy as np

project_root = r"d:\personal\projects\food-shelf-life-ai_1.0-main\food-shelf-life-ai_1.0-main"
sys.path.insert(0, project_root)

from backend.ml_pipeline import load_models, predict_data
from backend.main import validate_quality_acceptance, parse_numeric, ManualPredictionRequest, manual_predict, app
from backend.database import SessionLocal, Base, engine
from backend.models import PredictionLog

Base.metadata.create_all(bind=engine)

def run_all_tests():
    print("=" * 80)
    print("STARTING ACCEPTANCE LAYER & REGRESSION TEST SUITE")
    print("=" * 80)

    # 1. Check Model File Hashes (Must remain 100% unchanged)
    p1 = os.path.join(project_root, "models", "model_shelf_life.pkl")
    p2 = os.path.join(project_root, "models", "model_quality.pkl")

    h1 = hashlib.md5(open(p1, "rb").read()).hexdigest()
    h2 = hashlib.md5(open(p2, "rb").read()).hexdigest()

    expected_h1 = "0e6b79f42338c6c4af93bcba8ec161e6"
    expected_h2 = "358706a1a3b37d9aa2bc1483bebe50a2"

    assert h1 == expected_h1, f"CRITICAL: model_shelf_life.pkl changed! Hash: {h1}"
    assert h2 == expected_h2, f"CRITICAL: model_quality.pkl changed! Hash: {h2}"
    print("[PASS] TEST 27 & 28: Model file hashes are 100% UNCHANGED.")

    # 2. Check ML Prediction Baseline (Must remain 100% unchanged)
    sample_df = pd.DataFrame([{
        "Retort_Temperature": 121.0,
        "Holding_Time": 20.0,
        "F0": 25.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0,
        "pH": 6.5,
        "PV": 0.5,
        "TPC": 0.2,
        "O2": 18.0,
        "CO2": 1.0,
        "Moisture_Content": 68.0,
        "L_Value": 65.0,
        "a_Value": 4.0,
        "b_Value": 12.0
    }])
    preds, risks, _ = predict_data(sample_df, product_type=["retorted"])
    pred_shelf_life = preds[0]
    expected_shelf_life = 89.0593
    assert abs(pred_shelf_life - expected_shelf_life) < 1e-3, f"ML prediction changed! Got {pred_shelf_life}, expected {expected_shelf_life}"
    print(f"[PASS] TEST 26: ML Prediction remains 100% UNCHANGED ({pred_shelf_life:.4f} days).")

    # 3. RETORTED Acceptance Tests (Tests 1–14)
    # 1. PV = 10 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"PV": 10}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 1: {re}"

    # 2. PV = 10.01 -> REJECTED
    st, re = validate_quality_acceptance({"PV": 10.01}, "Retorted")
    assert st == "REJECTED" and any("PV" in r for r in re), f"Failed Test 2: {re}"

    # 3. TPC = 0 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"TPC": 0}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 3: {re}"

    # 4. TPC > 0 -> REJECTED
    st, re = validate_quality_acceptance({"TPC": 0.1}, "Retorted")
    assert st == "REJECTED" and any("TPC" in r for r in re), f"Failed Test 4: {re}"

    # 5. pH = 4.9 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"pH": 4.9}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 5: {re}"

    # 6. pH = 4.89 -> REJECTED
    st, re = validate_quality_acceptance({"pH": 4.89}, "Retorted")
    assert st == "REJECTED" and any("pH" in r for r in re), f"Failed Test 6: {re}"

    # 7. O2 = 25 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"O2": 25}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 7: {re}"

    # 8. O2 > 25 -> REJECTED
    st, re = validate_quality_acceptance({"O2": 25.1}, "Retorted")
    assert st == "REJECTED" and any("O2" in r for r in re), f"Failed Test 8: {re}"

    # 9. CO2 = 25 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"CO2": 25}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 9: {re}"

    # 10. CO2 > 25 -> REJECTED
    st, re = validate_quality_acceptance({"CO2": 25.1}, "Retorted")
    assert st == "REJECTED" and any("CO2" in r for r in re), f"Failed Test 10: {re}"

    # 11. Overall Acceptability = 5 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"Overall_Acceptability": 5}, "Retorted")
    assert st == "ACCEPTABLE", f"Failed Test 11: {re}"

    # 12. Overall Acceptability = 4.99 -> REJECTED
    st, re = validate_quality_acceptance({"Overall_Acceptability": 4.99}, "Retorted")
    assert st == "REJECTED" and any("Overall Acceptability" in r for r in re), f"Failed Test 12: {re}"

    # 13. Package Status = Normal -> not rejected by package rule
    st, re = validate_quality_acceptance({}, "Retorted", package_status="Normal")
    assert st == "ACCEPTABLE", f"Failed Test 13: {re}"

    # 14. Package Status = Seal Failure -> REJECTED
    st, re = validate_quality_acceptance({}, "Retorted", package_status="Seal_Failure")
    assert st == "REJECTED" and any("Package failure" in r for r in re), f"Failed Test 14: {re}"

    print("[PASS] TESTS 1-14: All Retorted acceptance rules verified.")

    # 4. UNRETORTED Acceptance Tests (Tests 15–25)
    # 15. PV = 10 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"PV": 10}, "Unretorted")
    assert st == "ACCEPTABLE", f"Failed Test 15: {re}"

    # 16. PV > 10 -> REJECTED
    st, re = validate_quality_acceptance({"PV": 10.01}, "Unretorted")
    assert st == "REJECTED" and any("PV" in r for r in re), f"Failed Test 16: {re}"

    # 17. TPC = 5 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"TPC": 5}, "Unretorted")
    assert st == "ACCEPTABLE", f"Failed Test 17: {re}"

    # 18. TPC > 5 -> REJECTED
    st, re = validate_quality_acceptance({"TPC": 5.1}, "Unretorted")
    assert st == "REJECTED" and any("TPC" in r for r in re), f"Failed Test 18: {re}"

    # 19. pH = 4.9 -> ACCEPTABLE
    st, re = validate_quality_acceptance({"pH": 4.9}, "Unretorted")
    assert st == "ACCEPTABLE", f"Failed Test 19: {re}"

    # 20. pH < 4.9 -> REJECTED
    st, re = validate_quality_acceptance({"pH": 4.89}, "Unretorted")
    assert st == "REJECTED" and any("pH" in r for r in re), f"Failed Test 20: {re}"

    # 21. Moisture abnormal -> must NOT reject solely because there is no Unretorted moisture threshold
    st, re = validate_quality_acceptance({"Moisture_Content": 20.0}, "Unretorted")
    assert st == "ACCEPTABLE", f"Failed Test 21: {re}"

    # 22. O2 > 25 -> REJECTED
    st, re = validate_quality_acceptance({"O2": 25.1}, "Unretorted")
    assert st == "REJECTED" and any("O2" in r for r in re), f"Failed Test 22: {re}"

    # 23. CO2 > 25 -> REJECTED
    st, re = validate_quality_acceptance({"CO2": 25.1}, "Unretorted")
    assert st == "REJECTED" and any("CO2" in r for r in re), f"Failed Test 23: {re}"

    # 24. Overall Acceptability < 5 -> REJECTED
    st, re = validate_quality_acceptance({"Overall_Acceptability": 4.5}, "Unretorted")
    assert st == "REJECTED" and any("Overall Acceptability" in r for r in re), f"Failed Test 24: {re}"

    # 25. Package failure -> REJECTED
    st, re = validate_quality_acceptance({}, "Unretorted", package_status="Bulged")
    assert st == "REJECTED" and any("Package failure" in r for r in re), f"Failed Test 25: {re}"

    print("[PASS] TESTS 15-25: All Unretorted acceptance rules verified.")

    # 5. Test Missing / Blank Values Handling (Section 8)
    st, re = validate_quality_acceptance({"PV": 12, "TPC": None, "pH": None}, "Retorted")
    assert st == "REJECTED"
    assert len(re) == 1 and "PV" in re[0], f"Missing parameters should not trigger rejection: {re}"
    print("[PASS] MISSING VALUES TEST: Blank/missing fields skipped properly, never cause false rejection.")

    # 6. Test Manual Prediction API endpoint
    db = SessionLocal()
    req = ManualPredictionRequest(
        Storage_Temperature=25.0,
        Storage_Day=0.0,
        pH=4.8, # pH < 4.9 -> REJECTED
        PV=12.0, # PV > 10 -> REJECTED
        Product_Type="Retorted",
        Package_Status="Normal"
    )
    res = manual_predict(req, db=db)
    assert res["product_status"] == "REJECTED"
    assert len(res["rejection_reasons"]) == 2
    assert "shelf_life" in res and res["shelf_life"] > 0 # Shelf life remains calculated ML value!
    print(f"[PASS] TEST 30: Manual Prediction API endpoint returns shelf_life={res['shelf_life']} alongside product_status={res['product_status']} and reasons={res['rejection_reasons']}.")
    db.close()

    print("=" * 80)
    print("ALL 31 TEST SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
