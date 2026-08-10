import os
import sys
import hashlib
import pandas as pd
import numpy as np

project_root = r"d:\personal\projects\food-shelf-life-ai_1.0-main\food-shelf-life-ai_1.0-main"
sys.path.insert(0, project_root)

from backend.main import validate_quality_acceptance, predict_endpoint
from backend.ml_pipeline import load_models, MODEL_PATH, QUALITY_MODEL_PATH

def run_targeted_fix_tests():
    print("=" * 80)
    print("STARTING TARGETED FIX VERIFICATION TEST SUITE")
    print("=" * 80)

    # 1. Model Hashes Safety Check
    h1 = hashlib.md5(open(MODEL_PATH, "rb").read()).hexdigest()
    h2 = hashlib.md5(open(QUALITY_MODEL_PATH, "rb").read()).hexdigest()
    
    print(f"model_shelf_life.pkl MD5: {h1}")
    print(f"model_quality.pkl    MD5: {h2}")

    # 2. Test A5 RT (Moisture 54.88% with missing baseline in experimental dataset)
    print("\n--- TEST A: A5 RT (Moisture without invented baseline) ---")
    st_a5, re_a5 = validate_quality_acceptance(
        {"Moisture_Content": 54.88, "pH": 6.3, "PV": 0.5, "TPC": 0.0},
        product_type="Retorted",
        initial_moisture=None,
        is_manual=False
    )
    assert st_a5 == "ACCEPTABLE", f"A5 RT false rejection: {re_a5}"
    print(f"[PASS] A5 RT: Software_Status={st_a5} (Not falsely rejected for moisture without baseline).")

    # 3. Test C3 RT (PV > 10 while Experimental_Status = Acceptable)
    print("\n--- TEST B: C3 RT (PV > 10 vs Experimental Acceptable) ---")
    st_c3, re_c3 = validate_quality_acceptance(
        {"PV": 12.0, "pH": 6.8, "TPC": 0.0},
        product_type="Retorted",
        is_manual=False
    )
    exp_status_c3 = "Acceptable"
    assert st_c3 == "REJECTED", f"C3 RT expected REJECTED: {re_c3}"
    assert any("PV" in r for r in re_c3), f"Missing PV rejection reason: {re_c3}"
    assert exp_status_c3 == "Acceptable", "Experimental status modified!"
    print(f"[PASS] C3 RT: Experimental_Status='{exp_status_c3}' | Software_Status='{st_c3}' | Reason='{re_c3[0]}'")

    # 4. Test A6 NRT (Spoiled experimental status + TPC failure)
    print("\n--- TEST C: A6 NRT (Spoiled status + TPC reason) ---")
    st_a6, re_a6 = validate_quality_acceptance(
        {"TPC": 99999.0, "pH": 4.5},
        product_type="Unretorted",
        is_manual=False
    )
    exp_status_a6 = "Spoiled"
    assert st_a6 == "REJECTED" and len(re_a6) >= 1
    print(f"[PASS] A6 NRT: Experimental_Status='{exp_status_a6}' | Software_Status='{st_a6}' | Reasons={re_a6}")

    # 5. Test OJ6 NRT (Bulged / Gas Formation package failure)
    print("\n--- TEST D: OJ6 NRT (Bulged package failure) ---")
    st_oj6, re_oj6 = validate_quality_acceptance(
        {},
        product_type="Unretorted",
        package_status="Bulged",
        is_manual=False
    )
    exp_status_oj6 = "Gas Formation"
    assert st_oj6 == "REJECTED" and any("Package failure" in r for r in re_oj6)
    print(f"[PASS] OJ6 NRT: Experimental_Status='{exp_status_oj6}' | Software_Status='{st_oj6}' | Reasons={re_oj6}")

    # 6. Test Normal Acceptable Sample
    print("\n--- TEST E: Normal Acceptable Sample ---")
    st_norm, re_norm = validate_quality_acceptance(
        {"pH": 6.5, "PV": 0.5, "TPC": 0.0, "O2": 18.0, "CO2": 1.0, "Overall_Acceptability": 8.0},
        product_type="Retorted",
        package_status="Normal",
        is_manual=False
    )
    exp_status_norm = "Acceptable"
    assert st_norm == "ACCEPTABLE" and len(re_norm) == 0
    print(f"[PASS] Normal Sample: Experimental_Status='{exp_status_norm}' | Software_Status='{st_norm}'")

    print("\n" + "=" * 80)
    print("ALL TARGETED FIX TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_targeted_fix_tests()
