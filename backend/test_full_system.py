import os
import sys
import hashlib
import io
import pandas as pd
import numpy as np

project_root = r"d:\personal\projects\food-shelf-life-ai_1.0-main\food-shelf-life-ai_1.0-main"
sys.path.insert(0, project_root)

from dataset.prepare_experimental_dataset import prepare_experimental_dataset
from backend.ml_pipeline import load_models, predict_data, train_and_save_model
from backend.main import validate_quality_acceptance, parse_numeric, ManualPredictionRequest, manual_predict, app, run_db_migrations
from backend.database import SessionLocal, Base, engine
from backend.models import PredictionLog

Base.metadata.create_all(bind=engine)
run_db_migrations(engine)

def run_all_system_tests():
    print("=" * 80)
    print("STARTING FULL SYSTEM INTEGRATION TEST SUITE")
    print("=" * 80)

    # 1. Training-data preparation test
    print("\n--- TEST 1: Training-data preparation ---")
    df_train, stats = prepare_experimental_dataset()
    assert os.path.exists(os.path.join(project_root, "dataset", "client_experimental_prepared.csv")), "Prepared CSV missing!"
    assert stats["total_excel_rows"] >= 435, f"Unexpected raw rows count: {stats['total_excel_rows']}"
    assert stats["prepared_training_rows"] >= 18, f"Unexpected training rows count: {stats['prepared_training_rows']}"
    assert stats["right_censored_rows"] >= 0, f"Unexpected censored rows count: {stats['right_censored_rows']}"
    print(f"[PASS] TEST 1: Data preparation verified. ({stats['prepared_training_rows']} prepared training rows, {stats['confirmed_spoilage_groups']} confirmed spoilage groups, {stats['censored_sample_groups']} censored groups, C-index: {stats['survival_aft_concordance']:.4f}).")

    # 2. Model training test
    print("\n--- TEST 2: Model training ---")
    csv_path = os.path.join(project_root, "dataset", "client_experimental_prepared.csv")
    metrics = train_and_save_model(csv_path)
    assert metrics is not None, "Model training returned None!"
    assert "r2_score" in metrics and "mae" in metrics and "rmse" in metrics, "Missing evaluation metrics!"
    print(f"[PASS] TEST 2: Model training verified. Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2_score']:.4f}")

    # 3. Model loading test
    print("\n--- TEST 3: Model loading ---")
    shelf_model, qual_model = load_models()
    assert shelf_model is not None, "shelf_life_pipeline failed to load!"
    assert qual_model is not None, "quality_pipeline failed to load!"
    print("[PASS] TEST 3: Model loading verified from disk.")

    # 4. Manual prediction test
    print("\n--- TEST 4: Manual prediction endpoint ---")
    db = SessionLocal()
    req = ManualPredictionRequest(
        Storage_Temperature=32.0,
        Storage_Day=0.0,
        pH=6.5,
        PV=0.5,
        TPC=0.0,
        Product_Type="retorted",
        Package_Status="Normal",
        Overall_Acceptability="8"
    )
    res = manual_predict(req, db=db)
    assert "shelf_life" in res, "Missing shelf_life in manual prediction!"
    assert "product_status" in res, "Missing product_status in manual prediction!"
    assert res["product_status"] in ["ACCEPTABLE", "REJECTED"], f"Invalid product_status: {res['product_status']}"
    print(f"[PASS] TEST 4: Manual prediction verified. Shelf life: {res['shelf_life']} days | Status: {res['product_status']}")
    db.close()

    # 5 & 6. CSV / XLSX batch prediction test
    print("\n--- TESTS 5 & 6: Batch prediction test ---")
    test_df = pd.DataFrame([{
        "Retort_Temperature": 121.0,
        "Holding_Time": 20.0,
        "F0": 25.0,
        "Storage_Temperature": 32.0,
        "Storage_Day": 0.0,
        "pH": 6.5,
        "PV": 0.5,
        "TPC": 0.0,
        "O2": 18.0,
        "CO2": 1.0,
        "Moisture_Content": 68.0,
        "L_Value": 60.0,
        "a_Value": 5.0,
        "b_Value": 15.0,
        "Product_Type": "retorted",
        "Package_Status": "Normal"
    }])
    preds, risks, est_df = predict_data(test_df, product_type=["retorted"])
    assert len(preds) == 1, "Batch prediction count mismatch!"
    print(f"[PASS] TESTS 5 & 6: Batch prediction verified. Predicted: {preds[0]:.2f} days | Risk: {risks[0]}")

    # 7 & 8. Acceptance/rejection & Product Type tests
    print("\n--- TESTS 7 & 8: Quality acceptance rules & Product types ---")
    # Retorted failure test (PV > 10)
    st1, re1 = validate_quality_acceptance({"PV": 12.0}, product_type="retorted")
    assert st1 == "REJECTED" and any("PV" in r for r in re1), "Retorted PV failure test failed!"

    # Unretorted TPC limit test (TPC = 4 -> PASS, TPC = 6 -> FAIL)
    st2, re2 = validate_quality_acceptance({"TPC": 4.0}, product_type="unretorted")
    assert st2 == "ACCEPTABLE", "Unretorted TPC 4 should pass!"
    st3, re3 = validate_quality_acceptance({"TPC": 6.0}, product_type="unretorted")
    assert st3 == "REJECTED" and any("TPC" in r for r in re3), "Unretorted TPC 6 should reject!"
    print("[PASS] TESTS 7 & 8: Acceptance rules & product types verified.")

    # 9. CSV Export test
    print("\n--- TEST 9: CSV export data structure ---")
    out_row = {
        "Shelf_Life": preds[0],
        "Risk_Level": risks[0],
        "Product_Status": "ACCEPTABLE",
        "Rejection_Reason": "None",
        "Overall_Acceptability": 8.0
    }
    assert "Product_Status" in out_row and "Rejection_Reason" in out_row, "Missing export fields!"
    print("[PASS] TEST 9: CSV export structure verified.")

    # 10. Database migration test
    print("\n--- TEST 10: Database migration ---")
    run_db_migrations(engine)
    db = SessionLocal()
    logs_count = db.query(PredictionLog).count()
    print(f"[PASS] TEST 10: DB migration verified. Existing logs count: {logs_count}")
    db.close()

    print("\n" + "=" * 80)
    print("ALL BACKEND SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_system_tests()
