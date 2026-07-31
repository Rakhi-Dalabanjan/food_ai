import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
from .database import SessionLocal
from .models import ModelMetrics

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model_shelf_life.pkl")
QUALITY_MODEL_PATH = os.path.join(MODEL_DIR, "model_quality.pkl")

BASE_FEATURES = [
    "Retort_Temperature",
    "Holding_Time",
    "F0",
    "Storage_Temperature",
    "Storage_Day",
]

QUALITY_FEATURES = [
    "pH",
    "PV",
    "TPC",
    "O2",
    "CO2",
    "Moisture_Content",
    "L_Value",
    "a_Value",
    "b_Value",
]

FEATURES = BASE_FEATURES + QUALITY_FEATURES
TARGET = "Shelf_Life_Remaining"

class MultiOutputQualityModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        import numpy as np
        preds = []
        for feat in QUALITY_FEATURES:
            preds.append(self.models[feat].predict(X))
        return np.column_stack(preds)


def get_risk_level(shelf_life: float) -> str:
    if shelf_life >= 70:
        return "Fresh"
    elif shelf_life >= 40:
        return "Monitor"
    elif shelf_life >= 15:
        return "Near Spoilage"
    else:
        return "Spoiled"


def train_and_save_model(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df = df.dropna()

    print("-" * 50)
    print("TRAINING DIAGNOSTICS")
    print("-" * 50)
    print(f"Dataset shape: {df.shape}")

    # Split: Train (70%), Validation (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # --- MODEL 1: Quality Parameters ---
    X_q_train = X_train[BASE_FEATURES]
    X_q_val = X_val[BASE_FEATURES]

    # Monotonic constraints for Model 1 (inputs: BASE_FEATURES)
    # Storage Temp (idx 3) and Storage Day (idx 4)
    quality_constraints = {
        "pH": [0, 0, 0, -1, -1],
        "PV": [0, 0, 0, 1, 1],
        "TPC": [0, 0, 0, 1, 1],
        "O2": [0, 0, 0, -1, -1],
        "CO2": [0, 0, 0, 1, 1],
        "Moisture_Content": [0, 0, 0, 0, 0],
        "L_Value": [0, 0, 0, 0, 0],
        "a_Value": [0, 0, 0, 0, 0],
        "b_Value": [0, 0, 0, 0, 0],
    }

    quality_models = {}
    q_r2_scores = []

    print("Training Model 1 (Quality Parameters)...")
    for feature in QUALITY_FEATURES:
        y_q_train = df.loc[X_train.index, feature]
        y_q_val = df.loc[X_val.index, feature]
        
        cst = quality_constraints[feature]
        quality_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HistGradientBoostingRegressor(monotonic_cst=cst, random_state=42)),
            ]
        )
        quality_pipe.fit(X_q_train, y_q_train)
        
        val_preds = quality_pipe.predict(X_q_val)
        val_r2 = r2_score(y_q_val, val_preds)
        q_r2_scores.append(val_r2)
        quality_models[feature] = quality_pipe
        print(f"Model 1 ({feature}) - Validation R² Score: {val_r2:.4f}")

    avg_q_r2 = float(np.mean(q_r2_scores))
    print(f"Model 1 Average Validation R² Score: {avg_q_r2:.4f}")

    quality_pipeline = MultiOutputQualityModel(quality_models)

    # --- MODEL 2: Shelf Life ---
    # FEATURES = BASE_FEATURES + QUALITY_FEATURES
    # BASE_FEATURES: Retort(0), Holding(0), F0(0), Temp(-1), Day(-1)
    # QUALITY_FEATURES: pH(1), PV(-1), TPC(-1), O2(1), CO2(-1), Moisture(0), L(0), a(0), b(0)
    monotonic_cst_m2 = [0, 0, 0, -1, -1, 1, -1, -1, 1, -1, 0, 0, 0, 0]

    shelf_life_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(monotonic_cst=monotonic_cst_m2, random_state=42),
            ),
        ]
    )

    print("Training Model 2 (Shelf Life)...")
    shelf_life_pipeline.fit(X_train, y_train)

    # Evaluate Model 2 on validation set
    y_val_pred = shelf_life_pipeline.predict(X_val)
    y_val_pred = np.clip(y_val_pred, 0, 90)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"Model 2 Validation Metrics:")
    print(f"R² Score: {val_r2:.4f}")
    print(f"MAE: {val_mae:.4f}")
    print(f"RMSE: {val_rmse:.4f}")

    # Evaluate Model 2 on test set (final hold-out metrics to log/save)
    y_test_pred = shelf_life_pipeline.predict(X_test)
    y_test_pred = np.clip(y_test_pred, 0, 90)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Model 2 Test Metrics (Hold-out):")
    print(f"R² Score: {test_r2:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")

    if val_r2 < 0.80:
        print("CRITICAL: R² < 0.80. Model deployment aborted.")
        return None

    # Permutation importance for Model 2 using validation set
    result = permutation_importance(shelf_life_pipeline, X_val, y_val, n_repeats=5, random_state=42)
    importances = np.maximum(0, result.importances_mean)
    if importances.sum() > 0:
        importances = importances / importances.sum()
    feature_importance_dict = {
        FEATURES[i]: float(importances[i]) for i in range(len(FEATURES))
    }

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(quality_pipeline, QUALITY_MODEL_PATH)
    joblib.dump(shelf_life_pipeline, MODEL_PATH)

    # Save metrics to DB (using validation set metrics)
    db = SessionLocal()
    metrics = ModelMetrics(
        mae=float(val_mae),
        rmse=float(val_rmse),
        r2_score=float(val_r2),
        feature_importances=json.dumps(feature_importance_dict),
    )
    db.add(metrics)
    db.commit()
    db.close()

    return {
        "mae": val_mae,
        "rmse": val_rmse,
        "r2_score": val_r2,
        "feature_importances": feature_importance_dict,
        "quality_r2": avg_q_r2,
    }


def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(QUALITY_MODEL_PATH):
        return None, None
    return joblib.load(QUALITY_MODEL_PATH), joblib.load(MODEL_PATH)


def get_row_penalty_multiplier(row) -> float:
    multiplier = 1.0

    # pH:
    # 6.2–6.8: No penalty
    # 5.5–6.1: Small reduction
    # 4.8–5.4: Moderate reduction
    # <4.8: Strong reduction
    ph = row.get("pH", 6.5)
    if ph >= 6.2 and ph <= 6.8:
        multiplier *= 1.0
    elif (ph >= 5.5 and ph < 6.2) or (ph > 6.8 and ph <= 7.2):
        multiplier *= 0.96
    elif (ph >= 4.8 and ph < 5.5) or (ph > 7.2 and ph <= 7.8):
        multiplier *= 0.88
    else:  # < 4.8 or > 7.8
        multiplier *= 0.70

    # Moisture (Moisture_Content):
    # 60–70: Normal
    # 45–59: Moderate impact
    # <45: Shelf life reduction
    # Do not classify aggressively.
    mc = row.get("Moisture_Content", 68.0)
    if mc >= 60.0 and mc <= 75.0:
        multiplier *= 1.0
    elif mc >= 45.0 and mc < 60.0:
        multiplier *= 0.96
    else:  # < 45.0 or > 75.0
        multiplier *= 0.90  # Bounded to 0.90 to avoid aggressive classification

    # PV:
    # < 1.2: Normal
    # 1.2–3.0: Small reduction
    # 3.0–5.5: Moderate reduction
    # > 5.5: Strong reduction
    pv = row.get("PV", 0.5)
    if pv <= 1.2:
        multiplier *= 1.0
    elif pv <= 3.0:
        multiplier *= 0.96
    elif pv <= 5.5:
        multiplier *= 0.88
    else:
        multiplier *= 0.70

    # TPC:
    # < 1.8: Normal
    # 1.8–3.5: Small reduction
    # 3.5–6.5: Moderate reduction
    # > 6.5: Strong reduction
    tpc = row.get("TPC", 0.2)
    if tpc <= 1.8:
        multiplier *= 1.0
    elif tpc <= 3.5:
        multiplier *= 0.96
    elif tpc <= 6.5:
        multiplier *= 0.88
    else:
        multiplier *= 0.70

    # O2:
    # >= 15.0: Normal
    # 10.0–15.0: Small reduction
    # 4.0–10.0: Moderate reduction
    # < 4.0: Strong reduction
    o2 = row.get("O2", 18.0)
    if o2 >= 15.0:
        multiplier *= 1.0
    elif o2 >= 10.0:
        multiplier *= 0.96
    elif o2 >= 4.0:
        multiplier *= 0.88
    else:
        multiplier *= 0.70

    # CO2:
    # <= 8.0: Normal
    # 8.0–12.0: Small reduction
    # 12.0–16.0: Moderate reduction
    # > 16.0: Strong reduction
    co2 = row.get("CO2", 1.0)
    if co2 <= 8.0:
        multiplier *= 1.0
    elif co2 <= 12.0:
        multiplier *= 0.96
    elif co2 <= 16.0:
        multiplier *= 0.88
    else:
        multiplier *= 0.70

    return multiplier


def get_row_risk_level(shelf_life: float, row) -> str:
    # 1. Base risk from shelf life
    if shelf_life >= 70:
        base_risk = "Fresh"
    elif shelf_life >= 40:
        base_risk = "Monitor"
    elif shelf_life >= 15:
        base_risk = "Near Spoilage"
    else:
        base_risk = "Spoiled"

    # 2. Indicator degradation levels
    degradations = []

    # pH
    ph = row.get("pH", 6.5)
    if ph >= 6.2 and ph <= 6.8:
        ph_deg = 0
    elif (ph >= 5.5 and ph < 6.2) or (ph > 6.8 and ph <= 7.2):
        ph_deg = 1
    elif (ph >= 4.8 and ph < 5.5) or (ph > 7.2 and ph <= 7.8):
        ph_deg = 2
    else:
        ph_deg = 3
    degradations.append(ph_deg)

    # Moisture (Moisture_Content)
    mc = row.get("Moisture_Content", 68.0)
    if mc >= 60.0 and mc <= 75.0:
        mc_deg = 0
    elif mc >= 45.0 and mc < 60.0:
        mc_deg = 1
    else:
        mc_deg = 2  # Treat as moderate impact, do not classify aggressively
    degradations.append(mc_deg)

    # PV
    pv = row.get("PV", 0.5)
    if pv <= 1.2:
        pv_deg = 0
    elif pv <= 3.0:
        pv_deg = 1
    elif pv <= 5.5:
        pv_deg = 2
    else:
        pv_deg = 3
    degradations.append(pv_deg)

    # TPC
    tpc = row.get("TPC", 0.2)
    if tpc <= 1.8:
        tpc_deg = 0
    elif tpc <= 3.5:
        tpc_deg = 1
    elif tpc <= 6.5:
        tpc_deg = 2
    else:
        tpc_deg = 3
    degradations.append(tpc_deg)

    # O2
    o2 = row.get("O2", 18.0)
    if o2 >= 15.0:
        o2_deg = 0
    elif o2 >= 10.0:
        o2_deg = 1
    elif o2 >= 4.0:
        o2_deg = 2
    else:
        o2_deg = 3
    degradations.append(o2_deg)

    # CO2
    co2 = row.get("CO2", 1.0)
    if co2 <= 8.0:
        co2_deg = 0
    elif co2 <= 12.0:
        co2_deg = 1
    elif co2 <= 16.0:
        co2_deg = 2
    else:
        co2_deg = 3
    degradations.append(co2_deg)

    num_degraded = sum(1 for d in degradations if d >= 1)
    num_moderate_or_strong = sum(1 for d in degradations if d >= 2)
    num_strong = sum(1 for d in degradations if d == 3)

    # Risk logic hierarchy
    # Fresh: Only if quality indicators mostly healthy (no moderate/strong degradations, and <= 1 small degradation)
    mostly_healthy = (num_moderate_or_strong == 0) and (num_degraded <= 1)

    # Spoiled: Major degradation (>= 2 strong, or >= 4 moderate/strong)
    is_spoiled = (num_strong >= 2) or (num_moderate_or_strong >= 4)

    # Near Spoilage: Multiple indicators degrade (>= 3 degraded or >= 1 strong)
    is_near_spoilage = (num_degraded >= 3) or (num_strong >= 1)

    # Monitor: If one or two indicators degrade (>= 1 degraded)
    is_monitor = (num_degraded >= 1)

    if is_spoiled:
        deg_risk = "Spoiled"
    elif is_near_spoilage:
        deg_risk = "Near Spoilage"
    elif is_monitor:
        deg_risk = "Monitor"
    else:
        deg_risk = "Fresh"

    # Severity lookup to ensure we use the worse (most severe) risk level
    severity = {"Fresh": 0, "Monitor": 1, "Near Spoilage": 2, "Spoiled": 3}
    
    if severity[deg_risk] > severity[base_risk]:
        return deg_risk
    return base_risk


def predict_data(df: pd.DataFrame):
    quality_pipeline, shelf_life_pipeline = load_models()
    if shelf_life_pipeline is None:
        raise Exception("Models not trained yet.")

    # Create a copy to work on
    working_df = df.copy()

    # Ensure all FEATURES exist in working_df
    for f in FEATURES:
        if f not in working_df.columns:
            working_df[f] = np.nan

    # 1. Fill missing BASE_FEATURES with Defaults first
    base_defaults = {
        "Retort_Temperature": 121.0,
        "Holding_Time": 20.0,
        "F0": 25.0,
        "Storage_Temperature": 25.0,
        "Storage_Day": 0.0,
    }
    for feat, default_val in base_defaults.items():
        if feat in working_df.columns:
            working_df[feat] = working_df[feat].fillna(default_val)
        else:
            working_df[feat] = default_val

    # 2. Generate missing QUALITY_FEATURES using quality_pipeline
    missing_quality = [qf for qf in QUALITY_FEATURES if working_df[qf].isnull().any()]
    if missing_quality and quality_pipeline is not None:
        X_base = working_df[BASE_FEATURES]
        predicted_quality = quality_pipeline.predict(X_base)
        predicted_quality_df = pd.DataFrame(
            predicted_quality, columns=QUALITY_FEATURES, index=df.index
        )
        for qf in QUALITY_FEATURES:
            working_df[qf] = working_df[qf].fillna(predicted_quality_df[qf])

        # Align prediction with UI calibration limits for generated quality values
        for idx in working_df.index:
            row = working_df.loc[idx]
            temp = row.get("Storage_Temperature", 25.0)
            days = row.get("Storage_Day", 0.0)

            if temp >= 45 or days >= 60:
                limits = {
                    "pH": (4.8, 5.5),
                    "PV": (5.0, 8.0),
                    "TPC": (8.0, 10.0),
                    "O2": (0.0, 4.0),
                    "CO2": (12.0, 20.0),
                    "L_Value": (45.0, 55.0),
                    "a_Value": (8.0, 12.0),
                    "b_Value": (14.0, 18.0),
                    "Moisture_Content": (50.0, 60.0),
                }
            elif temp >= 35 or days >= 30:
                limits = {
                    "pH": (5.9, 6.4),
                    "PV": (1.0, 2.0),
                    "TPC": (1.5, 3.0),
                    "O2": (12.0, 16.0),
                    "CO2": (3.0, 6.0),
                    "Moisture_Content": (60.0, 68.0),
                    "L_Value": (60.0, 65.0),
                    "a_Value": (4.0, 6.0),
                    "b_Value": (11.0, 14.0),
                }
            elif temp >= 30 or days >= 15:
                limits = {
                    "pH": (6.2, 6.6),
                    "PV": (0.5, 1.2),
                    "TPC": (0.8, 1.8),
                    "O2": (15.0, 18.0),
                    "CO2": (1.0, 3.0),
                    "Moisture_Content": (65.0, 70.0),
                    "L_Value": (63.0, 68.0),
                    "a_Value": (4.0, 6.0),
                    "b_Value": (12.0, 15.0),
                }
            else:
                limits = {
                    "pH": (6.7, 7.0),
                    "PV": (0.3, 0.7),
                    "TPC": (0.2, 1.0),
                    "O2": (18.0, 21.0),
                    "CO2": (0.0, 1.0),
                    "Moisture_Content": (68.0, 72.0),
                    "L_Value": (65.0, 75.0),
                    "a_Value": (3.0, 5.0),
                    "b_Value": (10.0, 14.0),
                }

            for key, (low, high) in limits.items():
                is_missing_in_original = (key not in df.columns) or pd.isnull(df.loc[idx, key])
                if is_missing_in_original and key in working_df.columns:
                    val = working_df.loc[idx, key]
                    if pd.notnull(val):
                        working_df.loc[idx, key] = float(np.clip(val, low, high))

    # 3. Fill any remaining missing quality features with Defaults
    quality_defaults = {
        "pH": 6.5,
        "PV": 0.5,
        "TPC": 0.2,
        "O2": 18.0,
        "CO2": 1.0,
        "Moisture_Content": 68.0,
        "L_Value": 65.0,
        "a_Value": 4.0,
        "b_Value": 12.0,
    }
    for feat, default_val in quality_defaults.items():
        if feat in working_df.columns:
            working_df[feat] = working_df[feat].fillna(default_val)
        else:
            working_df[feat] = default_val

    # Stage 2: Predict Shelf Life
    # Dampen the color features (L_Value, a_Value, b_Value) to reduce their dominance by 85%
    X_model = working_df.copy()
    X_model["L_Value"] = 65.0 + (working_df["L_Value"] - 65.0) * 0.15
    X_model["a_Value"] = 4.0 + (working_df["a_Value"] - 4.0) * 0.15
    X_model["b_Value"] = 12.0 + (working_df["b_Value"] - 12.0) * 0.15

    X = X_model[FEATURES]
    predictions = shelf_life_pipeline.predict(X)

    # User Calibration: Prediction [0, 90]
    predictions = np.clip(predictions, 0, 90)

    # Apply soft penalties to the predictions before calibration bounds
    multipliers = np.array([get_row_penalty_multiplier(working_df.iloc[i]) for i in range(len(working_df))])
    predictions = predictions * multipliers

    # Apply calibration based on Quality Score to avoid over-reducing shelf life
    # pH: fresh 6.6, spoiled 4.8
    ph_f = np.clip((working_df["pH"] - 4.8) / (6.6 - 4.8), 0.0, 1.0)
    # PV: fresh 0.5, spoiled 8.0
    pv_f = np.clip((8.0 - working_df["PV"]) / (8.0 - 0.5), 0.0, 1.0)
    # TPC: fresh 0.8, spoiled 10.0
    tpc_f = np.clip((10.0 - working_df["TPC"]) / (10.0 - 0.8), 0.0, 1.0)
    # O2: fresh 18, spoiled 0
    o2_f = np.clip(working_df["O2"] / 18.0, 0.0, 1.0)
    # CO2: fresh 1.0, spoiled 20.0
    co2_f = np.clip((20.0 - working_df["CO2"]) / (20.0 - 1.0), 0.0, 1.0)
    # Moisture: fresh 70, spoiled 50
    moist_f = np.clip((working_df["Moisture_Content"] - 50.0) / (70.0 - 50.0), 0.0, 1.0)

    qs = (ph_f + pv_f + tpc_f + o2_f + co2_f + moist_f) / 6.0
    
    # Scale quality score bounds using the penalty multiplier
    # so that bounds also contract downward if quality indicators are degraded
    qs_scaled = qs * multipliers

    days = working_df["Storage_Day"].values
    qs_val = qs_scaled.values if isinstance(qs_scaled, pd.Series) else qs_scaled
    
    # Calculate bounds based on quality score to keep shelf life realistic
    lower_bound = (90.0 - days) * (qs_val ** 0.7)
    upper_bound = (90.0 - days) * (qs_val ** 0.4)
    
    # If the food is heavily spoiled (QS < 0.3), disable lower bound
    lower_bound = np.where(qs_val >= 0.3, lower_bound, 0.0)
    
    lower_bound = np.maximum(0.0, lower_bound)
    upper_bound = np.maximum(0.0, upper_bound)
    
    # Bounded clip of predictions
    predictions = np.clip(predictions, lower_bound, upper_bound)
    predictions = np.clip(predictions, 0.0, 90.0)

    risk_levels = [get_row_risk_level(predictions[i], working_df.iloc[i]) for i in range(len(predictions))]

    return predictions, risk_levels, working_df[QUALITY_FEATURES]
