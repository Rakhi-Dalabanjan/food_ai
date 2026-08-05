from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd
import numpy as np
import shutil
import os
import json
from typing import List, Dict, Optional

from .database import engine, Base, get_db
from .models import PredictionLog, ModelMetrics
from .ml_pipeline import (
    train_and_save_model,
    predict_data,
    FEATURES,
    BASE_FEATURES,
    QUALITY_FEATURES,
    load_models,
    get_risk_level,
)

def run_db_migrations(engine):
    """Safely and idempotently adds missing V2 columns to the prediction_logs table."""
    import sqlite3
    import logging

    logger = logging.getLogger("DBMigration")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

    try:
        db_url = str(engine.url)
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
        else:
            db_path = engine.url.database

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check existing columns using SQLite PRAGMA
        cursor.execute("PRAGMA table_info(prediction_logs)")
        columns_info = cursor.fetchall()
        existing_columns = {row[1] for row in columns_info}

        columns_to_add = [
            ("product_type", "TEXT"),
            ("package_status", "TEXT"),
            ("oxidation_rate", "REAL")
        ]

        migrated_cols = []
        for col_name, col_type in columns_to_add:
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE prediction_logs ADD COLUMN {col_name} {col_type}")
                migrated_cols.append(col_name)
                logger.info(f"Database Migration: Column '{col_name}' ADDED to prediction_logs table.")

        conn.commit()
        conn.close()

        if migrated_cols:
            logger.info(f"Database Migration completed successfully for columns: {', '.join(migrated_cols)}")
        else:
            logger.info("Database Migration: skipped. All columns already exist.")

    except Exception as e:
        logger.error(f"Database Migration failed: {str(e)}")


Base.metadata.create_all(bind=engine)
run_db_migrations(engine)

app = FastAPI(title="ShelfLife AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_log.txt")
    try:
        with open(log_path, "a") as f:
            f.write(f"Exception during request to {request.url.path}:\n")
            traceback.print_exc(file=f)
            f.write("\n" + "="*80 + "\n")
    except Exception as log_err:
        print(f"Failed to log exception: {log_err}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}", "traceback": traceback.format_exc()},
    )

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAPPING_SUGGESTIONS = {
    "Retort_Temperature": ["retort_temperature", "retort_temp", "retorttemp", "retort", "temp_c"],
    "Holding_Time": ["holding_time", "holdingtime", "holding_time_min", "holding_min", "hold_time", "holdtime", "holdtimemin"],
    "F0": ["f0", "f0_value", "f0value"],
    "Storage_Temperature": ["storage_temperature", "storage_temp", "storagetemp", "temp", "temperature", "temp_c"],
    "Storage_Day": ["storage_day", "storage_days", "storageday", "storagedays", "storage_time_in_days", "storagetimeindays", "day", "days"],
    "pH": ["ph", "ph_value", "phvalue", "acidity"],
    "PV": ["pv", "pv_value", "peroxide", "peroxide_value"],
    "TPC": ["tpc", "tpc_value", "bacteria", "microbial_count", "microbial", "tvc_tpc", "tvc", "tvc_tpc_count"],
    "O2": ["o2", "oxygen", "o2_%"],
    "CO2": ["co2", "carbon_dioxide", "carbondioxide", "co2_%"],
    "Moisture_Content": ["moisture_content", "moisture", "moisturecontent", "moisture_%", "moisture_content_%"],
    "L_Value": ["l_value", "lvalue", "l", "l*"],
    "a_Value": ["a_value", "avalue", "a", "a*"],
    "b_Value": ["b_value", "bvalue", "b", "b*"],
}


def normalize_str(s: str) -> str:
    return "".join(e for e in str(s).lower() if e.isalnum())


def suggest_mapping(df: pd.DataFrame) -> Dict[str, str]:
    columns = df.columns.tolist()
    mapping = {}
    normalized_cols = {col: normalize_str(col) for col in columns}

    def get_col_mean(col):
        try:
            return pd.to_numeric(df[col], errors='coerce').mean()
        except:
            return None

    # 1. Exact / Direct Normalized Matches
    for feature in FEATURES:
        norm_feature = normalize_str(feature)
        match = next(
            (col for col, norm in normalized_cols.items() if norm == norm_feature),
            None
        )
        if match:
            mapping[feature] = match

    # 2. Smart Temperature Differentiation (Retort vs Storage)
    if "Retort_Temperature" not in mapping:
        # Check for retort keywords
        for col, norm in normalized_cols.items():
            if "retort" in norm:
                mapping["Retort_Temperature"] = col
                break
        # Fallback: if 'temp' in col name and mean value >= 60°C (processing temp)
        if "Retort_Temperature" not in mapping:
            for col, norm in normalized_cols.items():
                if "temp" in norm and "storage" not in norm:
                    mean_val = get_col_mean(col)
                    if mean_val is not None and mean_val >= 60:
                        mapping["Retort_Temperature"] = col
                        break

    if "Storage_Temperature" not in mapping:
        # Check for storage keywords
        for col, norm in normalized_cols.items():
            if "storage" in norm and "temp" in norm:
                mapping["Storage_Temperature"] = col
                break
        # Fallback: if 'temp' in col name, not assigned to Retort, and mean value < 60°C
        if "Storage_Temperature" not in mapping:
            for col, norm in normalized_cols.items():
                if "temp" in norm and col != mapping.get("Retort_Temperature"):
                    mean_val = get_col_mean(col)
                    if mean_val is None or mean_val < 60:
                        mapping["Storage_Temperature"] = col
                        break

    # 3. Alias Check for remaining features
    for feature in FEATURES:
        if feature in mapping:
            continue
        suggestions = MAPPING_SUGGESTIONS.get(feature, [])
        match = next(
            (col for col, norm in normalized_cols.items() if norm in [normalize_str(s) for s in suggestions]),
            None
        )
        if match:
            mapping[feature] = match

    return mapping


def clean_nans(data):
    if isinstance(data, dict):
        return {k: clean_nans(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nans(v) for v in data]
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    return data


class ManualPredictionRequest(BaseModel):
    Retort_Temperature: Optional[float] = None
    Holding_Time: Optional[float] = None
    F0: Optional[float] = None
    Storage_Temperature: Optional[float] = None
    Storage_Day: Optional[float] = None
    pH: Optional[float] = None
    PV: Optional[float] = None
    TPC: Optional[float] = None
    O2: Optional[float] = None
    CO2: Optional[float] = None
    Moisture_Content: Optional[float] = None
    L_Value: Optional[float] = None
    a_Value: Optional[float] = None
    b_Value: Optional[float] = None


@app.get("/health")
def health_check():
    _, shelf_life_pipeline = load_models()
    return {"status": "healthy", "model_loaded": shelf_life_pipeline is not None}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format.\n\nSupported:\nCSV (.csv)\nExcel (.xlsx)",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        preview = clean_nans(df.head(10).to_dict(orient="records"))
        columns = df.columns.tolist()
        suggested = suggest_mapping(df)

        return {
            "filename": file.filename,
            "columns": columns,
            "row_count": len(df),
            "preview": preview,
            "suggested_mapping": suggested,
            "features": FEATURES,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    mapping: str = Form(...),
    db: Session = Depends(get_db),
):
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format.\n\nSupported:\nCSV (.csv)\nExcel (.xlsx)",
        )

    file_path = os.path.join(UPLOAD_DIR, "predict_" + file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Parse mapping
        col_mapping = json.loads(mapping)

        # Apply mapping
        mapped_df = pd.DataFrame()
        provided_fields_count = 0

        for feature in FEATURES:
            uploaded_col = col_mapping.get(feature)
            if uploaded_col and uploaded_col in df.columns:
                mapped_df[feature] = pd.to_numeric(df[uploaded_col], errors="coerce")
                provided_fields_count += 1
            else:
                mapped_df[feature] = np.nan

        # Validation rule: at least 2 parameters in the file
        # We check per row later or just globally if the mapping has at least 2 fields
        if provided_fields_count < 2:
            raise HTTPException(
                status_code=400,
                detail="Please provide at least 2 parameters in your file.",
            )

        predictions, risk_levels, quality_df = predict_data(mapped_df)

        # Update original df with predictions and also estimated quality values
        df["Shelf_Life"] = predictions
        df["Risk_Level"] = risk_levels

        for feature in QUALITY_FEATURES:
            df[f"Estimated_{feature}"] = quality_df[feature].values

        # Log to DB (using quality_df results)
        for idx, row in mapped_df.head(100).iterrows():
            log = PredictionLog(
                retort_temperature=float(
                    row.get("Retort_Temperature")
                    if pd.notnull(row.get("Retort_Temperature"))
                    else 0
                ),
                holding_time=float(
                    row.get("Holding_Time")
                    if pd.notnull(row.get("Holding_Time"))
                    else 0
                ),
                f0=float(row.get("F0") if pd.notnull(row.get("F0")) else 0),
                storage_temperature=float(
                    row.get("Storage_Temperature")
                    if pd.notnull(row.get("Storage_Temperature"))
                    else 0
                ),
                storage_day=float(
                    row.get("Storage_Day") if pd.notnull(row.get("Storage_Day")) else 0
                ),
                ph=float(quality_df.loc[idx, "pH"]),
                pv=float(quality_df.loc[idx, "PV"]),
                tpc=float(quality_df.loc[idx, "TPC"]),
                o2=float(quality_df.loc[idx, "O2"]),
                co2=float(quality_df.loc[idx, "CO2"]),
                moisture_content=float(quality_df.loc[idx, "Moisture_Content"]),
                l_value=float(quality_df.loc[idx, "L_Value"]),
                a_value=float(quality_df.loc[idx, "a_Value"]),
                b_value=float(quality_df.loc[idx, "b_Value"]),
                shelf_life_remaining=float(df.loc[idx, "Shelf_Life"]),
                risk_level=str(df.loc[idx, "Risk_Level"]),
            )
            db.add(log)
        db.commit()

        return clean_nans(df.to_dict(orient="records"))
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/manual-predict")
def manual_predict(data: ManualPredictionRequest, db: Session = Depends(get_db)):
    try:
        # Filter out None values to let predict_data handle estimation
        input_dict = {k: v for k, v in data.dict().items() if v is not None}

        # New Rule: Minimum 1 parameter
        if len(input_dict) < 1:
            raise HTTPException(
                status_code=400, detail="Please provide at least 1 parameter."
            )

        df = pd.DataFrame([input_dict])

        predictions, risk_levels, quality_df = predict_data(df)

        shelf_life = float(predictions[0])
        risk_level = risk_levels[0]
        estimated_quality = quality_df.iloc[0].to_dict()

        log = PredictionLog(
            retort_temperature=data.Retort_Temperature,
            holding_time=data.Holding_Time,
            f0=data.F0,
            storage_temperature=data.Storage_Temperature,
            storage_day=data.Storage_Day,
            ph=estimated_quality["pH"],
            pv=estimated_quality["PV"],
            tpc=estimated_quality["TPC"],
            o2=estimated_quality["O2"],
            co2=estimated_quality["CO2"],
            moisture_content=estimated_quality["Moisture_Content"],
            l_value=estimated_quality["L_Value"],
            a_value=estimated_quality["a_Value"],
            b_value=estimated_quality["b_Value"],
            shelf_life_remaining=shelf_life,
            risk_level=risk_level,
        )
        db.add(log)
        db.commit()

        return {
            "shelf_life": round(shelf_life, 1),
            "risk_level": risk_level,
            "estimated_quality": estimated_quality,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Technical error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Unable to process prediction. Please try again."
        )


@app.post("/estimate-quality")
def estimate_quality(data: ManualPredictionRequest):
    try:
        input_dict = {k: v for k, v in data.dict().items() if v is not None}

        if len(input_dict) < 1:
            raise HTTPException(
                status_code=400, detail="Please provide at least 1 parameter."
            )

        # Default missing base features so they can be generated and returned
        defaults = {
            "Retort_Temperature": 121.0,
            "Holding_Time": 20.0,
            "F0": 25.0,
            "Storage_Temperature": 25.0,
            "Storage_Day": 0.0,
        }
        for k, v in defaults.items():
            if k not in input_dict:
                input_dict[k] = v

        df = pd.DataFrame([input_dict])
        _, _, quality_df = predict_data(df)

        # Merge the original inputs with the predicted ones for a full form set
        full_data = quality_df.iloc[0].to_dict()

        # --- CALIBRATION FIX ---
        temp = input_dict.get("Storage_Temperature", 25)
        days = input_dict.get("Storage_Day", 0)

        # Deterioration Rules based on User Cases
        if temp >= 45 or days >= 60:
            # Case 3: Extreme / Spoiled
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
            # Case 2: Moderate / Monitor (Target: 45-65 Days)
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
            # Case 1: Fresh-ish / Monitor (Target: 60-75 Days)
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
            # Super Fresh
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
            if key in full_data and key not in input_dict:
                full_data[key] = float(np.clip(full_data[key], low, high))

        for k, v in input_dict.items():
            full_data[k] = v

        return full_data
    except HTTPException:
        raise
    except Exception as e:
        print(f"Technical error during estimation: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Unable to generate values. Please try again."
        )


@app.post("/retrain")
async def retrain_model(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        metrics = train_and_save_model(file_path)
        return {"status": "Model retrained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    metrics = db.query(ModelMetrics).order_by(ModelMetrics.id.desc()).first()
    if not metrics:
        return {"mae": None, "rmse": None, "r2_score": None, "feature_importances": {}}

    import json

    return {
        "timestamp": metrics.timestamp,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2_score": metrics.r2_score,
        "feature_importances": (
            json.loads(metrics.feature_importances)
            if metrics.feature_importances
            else {}
        ),
    }


@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(100).all()
    return logs
