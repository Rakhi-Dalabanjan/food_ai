import os
import sys
import pandas as pd
import numpy as np
import logging
from lifelines import WeibullAFTFitter

project_root = r"d:\personal\projects\food-shelf-life-ai_1.0-main\food-shelf-life-ai_1.0-main"
sys.path.insert(0, project_root)

from backend.ml_pipeline import safe_float, FEATURES, BASE_FEATURES, QUALITY_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_experimental_dataset(excel_path=None, output_csv=None):
    if excel_path is None:
        excel_path = os.path.join(project_root, "My Experiment Readings M Tech (2).xlsx")
        if not os.path.exists(excel_path):
            excel_path = os.path.join(project_root, "My Experiment Readings M Tech.xlsx")
            
    if output_csv is None:
        output_csv = os.path.join(project_root, "dataset", "client_experimental_prepared.csv")

    logging.info(f"Loading raw experimental data from: {excel_path}")
    df_raw = pd.read_excel(excel_path, sheet_name=0)

    sample_col = "Sample ID"
    day_col = "Storage Time in Days"
    status_col = "Final Sample Status" if "Final Sample Status" in df_raw.columns else "Sample Status"
    obs_col = "Observation"
    pouch_col = "Package Integrity" if "Package Integrity" in df_raw.columns else "Pouch Condition"

    # Identify confirmed spoilage endpoints across sample groups
    spoilage_endpoints = {}
    spoilage_reasons = {}

    for sample_id, grp in df_raw.groupby(sample_col):
        if pd.isnull(sample_id) or str(sample_id).strip().lower() in ["nan", "none", "n/a"]:
            continue
            
        s_id_str = str(sample_id).strip()
        grp_sorted = grp.sort_values(by=[day_col])
        is_unretorted = ("NRT" in s_id_str or "unretort" in s_id_str.lower())
        
        spoil_day = None
        reasons_found = []
        
        for idx, row in grp_sorted.iterrows():
            day = safe_float(row[day_col])
            status = str(row.get(status_col, "")).strip().lower()
            obs = str(row.get(obs_col, "")).strip().lower()
            pouch = str(row.get(pouch_col, "")).strip().lower()
            ph = safe_float(row.get("pH"))
            pv = safe_float(row.get("PV"))
            tpc_val = safe_float(row.get("TPC (log CFU/g)") or row.get("TVC (TPC)"))
            tpc_str = str(row.get("TPC (log CFU/g)") or row.get("TVC (TPC)") or "").strip().upper()
            oa = safe_float(row.get("Overall Acceptability"))
            
            r = []
            if status in ["spoiled", "gas formation", "not acceptable"]:
                r.append(f"Sample Status: {row.get(status_col)}")
            if obs in ["gas formation", "bulged", "spoilage", "foul smell", "rancid"]:
                r.append(f"Observation: {row.get(obs_col)}")
            if pouch in ["bulged", "bulging", "seal failure", "leakage", "damaged", "damaged package", "damaged pouch"]:
                r.append(f"Pouch Condition: {row.get(pouch_col)}")
            if tpc_str == "TNTC" or (is_unretorted and tpc_val is not None and tpc_val > 5.0) or (not is_unretorted and tpc_val is not None and tpc_val > 0.0):
                r.append(f"TPC limit exceeded")
            if pv is not None and pv > 10.0:
                r.append(f"PV limit exceeded ({pv} > 10)")
            if ph is not None and ph > 0 and ph < 4.9:
                r.append(f"pH limit exceeded ({ph} < 4.9)")
            if oa is not None and oa > 0 and oa < 5.0:
                r.append(f"Overall Acceptability below limit ({oa} < 5)")
                
            if r:
                spoil_day = day
                reasons_found = r
                break
                
        if spoil_day is not None:
            spoilage_endpoints[s_id_str] = spoil_day
            spoilage_reasons[s_id_str] = "; ".join(reasons_found)

    # Fit Weibull Survival AFT Model on Group Durations & Event Indicators to handle right-censored data
    group_records = []
    for s_id_str, grp in df_raw.groupby(sample_col):
        if pd.isnull(s_id_str): continue
        s_key = str(s_id_str).strip()
        is_unretorted = ("NRT" in s_key or "unretort" in s_key.lower())
        
        if s_key in spoilage_endpoints:
            duration = spoilage_endpoints[s_key]
            event = 1
        else:
            duration = safe_float(grp[day_col].max())
            event = 0
            
        if duration == 0: duration = 0.5
        
        day0 = grp.iloc[0]
        group_records.append({
            "Sample_ID": s_key,
            "Duration": duration,
            "Event": event,
            "Is_Retorted": 0.0 if is_unretorted else 1.0,
            "F0": safe_float(day0.get("F0")) or (0.0 if is_unretorted else 25.0),
            "Storage_Temperature": safe_float(day0.get("Storage Temp (C)")) or 25.0
        })

    df_groups = pd.DataFrame(group_records)
    aft = WeibullAFTFitter()
    aft.fit(df_groups[["Duration", "Event", "Storage_Temperature", "F0"]], duration_col="Duration", event_col="Event")
    df_groups["Predicted_Lifespan"] = aft.predict_median(df_groups[["Duration", "Event", "Storage_Temperature", "F0"]])
    lifespan_map = dict(zip(df_groups["Sample_ID"], df_groups["Predicted_Lifespan"]))

    # Process all rows to build complete prepared dataset
    row_list = []
    censored_count = 0
    event_count = 0

    for idx, row in df_raw.iterrows():
        sample_id = row[sample_col]
        if pd.isnull(sample_id) or str(sample_id).strip().lower() in ["nan", "none", "n/a"]:
            continue
            
        s_id_str = str(sample_id).strip()
        curr_day = safe_float(row[day_col]) or 0.0
        is_unretorted = ("NRT" in s_id_str or "unretort" in s_id_str.lower())
        p_type = "Unretorted" if is_unretorted else "Retorted"
        
        if s_id_str in spoilage_endpoints:
            spoil_day = spoilage_endpoints[s_id_str]
            target_rem = max(0.0, spoil_day - curr_day)
            is_censored = 0
            event_count += 1
        else:
            pred_total = lifespan_map.get(s_id_str, 50.0)
            target_rem = max(0.0, pred_total - curr_day)
            is_censored = 1
            censored_count += 1
            
        r_dict = {
            "Sample_ID": s_id_str,
            "Product_Type": p_type,
            "Storage_Day": curr_day,
            "Shelf_Life_Remaining": target_rem,
            "Is_Censored": is_censored,
            "Retort_Temperature": safe_float(row.get("Temp (C)")) or (0.0 if is_unretorted else 121.0),
            "Holding_Time": safe_float(row.get("Hold Time (Min)")) or (0.0 if is_unretorted else 20.0),
            "F0": safe_float(row.get("F0")) or (0.0 if is_unretorted else 25.0),
            "Storage_Temperature": safe_float(row.get("Storage Temp (C)")) or 25.0,
            "pH": safe_float(row.get("pH")),
            "PV": safe_float(row.get("PV")),
            "TPC": safe_float(row.get("TPC (log CFU/g)") or row.get("TVC (TPC)")),
            "O2": safe_float(row.get("O2 %")),
            "CO2": safe_float(row.get("CO2 %")),
            "Moisture_Content": safe_float(row.get("Moisture Content %")),
            "L_Value": safe_float(row.get("L*")),
            "a_Value": safe_float(row.get("a*")),
            "b_Value": safe_float(row.get("b*")),
            "Overall_Acceptability": safe_float(row.get("Overall Acceptability"))
        }
        row_list.append(r_dict)

    df_train = pd.DataFrame(row_list)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_train.to_csv(output_csv, index=False)

    stats = {
        "total_excel_rows": len(df_raw),
        "prepared_training_rows": len(df_train),
        "confirmed_spoilage_events": event_count,
        "right_censored_rows": censored_count,
        "total_sample_groups": len(df_groups),
        "confirmed_spoilage_groups": len(spoilage_endpoints),
        "censored_sample_groups": len(df_groups) - len(spoilage_endpoints),
        "survival_aft_concordance": float(aft.concordance_index_),
        "spoilage_endpoints": spoilage_endpoints,
        "spoilage_reasons": spoilage_reasons
    }

    logging.info(f"Dataset preparation complete. Saved {len(df_train)} prepared rows to {output_csv}")
    logging.info(f"Confirmed spoilage groups: {len(spoilage_endpoints)} | Censored groups: {len(df_groups) - len(spoilage_endpoints)} | Survival C-index: {aft.concordance_index_:.4f}")
    return df_train, stats

if __name__ == "__main__":
    df_train, stats = prepare_experimental_dataset()
    print("Dataset Prep Stats:")
    for k, v in stats.items():
        if k not in ["spoilage_endpoints", "spoilage_reasons"]:
            print(f"  {k}: {v}")
