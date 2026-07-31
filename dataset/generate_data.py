import pandas as pd
import numpy as np
import os


def generate_synthetic_data(num_samples=1500, output_path="synthetic_food_data.csv"):
    np.random.seed(42)

    samples_per_cat = num_samples // 4
    all_rows = []

    # Risk level threshold logic:
    # Fresh > 80
    # Monitor [40, 80]
    # Near Spoilage [15, 40]
    # Spoiled < 15 (Target 0-10 per user request)

    categories = [
        {
            "name": "Fresh",
            "life_range": (85, 120),
            "temp_range": (4, 15),
            "day_range": (0, 10),
            "tpc_range": (0.1, 1.0),
        },
        {
            "name": "Monitor",
            "life_range": (45, 75),
            "temp_range": (15, 25),
            "day_range": (10, 30),
            "tpc_range": (1.0, 2.5),
        },
        {
            "name": "Near Spoilage",
            "life_range": (18, 35),
            "temp_range": (25, 35),
            "day_range": (30, 60),
            "tpc_range": (2.5, 4.5),
        },
        {
            "name": "Spoiled",
            "life_range": (0, 10),
            "temp_range": (35, 45),
            "day_range": (60, 120),
            "tpc_range": (4.5, 8.0),
        },
    ]

    for cat in categories:
        count = samples_per_cat
        life = np.random.uniform(cat["life_range"][0], cat["life_range"][1], count)
        temp = np.random.uniform(cat["temp_range"][0], cat["temp_range"][1], count)
        days = np.random.uniform(cat["day_range"][0], cat["day_range"][1], count)
        tpc = np.random.uniform(cat["tpc_range"][0], cat["tpc_range"][1], count)

        # Other features within reasonable bounds
        retort = np.random.uniform(110, 130, count)
        holding = np.random.uniform(10, 40, count)
        f0 = np.random.uniform(10, 40, count)
        ph = np.random.uniform(4.0, 7.0, count)
        pv = np.random.uniform(0.1, 8.0, count)
        o2 = np.random.uniform(0, 20, count)
        co2 = np.random.uniform(0, 15, count)
        moisture = np.random.uniform(20, 80, count)
        l_val = np.random.uniform(30, 80, count)
        a_val = np.random.uniform(-10, 20, count)
        b_val = np.random.uniform(0, 30, count)

        # Override for Spoiled category based on calibration requirements
        if cat["name"] == "Spoiled":
            pv = np.random.uniform(5.0, 10.0, count)
            o2 = np.random.uniform(0, 3.0, count)
            co2 = np.random.uniform(15.0, 25.0, count)
            # Ensure high storage days and bacterial count
            days = np.random.uniform(90, 120, count)
            tpc = np.random.uniform(8.0, 12.0, count)

        df_cat = pd.DataFrame(
            {
                "Retort_Temperature": np.round(retort, 1),
                "Holding_Time": np.round(holding, 1),
                "F0": np.round(f0, 2),
                "Storage_Temperature": np.round(temp, 1),
                "Storage_Day": np.round(days, 1),
                "pH": np.round(ph, 2),
                "PV": np.round(pv, 2),
                "TPC": np.round(tpc, 2),
                "O2": np.round(o2, 2),
                "CO2": np.round(co2, 2),
                "Moisture_Content": np.round(moisture, 1),
                "L_Value": np.round(l_val, 1),
                "a_Value": np.round(a_val, 1),
                "b_Value": np.round(b_val, 1),
                "Shelf_Life_Remaining": np.round(life, 1),
            }
        )
        all_rows.append(df_cat)

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Distribution Check
    def get_risk(val):
        if val > 80:
            return "Fresh"
        if val >= 40:
            return "Monitor"
        if val >= 15:
            return "Near Spoilage"
        return "Spoiled"

    counts = df["Shelf_Life_Remaining"].apply(get_risk).value_counts()
    print(f"Generated {len(df)} rows at {output_path}")
    print("-" * 30)
    print("Training Target Distribution:")
    for cat in ["Fresh", "Monitor", "Near Spoilage", "Spoiled"]:
        print(f"{cat} Count: {counts.get(cat, 0)}")
    print("-" * 30)
    return df


if __name__ == "__main__":
    generate_synthetic_data(1500, "dataset/synthetic_food_data.csv")
