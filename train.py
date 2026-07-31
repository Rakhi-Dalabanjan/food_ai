import os
import sys

# Add the project root to sys.path so we can import backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ml_pipeline import train_and_save_model
from backend.database import engine, Base
from backend.models import PredictionLog, ModelMetrics

# Initialize DB tables
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    dataset_path = "dataset/synthetic_food_data.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run dataset/generate_data.py first.")
        sys.exit(1)
        
    print("Training model...")
    metrics = train_and_save_model(dataset_path)
    print("Model trained successfully!")
    print(metrics)
