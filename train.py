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
    print("Preparing experimental training dataset from 'My Experiment Readings M Tech.xlsx'...")
    from dataset.prepare_experimental_dataset import prepare_experimental_dataset
    df_train, stats = prepare_experimental_dataset()
    
    dataset_path = "dataset/client_experimental_prepared.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Preparation failed.")
        sys.exit(1)
        
    print("Training ML model on client experimental data...")
    metrics = train_and_save_model(dataset_path)
    print("Model trained successfully!")
    print(metrics)
