# ShelfLife AI

ShelfLife AI is an end-to-end Machine Learning application built to predict the remaining shelf life of food products based on detailed quality parameters like Retort Temperature, Holding Time, F0 Value, and specific chemical metrics (pH, PV, TPC).

## Features

- **Strict, Simple UI:** Extremely straightforward navigation containing only Home, Manual Prediction, and CSV Prediction.
- **Manual Prediction:** Enter data directly into a form to instantly receive a shelf life prediction and corresponding Spoilage Risk Level (Fresh, Monitor, Near Spoilage, Spoiled).
- **Bulk CSV Upload:** Upload large batches of data for batch predictions, with CSV export capabilities.
- **Dynamic Risk Engine:** Classifies the risk level natively based on the predicted shelf life.
- **Model Metrics Dashboard:** Accessible from the Home page, offering visualization of MAE, RMSE, R² Score, Top 10 Feature Importances, and Prediction Distributions.
- **Retraining Support:** The backend allows seamless retraining of the model when new datasets are submitted.

## Tech Stack

- **Backend**: Python, FastAPI, Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer), Pandas, SQLite.
- **Frontend**: React (Vite), Tailwind CSS (v3), Recharts, React Router.

## API Endpoints

- `GET /health`: System health check.
- `POST /upload`: Upload a CSV file and preview data.
- `POST /manual-predict`: Predict shelf life for a single manual entry.
- `POST /predict`: Generate predictions for an uploaded CSV dataset.
- `POST /retrain`: Retrain the Random Forest model using the uploaded dataset path.
- `GET /metrics`: Fetch the latest model evaluation metrics.
- `GET /logs`: Fetch the recent prediction logs.

## Setup Instructions

### 1. Backend Setup

```bash
# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies if not already installed
pip install fastapi uvicorn pandas numpy scikit-learn joblib sqlalchemy python-multipart

# Generate the synthetic dataset
python dataset/generate_data.py

# Train the ML model
python train.py

# Start the FastAPI server
uvicorn backend.main:app --reload
```

API Documentation: `http://import.meta.env.VITE_API_URL/docs`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies (Tailwind v3 is used)
npm install

# Start the development server
npm run dev
```

Application URL: `http://localhost:5173`
