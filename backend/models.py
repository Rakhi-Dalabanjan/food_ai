from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from .database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    retort_temperature = Column(Float)
    holding_time = Column(Float)
    f0 = Column(Float)
    storage_temperature = Column(Float)
    storage_day = Column(Float)
    ph = Column(Float)
    pv = Column(Float)
    tpc = Column(Float)
    o2 = Column(Float)
    co2 = Column(Float)
    moisture_content = Column(Float)
    l_value = Column(Float)
    a_value = Column(Float)
    b_value = Column(Float)
    shelf_life_remaining = Column(Float)
    risk_level = Column(String)
    product_type = Column(String, nullable=True)
    package_status = Column(String, nullable=True)
    oxidation_rate = Column(Float, nullable=True)
    product_status = Column(String, nullable=True)
    rejection_reasons = Column(String, nullable=True)
    overall_acceptability = Column(Float, nullable=True)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    mae = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    feature_importances = Column(String)
