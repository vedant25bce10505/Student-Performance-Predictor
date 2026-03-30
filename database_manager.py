"""
database_manager.py
-------------------
SQLite / SQLAlchemy layer for storing and querying predictions.
"""

import os
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "database", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

Base = declarative_base()


class Prediction(Base):
    """ORM model for a single prediction record."""

    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    study_hours     = Column(Float)
    attendance      = Column(Float)
    previous_grade  = Column(Float)
    parent_education = Column(String(50))
    predicted_grade = Column(String(5))
    confidence      = Column(Float)
    pass_fail       = Column(String(10))
    model_used      = Column(String(50))

    def __repr__(self):
        return (f"<Prediction id={self.id} grade={self.predicted_grade} "
                f"confidence={self.confidence:.1f}% model={self.model_used}>")


class DatabaseManager:
    """CRUD interface for the predictions database."""

    def __init__(self, db_path: str = DB_PATH):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Database ready at %s", db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_prediction(self, student_data: dict, prediction_result: dict) -> int:
        """Persist a prediction to the database; return the new record ID."""
        record = Prediction(
            study_hours     = student_data.get("study_hours"),
            attendance      = student_data.get("attendance"),
            previous_grade  = student_data.get("previous_grade"),
            parent_education = student_data.get("parent_education"),
            predicted_grade = prediction_result.get("prediction"),
            confidence      = prediction_result.get("confidence"),
            pass_fail       = prediction_result.get("pass_fail"),
            model_used      = prediction_result.get("model_used"),
        )
        with self.Session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("Saved prediction id=%d", record.id)
            return record.id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_predictions(self) -> pd.DataFrame:
        """Return all stored predictions as a DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM predictions ORDER BY timestamp DESC"), conn)

    def get_recent_predictions(self, n: int = 10) -> pd.DataFrame:
        """Return the n most recent predictions."""
        with self.engine.connect() as conn:
            return pd.read_sql(
                text(f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {n}"), conn
            )

    def get_statistics(self) -> dict:
        """Summary statistics over all stored predictions."""
        df = self.get_all_predictions()
        if df.empty:
            return {"total": 0}
        return {
            "total":        len(df),
            "pass_rate":    round((df["pass_fail"] == "Pass").mean() * 100, 2),
            "avg_confidence": round(df["confidence"].mean(), 2),
            "grade_distribution": df["predicted_grade"].value_counts().to_dict(),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(self, filepath: str) -> str:
        """Export all predictions to a CSV file."""
        df = self.get_all_predictions()
        df.to_csv(filepath, index=False)
        logger.info("Exported %d records → %s", len(df), filepath)
        return filepath


if __name__ == "__main__":
    db = DatabaseManager()
    print("DatabaseManager ready. Stats:", db.get_statistics())
