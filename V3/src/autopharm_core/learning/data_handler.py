import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3  # Simplified DB for demo (replace with InfluxDB in production)
import os


# Simplified types for notebook demo (would import from ..common.types in full implementation)
class StateVector:
    def __init__(self, timestamp: float, cmas: Dict[str, float], cpps: Dict[str, float]):
        self.timestamp = timestamp
        self.cmas = cmas
        self.cpps = cpps


class TrainingMetrics:
    def __init__(
        self,
        model_version: str,
        validation_loss: float,
        training_duration_seconds: float,
        dataset_size: int,
    ):
        self.model_version = model_version
        self.validation_loss = validation_loss
        self.training_duration_seconds = training_duration_seconds
        self.dataset_size = dataset_size


class DataHandler:
    """
    Handles data storage, retrieval, and preprocessing for online learning.
    Interfaces with time-series database to manage operational data.
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize database connection.

        Args:
            db_connection_string: Database connection string or path
        """
        self.db_path = db_connection_string
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Process data table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS process_data (
                    timestamp REAL PRIMARY KEY,
                    d50 REAL,
                    lod REAL,
                    spray_rate REAL,
                    air_flow REAL,
                    carousel_speed REAL,
                    specific_energy REAL,
                    froude_number_proxy REAL
                )
            """
            )

            # Model performance table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    timestamp REAL,
                    model_version TEXT,
                    validation_loss REAL,
                    dataset_size INTEGER,
                    training_duration REAL
                )
            """
            )

            conn.commit()

    def log_trajectory(self, trajectory: List[StateVector]):
        """
        Log a completed trajectory to the database.

        Args:
            trajectory: List of StateVector observations
        """
        with sqlite3.connect(self.db_path) as conn:
            for state in trajectory:
                # Calculate soft sensors
                specific_energy = (
                    state.cpps.get("spray_rate", 0.0) * state.cpps.get("carousel_speed", 0.0)
                ) / 1000.0
                froude_number_proxy = (state.cpps.get("carousel_speed", 0.0) ** 2) / 9.81

                # Prepare data row
                data_row = {
                    "timestamp": state.timestamp,
                    "d50": state.cmas.get("d50", 0.0),
                    "lod": state.cmas.get("lod", 0.0),
                    "spray_rate": state.cpps.get("spray_rate", 0.0),
                    "air_flow": state.cpps.get("air_flow", 0.0),
                    "carousel_speed": state.cpps.get("carousel_speed", 0.0),
                    "specific_energy": specific_energy,
                    "froude_number_proxy": froude_number_proxy,
                }

                # Insert with conflict resolution
                conn.execute(
                    """
                    INSERT OR REPLACE INTO process_data 
                    (timestamp, d50, lod, spray_rate, air_flow, carousel_speed, 
                     specific_energy, froude_number_proxy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    tuple(data_row.values()),
                )

            conn.commit()

    def fetch_recent_data(self, duration_hours: int = 24) -> pd.DataFrame:
        """
        Fetch recent operational data for retraining.

        Args:
            duration_hours: Number of hours of recent data to fetch

        Returns:
            pd.DataFrame: Recent process data
        """
        end_time = datetime.now().timestamp()
        start_time = end_time - (duration_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM process_data 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """

            df = pd.read_sql_query(query, conn, params=(start_time, end_time))

        return df

    def fetch_all_data(self) -> pd.DataFrame:
        """Fetch all available data for training."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM process_data ORDER BY timestamp"
            df = pd.read_sql_query(query, conn)
        return df

    def log_training_metrics(self, metrics: TrainingMetrics):
        """
        Log model training metrics to the database.

        Args:
            metrics: Training metrics to log
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_performance 
                (timestamp, model_version, validation_loss, dataset_size, training_duration)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().timestamp(),
                    metrics.model_version,
                    metrics.validation_loss,
                    metrics.dataset_size,
                    metrics.training_duration_seconds,
                ),
            )
            conn.commit()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Process data stats
            process_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_record,
                    MAX(timestamp) as latest_record
                FROM process_data
            """
            ).fetchone()

            # Model performance stats
            model_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_training_runs,
                    COUNT(DISTINCT model_version) as unique_models,
                    AVG(validation_loss) as avg_validation_loss
                FROM model_performance
            """
            ).fetchone()

        return {
            "process_data": {
                "total_records": process_stats[0],
                "time_span_hours": (
                    (process_stats[2] - process_stats[1]) / 3600 if process_stats[1] else 0
                ),
                "earliest_record": (
                    datetime.fromtimestamp(process_stats[1]) if process_stats[1] else None
                ),
                "latest_record": (
                    datetime.fromtimestamp(process_stats[2]) if process_stats[2] else None
                ),
            },
            "model_performance": {
                "total_training_runs": model_stats[0],
                "unique_models": model_stats[1],
                "average_validation_loss": model_stats[2],
            },
        }
