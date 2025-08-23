"""
Type definitions and data contracts for AutoPharm V3.

This module defines all shared data structures used throughout
the AutoPharm framework for clear API contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel


class StateVector(BaseModel):
    """Represents a single timestep of process state"""

    timestamp: float
    cmas: Dict[str, float]  # Critical Material Attributes: {'d50': 380.0, 'lod': 1.8}
    cpps: Dict[
        str, float
    ]  # Critical Process Parameters: {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}

    class Config:
        arbitrary_types_allowed = True


class ControlAction(BaseModel):
    """Control action with metadata"""

    timestamp: float
    cpp_setpoints: Dict[str, float]  # New setpoints to apply
    action_id: str  # Unique identifier for this action
    confidence: float  # Confidence score [0.0, 1.0]

    class Config:
        arbitrary_types_allowed = True


class ModelPrediction(BaseModel):
    """Probabilistic model prediction with uncertainty"""

    mean: np.ndarray  # Shape: (horizon, n_cma_features)
    std: np.ndarray  # Shape: (horizon, n_cma_features)
    horizon: int  # Number of future timesteps predicted
    feature_names: List[str]  # Names of CMA features predicted

    class Config:
        arbitrary_types_allowed = True


class TrainingMetrics(BaseModel):
    """Model training performance metrics"""

    model_version: str
    validation_loss: float
    training_duration_seconds: float
    dataset_size: int
    hyperparameters: Dict[str, Any] = {}


class DecisionExplanation(BaseModel):
    """Human-interpretable explanation for a control decision"""

    decision_id: str
    control_action: ControlAction
    narrative: str  # Human-readable explanation
    feature_attributions: Dict[str, float]  # Feature name -> SHAP value
    confidence_factors: Dict[str, float]  # Factors affecting confidence
    alternatives_considered: int  # Number of alternative actions evaluated
