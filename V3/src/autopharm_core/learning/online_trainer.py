import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# Simplified types for demo (would import from ..common.types in full implementation)
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


# Simplified model for demonstration (would use ProbabilisticTransformer in full implementation)
class SimpleProcessModel(nn.Module):
    """Simplified neural network model for demonstration purposes."""

    def __init__(self, input_features: int = 5, output_features: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_features),
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        """Make predictions (compatibility method)."""
        with torch.no_grad():
            return self.forward(x)


class OnlineTrainer:
    """
    Manages continuous model training, validation, and deployment.
    Handles model versioning and performance monitoring.
    """

    def __init__(self, model_registry_path: str, config: Dict[str, Any]):
        """
        Initialize the online trainer.

        Args:
            model_registry_path: Path to store versioned models
            config: Training configuration
        """
        self.model_registry_path = model_registry_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create registry directory
        os.makedirs(model_registry_path, exist_ok=True)

        # Training history
        self.training_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def should_retrain(
        self, current_performance: Dict[str, float], threshold_config: Dict[str, float]
    ) -> bool:
        """
        Determine if model retraining is needed based on performance metrics.

        Args:
            current_performance: Current model performance metrics
            threshold_config: Performance thresholds for triggering retraining

        Returns:
            bool: True if retraining is recommended
        """
        # Check validation loss degradation
        validation_loss = current_performance.get("validation_loss", float("inf"))
        loss_threshold = threshold_config.get("max_validation_loss", 0.1)

        if validation_loss > loss_threshold:
            print(f"Retraining triggered: validation_loss {validation_loss:.4f} > {loss_threshold}")
            return True

        # Check prediction accuracy
        prediction_accuracy = current_performance.get("prediction_accuracy", 0.0)
        accuracy_threshold = threshold_config.get("min_prediction_accuracy", 0.85)

        if prediction_accuracy < accuracy_threshold:
            print(
                f"Retraining triggered: prediction_accuracy {prediction_accuracy:.4f} < {accuracy_threshold}"
            )
            return True

        # Check time since last training
        last_training_time = current_performance.get("last_training_timestamp", 0)
        current_time = datetime.now().timestamp()
        max_training_interval = threshold_config.get("max_training_interval_hours", 24) * 3600

        if (current_time - last_training_time) > max_training_interval:
            print(f"Retraining triggered: training interval exceeded")
            return True

        return False

    def run_training_job(
        self, training_data: pd.DataFrame, current_model: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, TrainingMetrics]:
        """
        Execute a complete training and validation run.

        Args:
            training_data: Training dataset
            current_model: Existing model to fine-tune (optional)

        Returns:
            Tuple[nn.Module, TrainingMetrics]: (new_model, metrics)
        """
        start_time = datetime.now()
        print(f"Starting training job with {len(training_data)} training samples")

        # Prepare datasets
        train_loader, val_loader = self._prepare_dataloaders(training_data)

        # Initialize or load model
        if current_model is None:
            model = self._initialize_new_model()
        else:
            model = current_model

        model = model.to(self.device)

        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = self.config.get("early_stopping_patience", 10)

        for epoch in range(self.config.get("max_epochs", 50)):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Restore best model
        model.load_state_dict(best_model_state)

        # Final validation
        final_val_loss, final_metrics = self._validate_epoch(model, val_loader, criterion)

        # Create training metrics
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        model_version = f"v{int(datetime.now().timestamp())}"

        training_metrics = TrainingMetrics(
            model_version=model_version,
            validation_loss=final_val_loss,
            training_duration_seconds=training_duration,
            dataset_size=len(training_data),
        )

        # Save model
        self._save_model(model, model_version, training_metrics)

        # Update training history
        self.training_history.append(
            {
                "timestamp": end_time.timestamp(),
                "metrics": training_metrics,
                "final_metrics": final_metrics,
            }
        )

        print(f"Training completed: {model_version}, val_loss={final_val_loss:.4f}")

        return model, training_metrics

    def _prepare_dataloaders(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare PyTorch dataloaders from pandas DataFrame."""

        # Chronological split
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size].copy()
        val_data = data.iloc[train_size:].copy()

        # Scale features
        input_columns = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]
        output_columns = ["d50", "lod"]

        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        # Fit on training data only
        train_inputs_scaled = self.input_scaler.fit_transform(train_data[input_columns])
        train_outputs_scaled = self.output_scaler.fit_transform(train_data[output_columns])

        val_inputs_scaled = self.input_scaler.transform(val_data[input_columns])
        val_outputs_scaled = self.output_scaler.transform(val_data[output_columns])

        # Create tensors
        train_inputs_tensor = torch.tensor(train_inputs_scaled, dtype=torch.float32)
        train_outputs_tensor = torch.tensor(train_outputs_scaled, dtype=torch.float32)
        val_inputs_tensor = torch.tensor(val_inputs_scaled, dtype=torch.float32)
        val_outputs_tensor = torch.tensor(val_outputs_scaled, dtype=torch.float32)

        # Create datasets
        train_dataset = TensorDataset(train_inputs_tensor, train_outputs_tensor)
        val_dataset = TensorDataset(val_inputs_tensor, val_outputs_tensor)

        # Create dataloaders
        batch_size = self.config.get("batch_size", 32)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _initialize_new_model(self) -> nn.Module:
        """Initialize a new model with configured hyperparameters."""
        model_config = self.config.get("model_hyperparameters", {})

        model = SimpleProcessModel(
            input_features=5,  # CPPs + soft sensors
            output_features=2,  # CMAs
            hidden_dim=model_config.get("hidden_dim", 64),
        )

        return model

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(
        self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Store for metrics calculation
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Calculate additional metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = {
            "mse": mean_squared_error(all_targets, all_predictions),
            "mae": mean_absolute_error(all_targets, all_predictions),
        }

        avg_loss = total_loss / num_batches
        return avg_loss, metrics

    def _save_model(self, model: nn.Module, model_version: str, metrics: TrainingMetrics):
        """Save model and metadata to registry."""
        model_dir = os.path.join(self.model_registry_path, model_version)
        os.makedirs(model_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # Save scalers
        scalers = {"input_scaler": self.input_scaler, "output_scaler": self.output_scaler}
        scalers_path = os.path.join(model_dir, "scalers.pkl")
        joblib.dump(scalers, scalers_path)

        # Save training metrics
        metrics_path = os.path.join(model_dir, "training_metrics.pkl")
        joblib.dump(metrics, metrics_path)

        print(f"Model saved: {model_path}")

    def get_training_history(self) -> list:
        """Get complete training history."""
        return self.training_history.copy()

    def get_best_model_version(self) -> Optional[str]:
        """Get the version string of the best performing model."""
        if not self.training_history:
            return None

        best_entry = min(self.training_history, key=lambda x: x["metrics"].validation_loss)

        return best_entry["metrics"].model_version
