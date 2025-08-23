"""
Test suite for intelligent optimizer reset functionality in RobustMPCController.

This module validates the optimizer reset mechanism that provides fresh GA
populations when setpoints change significantly, improving pharmaceutical
process control optimization quality.
"""

import pytest
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from V2.robust_mpc.core import RobustMPCController


class TestOptimizerReset:
    """Test suite for optimizer reset functionality."""

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""

        class MockModel:
            def to(self, device):
                return self

            def predict_distribution(self, *args, **kwargs):
                return torch.zeros(1, 10, 2), torch.ones(1, 10, 2) * 0.1

        return MockModel()

    @pytest.fixture
    def mock_estimator(self):
        """Mock estimator for testing."""

        class MockEstimator:
            def estimate(self, measurement, control):
                return measurement

        return MockEstimator()

    @pytest.fixture
    def test_config(self):
        """Standard test configuration with optimizer reset parameters."""
        return {
            "cma_names": ["d50", "lod"],
            "cpp_names": ["spray_rate", "air_flow", "carousel_speed"],
            "cpp_full_names": [
                "spray_rate",
                "air_flow",
                "carousel_speed",
                "specific_energy",
                "froude_number_proxy",
            ],
            "horizon": 5,
            "lookback": 15,
            "integral_gain": 0.1,
            "mc_samples": 10,
            "risk_beta": 1.5,
            "verbose": False,
            "history_buffer_size": 50,
            "reset_optimizer_on_setpoint_change": True,
            "setpoint_change_threshold": 0.05,  # 5% relative change
            "cpp_constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 200.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 50.0},
            },
            "ga_config": {
                "population_size": 20,
                "num_generations": 5,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
            },
        }

    @pytest.fixture
    def test_scalers(self):
        """Comprehensive scalers for testing."""
        scalers = {}

        # CMA scalers
        scalers["d50"] = MinMaxScaler().fit(np.array([[300], [600]]))
        scalers["lod"] = MinMaxScaler().fit(np.array([[0.5], [3.0]]))

        # CPP scalers
        cpp_vars = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]
        for var in cpp_vars:
            scalers[var] = MinMaxScaler().fit(np.array([[0], [1000]]))

        return scalers

    @pytest.fixture
    def tracking_optimizer(self):
        """Create tracking optimizer to monitor reset behavior."""

        class TrackingOptimizer:
            def __init__(self, param_bounds, config):
                self.param_bounds = param_bounds
                self.config = config
                self.creation_count = 0
                self.optimization_count = 0
                TrackingOptimizer.total_creations = (
                    getattr(TrackingOptimizer, "total_creations", 0) + 1
                )
                self.instance_id = TrackingOptimizer.total_creations

            def optimize(self, fitness_func=None):
                self.optimization_count += 1
                # Return valid scaled solution within bounds
                horizon = self.config["horizon"]
                num_cpps = self.config["num_cpps"]
                solution = np.random.rand(horizon, num_cpps) * 0.5 + 0.25  # Values in [0.25, 0.75]
                return solution

        return TrackingOptimizer

    def test_setpoint_tracking_initialization(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test that setpoint tracking is properly initialized."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify setpoint tracking is initialized to None
        assert controller._last_setpoint is None, "Setpoint tracking should be initialized to None"

    def test_setpoint_change_detection_small(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test that small setpoint changes do not trigger optimizer reset."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # First setpoint
        setpoint1 = np.array([450.0, 1.8])
        should_reset1 = controller._should_reset_optimizer(setpoint1)
        assert not should_reset1, "Should not reset on first setpoint"

        # Update tracking
        controller._last_setpoint = setpoint1.copy()

        # Small change (2% change, below 5% threshold)
        setpoint2 = np.array([459.0, 1.836])  # ~2% change
        should_reset2 = controller._should_reset_optimizer(setpoint2)
        assert not should_reset2, "Should not reset for small setpoint changes"

    def test_setpoint_change_detection_large(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test that large setpoint changes trigger optimizer reset."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Set initial setpoint
        setpoint1 = np.array([450.0, 1.8])
        controller._last_setpoint = setpoint1.copy()

        # Large change (10% change, above 5% threshold)
        setpoint2 = np.array([495.0, 1.98])  # ~10% change
        should_reset2 = controller._should_reset_optimizer(setpoint2)
        assert should_reset2, "Should reset for large setpoint changes"

    def test_optimizer_reset_disabled(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test behavior when optimizer reset is disabled."""
        config = test_config.copy()
        config["reset_optimizer_on_setpoint_change"] = False

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=config,
            scalers=test_scalers,
        )

        # Set initial setpoint
        setpoint1 = np.array([450.0, 1.8])
        controller._last_setpoint = setpoint1.copy()

        # Large change that would normally trigger reset
        setpoint2 = np.array([600.0, 2.5])  # Large change
        should_reset = controller._should_reset_optimizer(setpoint2)
        assert not should_reset, "Should not reset when feature is disabled"

    def test_optimizer_reset_mechanism(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test that _reset_optimizer() creates new optimizer instance."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Get initial optimizer instance ID
        initial_instance_id = controller.optimizer.instance_id

        # Reset optimizer
        controller._reset_optimizer()

        # Verify new instance was created
        new_instance_id = controller.optimizer.instance_id
        assert new_instance_id != initial_instance_id, "Reset should create new optimizer instance"

    def test_integrated_suggest_action_with_reset(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test full suggest_action integration with optimizer reset."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Test inputs
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])

        # Initial optimizer instance
        initial_instance_id = controller.optimizer.instance_id

        # First control step
        setpoint1 = np.array([450.0, 1.8])
        action1 = controller.suggest_action(measurement, control_input, setpoint1)

        # Should still have same optimizer (no reset on first call)
        assert controller.optimizer.instance_id == initial_instance_id
        assert isinstance(action1, np.ndarray)
        assert action1.shape == (3,)

        # Second control step with large setpoint change
        setpoint2 = np.array([500.0, 2.0])  # Significant change
        action2 = controller.suggest_action(measurement, control_input, setpoint2)

        # Should have new optimizer instance (reset occurred)
        assert controller.optimizer.instance_id != initial_instance_id
        assert isinstance(action2, np.ndarray)
        assert action2.shape == (3,)

    def test_configuration_validation(
        self, mock_model, mock_estimator, test_scalers, tracking_optimizer
    ):
        """Test validation of optimizer reset configuration parameters."""

        # Test invalid setpoint_change_threshold
        invalid_config1 = {
            "cma_names": ["d50", "lod"],
            "cpp_names": ["spray_rate", "air_flow", "carousel_speed"],
            "cpp_full_names": [
                "spray_rate",
                "air_flow",
                "carousel_speed",
                "specific_energy",
                "froude_number_proxy",
            ],
            "horizon": 5,
            "lookback": 15,
            "setpoint_change_threshold": -0.1,  # Invalid: negative
            "cpp_constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 200.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 50.0},
            },
            "ga_config": {"population_size": 10, "num_generations": 3},
        }

        with pytest.raises(
            ValueError, match="'setpoint_change_threshold' must be a positive number"
        ):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=tracking_optimizer,
                config=invalid_config1,
                scalers=test_scalers,
            )

        # Test invalid reset_optimizer_on_setpoint_change
        invalid_config2 = {
            "cma_names": ["d50", "lod"],
            "cpp_names": ["spray_rate", "air_flow", "carousel_speed"],
            "cpp_full_names": [
                "spray_rate",
                "air_flow",
                "carousel_speed",
                "specific_energy",
                "froude_number_proxy",
            ],
            "horizon": 5,
            "lookback": 15,
            "reset_optimizer_on_setpoint_change": "yes",  # Invalid: not boolean
            "cpp_constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 200.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 50.0},
            },
            "ga_config": {"population_size": 10, "num_generations": 3},
        }

        with pytest.raises(
            ValueError, match="'reset_optimizer_on_setpoint_change' must be a boolean"
        ):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=tracking_optimizer,
                config=invalid_config2,
                scalers=test_scalers,
            )

    def test_custom_threshold_configuration(
        self, mock_model, mock_estimator, test_config, test_scalers, tracking_optimizer
    ):
        """Test custom setpoint change threshold configuration."""
        # Set custom threshold to 10%
        config = test_config.copy()
        config["setpoint_change_threshold"] = 0.10

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=tracking_optimizer,
            config=config,
            scalers=test_scalers,
        )

        # Set initial setpoint
        setpoint1 = np.array([450.0, 1.8])
        controller._last_setpoint = setpoint1.copy()

        # 7% change (below 10% threshold)
        setpoint2 = np.array([481.5, 1.926])
        should_reset = controller._should_reset_optimizer(setpoint2)
        assert not should_reset, "Should not reset with 7% change when threshold is 10%"

        # 12% change (above 10% threshold)
        setpoint3 = np.array([504.0, 2.016])
        should_reset = controller._should_reset_optimizer(setpoint3)
        assert should_reset, "Should reset with 12% change when threshold is 10%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
