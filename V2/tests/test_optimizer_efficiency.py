"""
Test suite for optimizer efficiency improvements in RobustMPCController.

This module validates that the optimizer is properly instantiated as an instance
variable and reused across control steps, rather than being created fresh each time.
"""

import pytest
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from V2.robust_mpc.core import RobustMPCController
from V2.robust_mpc.optimizers import GeneticOptimizer


class TestOptimizerEfficiency:
    """Test suite for optimizer efficiency improvements."""

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""

        class MockModel:
            def to(self, device):
                return self

            def predict_distribution(self, *args, **kwargs):
                return np.zeros((1, 10, 2)), np.ones((1, 10, 2)) * 0.1

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
        """Standard test configuration."""
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
            "horizon": 10,
            "lookback": 15,
            "integral_gain": 0.1,
            "mc_samples": 30,
            "risk_beta": 1.5,
            "verbose": False,
            "history_buffer_size": 50,
            "cpp_constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 200.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 50.0},
            },
            "ga_config": {
                "population_size": 30,
                "num_generations": 5,  # Small for fast testing
                "cx_prob": 0.7,
                "mut_prob": 0.2,
            },
        }

    @pytest.fixture
    def test_scalers(self):
        """Realistic pharmaceutical scalers."""
        scalers = {}

        # CMA scalers
        d50_scaler = MinMaxScaler()
        d50_scaler.fit(np.array([[300], [600]]))
        scalers["d50"] = d50_scaler

        lod_scaler = MinMaxScaler()
        lod_scaler.fit(np.array([[0.5], [3.0]]))
        scalers["lod"] = lod_scaler

        # CPP scalers
        for name in [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]:
            scaler = MinMaxScaler()
            scaler.fit(np.array([[0], [100]]))
            scalers[name] = scaler

        return scalers

    def test_optimizer_instance_creation(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that optimizer is created as instance variable."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify optimizer is created as instance variable
        assert hasattr(controller, "optimizer")
        assert controller.optimizer is not None
        assert isinstance(controller.optimizer, GeneticOptimizer)

        # Verify optimizer has correct configuration (includes added horizon and num_cpps)
        expected_config = test_config["ga_config"].copy()
        expected_config["horizon"] = test_config["horizon"]
        expected_config["num_cpps"] = len(test_config["cpp_names"])
        assert controller.optimizer.config == expected_config
        assert len(controller.optimizer.param_bounds) == test_config["horizon"] * len(
            test_config["cpp_names"]
        )

    def test_optimizer_reuse_across_steps(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that the same optimizer instance is reused across control steps."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Store reference to original optimizer
        original_optimizer = controller.optimizer
        optimizer_id = id(controller.optimizer)

        # Perform multiple control steps
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        for i in range(3):
            controller.suggest_action(measurement, control_input, setpoint)

            # Verify same optimizer instance is still being used
            assert controller.optimizer is original_optimizer
            assert id(controller.optimizer) == optimizer_id

    def test_fitness_function_injection(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that fitness function is properly injected at optimization time."""

        # Create tracking optimizer to verify fitness function passing
        class TrackingOptimizer:
            def __init__(self, param_bounds, config):
                self.param_bounds = param_bounds
                self.config = config
                self.fitness_functions_received = []

            def optimize(self, fitness_func=None):
                self.fitness_functions_received.append(fitness_func)
                return np.array([[120.0, 500.0, 25.0]] * test_config["horizon"])

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=TrackingOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Perform control steps with different setpoints
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoints = [np.array([450.0, 1.8]), np.array([460.0, 1.9]), np.array([440.0, 1.7])]

        for setpoint in setpoints:
            controller.suggest_action(measurement, control_input, setpoint)

        # Verify fitness functions were passed to optimizer
        assert len(controller.optimizer.fitness_functions_received) == len(setpoints)
        for fitness_func in controller.optimizer.fitness_functions_received:
            assert callable(fitness_func)

    def test_performance_improvement(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test that optimizer reuse provides performance improvement."""

        # Create instrumented optimizer to track instantiation overhead
        class InstrumentedOptimizer:
            instantiation_count = 0

            def __init__(self, param_bounds, config):
                InstrumentedOptimizer.instantiation_count += 1
                self.param_bounds = param_bounds
                self.config = config
                self.instantiation_time = time.time()

            def optimize(self, fitness_func=None):
                return np.array([[120.0, 500.0, 25.0]] * test_config["horizon"])

        # Reset counter
        InstrumentedOptimizer.instantiation_count = 0

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=InstrumentedOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify only one instantiation during controller creation
        assert InstrumentedOptimizer.instantiation_count == 1

        # Perform multiple control steps
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        num_steps = 5
        for i in range(num_steps):
            controller.suggest_action(measurement, control_input, setpoint)

        # Verify no additional instantiations occurred
        assert (
            InstrumentedOptimizer.instantiation_count == 1
        ), f"Expected 1 instantiation, got {InstrumentedOptimizer.instantiation_count}"

    def test_backward_compatibility_with_none_optimizer(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that None optimizer_class is handled gracefully."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify optimizer is None
        assert controller.optimizer is None

        # Verify suggest_action works and returns fallback
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        result = controller.suggest_action(measurement, control_input, setpoint)

        # Should return fallback action (safe control values)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(test_config["cpp_names"]),)
        assert np.all(np.isfinite(result))

    def test_genetic_optimizer_fitness_function_handling(self, test_config):
        """Test GeneticOptimizer with new fitness function injection interface."""

        def test_fitness_function(control_plan):
            """Simple test fitness function."""
            return np.sum(control_plan**2)

        # Create parameter bounds for testing
        param_bounds = [(80, 200), (400, 700), (20, 50)] * test_config["horizon"]

        # Create complete GA config with required parameters
        ga_config = test_config["ga_config"].copy()
        ga_config["horizon"] = test_config["horizon"]
        ga_config["num_cpps"] = len(test_config["cpp_names"])

        # Test new interface - no fitness function in init
        optimizer = GeneticOptimizer(param_bounds, ga_config)

        # Test optimization with fitness function passed at runtime
        result = optimizer.optimize(test_fitness_function)

        # Verify result shape and validity
        expected_shape = (test_config["horizon"], len(test_config["cpp_names"]))
        assert result.shape == expected_shape
        assert np.all(np.isfinite(result))

        # Verify bounds are respected
        for i, (min_val, max_val) in enumerate(param_bounds):
            flat_result = result.flatten()
            assert min_val <= flat_result[i] <= max_val

    def test_optimizer_error_handling(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test optimizer error handling with instance variables."""

        # Create optimizer that always fails
        class FailingOptimizer:
            def __init__(self, param_bounds, config):
                self.param_bounds = param_bounds
                self.config = config

            def optimize(self, fitness_func=None):
                raise RuntimeError("Optimizer failed intentionally")

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=FailingOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify optimizer is created
        assert isinstance(controller.optimizer, FailingOptimizer)

        # Test fallback behavior when optimizer fails
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        result = controller.suggest_action(measurement, control_input, setpoint)

        # Should return safe fallback action
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(test_config["cpp_names"]),)
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
