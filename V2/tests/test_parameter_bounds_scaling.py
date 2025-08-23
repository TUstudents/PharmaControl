"""
Test suite for parameter bounds scaling fix in RobustMPCController.

This module validates that the critical bug in parameter bounds scaling has been
fixed and that the GeneticOptimizer now works correctly in scaled space.
"""

import os
import sys

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from V2.robust_mpc.core import RobustMPCController
from V2.robust_mpc.optimizers import GeneticOptimizer


class TestParameterBoundsScaling:
    """Test suite for parameter bounds scaling functionality."""

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
        """Standard test configuration with known constraints."""
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
                "population_size": 20,
                "num_generations": 5,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
            },
        }

    @pytest.fixture
    def test_scalers(self):
        """Realistic pharmaceutical scalers with known ranges."""
        scalers = {}

        # CMA scalers
        d50_scaler = MinMaxScaler()
        d50_scaler.fit(np.array([[300], [600]]))
        scalers["d50"] = d50_scaler

        lod_scaler = MinMaxScaler()
        lod_scaler.fit(np.array([[0.5], [3.0]]))
        scalers["lod"] = lod_scaler

        # CPP scalers with known ranges for testing
        spray_scaler = MinMaxScaler()
        spray_scaler.fit(np.array([[50], [250]]))  # Range: 50-250 g/min
        scalers["spray_rate"] = spray_scaler

        air_scaler = MinMaxScaler()
        air_scaler.fit(np.array([[300], [800]]))  # Range: 300-800 m³/h
        scalers["air_flow"] = air_scaler

        speed_scaler = MinMaxScaler()
        speed_scaler.fit(np.array([[10], [60]]))  # Range: 10-60 rpm
        scalers["carousel_speed"] = speed_scaler

        # Soft sensor scalers
        for name in ["specific_energy", "froude_number_proxy"]:
            scaler = MinMaxScaler()
            scaler.fit(np.array([[0], [100]]))
            scalers[name] = scaler

        return scalers

    def test_parameter_bounds_are_scaled(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that _get_param_bounds() returns scaled bounds in [0,1] range."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        param_bounds = controller._get_param_bounds()

        # Verify structure
        expected_num_bounds = test_config["horizon"] * len(test_config["cpp_names"])
        assert len(param_bounds) == expected_num_bounds

        # Verify all bounds are in scaled space [0,1]
        for min_bound, max_bound in param_bounds:
            assert 0.0 <= min_bound <= 1.0, f"Min bound {min_bound} not in [0,1]"
            assert 0.0 <= max_bound <= 1.0, f"Max bound {max_bound} not in [0,1]"
            assert min_bound <= max_bound, f"Min {min_bound} > Max {max_bound}"

    def test_parameter_bounds_scaling_correctness(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that parameter bounds are correctly scaled using fitted scalers."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        param_bounds = controller._get_param_bounds()

        # Check specific scaling for first set of parameters
        # spray_rate: constraint 80-200, scaler range 50-250
        spray_min_expected = test_scalers["spray_rate"].transform([[80.0]])[0, 0]
        spray_max_expected = test_scalers["spray_rate"].transform([[200.0]])[0, 0]

        spray_min_actual, spray_max_actual = param_bounds[0]  # First parameter
        assert np.isclose(
            spray_min_actual, spray_min_expected
        ), f"Spray rate min: expected {spray_min_expected}, got {spray_min_actual}"
        assert np.isclose(
            spray_max_actual, spray_max_expected
        ), f"Spray rate max: expected {spray_max_expected}, got {spray_max_actual}"

        # air_flow: constraint 400-700, scaler range 300-800
        air_min_expected = test_scalers["air_flow"].transform([[400.0]])[0, 0]
        air_max_expected = test_scalers["air_flow"].transform([[700.0]])[0, 0]

        air_min_actual, air_max_actual = param_bounds[1]  # Second parameter
        assert np.isclose(
            air_min_actual, air_min_expected
        ), f"Air flow min: expected {air_min_expected}, got {air_min_actual}"
        assert np.isclose(
            air_max_actual, air_max_expected
        ), f"Air flow max: expected {air_max_expected}, got {air_max_actual}"

    def test_unscale_cpp_plan_correctness(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that _unscale_cpp_plan() correctly reverses scaling."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Create test scaled plan (values in [0,1])
        horizon = test_config["horizon"]
        num_cpps = len(test_config["cpp_names"])
        scaled_plan = np.random.rand(horizon, num_cpps)  # Random values in [0,1]

        # Unscale the plan
        unscaled_plan = controller._unscale_cpp_plan(scaled_plan)

        # Verify shape
        assert unscaled_plan.shape == scaled_plan.shape

        # Verify that unscaling is correct by re-scaling and comparing
        for i, cpp_name in enumerate(test_config["cpp_names"]):
            scaler = test_scalers[cpp_name]

            # Manual unscaling for comparison
            manual_unscaled = scaler.inverse_transform(scaled_plan[:, i].reshape(-1, 1)).flatten()

            assert np.allclose(
                unscaled_plan[:, i], manual_unscaled
            ), f"Unscaling incorrect for {cpp_name}"

    def test_scaling_roundtrip_consistency(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that scaling -> unscaling is consistent."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Create test unscaled plan with constraint values
        horizon = test_config["horizon"]
        original_plan = np.array(
            [
                [120.0, 500.0, 35.0],  # Within constraints
                [150.0, 600.0, 40.0],
                [100.0, 450.0, 25.0],
                [180.0, 650.0, 45.0],
                [90.0, 550.0, 30.0],
            ]
        )

        # Scale then unscale
        scaled_plan = controller._scale_cpp_plan(original_plan)
        roundtrip_plan = controller._unscale_cpp_plan(scaled_plan)

        # Should be very close to original
        assert np.allclose(
            original_plan, roundtrip_plan, atol=1e-10
        ), "Scaling roundtrip not consistent"

    def test_optimizer_with_scaled_bounds(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that optimizer works correctly with scaled parameter bounds."""

        # Create tracking optimizer to verify bounds and solution range
        class BoundsTrackingOptimizer:
            def __init__(self, param_bounds, config):
                self.param_bounds = param_bounds
                self.config = config
                self.solutions_generated = []

            def optimize(self, fitness_func=None):
                # Generate solution within bounds
                horizon = self.config["horizon"]
                num_cpps = self.config["num_cpps"]

                solution = np.zeros((horizon, num_cpps))
                param_idx = 0

                for h in range(horizon):
                    for c in range(num_cpps):
                        min_bound, max_bound = self.param_bounds[param_idx]
                        # Generate value within scaled bounds
                        solution[h, c] = np.random.uniform(min_bound, max_bound)
                        param_idx += 1

                self.solutions_generated.append(solution.copy())
                return solution

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=BoundsTrackingOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Test control step
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        action = controller.suggest_action(measurement, control_input, setpoint)

        # Verify optimizer received scaled bounds
        optimizer = controller.optimizer
        for min_bound, max_bound in optimizer.param_bounds:
            assert 0.0 <= min_bound <= 1.0
            assert 0.0 <= max_bound <= 1.0

        # Verify optimizer generated scaled solution
        assert len(optimizer.solutions_generated) == 1
        scaled_solution = optimizer.solutions_generated[0]
        assert np.all((scaled_solution >= 0.0) & (scaled_solution <= 1.0))

        # Verify controller returned unscaled action
        assert isinstance(action, np.ndarray)
        assert action.shape == (len(test_config["cpp_names"]),)

        # Action should be in physical units (outside [0,1] for our test scalers)
        # Since our constraint ranges don't map to [0,1] in physical space
        spray_rate, air_flow, carousel_speed = action
        assert 50 <= spray_rate <= 250  # Physical scaler range
        assert 300 <= air_flow <= 800  # Physical scaler range
        assert 10 <= carousel_speed <= 60  # Physical scaler range

    def test_fitness_function_with_scaled_inputs(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that fitness function correctly handles scaled control plans."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Setup mock data
        past_cmas_scaled = np.random.rand(15, 2)
        past_cpps_scaled = np.random.rand(15, 5)  # Including soft sensors
        target_cmas_unscaled = np.array([450.0, 1.8])

        # Get fitness function
        fitness_func = controller._get_fitness_function(
            past_cmas_scaled, past_cpps_scaled, target_cmas_unscaled
        )

        # Create scaled control plan (what GA would provide)
        horizon = test_config["horizon"]
        num_cpps = len(test_config["cpp_names"])
        scaled_control_plan = np.random.rand(horizon, num_cpps)

        # Fitness function should handle scaled input without error
        try:
            cost = fitness_func(scaled_control_plan)
            assert isinstance(cost, (float, int))
            assert cost >= 0.0  # Cost should be non-negative
        except Exception as e:
            pytest.fail(f"Fitness function failed with scaled input: {e}")

    def test_error_handling_missing_scalers(self, mock_model, mock_estimator, test_config):
        """Test error handling when scalers are missing."""
        # Create incomplete scalers (missing spray_rate)
        incomplete_scalers = {
            "d50": MinMaxScaler().fit(np.array([[300], [600]])),
            "lod": MinMaxScaler().fit(np.array([[0.5], [3.0]])),
            "air_flow": MinMaxScaler().fit(np.array([[300], [800]])),
            "carousel_speed": MinMaxScaler().fit(np.array([[10], [60]])),
            # Missing 'spray_rate' scaler
        }

        # Should raise error during initialization
        with pytest.raises(ValueError, match="Missing scalers for.*spray_rate"):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=GeneticOptimizer,
                config=test_config,
                scalers=incomplete_scalers,
            )

    def test_bounds_ordering_with_reversed_scalers(self, mock_model, mock_estimator, test_config):
        """Test that bounds are properly ordered even with reversed scaler ranges."""
        # Create scaler with reversed range (max < min in fitted data)
        reversed_scalers = {
            "d50": MinMaxScaler().fit(np.array([[300], [600]])),
            "lod": MinMaxScaler().fit(np.array([[0.5], [3.0]])),
            "spray_rate": MinMaxScaler().fit(np.array([[250], [50]])),  # Reversed!
            "air_flow": MinMaxScaler().fit(np.array([[300], [800]])),
            "carousel_speed": MinMaxScaler().fit(np.array([[10], [60]])),
        }

        for name in ["specific_energy", "froude_number_proxy"]:
            reversed_scalers[name] = MinMaxScaler().fit(np.array([[0], [100]]))

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=GeneticOptimizer,
            config=test_config,
            scalers=reversed_scalers,
        )

        param_bounds = controller._get_param_bounds()

        # All bounds should still be properly ordered
        for min_bound, max_bound in param_bounds:
            assert min_bound <= max_bound, f"Bounds not ordered: {min_bound} > {max_bound}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
