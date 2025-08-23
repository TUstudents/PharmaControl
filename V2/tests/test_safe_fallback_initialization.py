"""
Test suite for safe fallback action initialization in RobustMPCController.

This module validates the critical safety fix for unsafe None initialization
of fallback actions, ensuring pharmaceutical process control is safe from
the very first control step.
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


class TestSafeFallbackInitialization:
    """Test suite for safe fallback action initialization."""

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
        """Standard test configuration with known constraint bounds."""
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

        # CPP scalers (including all soft sensors)
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

    def test_fallback_action_initialized_on_startup(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that fallback action is properly initialized during controller creation."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # Verify fallback action is not None after initialization
        assert (
            controller._last_successful_action is not None
        ), "Fallback action should be initialized, not None"

        # Verify it's a proper numpy array
        assert isinstance(
            controller._last_successful_action, np.ndarray
        ), "Fallback action should be numpy array"

        # Verify correct dimensions
        expected_size = len(test_config["cpp_names"])
        assert controller._last_successful_action.shape == (
            expected_size,
        ), f"Fallback action shape should be ({expected_size},), got {controller._last_successful_action.shape}"

        # Verify all values are finite
        assert np.all(
            np.isfinite(controller._last_successful_action)
        ), "All fallback action values should be finite"

    def test_safe_default_action_values(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that safe default action uses correct constraint midpoints."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        fallback_action = controller._last_successful_action

        # Expected midpoint values
        expected_spray_rate = (80.0 + 200.0) / 2.0  # 140.0
        expected_air_flow = (400.0 + 700.0) / 2.0  # 550.0
        expected_carousel_speed = (20.0 + 50.0) / 2.0  # 35.0

        expected_values = np.array(
            [expected_spray_rate, expected_air_flow, expected_carousel_speed]
        )

        # Verify calculated values match expected midpoints
        np.testing.assert_array_equal(
            fallback_action,
            expected_values,
            err_msg="Safe default action should use constraint midpoints",
        )

    def test_calculate_safe_default_action_method(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test the _calculate_safe_default_action method directly."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # Call method directly
        safe_action = controller._calculate_safe_default_action()

        # Verify properties
        assert isinstance(safe_action, np.ndarray), "Should return numpy array"
        assert safe_action.shape == (len(test_config["cpp_names"]),), "Correct shape"
        assert np.all(np.isfinite(safe_action)), "All values should be finite"

        # Verify values are within constraint bounds
        cpp_config = test_config["cpp_constraints"]
        for i, name in enumerate(test_config["cpp_names"]):
            min_val = cpp_config[name]["min_val"]
            max_val = cpp_config[name]["max_val"]
            value = safe_action[i]

            assert (
                min_val <= value <= max_val
            ), f"{name} value {value} not within bounds [{min_val}, {max_val}]"

            # Verify it's exactly the midpoint
            expected_midpoint = (min_val + max_val) / 2.0
            assert np.isclose(
                value, expected_midpoint
            ), f"{name} should be midpoint {expected_midpoint}, got {value}"

    def test_first_step_optimizer_failure_safety(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """CRITICAL TEST: Verify safe fallback on very first control step when optimizer fails."""

        # Create always-failing optimizer for worst-case scenario
        class AlwaysFailingOptimizer:
            def __init__(self, param_bounds, config):
                pass

            def optimize(self, fitness_func=None):
                raise RuntimeError("Optimizer failure on first step")

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=AlwaysFailingOptimizer,
            config=test_config,
            scalers=test_scalers,
        )

        # Test first control step with optimizer failure
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])

        # This should NOT fail and should return safe fallback action
        action = controller.suggest_action(measurement, control_input, setpoint)

        # Verify we got a valid action
        assert isinstance(action, np.ndarray), "Should return valid numpy array"
        assert action.shape == (len(test_config["cpp_names"]),), "Correct action shape"
        assert np.all(np.isfinite(action)), "All action values should be finite"

        # Verify action satisfies constraints
        cpp_config = test_config["cpp_constraints"]
        for i, name in enumerate(test_config["cpp_names"]):
            min_val = cpp_config[name]["min_val"]
            max_val = cpp_config[name]["max_val"]
            value = action[i]

            assert (
                min_val <= value <= max_val
            ), f"Action {name}={value} violates constraints [{min_val}, {max_val}]"

        # Verify it used the pre-initialized safe defaults (constraint midpoints)
        expected_action = np.array([140.0, 550.0, 35.0])  # Midpoints
        np.testing.assert_array_equal(
            action,
            expected_action,
            err_msg="Should use pre-initialized safe defaults on first step optimizer failure",
        )

    def test_fallback_action_validation(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that fallback action passes controller's own validation."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # The pre-initialized fallback action should pass validation
        fallback_action = controller._last_successful_action
        is_valid = controller._validate_control_action(fallback_action)

        assert is_valid, "Pre-initialized fallback action should pass controller validation"

    def test_missing_constraint_configuration_error(self, mock_model, mock_estimator, test_scalers):
        """Test error handling when constraints are missing for required CPPs."""
        # Config missing constraints for carousel_speed
        bad_config = {
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
            "cpp_constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 200.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                # Missing carousel_speed constraints
            },
            "ga_config": {
                "population_size": 20,
                "num_generations": 5,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
            },
        }

        with pytest.raises(
            ValueError, match="Missing constraint configuration for CPP 'carousel_speed'"
        ):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=bad_config,
                scalers=test_scalers,
            )

    def test_invalid_constraint_bounds_error(self, mock_model, mock_estimator, test_scalers):
        """Test error handling when constraint bounds are invalid (min >= max)."""
        # Config with invalid bounds (min >= max)
        bad_config = {
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
            "cpp_constraints": {
                "spray_rate": {"min_val": 200.0, "max_val": 80.0},  # Invalid: min > max
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

        with pytest.raises(
            ValueError,
            match="Invalid constraint bounds for 'spray_rate'.*min_val=200.0 >= max_val=80.0",
        ):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=bad_config,
                scalers=test_scalers,
            )

    def test_get_fallback_action_uses_new_method(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test that _get_fallback_action properly leverages the new safe default calculation."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # Test fallback when no current input provided (worst case)
        fallback_action = controller._get_fallback_action(None)

        # Should get safe defaults (constraint midpoints)
        expected_action = np.array([140.0, 550.0, 35.0])
        np.testing.assert_array_equal(
            fallback_action,
            expected_action,
            err_msg="Fallback action should use safe default calculation method",
        )

    def test_fallback_strategy_hierarchy(
        self, mock_model, mock_estimator, test_config, test_scalers
    ):
        """Test the three-tier fallback strategy works correctly."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers,
        )

        # Strategy 1: Should use pre-initialized last successful action (safe defaults)
        fallback_1 = controller._get_fallback_action(None)
        expected_safe_defaults = np.array([140.0, 550.0, 35.0])
        np.testing.assert_array_equal(fallback_1, expected_safe_defaults)

        # Strategy 2: Should use valid current input
        valid_current_input = np.array([100.0, 500.0, 25.0])
        fallback_2 = controller._get_fallback_action(valid_current_input)
        np.testing.assert_array_equal(fallback_2, expected_safe_defaults)  # Strategy 1 wins

        # Temporarily set last successful action to None to test Strategy 2
        original_action = controller._last_successful_action
        controller._last_successful_action = None
        fallback_3 = controller._get_fallback_action(valid_current_input)
        np.testing.assert_array_equal(fallback_3, valid_current_input)  # Strategy 2 used

        # Restore
        controller._last_successful_action = original_action


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
