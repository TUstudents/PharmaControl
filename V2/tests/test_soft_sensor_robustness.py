"""
Test suite for soft sensor calculation robustness in RobustMPCController.

This module validates the critical fix for hardcoded soft sensor indices,
ensuring that soft sensor calculations are immune to column order changes
and properly validate required configuration elements.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from V2.robust_mpc.core import RobustMPCController


class TestSoftSensorRobustness:
    """Test suite for robust soft sensor calculations."""

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
    def base_config(self):
        """Base configuration for soft sensor testing."""
        return {
            "cma_names": ["d50", "lod"],
            "cpp_names": ["spray_rate", "air_flow", "carousel_speed"],
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
        d50_scaler = MinMaxScaler()
        d50_scaler.fit(np.array([[300], [600]]))
        scalers["d50"] = d50_scaler

        lod_scaler = MinMaxScaler()
        lod_scaler.fit(np.array([[0.5], [3.0]]))
        scalers["lod"] = lod_scaler

        # CPP scalers (including all possible soft sensors)
        cpp_vars = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]
        for var in cpp_vars:
            scaler = MinMaxScaler()
            scaler.fit(np.array([[0], [1000]]))  # Wide range for all variables
            scalers[var] = scaler

        return scalers

    def test_original_column_order(self, mock_model, mock_estimator, base_config, test_scalers):
        """Test soft sensors work with original column order."""
        # Standard column order from config.yaml
        config = base_config.copy()
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        # Test soft sensor calculations
        test_plan = np.array(
            [[120.0, 500.0, 30.0], [150.0, 600.0, 35.0]]  # spray_rate, air_flow, carousel_speed
        )

        scaled_plan = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)

        # Verify shape
        assert scaled_plan.shape == (2, 5)  # horizon=2, 5 total features

        # Verify all values are finite and in [0,1] range (scaled)
        assert np.all(np.isfinite(scaled_plan))
        assert np.all(scaled_plan >= 0.0)
        assert np.all(scaled_plan <= 1.0)

    def test_reversed_column_order(self, mock_model, mock_estimator, base_config, test_scalers):
        """Test immunity to reversed column order - CRITICAL TEST."""
        # Reverse the column order to trigger the original bug
        config = base_config.copy()
        config["cpp_full_names"] = [
            "froude_number_proxy",
            "specific_energy",
            "carousel_speed",
            "air_flow",
            "spray_rate",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        # Test with known values
        test_plan = np.array(
            [[120.0, 500.0, 30.0]]  # spray_rate=120, air_flow=500, carousel_speed=30
        )

        # This should NOT fail and should calculate correct soft sensors
        scaled_plan = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)

        # Verify calculations by unscaling and checking soft sensor values
        # Convert scaled plan back for validation
        plan_df = pd.DataFrame(scaled_plan, columns=config["cpp_full_names"])

        # Unscale to check calculation correctness
        for col in config["cpp_full_names"]:
            scaler = test_scalers[col]
            plan_df[col] = scaler.inverse_transform(plan_df[col].values.reshape(-1, 1)).flatten()

        # Expected soft sensor values
        expected_specific_energy = (120.0 * 30.0) / 1000.0  # 3.6
        expected_froude_number = (30.0**2) / 9.81  # ~91.7

        # Verify correct calculations (allowing for scaling/unscaling precision)
        assert np.isclose(plan_df["specific_energy"].iloc[0], expected_specific_energy, rtol=1e-3)
        assert np.isclose(plan_df["froude_number_proxy"].iloc[0], expected_froude_number, rtol=1e-3)

    def test_random_column_order(self, mock_model, mock_estimator, base_config, test_scalers):
        """Test various random column orderings."""
        # Scrambled order
        config = base_config.copy()
        config["cpp_full_names"] = [
            "air_flow",
            "specific_energy",
            "spray_rate",
            "froude_number_proxy",
            "carousel_speed",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        test_plan = np.array([[100.0, 450.0, 25.0]])  # spray=100, air=450, carousel=25

        # Should work without errors
        scaled_plan = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)
        assert scaled_plan.shape == (1, 5)
        assert np.all(np.isfinite(scaled_plan))

    def test_missing_required_base_variable(
        self, mock_model, mock_estimator, base_config, test_scalers
    ):
        """Test error handling when required base variables are missing."""
        # Missing spray_rate - should fail because spray_rate is in cpp_names but not cpp_full_names
        config = base_config.copy()
        config["cpp_full_names"] = [
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        with pytest.raises(ValueError, match="CPP 'spray_rate' not found in cpp_full_names"):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=config,
                scalers=test_scalers,
            )

    def test_soft_sensor_base_variable_validation(
        self, mock_model, mock_estimator, base_config, test_scalers
    ):
        """Test soft sensor base variable validation specifically."""
        # Test configuration that passes basic validation but fails soft sensor validation
        config = base_config.copy()
        # Include all cpp_names in cpp_full_names but exclude required soft sensor base
        config["cpp_names"] = ["air_flow", "carousel_speed"]  # Missing spray_rate
        config["cpp_full_names"] = [
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        with pytest.raises(ValueError, match="Missing required base variables.*spray_rate"):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=config,
                scalers=test_scalers,
            )

    def test_missing_required_soft_sensor(
        self, mock_model, mock_estimator, base_config, test_scalers
    ):
        """Test error handling when soft sensor variables are missing."""
        # Missing specific_energy - should fail at initialization
        config = base_config.copy()
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "froude_number_proxy",
        ]

        with pytest.raises(
            ValueError, match="Missing required soft sensor variables.*specific_energy"
        ):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=config,
                scalers=test_scalers,
            )

    def test_soft_sensor_calculation_accuracy(
        self, mock_model, mock_estimator, base_config, test_scalers
    ):
        """Test mathematical accuracy of soft sensor calculations."""
        config = base_config.copy()
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        # Test with multiple known values
        test_data = [
            (100.0, 400.0, 20.0),  # spray, air, carousel
            (150.0, 600.0, 35.0),
            (200.0, 700.0, 45.0),
        ]

        for spray_rate, air_flow, carousel_speed in test_data:
            test_plan = np.array([[spray_rate, air_flow, carousel_speed]])

            # Get the DataFrame approach results
            scaled_plan = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)

            # Unscale for verification
            plan_df = pd.DataFrame(scaled_plan, columns=config["cpp_full_names"])
            for col in config["cpp_full_names"]:
                scaler = test_scalers[col]
                plan_df[col] = scaler.inverse_transform(
                    plan_df[col].values.reshape(-1, 1)
                ).flatten()

            # Expected calculations
            expected_specific_energy = (spray_rate * carousel_speed) / 1000.0
            expected_froude_number = (carousel_speed**2) / 9.81

            # Verify accuracy
            assert np.isclose(
                plan_df["specific_energy"].iloc[0], expected_specific_energy, rtol=1e-3
            ), f"Specific energy calculation incorrect for inputs {test_data}"
            assert np.isclose(
                plan_df["froude_number_proxy"].iloc[0], expected_froude_number, rtol=1e-3
            ), f"Froude number calculation incorrect for inputs {test_data}"

    def test_cpp_not_in_full_names_error(
        self, mock_model, mock_estimator, base_config, test_scalers
    ):
        """Test error when cpp_names contains variables not in cpp_full_names."""
        config = base_config.copy()
        config["cpp_names"] = ["spray_rate", "air_flow", "carousel_speed", "unknown_variable"]
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        with pytest.raises(ValueError, match="CPP 'unknown_variable' not found in cpp_full_names"):
            RobustMPCController(
                model=mock_model,
                estimator=mock_estimator,
                optimizer_class=None,
                config=config,
                scalers=test_scalers,
            )

    def test_edge_case_single_row(self, mock_model, mock_estimator, base_config, test_scalers):
        """Test edge case with single time step."""
        config = base_config.copy()
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        # Single row test
        test_plan = np.array([[120.0, 500.0, 30.0]])
        scaled_plan = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)

        assert scaled_plan.shape == (1, 5)
        assert np.all(np.isfinite(scaled_plan))

    def test_performance_comparison(self, mock_model, mock_estimator, base_config, test_scalers):
        """Test that pandas-based approach doesn't significantly degrade performance."""
        import time

        config = base_config.copy()
        config["cpp_full_names"] = [
            "spray_rate",
            "air_flow",
            "carousel_speed",
            "specific_energy",
            "froude_number_proxy",
        ]

        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=config,
            scalers=test_scalers,
        )

        # Large test plan to measure performance
        large_plan = np.random.rand(100, 3) * 100 + 50  # 100 time steps

        start_time = time.time()
        for _ in range(10):  # Multiple runs
            scaled_plan = controller._scale_cpp_plan(large_plan, with_soft_sensors=True)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 10 runs)
        assert execution_time < 1.0, f"Performance too slow: {execution_time} seconds"
        assert scaled_plan.shape == (100, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
