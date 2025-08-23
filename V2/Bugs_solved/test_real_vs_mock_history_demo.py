#!/usr/bin/env python3
"""
Demonstration of Real vs Mock History Impact on MPC Performance

This script shows the critical difference between the old unrealistic mock history
generation and the new real trajectory tracking for pharmaceutical process control.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add project paths
sys.path.insert(0, os.path.abspath("."))

from robust_mpc.core import RobustMPCController
from robust_mpc.data_buffer import DataBuffer, StartupHistoryGenerator


def create_test_scalers():
    """Create realistic pharmaceutical scalers."""
    scalers = {}

    # CMA scalers for d50 (particle size) and LOD (moisture)
    d50_scaler = MinMaxScaler()
    d50_scaler.fit(np.array([[300], [600]]))  # 300μm range
    scalers["d50"] = d50_scaler

    lod_scaler = MinMaxScaler()
    lod_scaler.fit(np.array([[0.5], [3.0]]))  # 2.5% range
    scalers["lod"] = lod_scaler

    # CPP scalers for spray_rate, air_flow, carousel_speed
    spray_scaler = MinMaxScaler()
    spray_scaler.fit(np.array([[80], [200]]))  # g/min
    scalers["spray_rate"] = spray_scaler

    air_scaler = MinMaxScaler()
    air_scaler.fit(np.array([[400], [700]]))  # m³/h
    scalers["air_flow"] = air_scaler

    speed_scaler = MinMaxScaler()
    speed_scaler.fit(np.array([[20], [50]]))  # rpm
    scalers["carousel_speed"] = speed_scaler

    # Soft sensor scalers
    for name in ["specific_energy", "froude_number_proxy"]:
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [100]]))
        scalers[name] = scaler

    return scalers


def simulate_realistic_trajectory():
    """Simulate a realistic pharmaceutical control trajectory.

    Scenario: Process starts at low spray rate, controller gradually increases
    it to reach target particle size, creating a realistic control trajectory.
    """
    steps = 30

    # Control trajectory: gradual increase in spray rate
    spray_rates = np.linspace(90, 170, steps)  # 90 → 170 g/min
    air_flows = np.full(steps, 550.0)  # Constant air flow
    carousel_speeds = np.full(steps, 30.0)  # Constant carousel speed

    # Process response: particle size increases with spray rate (simplified)
    d50_values = 400 + (spray_rates - 90) * 0.8 + np.random.normal(0, 3, steps)
    lod_values = 1.5 + (spray_rates - 90) * 0.003 + np.random.normal(0, 0.05, steps)

    control_trajectory = np.column_stack([spray_rates, air_flows, carousel_speeds])
    measurement_trajectory = np.column_stack([d50_values, lod_values])

    return control_trajectory, measurement_trajectory


def demonstrate_mock_vs_real_history():
    """Demonstrate the critical difference between mock and real history."""

    print("=" * 80)
    print("DEMONSTRATION: Mock vs Real History in Pharmaceutical MPC")
    print("=" * 80)
    print()

    # Setup test configuration
    config = {
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
        "verbose": True,
        "history_buffer_size": 50,
    }

    scalers = create_test_scalers()

    # Create mock model and estimator
    class MockModel:
        def to(self, device):
            return self

        def predict_distribution(self, *args, **kwargs):
            return np.zeros((1, 10, 2)), np.ones((1, 10, 2)) * 0.1

    class MockEstimator:
        def estimate(self, measurement, control):
            return measurement

    # Create controller with real history buffer
    controller = RobustMPCController(
        model=MockModel(),
        estimator=MockEstimator(),
        optimizer_class=None,
        config=config,
        scalers=scalers,
    )

    # Generate realistic control trajectory
    control_trajectory, measurement_trajectory = simulate_realistic_trajectory()

    print("Scenario: Pharmaceutical granulation process control")
    print(f"- Process starts at spray_rate = {control_trajectory[0, 0]:.1f} g/min")
    print(f"- Controller gradually increases to {control_trajectory[-1, 0]:.1f} g/min")
    print(
        f"- Particle size (d50) responds from {measurement_trajectory[0, 0]:.1f} to {measurement_trajectory[-1, 0]:.1f} μm"
    )
    print()

    # Phase 1: Build up real history
    print("Phase 1: Building real trajectory history...")
    for i, (control, measurement) in enumerate(
        zip(control_trajectory[:20], measurement_trajectory[:20])
    ):
        controller.history_buffer.add_measurement(measurement)
        controller.history_buffer.add_control_action(control)

        if i % 5 == 0:
            print(f"  Step {i:2d}: spray_rate={control[0]:5.1f}, d50={measurement[0]:5.1f}")

    print()

    # Phase 2: Compare historical data
    print("Phase 2: Comparing Mock vs Real History at current state...")
    print()

    # Current state after trajectory
    current_measurement = measurement_trajectory[19]  # Latest measurement
    current_control = control_trajectory[19]  # Latest control

    print(
        f"Current state: d50={current_measurement[0]:.1f}μm, spray_rate={current_control[0]:.1f}g/min"
    )
    print()

    # Get real history from buffer
    real_cmas, real_cpps = controller.history_buffer.get_model_inputs(lookback=15)

    # Create mock history (old approach) for comparison
    mock_generator = StartupHistoryGenerator(
        cma_features=2,
        cpp_features=3,
        initial_cma_state=np.array([450.0, 1.8]),  # Default baselines
        initial_cpp_state=np.array([130.0, 550.0, 30.0]),  # Hardcoded baselines
    )
    mock_cmas, mock_cpps = mock_generator.generate_startup_history(15)

    # Analysis
    print("CRITICAL COMPARISON:")
    print("-" * 50)

    # Extract spray rates (first CPP column)
    real_spray_history = real_cpps[:, 0]  # This will be scaled, but shows trend
    mock_spray_history = mock_cpps[:, 0]  # Shows baseline convergence

    print(f"Real History Summary:")
    print(
        f"  - Reflects actual trajectory: {control_trajectory[5, 0]:.1f} → {control_trajectory[19, 0]:.1f} g/min"
    )
    print(f"  - Shows increasing trend from controller actions")
    print(f"  - Model sees how process actually reached current state")
    print()

    print(f"Mock History Summary (OLD APPROACH):")
    print(f"  - Uses hardcoded baseline: ~130 g/min ± noise")
    print(f"  - Ignores actual trajectory completely")
    print(f"  - Model has no knowledge of how process reached current state")
    print()

    # Calculate the difference in information content
    real_trajectory_range = np.max(real_cmas[:, 0]) - np.min(real_cmas[:, 0])
    mock_trajectory_range = np.max(mock_cmas[:, 0]) - np.min(mock_cmas[:, 0])

    print(f"Information Content:")
    print(f"  - Real history d50 range: {real_trajectory_range:.1f} μm (informative)")
    print(f"  - Mock history d50 range: {mock_trajectory_range:.1f} μm (artificial noise)")
    print()

    # Impact on model predictions
    print("IMPACT ON MODEL PREDICTIONS:")
    print("-" * 50)
    print("Real History → Model understands:")
    print("  ✅ Process has been increasing spray rate")
    print("  ✅ Particle size has been responding positively")
    print("  ✅ Trajectory shows control effectiveness")
    print("  ✅ Predictions based on actual process dynamics")
    print()

    print("Mock History → Model believes:")
    print("  ❌ Process has been operating at baseline conditions")
    print("  ❌ No knowledge of recent control actions")
    print("  ❌ Predictions based on fabricated, unrepresentative history")
    print("  ❌ Cannot predict trajectory continuation accurately")
    print()

    # Production Impact
    print("PRODUCTION IMPACT:")
    print("-" * 50)
    print("With Real History:")
    print("  🎯 Accurate model predictions")
    print("  🎯 Effective control strategies")
    print("  🎯 Proper trajectory planning")
    print("  🎯 Reliable pharmaceutical batch quality")
    print()

    print("With Mock History:")
    print("  ⚠️  Inaccurate model predictions")
    print("  ⚠️  Suboptimal control decisions")
    print("  ⚠️  Poor trajectory planning")
    print("  ⚠️  Risk of batch failures and regulatory issues")
    print()

    print("=" * 80)
    print("CONCLUSION: Real history tracking is ESSENTIAL for production MPC")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_mock_vs_real_history()
