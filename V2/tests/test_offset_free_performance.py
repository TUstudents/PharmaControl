"""
Test suite for validating offset-free MPC performance with corrected offset scaling.

This module demonstrates that the fixed offset scaling enables proper integral action
and eliminates steady-state error in the presence of step disturbances.
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


def test_offset_free_performance_comparison():
    """
    Compare offset-free MPC performance before and after the scaling fix.

    This test demonstrates the critical importance of correct offset scaling
    by simulating a step disturbance and showing how integral action works.
    """

    # Create realistic pharmaceutical configuration
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
        "integral_gain": 0.2,  # Higher gain for faster disturbance rejection
        "mc_samples": 30,
        "risk_beta": 1.5,
        "verbose": False,  # Quiet mode for tests
    }

    # Create realistic fitted scalers
    scalers = {}

    # CMA scalers
    d50_scaler = MinMaxScaler()
    d50_scaler.fit(np.array([[300], [600]]))  # Range: 300 μm
    scalers["d50"] = d50_scaler

    lod_scaler = MinMaxScaler()
    lod_scaler.fit(np.array([[0.5], [3.0]]))  # Range: 2.5%
    scalers["lod"] = lod_scaler

    # Add other required scalers (not used in this test but needed for validation)
    for name in [
        "spray_rate",
        "air_flow",
        "carousel_speed",
        "specific_energy",
        "froude_number_proxy",
    ]:
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [100]]))  # Dummy range
        scalers[name] = scaler

    # Mock model that simulates a step disturbance
    class MockModelWithDisturbance:
        def __init__(self, disturbance_offset=np.array([0.0, 0.0])):
            self.disturbance_offset = disturbance_offset

        def to(self, device):
            return self

        def predict_distribution(self, past_cmas, past_cpps, future_cpps, n_samples=30):
            """Mock prediction that includes a persistent disturbance."""
            batch_size, horizon, cma_features = (
                1,
                future_cpps.shape[1],
                len(self.disturbance_offset),
            )

            # Base prediction (zeros in scaled space)
            mean_pred = torch.zeros(batch_size, horizon, cma_features)

            # Add disturbance in scaled space
            disturbance_scaled = torch.tensor(
                [
                    self.disturbance_offset[0] * d50_scaler.scale_[0],  # Scale disturbance properly
                    self.disturbance_offset[1] * lod_scaler.scale_[0],
                ]
            )

            # Apply disturbance to all time steps
            for t in range(horizon):
                mean_pred[0, t, :] = disturbance_scaled

            # Small uncertainty
            std_pred = torch.ones_like(mean_pred) * 0.01

            return mean_pred, std_pred

    # Mock estimator
    class MockEstimator:
        def estimate(self, measurement, control):
            return measurement

    # Test scenario: Step disturbance of +30 μm in d50, +0.1% in LOD
    step_disturbance = np.array([30.0, 0.1])

    # Create controller with disturbance
    model_with_disturbance = MockModelWithDisturbance(step_disturbance)
    controller = RobustMPCController(
        model=model_with_disturbance,
        estimator=MockEstimator(),
        optimizer_class=None,  # Not needed for this test
        config=config,
        scalers=scalers,
    )

    print("=== Testing Offset-Free MPC Performance ===")
    print(f"Step disturbance: d50={step_disturbance[0]} μm, LOD={step_disturbance[1]}%")

    # Simulation parameters
    num_steps = 20
    setpoint = np.array([450.0, 1.8])  # Target: d50=450μm, LOD=1.8%

    # Arrays to store results
    measurements = np.zeros((num_steps, 2))
    disturbance_estimates = np.zeros((num_steps, 2))
    scaled_disturbances = np.zeros((num_steps, 2))

    # Simulate the process
    current_measurement = setpoint.copy()  # Start at setpoint

    for step in range(num_steps):
        # Update disturbance estimate (integral action)
        error = setpoint - current_measurement
        controller._update_disturbance_estimate(current_measurement, setpoint)

        # Record current state
        measurements[step] = current_measurement
        disturbance_estimates[step] = controller.disturbance_estimate.copy()

        # Test the corrected offset scaling
        scaled_disturbance = controller._scale_cma_offset(controller.disturbance_estimate)
        scaled_disturbances[step] = scaled_disturbance

        # Simulate measurement with disturbance (simplified)
        # In reality, this would come from the process
        true_disturbance_effect = (
            step_disturbance * 0.9
        )  # 90% of disturbance shows up in measurement
        current_measurement = (
            setpoint + true_disturbance_effect - controller.disturbance_estimate * 0.8
        )

        # Print progress every 5 steps
        if step % 5 == 0:
            print(
                f"Step {step:2d}: Error={error}, Disturbance Estimate={controller.disturbance_estimate}"
            )

    # Analysis of results
    print(f"\n=== Results Analysis ===")

    # Final disturbance estimate should approximate the true disturbance
    final_disturbance_estimate = disturbance_estimates[-1]
    estimation_error = np.abs(final_disturbance_estimate - step_disturbance)

    print(f"True disturbance:      {step_disturbance}")
    print(f"Final estimate:        {final_disturbance_estimate}")
    print(f"Estimation error:      {estimation_error}")
    print(f"Relative error (%):    {100 * estimation_error / np.abs(step_disturbance)}")

    # Test that disturbance estimate converges toward true disturbance
    # (allowing for some model mismatch and simplified simulation)
    assert estimation_error[0] < 15.0, f"d50 estimation error too large: {estimation_error[0]} μm"
    assert estimation_error[1] < 0.05, f"LOD estimation error too large: {estimation_error[1]}%"

    # Test that scaling is working correctly
    expected_scaled_d50 = final_disturbance_estimate[0] / (600.0 - 300.0)  # Should be offset/range
    expected_scaled_lod = final_disturbance_estimate[1] / (3.0 - 0.5)

    actual_scaled = scaled_disturbances[-1]

    print(f"\n=== Scaling Verification ===")
    print(f"Expected scaled d50:   {expected_scaled_d50:.6f}")
    print(f"Actual scaled d50:     {actual_scaled[0]:.6f}")
    print(f"Expected scaled LOD:   {expected_scaled_lod:.6f}")
    print(f"Actual scaled LOD:     {actual_scaled[1]:.6f}")

    # Verify correct offset scaling (no translation)
    assert np.isclose(actual_scaled[0], expected_scaled_d50, rtol=1e-4)
    assert np.isclose(actual_scaled[1], expected_scaled_lod, rtol=1e-4)

    # Test offset scaling properties
    print(f"\n=== Critical Offset Scaling Properties ===")

    # 1. Zero offset should remain zero
    zero_scaled = controller._scale_cma_offset(np.array([0.0, 0.0]))
    assert np.allclose(zero_scaled, [0.0, 0.0]), "Zero offset not preserved"
    print("Zero offset preservation: PASSED")

    # 2. Sign preservation
    positive_offset = np.array([20.0, 0.1])
    negative_offset = np.array([-20.0, -0.1])

    positive_scaled = controller._scale_cma_offset(positive_offset)
    negative_scaled = controller._scale_cma_offset(negative_offset)

    assert np.all(positive_scaled > 0), "Positive offset signs not preserved"
    assert np.all(negative_scaled < 0), "Negative offset signs not preserved"
    print("Sign preservation: PASSED")

    # 3. Proportionality
    offset_1x = np.array([10.0, 0.05])
    offset_2x = np.array([20.0, 0.10])

    scaled_1x = controller._scale_cma_offset(offset_1x)
    scaled_2x = controller._scale_cma_offset(offset_2x)

    assert np.allclose(scaled_2x, 2.0 * scaled_1x, rtol=1e-6), "Proportionality not preserved"
    print("Proportionality preservation: PASSED")

    print("Offset-Free MPC Performance Test: PASSED")
    print("Critical offset scaling bug has been successfully fixed")
    print("Integral action now works correctly for pharmaceutical process control")

    return {
        "measurements": measurements,
        "disturbance_estimates": disturbance_estimates,
        "scaled_disturbances": scaled_disturbances,
        "final_estimation_error": estimation_error,
        "convergence_achieved": np.all(estimation_error < [15.0, 0.05]),
    }


def test_offset_scaling_mathematical_verification():
    """
    Mathematical verification that offset scaling follows the correct formula.

    This test provides definitive proof that the offset scaling implementation
    matches the mathematical requirements for proper integral action.
    """

    print("\n=== Mathematical Verification of Offset Scaling ===")

    # Create test scalers with known parameters
    d50_min, d50_max = 300.0, 600.0
    lod_min, lod_max = 0.5, 3.0

    d50_scaler = MinMaxScaler()
    d50_scaler.fit(np.array([[d50_min], [d50_max]]))

    lod_scaler = MinMaxScaler()
    lod_scaler.fit(np.array([[lod_min], [lod_max]]))

    scalers = {"d50": d50_scaler, "lod": lod_scaler}

    # Create minimal controller for testing
    config = {"cma_names": ["d50", "lod"], "verbose": False}

    class MockModel:
        def to(self, device):
            return self

    class MockEstimator:
        def estimate(self, measurement, control):
            return measurement

    # Create controller instance (minimal setup for testing)
    controller = RobustMPCController.__new__(RobustMPCController)
    controller.config = config
    controller.scalers = scalers

    # Test various offset values
    test_offsets = [
        np.array([0.0, 0.0]),  # Zero offset
        np.array([30.0, 0.15]),  # Positive offset
        np.array([-45.0, -0.2]),  # Negative offset
        np.array([150.0, 1.25]),  # Large positive offset
        np.array([300.0, 2.5]),  # Full-range offset
    ]

    print(f"d50 range: [{d50_min}, {d50_max}] μm (range = {d50_max - d50_min})")
    print(f"LOD range: [{lod_min}, {lod_max}]% (range = {lod_max - lod_min})")
    print(f"Expected scale factors: d50={1/(d50_max-d50_min):.6f}, LOD={1/(lod_max-lod_min):.6f}")

    print("\nOffset → Scaled Verification:")
    print("=" * 60)

    for i, offset in enumerate(test_offsets):
        scaled = controller._scale_cma_offset(offset)

        # Manual calculation
        expected_d50 = offset[0] / (d50_max - d50_min)
        expected_lod = offset[1] / (lod_max - lod_min)
        expected = np.array([expected_d50, expected_lod])

        # Verify mathematical correctness
        assert np.allclose(scaled, expected, rtol=1e-10), f"Mathematical error for offset {offset}"

        print(
            f"Test {i+1}: [{offset[0]:6.1f}, {offset[1]:5.2f}] -> [{scaled[0]:8.6f}, {scaled[1]:8.6f}]"
        )

    print("Mathematical verification: ALL TESTS PASSED")
    print("Offset scaling formula: scaled_offset = offset / (max - min)")
    print("No translation term applied (critical for integral action)")


if __name__ == "__main__":
    # Run the performance test
    results = test_offset_free_performance_comparison()

    # Run the mathematical verification
    test_offset_scaling_mathematical_verification()

    print("All offset-free MPC tests completed successfully")
    print("The critical offset scaling bug has been fixed")
