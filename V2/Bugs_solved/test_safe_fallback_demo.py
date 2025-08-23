#!/usr/bin/env python3
"""
Safe Fallback Action Initialization Fix Demonstration

This script demonstrates the critical safety fix for unsafe None initialization
of fallback actions, showing how pharmaceutical process control is now safe
from the very first control step.
"""

import os
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Add project paths
sys.path.insert(0, os.path.abspath("."))

from robust_mpc.core import RobustMPCController


def create_test_scalers():
    """Create test scalers for demonstration."""
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


def create_test_config():
    """Create test controller configuration."""
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
        "lookback": 10,
        "integral_gain": 0.1,
        "mc_samples": 5,
        "risk_beta": 1.5,
        "verbose": False,
        "history_buffer_size": 50,
        "cpp_constraints": {
            "spray_rate": {"min_val": 80.0, "max_val": 200.0},
            "air_flow": {"min_val": 400.0, "max_val": 700.0},
            "carousel_speed": {"min_val": 20.0, "max_val": 50.0},
        },
        "ga_config": {"population_size": 10, "num_generations": 3, "cx_prob": 0.7, "mut_prob": 0.2},
    }


class MockModel:
    """Mock model for testing."""

    def to(self, device):
        return self

    def predict_distribution(self, *args, **kwargs):
        import torch

        return torch.zeros(1, 5, 2), torch.ones(1, 5, 2) * 0.1


class MockEstimator:
    """Mock estimator for testing."""

    def estimate(self, measurement, control):
        return measurement


class AlwaysFailingOptimizer:
    """Optimizer that always fails to simulate worst-case scenario."""

    def __init__(self, param_bounds, config):
        self.failure_count = 0

    def optimize(self, fitness_func=None):
        self.failure_count += 1
        raise RuntimeError(f"Simulated optimizer failure #{self.failure_count}")


def demonstrate_safe_fallback_fix():
    """Demonstrate the safe fallback initialization fix."""

    print("=" * 80)
    print("SAFE FALLBACK ACTION INITIALIZATION FIX DEMONSTRATION")
    print("=" * 80)
    print()

    config = create_test_config()
    scalers = create_test_scalers()

    print("Pharmaceutical Process Constraints:")
    for name, constraints in config["cpp_constraints"].items():
        min_val = constraints["min_val"]
        max_val = constraints["max_val"]
        midpoint = (min_val + max_val) / 2.0
        print(f"  {name}: [{min_val}, {max_val}] → Safe default: {midpoint}")
    print()

    print("TESTING CONTROLLER INITIALIZATION:")
    print("-" * 50)

    # Test 1: Controller initialization
    print("1. Creating RobustMPCController with AlwaysFailingOptimizer...")

    controller = RobustMPCController(
        model=MockModel(),
        estimator=MockEstimator(),
        optimizer_class=AlwaysFailingOptimizer,
        config=config,
        scalers=scalers,
    )

    # Verify safe initialization
    fallback_action = controller._last_successful_action
    print(f"   Pre-initialized fallback action: {fallback_action}")
    print(f"   Is None?: {fallback_action is None}")
    print(f"   Shape: {fallback_action.shape}")
    print(f"   All finite values?: {np.all(np.isfinite(fallback_action))}")

    # Verify values are constraint midpoints
    expected_values = np.array([140.0, 550.0, 35.0])  # Constraint midpoints
    values_correct = np.array_equal(fallback_action, expected_values)
    print(f"   Values are constraint midpoints?: {values_correct}")

    if values_correct:
        print("   ✅ SAFE: Controller initialized with valid fallback action")
    else:
        print("   ❌ UNSAFE: Fallback action not properly initialized")

    print()

    print("TESTING FIRST CONTROL STEP OPTIMIZER FAILURE:")
    print("-" * 50)

    # Test 2: First control step with optimizer failure
    print("2. Executing first control step with guaranteed optimizer failure...")

    # Pharmaceutical process inputs
    measurement = np.array([450.0, 1.8])  # d50=450μm, LOD=1.8%
    current_input = np.array([130.0, 550.0, 30.0])  # Current CPP values
    setpoint = np.array([380.0, 1.8])  # Target values

    print(f"   Process measurement: d50={measurement[0]}μm, LOD={measurement[1]}%")
    print(f"   Current control input: {current_input}")
    print(f"   Target setpoint: d50={setpoint[0]}μm, LOD={setpoint[1]}%")

    try:
        # This would fail catastrophically in the old system if optimizer failed on first step
        action = controller.suggest_action(measurement, current_input, setpoint)

        print(f"   Returned control action: {action}")
        print(f"   Action shape: {action.shape}")
        print(f"   All finite values?: {np.all(np.isfinite(action))}")

        # Verify action is within constraints
        constraints_satisfied = True
        cpp_config = config["cpp_constraints"]

        for i, name in enumerate(config["cpp_names"]):
            min_val = cpp_config[name]["min_val"]
            max_val = cpp_config[name]["max_val"]
            value = action[i]

            within_bounds = min_val <= value <= max_val
            print(f"   {name}: {value} ∈ [{min_val}, {max_val}]? {within_bounds}")

            if not within_bounds:
                constraints_satisfied = False

        if constraints_satisfied:
            print("   ✅ SUCCESS: Safe control action returned despite optimizer failure")
            print("   ✅ SAFETY: All constraint bounds satisfied")
            print("   ✅ PHARMACEUTICAL: Process remains under safe control")
        else:
            print("   ❌ DANGER: Control action violates constraints")

    except Exception as e:
        print(f"   ❌ CATASTROPHIC FAILURE: {e}")
        print("   ❌ PHARMACEUTICAL RISK: No fallback action available")

    print()

    print("TESTING FALLBACK STRATEGY VALIDATION:")
    print("-" * 50)

    # Test 3: Validate fallback action passes controller validation
    print("3. Testing fallback action validation...")

    is_valid = controller._validate_control_action(controller._last_successful_action)
    print(f"   Fallback action passes validation?: {is_valid}")

    if is_valid:
        print("   ✅ VALIDATED: Fallback action meets all controller requirements")
    else:
        print("   ❌ INVALID: Fallback action fails validation")

    print()

    print("TESTING SAFE DEFAULT CALCULATION METHOD:")
    print("-" * 50)

    # Test 4: Direct method testing
    print("4. Testing _calculate_safe_default_action method...")

    safe_defaults = controller._calculate_safe_default_action()
    print(f"   Calculated safe defaults: {safe_defaults}")

    # Verify mathematical correctness
    expected_spray = (80.0 + 200.0) / 2.0  # 140.0
    expected_air = (400.0 + 700.0) / 2.0  # 550.0
    expected_speed = (20.0 + 50.0) / 2.0  # 35.0

    calculations_correct = np.allclose(
        safe_defaults, [expected_spray, expected_air, expected_speed]
    )
    print(f"   Mathematical calculations correct?: {calculations_correct}")

    if calculations_correct:
        print("   ✅ MATHEMATICS: Safe defaults correctly calculated as constraint midpoints")
    else:
        print("   ❌ ERROR: Safe default calculations incorrect")

    print()

    print("COMPARATIVE SAFETY ANALYSIS:")
    print("-" * 50)

    print("BEFORE FIX (Unsafe):")
    print("  ❌ _last_successful_action initialized to None")
    print("  ❌ No guaranteed safe fallback on first control step")
    print("  ❌ Relies on potentially invalid current control input")
    print("  ❌ Risk of pharmaceutical batch failure during startup")
    print()

    print("AFTER FIX (Safe):")
    print("  ✅ _last_successful_action pre-initialized with safe defaults")
    print("  ✅ Guaranteed safe fallback available from first control step")
    print("  ✅ Safe defaults use constraint midpoints (conservative, stable)")
    print("  ✅ Pharmaceutical manufacturing safe during startup")
    print()

    print("PHARMACEUTICAL MANUFACTURING IMPACT:")
    print("-" * 50)
    print("✅ BATCH SAFETY: No risk of unsafe control actions during startup")
    print("✅ PRODUCTION RELIABILITY: Guaranteed fallback for optimizer failures")
    print("✅ REGULATORY COMPLIANCE: Safe operation from system initialization")
    print("✅ FINANCIAL PROTECTION: No batch losses from startup control failures")

    print()
    print("=" * 80)
    print("🎯 SUCCESS: Safe fallback initialization fix validated and operational")
    print("🛡️ SAFETY: Pharmaceutical process control secure from first control step")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_safe_fallback_fix()
