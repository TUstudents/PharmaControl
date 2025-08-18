#!/usr/bin/env python3
"""
Soft Sensor Robustness Fix Demonstration

This script demonstrates that the critical hardcoded indices bug has been fixed
and that soft sensor calculations are now immune to column order changes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add project paths
sys.path.insert(0, os.path.abspath('.'))

from robust_mpc.core import RobustMPCController


def create_test_scalers():
    """Create test scalers for demonstration."""
    scalers = {}
    
    # CMA scalers
    scalers['d50'] = MinMaxScaler().fit(np.array([[300], [600]]))
    scalers['lod'] = MinMaxScaler().fit(np.array([[0.5], [3.0]]))
    
    # CPP scalers - wide range for all variables
    cpp_vars = ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
    for var in cpp_vars:
        scalers[var] = MinMaxScaler().fit(np.array([[0], [1000]]))
    
    return scalers


def create_base_config():
    """Create base controller configuration."""
    return {
        'cma_names': ['d50', 'lod'],
        'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
        'horizon': 3,
        'lookback': 10,
        'integral_gain': 0.1,
        'mc_samples': 5,
        'risk_beta': 1.5,
        'verbose': False,
        'history_buffer_size': 50,
        'cpp_constraints': {
            'spray_rate': {'min_val': 80.0, 'max_val': 200.0},
            'air_flow': {'min_val': 400.0, 'max_val': 700.0},
            'carousel_speed': {'min_val': 20.0, 'max_val': 50.0}
        },
        'ga_config': {
            'population_size': 10,
            'num_generations': 3,
            'cx_prob': 0.7,
            'mut_prob': 0.2
        }
    }


class MockModel:
    """Mock model for testing."""
    def to(self, device): 
        return self
    def predict_distribution(self, *args, **kwargs):
        import torch
        return torch.zeros(1, 3, 2), torch.ones(1, 3, 2) * 0.1


class MockEstimator:
    """Mock estimator for testing."""
    def estimate(self, measurement, control):
        return measurement


def test_column_order_robustness():
    """Demonstrate immunity to column order changes."""
    
    print("=" * 80)
    print("SOFT SENSOR ROBUSTNESS FIX DEMONSTRATION")
    print("=" * 80)
    print()
    
    scalers = create_test_scalers()
    base_config = create_base_config()
    
    # Test data: spray_rate=120, air_flow=500, carousel_speed=30
    test_plan = np.array([[120.0, 500.0, 30.0]])
    
    # Expected soft sensor values
    expected_specific_energy = (120.0 * 30.0) / 1000.0  # 3.6
    expected_froude_number = (30.0 ** 2) / 9.81         # ~91.7
    
    print("Test Input Data:")
    print(f"  spray_rate = 120.0 g/min")
    print(f"  air_flow = 500.0 m³/h")
    print(f"  carousel_speed = 30.0 rpm")
    print()
    print("Expected Soft Sensor Values:")
    print(f"  specific_energy = (120 × 30) / 1000 = {expected_specific_energy:.2f}")
    print(f"  froude_number_proxy = (30²) / 9.81 = {expected_froude_number:.2f}")
    print()
    
    # Test different column orders
    test_configurations = [
        {
            'name': 'Original Order',
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
        },
        {
            'name': 'Reversed Order (Original Bug Trigger)',
            'cpp_full_names': ['froude_number_proxy', 'specific_energy', 'carousel_speed', 'air_flow', 'spray_rate']
        },
        {
            'name': 'Random Order #1',
            'cpp_full_names': ['air_flow', 'specific_energy', 'spray_rate', 'froude_number_proxy', 'carousel_speed']
        },
        {
            'name': 'Random Order #2', 
            'cpp_full_names': ['carousel_speed', 'froude_number_proxy', 'air_flow', 'spray_rate', 'specific_energy']
        }
    ]
    
    print("TESTING DIFFERENT COLUMN ORDERS:")
    print("-" * 50)
    
    all_results_correct = True
    
    for i, test_config in enumerate(test_configurations, 1):
        print(f"{i}. {test_config['name']}")
        print(f"   cpp_full_names: {test_config['cpp_full_names']}")
        
        # Create config with this column order
        config = base_config.copy()
        config['cpp_full_names'] = test_config['cpp_full_names']
        
        try:
            # Create controller with this configuration
            controller = RobustMPCController(
                model=MockModel(),
                estimator=MockEstimator(),
                optimizer_class=None,
                config=config,
                scalers=scalers
            )
            
            # Calculate soft sensors
            scaled_result = controller._scale_cpp_plan(test_plan, with_soft_sensors=True)
            
            # Unscale to verify calculation correctness
            result_df = pd.DataFrame(scaled_result, columns=config['cpp_full_names'])
            for col in config['cpp_full_names']:
                scaler = scalers[col]
                result_df[col] = scaler.inverse_transform(result_df[col].values.reshape(-1, 1)).flatten()
            
            # Extract calculated values
            calc_specific_energy = result_df['specific_energy'].iloc[0]
            calc_froude_number = result_df['froude_number_proxy'].iloc[0]
            
            # Verify correctness
            specific_energy_correct = np.isclose(calc_specific_energy, expected_specific_energy, rtol=1e-3)
            froude_number_correct = np.isclose(calc_froude_number, expected_froude_number, rtol=1e-3)
            
            if specific_energy_correct and froude_number_correct:
                print(f"   ✅ CORRECT: specific_energy={calc_specific_energy:.2f}, froude_number={calc_froude_number:.2f}")
            else:
                print(f"   ❌ INCORRECT: specific_energy={calc_specific_energy:.2f}, froude_number={calc_froude_number:.2f}")
                all_results_correct = False
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            all_results_correct = False
        
        print()
    
    print("CONFIGURATION VALIDATION TESTS:")
    print("-" * 50)
    
    # Test missing required base variables
    print("1. Testing Missing Required Base Variable (spray_rate)")
    config_missing_base = base_config.copy()
    config_missing_base['cpp_names'] = ['air_flow', 'carousel_speed'] 
    config_missing_base['cpp_full_names'] = ['air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
    
    try:
        RobustMPCController(
            model=MockModel(),
            estimator=MockEstimator(),
            optimizer_class=None,
            config=config_missing_base,
            scalers=scalers
        )
        print("   ❌ FAILED: Should have caught missing spray_rate")
        all_results_correct = False
    except ValueError as e:
        if "spray_rate" in str(e):
            print("   ✅ CORRECT: Caught missing spray_rate requirement")
        else:
            print(f"   ❌ WRONG ERROR: {e}")
            all_results_correct = False
    
    # Test missing soft sensor variables
    print("2. Testing Missing Soft Sensor Variable (specific_energy)")
    config_missing_soft = base_config.copy()
    config_missing_soft['cpp_full_names'] = ['spray_rate', 'air_flow', 'carousel_speed', 'froude_number_proxy']
    
    try:
        RobustMPCController(
            model=MockModel(),
            estimator=MockEstimator(),
            optimizer_class=None,
            config=config_missing_soft,
            scalers=scalers
        )
        print("   ❌ FAILED: Should have caught missing specific_energy")
        all_results_correct = False
    except ValueError as e:
        if "specific_energy" in str(e):
            print("   ✅ CORRECT: Caught missing specific_energy requirement")
        else:
            print(f"   ❌ WRONG ERROR: {e}")
            all_results_correct = False
    
    print()
    print("PERFORMANCE AND RELIABILITY:")
    print("-" * 50)
    
    # Test with larger dataset
    config = base_config.copy()
    config['cpp_full_names'] = ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
    controller = RobustMPCController(
        model=MockModel(),
        estimator=MockEstimator(), 
        optimizer_class=None,
        config=config,
        scalers=scalers
    )
    
    # Large test plan
    large_plan = np.random.rand(100, 3) * 50 + 75  # 100 time steps
    
    import time
    start_time = time.time()
    scaled_result = controller._scale_cpp_plan(large_plan, with_soft_sensors=True)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Performance Test: 100 time steps processed in {execution_time:.4f} seconds")
    print(f"Result shape: {scaled_result.shape}")
    print(f"All values finite: {np.all(np.isfinite(scaled_result))}")
    print(f"All values in [0,1]: {np.all((scaled_result >= 0) & (scaled_result <= 1))}")
    
    print()
    print("=" * 80)
    if all_results_correct:
        print("🎉 SUCCESS: All tests passed! Soft sensor calculations are robust to column order changes.")
        print("✅ The critical hardcoded indices bug has been fixed.")
        print("✅ Pharmaceutical process control is now safe from configuration changes.")
    else:
        print("⚠️  FAILURE: Some tests failed. The fix may not be complete.")
    print("=" * 80)


if __name__ == "__main__":
    test_column_order_robustness()