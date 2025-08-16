#!/usr/bin/env python3
"""
Quick test to validate the corrected bias-augmented Kalman filter implementation.
"""

import numpy as np
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.abspath('V1/src/'))
sys.path.insert(0, os.path.abspath('V2/'))

from plant_simulator import AdvancedPlantSimulator
from robust_mpc.estimators import KalmanStateEstimator, BiasAugmentedKalmanStateEstimator

def test_bias_correction():
    print("🧪 Testing Corrected Bias-Augmented Kalman Filter")
    print("=" * 60)
    
    # Simulate simple system matrices (derived from V1 data)
    A = np.array([[0.896, 0.050], [0.000, 0.997]])  # Approximate from V1
    B = np.array([[0.305, -0.000, -0.550], [-0.000, -0.000, 0.001]])  # Approximate
    
    # Control variables order
    control_vars = ['spray_rate', 'air_flow', 'carousel_speed']
    
    # Initialize plant simulator
    plant = AdvancedPlantSimulator()
    
    # Initial conditions
    initial_cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
    initial_state = np.array([plant.state['d50'], plant.state['lod']])
    
    print(f"Initial plant state: d50={initial_state[0]:.1f} μm, lod={initial_state[1]:.3f}%")
    
    # Initialize both estimators
    estimator_original = KalmanStateEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=initial_state,
        process_noise_std=1.0,
        measurement_noise_std=15.0
    )
    
    estimator_corrected = BiasAugmentedKalmanStateEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=initial_state,
        process_noise_std=1.0,
        measurement_noise_std=15.0,
        bias_process_noise_std=0.1
    )
    
    # Run simulation
    cpps = initial_cpps.copy()
    results = []
    
    print(f"\n⏳ Running simulation for 100 steps...")
    
    for t in range(100):
        # Step the plant
        true_state_dict = plant.step(cpps)
        true_state = np.array([true_state_dict['d50'], true_state_dict['lod']])
        
        # Create noisy measurement
        measurement = true_state + np.random.normal(0, 15, size=2)
        
        # Control input vector
        control_input = np.array([cpps[k] for k in control_vars])
        
        # Get estimates
        original_est = estimator_original.estimate(measurement, control_input)
        corrected_est = estimator_corrected.get_bias_corrected_estimate(measurement, control_input)
        bias_est = estimator_corrected.get_current_bias_estimate()
        
        results.append({
            'step': t,
            'true_d50': true_state[0],
            'measured_d50': measurement[0],
            'original_d50': original_est[0],
            'corrected_d50': corrected_est[0],
            'bias_d50': bias_est[0]
        })
        
        # Print progress
        if t % 25 == 0:
            print(f"  Step {t:2d}: True={true_state[0]:6.1f}, Original={original_est[0]:6.1f}, "
                  f"Corrected={corrected_est[0]:6.1f}, Bias={bias_est[0]:6.1f}")
    
    # Analyze results (last 20 steps for steady-state)
    steady_state = results[-20:]
    
    true_mean = np.mean([r['true_d50'] for r in steady_state])
    original_mean = np.mean([r['original_d50'] for r in steady_state])
    corrected_mean = np.mean([r['corrected_d50'] for r in steady_state])
    bias_mean = np.mean([r['bias_d50'] for r in steady_state])
    
    original_error = abs(original_mean - true_mean)
    corrected_error = abs(corrected_mean - true_mean)
    
    print(f"\n📊 Steady-State Analysis (last 20 steps):")
    print(f"  True state mean:      {true_mean:8.2f} μm")
    print(f"  Original filter:      {original_mean:8.2f} μm  (error: {original_error:6.2f} μm)")
    print(f"  Corrected filter:     {corrected_mean:8.2f} μm  (error: {corrected_error:6.2f} μm)")
    print(f"  Estimated bias:       {bias_mean:8.2f} μm")
    
    improvement = max(0, (original_error - corrected_error) / max(original_error, 1e-6) * 100)
    print(f"  Error improvement:    {improvement:8.1f}%")
    
    # Check if bias correction is working
    if corrected_error < original_error * 0.5:  # At least 50% improvement
        print(f"\n✅ SUCCESS: Bias correction is working!")
        print(f"   Corrected filter error ({corrected_error:.2f}) < Original error ({original_error:.2f})")
    else:
        print(f"\n❌ ISSUE: Bias correction not effective enough")
        print(f"   Corrected error: {corrected_error:.2f} μm")
        print(f"   Original error:  {original_error:.2f} μm")
    
    return {
        'original_error': original_error,
        'corrected_error': corrected_error,
        'bias_estimate': bias_mean,
        'improvement_percent': improvement
    }

if __name__ == "__main__":
    test_bias_correction()