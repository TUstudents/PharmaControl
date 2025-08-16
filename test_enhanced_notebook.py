#!/usr/bin/env python3
"""
Test the enhanced bias-corrected Kalman filter implementation from the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add paths
sys.path.insert(0, os.path.abspath('V1/src/'))
sys.path.insert(0, os.path.abspath('V2/'))

# Set random seed for reproducibility
np.random.seed(42)

def test_enhanced_bias_correction():
    print("🧪 Testing Enhanced Bias-Corrected Kalman Filter")
    print("=" * 60)
    
    # Import required modules
    try:
        from plant_simulator import AdvancedPlantSimulator
        from sklearn.linear_model import LinearRegression
        from robust_mpc.estimators import KalmanStateEstimator, BiasAugmentedKalmanStateEstimator
        print("✓ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create synthetic data since V1 data might not be available
    print("\n📊 Creating synthetic granulation data...")
    n_samples = 2000
    spray_rate = np.random.uniform(80, 180, n_samples)
    air_flow = np.random.uniform(400, 700, n_samples)
    carousel_speed = np.random.uniform(20, 40, n_samples)
    
    # Simple nonlinear relationships with noise + IMPORTANT INTERCEPT
    d50 = 25.0 + 400 + 0.5*spray_rate + 0.1*air_flow - 2*carousel_speed + np.random.normal(0, 10, n_samples)
    lod = 0.05 + 2 + 0.01*spray_rate - 0.001*air_flow + 0.02*carousel_speed + np.random.normal(0, 0.2, n_samples)
    
    df_full = pd.DataFrame({
        'd50': d50, 'lod': lod,
        'spray_rate': spray_rate, 'air_flow': air_flow, 'carousel_speed': carousel_speed
    })
    
    # Define variables
    state_vars = ['d50', 'lod']
    control_vars = ['spray_rate', 'air_flow', 'carousel_speed']
    
    # Prepare data for linear regression
    X = pd.concat([df_full[state_vars].shift(1), df_full[control_vars]], axis=1).dropna()
    y = df_full[state_vars][1:]
    
    # Fit linear model WITH intercept
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X, y)
    
    num_states = len(state_vars)
    transition_matrix_A = lin_reg.coef_[:, :num_states]
    control_matrix_B = lin_reg.coef_[:, num_states:]
    intercept_C = lin_reg.intercept_
    
    print(f"✓ Created synthetic dataset with {len(df_full)} samples")
    print(f"🚨 CRITICAL Intercept: {intercept_C} (will be ignored by naive filter)")
    print(f"   Expected bias: ~{intercept_C[0]:.1f} μm for d50")
    
    # Initialize plant and estimators
    plant = AdvancedPlantSimulator()
    initial_state = np.array([plant.state['d50'], plant.state['lod']])
    print(f"🌱 Plant initial state: d50={initial_state[0]:.1f} μm, lod={initial_state[1]:.3f}%")
    
    # Create both estimators
    estimator_naive = KalmanStateEstimator(
        transition_matrix=transition_matrix_A,
        control_matrix=control_matrix_B,
        initial_state_mean=initial_state,
        process_noise_std=1.0,
        measurement_noise_std=15.0
    )
    
    estimator_robust = BiasAugmentedKalmanStateEstimator(
        transition_matrix=transition_matrix_A,
        control_matrix=control_matrix_B,
        initial_state_mean=initial_state,
        process_noise_std=1.0,
        measurement_noise_std=15.0,
        bias_process_noise_std=0.1
    )
    
    # Run simulation
    cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
    log = []
    
    print("\n⏳ Running comparative simulation...")
    
    for t in range(150):
        if t == 75:
            cpps = {'spray_rate': 150.0, 'air_flow': 600.0, 'carousel_speed': 35.0}
            print(f"  Step {t}: Control input changed")
        
        # Get true state
        true_state_dict = plant.step(cpps)
        true_vec = np.array([true_state_dict['d50'], true_state_dict['lod']])
        
        # Add sensor noise
        noisy_vec = true_vec + np.random.normal(0, 15, size=2)
        
        # Control vector
        control_vec = np.array([cpps[var] for var in control_vars])
        
        # Get estimates
        naive_est = estimator_naive.estimate(noisy_vec, control_vec)
        robust_est, bias_est = estimator_robust.estimate(noisy_vec, control_vec)
        
        log.append({
            'time': t,
            'true_d50': true_vec[0],
            'measured_d50': noisy_vec[0],
            'naive_d50': naive_est[0],
            'robust_d50': robust_est[0],
            'bias_d50': bias_est[0]
        })
        
        # Print progress
        if t % 25 == 0 or t == 149:
            naive_err = naive_est[0] - true_vec[0]
            robust_err = robust_est[0] - true_vec[0]
            print(f"  Step {t:3d}: True={true_vec[0]:6.1f}, "
                  f"Naive={naive_est[0]:6.1f}(err:{naive_err:5.1f}), "
                  f"Robust={robust_est[0]:6.1f}(err:{robust_err:5.1f}), "
                  f"Bias={bias_est[0]:6.1f}")
    
    df_results = pd.DataFrame(log)
    
    # Analyze steady-state performance (last 25 steps)
    steady_state = df_results.tail(25)
    
    true_mean = steady_state['true_d50'].mean()
    naive_mean = steady_state['naive_d50'].mean()
    robust_mean = steady_state['robust_d50'].mean()
    bias_mean = steady_state['bias_d50'].mean()
    
    naive_error = abs(naive_mean - true_mean)
    robust_error = abs(robust_mean - true_mean)
    improvement = max(0, (naive_error - robust_error) / max(naive_error, 1e-6) * 100)
    
    print(f"\n📊 STEADY-STATE ANALYSIS (last 25 steps):")
    print(f"  True state mean:       {true_mean:8.2f} μm")
    print(f"  Naive filter mean:     {naive_mean:8.2f} μm  (error: {naive_error:6.2f} μm)")
    print(f"  Robust filter mean:    {robust_mean:8.2f} μm  (error: {robust_error:6.2f} μm)")
    print(f"  Learned bias:          {bias_mean:8.2f} μm")
    print(f"  Model intercept:       {intercept_C[0]:8.2f} μm")
    print(f"  Error improvement:     {improvement:8.1f}%")
    
    # Test step function response
    print(f"\n🎯 STEP FUNCTION TEST:")
    test_step_response()
    
    # Summary
    if robust_error < naive_error * 0.3:  # At least 70% improvement
        print(f"\n✅ SUCCESS: Bias correction is highly effective!")
        print(f"   Robust filter error ({robust_error:.2f}) << Naive error ({naive_error:.2f})")
        print(f"   Improvement: {improvement:.1f}%")
    elif robust_error < naive_error * 0.7:  # At least 30% improvement
        print(f"\n⚠️  PARTIAL SUCCESS: Some bias correction achieved")
        print(f"   Improvement: {improvement:.1f}%")
    else:
        print(f"\n❌ ISSUE: Bias correction not effective enough")
        print(f"   Need to investigate further")
    
    return {
        'naive_error': naive_error,
        'robust_error': robust_error,
        'improvement': improvement,
        'learned_bias': bias_mean,
        'model_intercept': intercept_C[0]
    }

def test_step_response():
    """Test bias correction on a simple step function."""
    from robust_mpc.estimators import KalmanStateEstimator, BiasAugmentedKalmanStateEstimator
    
    # Create step signal: 100 → 200 at step 30
    n_steps = 60
    true_signal = np.ones(n_steps) * 100.0
    true_signal[30:] = 200.0
    
    # Simple identity dynamics
    A_test = np.array([[1.0]])
    B_test = np.array([[0.0]])
    initial_test = np.array([100.0])
    
    # Create estimators
    naive_test = KalmanStateEstimator(
        transition_matrix=A_test,
        control_matrix=B_test,
        initial_state_mean=initial_test,
        process_noise_std=0.1,
        measurement_noise_std=10.0
    )
    
    robust_test = BiasAugmentedKalmanStateEstimator(
        transition_matrix=A_test,
        control_matrix=B_test,
        initial_state_mean=initial_test,
        process_noise_std=0.1,
        measurement_noise_std=10.0,
        bias_process_noise_std=2.0  # Faster adaptation
    )
    
    # Run test
    results = []
    control_zero = np.array([0.0])
    
    for t in range(n_steps):
        true_val = true_signal[t]
        noisy_val = true_val + np.random.normal(0, 10)
        
        naive_est = naive_test.estimate(np.array([noisy_val]), control_zero)
        robust_est, bias_est = robust_test.estimate(np.array([noisy_val]), control_zero)
        
        results.append({
            'step': t,
            'true': true_val,
            'naive': naive_est[0],
            'robust': robust_est[0],
            'bias': bias_est[0]
        })
    
    df_step = pd.DataFrame(results)
    
    # Analyze pre/post step performance
    pre_step = df_step[df_step['step'] < 30]
    post_step = df_step[df_step['step'] >= 45]  # Allow settling
    
    naive_error_post = abs(post_step['naive'].mean() - post_step['true'].mean())
    robust_error_post = abs(post_step['robust'].mean() - post_step['true'].mean())
    final_bias = df_step['bias'].iloc[-1]
    
    print(f"  Post-step naive error:     {naive_error_post:6.1f}")
    print(f"  Post-step robust error:    {robust_error_post:6.1f}")
    print(f"  Final bias estimate:       {final_bias:6.1f} (expected ~100)")
    
    if robust_error_post < naive_error_post * 0.5:
        print(f"  ✅ Step test PASSED: {robust_error_post:.1f} < {naive_error_post:.1f}")
    else:
        print(f"  ❌ Step test FAILED: {robust_error_post:.1f} >= {naive_error_post:.1f}")

if __name__ == "__main__":
    test_enhanced_bias_correction()