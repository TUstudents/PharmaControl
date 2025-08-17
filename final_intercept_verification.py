#!/usr/bin/env python3
"""
Final verification of intercept-based Kalman filter implementation.
Testing with scenarios where intercept correction should clearly demonstrate its effect.
"""

import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.abspath('V1/src/'))
sys.path.insert(0, os.path.abspath('V2/'))

from robust_mpc.estimators import (
    KalmanStateEstimator, 
    ProcessBiasKalmanEstimator,
    MeasurementBiasKalmanEstimator
)

def test_constant_offset_scenario():
    """Test scenario where measurement has a constant systematic offset (MEASUREMENT BIAS)."""
    
    print("🧪 Testing Constant Offset Scenario (MEASUREMENT BIAS)")
    print("=" * 60)
    
    # Scenario: True system is always at 150, but measurements are systematically high by 30
    # This is a MEASUREMENT BIAS problem (sensor calibration error)
    true_value = 150.0
    systematic_offset = 30.0  # Measurements are consistently 30 units too high
    
    # Generate measurements with systematic offset
    n_steps = 50
    measurements = []
    np.random.seed(42)
    
    for i in range(n_steps):
        # measurement = true_value + systematic_offset + noise
        measurement = true_value + systematic_offset + np.random.normal(0, 5)
        measurements.append(measurement)
    
    print(f"📊 Scenario setup:")
    print(f"  True system value: {true_value}")
    print(f"  Systematic measurement offset: +{systematic_offset}")
    print(f"  Expected measurement range: ~{true_value + systematic_offset} ± noise")
    print(f"  Expected corrected estimate: ~{true_value}")
    print(f"  This simulates SENSOR BIAS, not process bias")
    
    # Setup filters
    A = np.array([[1.0]])  # Identity dynamics
    B = np.array([[0.0]])  # No control
    control = np.array([0.0])
    
    # Naive filter: will track the biased measurements
    naive_filter = KalmanStateEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=np.array([true_value + systematic_offset]),  # Start near measurements
        process_noise_std=0.1,
        measurement_noise_std=5.0
    )
    
    # Measurement bias filter: should correct for the systematic sensor offset
    measurement_bias_filter = MeasurementBiasKalmanEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=np.array([true_value]),  # Start at TRUE value
        intercept_term=np.array([systematic_offset]),  # Known measurement bias
        process_noise_std=0.1,
        measurement_noise_std=5.0
    )
    
    # Run simulation
    naive_estimates = []
    measurement_bias_estimates = []
    
    print(f"\n⏳ Running simulation...")
    
    for i, measurement in enumerate(measurements):
        naive_est = naive_filter.estimate(np.array([measurement]), control)
        bias_corrected_est = measurement_bias_filter.estimate(np.array([measurement]), control)
        
        naive_estimates.append(naive_est[0])
        measurement_bias_estimates.append(bias_corrected_est[0])
        
        if i % 10 == 0 or i == len(measurements) - 1:
            print(f"  Step {i:2d}: Measurement={measurement:6.1f}, "
                  f"Naive={naive_est[0]:6.1f}, "
                  f"Bias-Corrected={bias_corrected_est[0]:6.1f}")
    
    # Calculate errors relative to true value
    naive_errors = [abs(est - true_value) for est in naive_estimates]
    bias_corrected_errors = [abs(est - true_value) for est in measurement_bias_estimates]
    
    naive_avg_error = np.mean(naive_errors)
    bias_corrected_avg_error = np.mean(bias_corrected_errors)
    
    print(f"\n📊 Performance Analysis:")
    print(f"  Naive average error:        {naive_avg_error:6.2f}")
    print(f"  Bias-corrected average error: {bias_corrected_avg_error:6.2f}")
    print(f"  Error reduction:            {naive_avg_error - bias_corrected_avg_error:6.2f}")
    print(f"  Improvement:                {(naive_avg_error - bias_corrected_avg_error)/naive_avg_error*100:5.1f}%")
    
    # Check final convergence values
    final_naive = np.mean(naive_estimates[-10:])
    final_bias_corrected = np.mean(measurement_bias_estimates[-10:])
    
    print(f"\n📊 Final Convergence:")
    print(f"  True value:                 {true_value:6.1f}")
    print(f"  Naive final estimate:       {final_naive:6.1f} (error: {abs(final_naive - true_value):5.1f})")
    print(f"  Bias-corrected final estimate: {final_bias_corrected:6.1f} (error: {abs(final_bias_corrected - true_value):5.1f})")
    
    # Success criteria
    success = bias_corrected_avg_error < naive_avg_error * 0.2  # Expect massive improvement
    
    if success:
        print(f"\n✅ SUCCESS: Measurement bias correction is working effectively!")
        print(f"   Bias correction provides significant error reduction")
    else:
        print(f"\n❌ FAILURE: Measurement bias correction not providing expected benefit")
        print(f"   Expected bias correction to remove systematic sensor offset")
    
    return success, naive_avg_error, bias_corrected_avg_error

def test_process_bias_scenario():
    """Test with actual process bias (missing intercept in dynamics)."""
    
    print(f"\n🧪 Testing Process Bias Scenario")
    print("=" * 50)
    
    # Scenario: Process has missing intercept that our model doesn't capture
    # This simulates the pharmaceutical granulation case (missing sklearn intercept)
    initial_value = 100.0
    missing_intercept = 5.0  # Process grows by 5 per step (missing from our model)
    
    n_steps = 30
    
    print(f"📊 Process bias scenario:")
    print(f"  True process: x[k+1] = x[k] + {missing_intercept} + noise")
    print(f"  Our model: x[k+1] = x[k] + noise (missing intercept)")
    print(f"  Measurements: y[k] = x[k] + noise (perfect sensors)")
    print(f"  Expected: filter should learn bias ≈ {missing_intercept}")
    
    # Simulate the true process with missing intercept
    np.random.seed(42)
    x_true = [initial_value]
    measurements = []
    
    for i in range(n_steps):
        # True process evolution with missing intercept
        if i > 0:
            x_next = x_true[-1] + missing_intercept + np.random.normal(0, 0.5)
            x_true.append(x_next)
        
        # Perfect measurement of true state
        measurement = x_true[i] + np.random.normal(0, 2.0)
        measurements.append(measurement)
    
    print(f"  True states (first 3): {x_true[:3]}")
    print(f"  True states (last 3): {x_true[-3:]}")
    print(f"  Expected final state: ~{initial_value + n_steps * missing_intercept}")
    
    # Setup filters
    A = np.array([[1.0]])  # Our (incorrect) model: x[k+1] = x[k]
    B = np.array([[0.0]])  # No control
    control = np.array([0.0])
    
    # Naive filter: uses incorrect model without intercept
    naive_filter = KalmanStateEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=np.array([initial_value]),
        process_noise_std=0.5,
        measurement_noise_std=2.0
    )
    
    # Process bias filter: corrects for missing intercept in dynamics
    process_bias_filter = ProcessBiasKalmanEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=np.array([initial_value]),
        intercept_term=np.array([missing_intercept]),  # Known missing intercept
        process_noise_std=0.5,
        measurement_noise_std=2.0
    )
    
    # Run simulation
    naive_estimates = []
    process_bias_estimates = []
    
    print(f"\n⏳ Running process bias simulation...")
    
    for i, measurement in enumerate(measurements):
        true_state = x_true[i]
        
        naive_est = naive_filter.estimate(np.array([measurement]), control)
        process_bias_est = process_bias_filter.estimate(np.array([measurement]), control)
        
        naive_estimates.append(naive_est[0])
        process_bias_estimates.append(process_bias_est[0])
        
        if i % 5 == 0 or i == len(measurements) - 1:
            print(f"  Step {i:2d}: True={true_state:6.1f}, "
                  f"Naive={naive_est[0]:6.1f}, "
                  f"Process-Bias={process_bias_est[0]:6.1f}")
    
    # Calculate tracking errors relative to true states
    naive_errors = [abs(naive_estimates[i] - x_true[i]) for i in range(n_steps)]
    process_bias_errors = [abs(process_bias_estimates[i] - x_true[i]) for i in range(n_steps)]
    
    naive_avg_error = np.mean(naive_errors)
    process_bias_avg_error = np.mean(process_bias_errors)
    
    final_naive = np.mean(naive_estimates[-5:])
    final_process_bias = np.mean(process_bias_estimates[-5:])
    final_true = np.mean(x_true[-5:])
    
    print(f"\n📊 Process Bias Correction Analysis:")
    print(f"  Naive average error:           {naive_avg_error:6.2f}")
    print(f"  Process-bias average error:    {process_bias_avg_error:6.2f}")
    print(f"  Error reduction:               {naive_avg_error - process_bias_avg_error:6.2f}")
    print(f"  Improvement:                   {(naive_avg_error - process_bias_avg_error)/naive_avg_error*100:5.1f}%")
    
    print(f"\n📊 Final Convergence:")
    print(f"  Final true state:              {final_true:6.1f}")
    print(f"  Naive final estimate:          {final_naive:6.1f} (error: {abs(final_naive - final_true):5.1f})")
    print(f"  Process-bias final estimate:   {final_process_bias:6.1f} (error: {abs(final_process_bias - final_true):5.1f})")
    
    # Success criteria
    success = process_bias_avg_error < naive_avg_error * 0.5
    
    if success:
        print(f"\n✅ SUCCESS: Process bias correction is working effectively!")
        print(f"   Process bias filter correctly handles missing intercept")
    else:
        print(f"\n⚠️  PARTIAL: Some improvement but not as dramatic as expected")
        print(f"   Process bias correction shows benefit but may need tuning")
    
    return success

def main():
    """Run corrected bias verification tests."""
    
    print("🔍 CORRECTED BIAS MODEL VERIFICATION")
    print("=" * 60)
    print("Testing both measurement bias and process bias paradigms")
    
    # Test 1: Measurement bias scenario (sensor calibration error)
    success1, naive_err1, bias_corrected_err1 = test_constant_offset_scenario()
    
    # Test 2: Process bias scenario (missing model dynamics)
    success2 = test_process_bias_scenario()
    
    print(f"\n" + "=" * 60)
    print("CORRECTED IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    if success1:
        print(f"✅ MEASUREMENT BIAS: Working correctly")
        print(f"   - Successfully corrects sensor calibration errors")
        print(f"   - Provides {(naive_err1 - bias_corrected_err1)/naive_err1*100:.1f}% error reduction")
        print(f"   - Subtracts bias from measurement before filtering")
    else:
        print(f"❌ MEASUREMENT BIAS: Issues remain")
    
    if success2:
        print(f"✅ PROCESS BIAS: Working correctly")
        print(f"   - Successfully corrects missing model dynamics")
        print(f"   - Adds intercept to process evolution equation")
        print(f"   - Appropriate for pharmaceutical granulation case")
    else:
        print(f"⚠️  PROCESS BIAS: Some improvement but could be better")
    
    print(f"\n💡 Key Insights from Corrected Implementation:")
    print(f"   1. Different bias types require different mathematical models")
    print(f"   2. Measurement bias: y = x + bias (corrected by subtracting bias)")
    print(f"   3. Process bias: x[k+1] = A*x[k] + bias (corrected by adding to dynamics)")
    print(f"   4. Model-scenario mismatch causes filter instability and divergence")
    
    print(f"\n🎯 Usage Guidelines:")
    print(f"   - MeasurementBiasKalmanEstimator: For sensor calibration errors")
    print(f"   - ProcessBiasKalmanEstimator: For missing sklearn intercept")
    print(f"   - BiasAugmentedKalmanStateEstimator: For adaptive/unknown bias")
    
    overall_success = success1 and success2
    if overall_success:
        print(f"\n🎉 OVERALL: Both bias correction paradigms verified successfully!")
    else:
        print(f"\n⚠️  OVERALL: Some implementations need further tuning")

if __name__ == "__main__":
    main()