"""
Test suite for offset scaling functionality in RobustMPCController.

This module contains comprehensive tests to validate the critical fix for disturbance
estimate scaling, ensuring proper offset-free MPC functionality.
"""

import pytest
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from V2.robust_mpc.core import RobustMPCController


class TestOffsetScaling:
    """Test suite for offset scaling functionality."""
    
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
        """Standard test configuration."""
        return {
            'cma_names': ['d50', 'lod'],
            'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
            'horizon': 10,
            'lookback': 15,
            'integral_gain': 0.1,
            'mc_samples': 30,
            'risk_beta': 1.5,
            'cpp_constraints': {
                'spray_rate': {'min_val': 80.0, 'max_val': 200.0},
                'air_flow': {'min_val': 400.0, 'max_val': 700.0},
                'carousel_speed': {'min_val': 20.0, 'max_val': 50.0}
            },
            'ga_config': {
                'population_size': 30,
                'num_generations': 15,
                'cx_prob': 0.7,
                'mut_prob': 0.2
            }
        }
    
    @pytest.fixture
    def test_scalers(self):
        """Realistic pharmaceutical scalers."""
        scalers = {}
        
        # CMA scalers with known ranges
        d50_scaler = MinMaxScaler()
        d50_scaler.fit(np.array([[300], [600]]))  # Range: 300 μm
        scalers['d50'] = d50_scaler
        
        lod_scaler = MinMaxScaler()
        lod_scaler.fit(np.array([[0.5], [3.0]]))  # Range: 2.5%
        scalers['lod'] = lod_scaler
        
        # CPP scalers
        spray_scaler = MinMaxScaler()
        spray_scaler.fit(np.array([[80], [200]]))
        scalers['spray_rate'] = spray_scaler
        
        air_scaler = MinMaxScaler()
        air_scaler.fit(np.array([[400], [700]]))
        scalers['air_flow'] = air_scaler
        
        speed_scaler = MinMaxScaler()
        speed_scaler.fit(np.array([[20], [50]]))
        scalers['carousel_speed'] = speed_scaler
        
        # Soft sensor scalers
        energy_scaler = MinMaxScaler()
        energy_scaler.fit(np.array([[1.6], [10.0]]))
        scalers['specific_energy'] = energy_scaler
        
        froude_scaler = MinMaxScaler()
        froude_scaler.fit(np.array([[40.8], [254.9]]))
        scalers['froude_number_proxy'] = froude_scaler
        
        return scalers
    
    @pytest.fixture
    def controller(self, mock_model, mock_estimator, test_config, test_scalers):
        """RobustMPCController instance for testing."""
        return RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,
            config=test_config,
            scalers=test_scalers
        )
    
    def test_offset_scaling_mathematical_correctness(self, controller, test_scalers):
        """Test that offset scaling follows correct mathematical formula."""
        # Test offset: +50 μm in d50, +0.2% in LOD
        offset_unscaled = np.array([50.0, 0.2])
        
        # Get scaled offset
        offset_scaled = controller._scale_cma_offset(offset_unscaled)
        
        # Manually calculate expected values
        d50_scaler = test_scalers['d50']
        lod_scaler = test_scalers['lod']
        
        # Expected: offset * scale_factor (no translation)
        expected_d50 = 50.0 * d50_scaler.scale_[0]  # 50.0 / 300.0 = 0.1667
        expected_lod = 0.2 * lod_scaler.scale_[0]   # 0.2 / 2.5 = 0.08
        
        # Verify mathematical correctness
        assert np.isclose(offset_scaled[0], expected_d50, rtol=1e-6)
        assert np.isclose(offset_scaled[1], expected_lod, rtol=1e-6)
        
        # Verify proportional scaling
        assert np.isclose(offset_scaled[0], 50.0 / 300.0, rtol=1e-6)
        assert np.isclose(offset_scaled[1], 0.2 / 2.5, rtol=1e-6)
    
    def test_offset_scaling_vs_value_scaling_difference(self, controller, test_scalers):
        """Test that offset scaling differs from value scaling as expected."""
        # Test with a known value that will show the translation difference
        test_vector = np.array([400.0, 1.5])  # Values for d50=400μm, LOD=1.5%
        
        # Scale as value (with translation)
        value_scaled = controller._scale_cma_vector(test_vector)
        
        # Scale as offset (no translation)
        offset_scaled = controller._scale_cma_offset(test_vector)
        
        # Values should be different due to translation
        assert not np.allclose(value_scaled, offset_scaled)
        
        # Calculate expected values manually for verification
        d50_scaler = test_scalers['d50']  # Range 300-600, so 400 should give (400-300)/(600-300) = 100/300 = 0.333
        lod_scaler = test_scalers['lod']  # Range 0.5-3.0, so 1.5 should give (1.5-0.5)/(3.0-0.5) = 1.0/2.5 = 0.4
        
        # For value scaling (with translation): (value - min) / (max - min)
        expected_value_d50 = (400.0 - 300.0) / (600.0 - 300.0)  # = 0.333
        expected_value_lod = (1.5 - 0.5) / (3.0 - 0.5)          # = 0.4
        
        # For offset scaling (no translation): value / (max - min)
        expected_offset_d50 = 400.0 / (600.0 - 300.0)  # = 1.333
        expected_offset_lod = 1.5 / (3.0 - 0.5)        # = 0.6
        
        # Verify value scaling
        assert np.isclose(value_scaled[0], expected_value_d50, rtol=1e-6)
        assert np.isclose(value_scaled[1], expected_value_lod, rtol=1e-6)
        
        # Verify offset scaling
        assert np.isclose(offset_scaled[0], expected_offset_d50, rtol=1e-6)
        assert np.isclose(offset_scaled[1], expected_offset_lod, rtol=1e-6)
        
        # Offset scaling should be higher than value scaling for positive values
        assert offset_scaled[0] > value_scaled[0]
        assert offset_scaled[1] > value_scaled[1]
    
    def test_zero_offset_preservation(self, controller):
        """Test that zero offsets remain zero after scaling."""
        zero_offset = np.array([0.0, 0.0])
        scaled_zero = controller._scale_cma_offset(zero_offset)
        
        # Zero offsets should remain zero (critical for integral action)
        assert np.allclose(scaled_zero, np.array([0.0, 0.0]))
    
    def test_offset_sign_preservation(self, controller):
        """Test that offset signs are preserved during scaling."""
        # Test positive and negative offsets
        positive_offset = np.array([50.0, 0.2])
        negative_offset = np.array([-50.0, -0.2])
        
        positive_scaled = controller._scale_cma_offset(positive_offset)
        negative_scaled = controller._scale_cma_offset(negative_offset)
        
        # Signs should be preserved
        assert np.all(positive_scaled > 0)
        assert np.all(negative_scaled < 0)
        
        # Magnitudes should be equal
        assert np.allclose(np.abs(positive_scaled), np.abs(negative_scaled))
    
    def test_offset_proportionality(self, controller):
        """Test that offset scaling preserves proportional relationships."""
        # Test different magnitude offsets
        offset_1x = np.array([30.0, 0.1])
        offset_2x = np.array([60.0, 0.2])  # Double magnitude
        
        scaled_1x = controller._scale_cma_offset(offset_1x)
        scaled_2x = controller._scale_cma_offset(offset_2x)
        
        # Proportional relationship should be preserved
        assert np.allclose(scaled_2x, 2.0 * scaled_1x, rtol=1e-6)
    
    def test_offset_scaling_input_validation(self, controller):
        """Test input validation for offset scaling."""
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Expected 1D offset vector"):
            controller._scale_cma_offset(np.array([[1.0, 2.0]]))  # 2D array
        
        # Test wrong type
        with pytest.raises(TypeError):
            controller._scale_cma_offset([1.0, 2.0])  # List instead of array
        
        # Test empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            controller._scale_cma_offset(np.array([]))
        
        # Test non-finite values
        with pytest.raises(ValueError, match="Input contains non-finite values"):
            controller._scale_cma_offset(np.array([np.nan, 1.0]))
    
    def test_integral_action_with_correct_scaling(self, controller):
        """Test that integral action works correctly with proper offset scaling."""
        # Set up a disturbance estimate
        controller.disturbance_estimate = np.array([30.0, 0.15])  # Realistic disturbance
        
        # Create fitness function with mock data
        past_cmas_scaled = np.random.rand(15, 2)
        past_cpps_scaled = np.random.rand(15, 5)
        target_plan = np.array([[450.0, 1.8]] * 10)  # Constant setpoint
        
        fitness_func = controller._get_fitness_function(
            past_cmas_scaled, past_cpps_scaled, target_plan
        )
        
        # Test that fitness function can be called without error
        test_control_plan = np.array([130.0, 550.0, 30.0] * 10)  # Flattened control plan
        try:
            cost = fitness_func(test_control_plan)
            assert isinstance(cost, float)
            assert np.isfinite(cost)
        except Exception as e:
            pytest.fail(f"Fitness function failed with proper offset scaling: {e}")
    
    def test_disturbance_estimate_integration(self, controller):
        """Test integration of disturbance estimate with model predictions."""
        # Set known disturbance
        known_disturbance = np.array([25.0, 0.1])
        controller.disturbance_estimate = known_disturbance
        
        # Mock prediction data
        past_cmas_scaled = np.zeros((15, 2))
        past_cpps_scaled = np.zeros((15, 5))
        target_plan = np.array([[450.0, 1.8]] * 10)
        
        # Create fitness function
        fitness_func = controller._get_fitness_function(
            past_cmas_scaled, past_cpps_scaled, target_plan
        )
        
        # Test control plan
        control_plan = np.array([130.0, 550.0, 30.0] * 10)
        
        # Should execute without error with proper offset scaling
        cost = fitness_func(control_plan)
        assert np.isfinite(cost)

    def test_offset_scaling_pharmaceutical_realism(self, controller, test_scalers):
        """Test offset scaling with realistic pharmaceutical disturbances."""
        # Realistic disturbances for granulation process
        disturbances = [
            np.array([10.0, 0.05]),   # Small disturbance: +10μm d50, +0.05% LOD
            np.array([50.0, 0.2]),    # Medium disturbance: +50μm d50, +0.2% LOD
            np.array([-30.0, -0.15]), # Negative disturbance: -30μm d50, -0.15% LOD
        ]
        
        for disturbance in disturbances:
            scaled_dist = controller._scale_cma_offset(disturbance)
            
            # Scaled offsets should be reasonable magnitudes
            assert np.all(np.abs(scaled_dist) < 1.0)  # Should be within reasonable scale
            assert np.all(np.abs(scaled_dist) > 0.0)  # Should not be zero (unless input is zero)
            
            # Test that scaling is reversible (approximately)
            d50_scaler = test_scalers['d50']
            lod_scaler = test_scalers['lod']
            
            unscaled_d50 = scaled_dist[0] / d50_scaler.scale_[0]
            unscaled_lod = scaled_dist[1] / lod_scaler.scale_[0]
            
            assert np.isclose(unscaled_d50, disturbance[0], rtol=1e-6)
            assert np.isclose(unscaled_lod, disturbance[1], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])