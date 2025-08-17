"""
Test suite for error handling in RobustMPCController.

This module validates that the controller handles optimizer failures gracefully
and implements appropriate fallback strategies for industrial deployment.
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


class TestErrorHandling:
    """Test suite for error handling functionality."""
    
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
        """Standard test configuration with constraints."""
        return {
            'cma_names': ['d50', 'lod'],
            'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
            'horizon': 10,
            'lookback': 15,
            'integral_gain': 0.1,
            'mc_samples': 30,
            'risk_beta': 1.5,
            'verbose': False,
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
        
        # CMA scalers
        d50_scaler = MinMaxScaler()
        d50_scaler.fit(np.array([[300], [600]]))
        scalers['d50'] = d50_scaler
        
        lod_scaler = MinMaxScaler()
        lod_scaler.fit(np.array([[0.5], [3.0]]))
        scalers['lod'] = lod_scaler
        
        # CPP scalers
        for name in ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']:
            scaler = MinMaxScaler()
            scaler.fit(np.array([[0], [100]]))
            scalers[name] = scaler
        
        return scalers
    
    def test_optimizer_failure_handling(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test that optimizer failures are handled gracefully."""
        
        # Create a failing optimizer class
        class FailingOptimizer:
            def __init__(self, fitness_func, param_bounds, config):
                self.fitness_func = fitness_func
                self.param_bounds = param_bounds
                self.config = config
                
            def optimize(self):
                raise RuntimeError("Genetic algorithm failed to converge")
        
        # Create controller with failing optimizer
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=FailingOptimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Test inputs
        noisy_measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # Should not raise exception - should return fallback action
        action = controller.suggest_action(noisy_measurement, control_input, setpoint)
        
        # Validate fallback action
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)  # Number of CPPs
        assert np.all(np.isfinite(action))
        
        # Should be within constraint bounds (safe defaults)
        assert 80.0 <= action[0] <= 200.0   # spray_rate
        assert 400.0 <= action[1] <= 700.0  # air_flow  
        assert 20.0 <= action[2] <= 50.0    # carousel_speed
    
    def test_successful_then_failing_optimizer(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test fallback to last successful action after initial success."""
        
        class SometimesFailingOptimizer:
            def __init__(self, fitness_func, param_bounds, config):
                self.fitness_func = fitness_func
                self.param_bounds = param_bounds
                self.config = config
                self.call_count = 0
                
            def optimize(self):
                self.call_count += 1
                if self.call_count == 1:
                    # First call succeeds
                    return np.array([[120.0, 500.0, 25.0]] * 10)  # Horizon length
                else:
                    # Subsequent calls fail
                    raise RuntimeError("Optimization failed")
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=SometimesFailingOptimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Test inputs
        noisy_measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # First call should succeed
        action1 = controller.suggest_action(noisy_measurement, control_input, setpoint)
        expected_action = np.array([120.0, 500.0, 25.0])
        assert np.allclose(action1, expected_action)
        
        # Second call should fail but return last successful action
        action2 = controller.suggest_action(noisy_measurement, control_input, setpoint)
        assert np.allclose(action2, expected_action)  # Should be same as first successful action
    
    def test_fallback_action_validation(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test validation of fallback actions."""
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=None,  # Not used in this test
            config=test_config,
            scalers=test_scalers
        )
        
        # Test valid action
        valid_action = np.array([120.0, 500.0, 25.0])
        assert controller._validate_control_action(valid_action) == True
        
        # Test invalid actions
        assert controller._validate_control_action(None) == False
        assert controller._validate_control_action([120.0, 500.0, 25.0]) == False  # List instead of array
        assert controller._validate_control_action(np.array([120.0, 500.0])) == False  # Wrong size
        assert controller._validate_control_action(np.array([np.nan, 500.0, 25.0])) == False  # NaN
        assert controller._validate_control_action(np.array([50.0, 500.0, 25.0])) == False  # Below min constraint
        assert controller._validate_control_action(np.array([250.0, 500.0, 25.0])) == False  # Above max constraint
    
    def test_safe_default_fallback(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test that safe default fallback uses constraint midpoints."""
        
        class AlwaysFailingOptimizer:
            def __init__(self, fitness_func, param_bounds, config):
                pass
                
            def optimize(self):
                raise RuntimeError("Always fails")
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=AlwaysFailingOptimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Test with invalid current control input to force safe defaults
        noisy_measurement = np.array([450.0, 1.8])
        invalid_control_input = np.array([1000.0, 1000.0, 1000.0])  # Way out of bounds
        setpoint = np.array([450.0, 1.8])
        
        action = controller.suggest_action(noisy_measurement, invalid_control_input, setpoint)
        
        # Should be constraint midpoints
        expected_spray = (80.0 + 200.0) / 2.0    # 140.0
        expected_air = (400.0 + 700.0) / 2.0     # 550.0 
        expected_speed = (20.0 + 50.0) / 2.0     # 35.0
        
        expected_action = np.array([expected_spray, expected_air, expected_speed])
        assert np.allclose(action, expected_action)
    
    def test_null_optimizer_result_handling(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test handling of optimizer returning None or empty result."""
        
        class NullOptimizer:
            def __init__(self, fitness_func, param_bounds, config):
                pass
                
            def optimize(self):
                return None  # Invalid result
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=NullOptimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Test inputs
        noisy_measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # Should handle None result gracefully
        action = controller.suggest_action(noisy_measurement, control_input, setpoint)
        
        # Should return valid fallback action
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])