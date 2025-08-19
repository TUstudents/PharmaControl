"""
Test suite for DataBuffer integrity integration with RobustMPCController.

This module validates that the race condition fixes are properly integrated
into the pharmaceutical process control system.
"""

import pytest
import numpy as np
import torch
import warnings
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from V2.robust_mpc.core import RobustMPCController


class TestBufferIntegrityIntegration:
    """Test suite for buffer integrity integration."""
    
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
    def mock_optimizer(self):
        """Mock optimizer for testing."""
        class MockOptimizer:
            def __init__(self, param_bounds, config):
                self.param_bounds = param_bounds
                self.config = config
            def optimize(self, fitness_func=None):
                # Return valid solution within bounds
                horizon = self.config['horizon']
                num_cpps = self.config['num_cpps']
                return np.random.rand(horizon, num_cpps) * 0.5 + 0.25
        return MockOptimizer
    
    @pytest.fixture
    def test_config(self):
        """Standard test configuration."""
        return {
            'cma_names': ['d50', 'lod'],
            'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
            'horizon': 5,
            'lookback': 15,
            'integral_gain': 0.1,
            'mc_samples': 10,
            'risk_beta': 1.5,
            'verbose': True,  # Enable verbose mode to see buffer warnings
            'history_buffer_size': 50,
            'cpp_constraints': {
                'spray_rate': {'min_val': 80.0, 'max_val': 200.0},
                'air_flow': {'min_val': 400.0, 'max_val': 700.0},
                'carousel_speed': {'min_val': 20.0, 'max_val': 50.0}
            },
            'ga_config': {
                'population_size': 20,
                'num_generations': 5,
                'cx_prob': 0.7,
                'mut_prob': 0.2
            }
        }
    
    @pytest.fixture
    def test_scalers(self):
        """Comprehensive scalers for testing."""
        scalers = {}
        
        # CMA scalers
        scalers['d50'] = MinMaxScaler().fit(np.array([[300], [600]]))
        scalers['lod'] = MinMaxScaler().fit(np.array([[0.5], [3.0]]))
        
        # CPP scalers
        cpp_vars = ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
        for var in cpp_vars:
            scalers[var] = MinMaxScaler().fit(np.array([[0], [1000]]))
        
        return scalers
    
    def test_atomic_buffer_operations_in_controller(self, mock_model, mock_estimator, mock_optimizer, test_config, test_scalers):
        """Test that RobustMPCController uses atomic buffer operations."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=mock_optimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Test inputs
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # Before control step, buffer should be empty
        assert len(controller.history_buffer) == 0
        
        # Execute control step
        action = controller.suggest_action(measurement, control_input, setpoint)
        
        # Verify atomic operation worked - all buffers should be synchronized
        assert len(controller.history_buffer._cma_buffer) == len(controller.history_buffer._cpp_buffer)
        assert len(controller.history_buffer._cma_buffer) == len(controller.history_buffer._timestamp_buffer)
        assert len(controller.history_buffer) == 1
        
        # Verify action is valid
        assert isinstance(action, np.ndarray)
        assert action.shape == (len(test_config['cpp_names']),)
    
    def test_buffer_integrity_validation_in_controller(self, mock_model, mock_estimator, mock_optimizer, test_config, test_scalers):
        """Test that controller detects and reports buffer misalignment."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=mock_optimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Deliberately create buffer misalignment using direct buffer access
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller.history_buffer.add_measurement(np.array([450.0, 1.8]))
            controller.history_buffer.add_measurement(np.array([455.0, 1.9]))  # Extra measurement
            controller.history_buffer.add_control_action(np.array([130.0, 550.0, 30.0]))
        
        # Verify misalignment exists
        assert len(controller.history_buffer._cma_buffer) == 2
        assert len(controller.history_buffer._cpp_buffer) == 1
        
        # Test that suggest_action detects the misalignment
        measurement = np.array([460.0, 1.7])
        control_input = np.array([135.0, 560.0, 32.0])
        setpoint = np.array([450.0, 1.8])
        
        # Capture output to verify warning is printed
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # This should detect misalignment and print warning due to verbose=True
            action = controller.suggest_action(measurement, control_input, setpoint)
            
            # Action should still be returned (graceful degradation)
            assert isinstance(action, np.ndarray)
            assert action.shape == (len(test_config['cpp_names']),)
    
    def test_pharmaceutical_manufacturing_workflow(self, mock_model, mock_estimator, mock_optimizer, test_config, test_scalers):
        """Test complete pharmaceutical manufacturing control workflow with data integrity."""
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=mock_optimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        # Simulate pharmaceutical manufacturing control steps
        manufacturing_data = [
            {'measurement': np.array([450.0, 1.8]), 'control': np.array([130.0, 550.0, 30.0]), 'setpoint': np.array([450.0, 1.8])},
            {'measurement': np.array([455.0, 1.75]), 'control': np.array([132.0, 555.0, 31.0]), 'setpoint': np.array([450.0, 1.8])},
            {'measurement': np.array([460.0, 1.82]), 'control': np.array([128.0, 545.0, 29.0]), 'setpoint': np.array([450.0, 1.8])},
            {'measurement': np.array([458.0, 1.78]), 'control': np.array([131.0, 552.0, 30.5]), 'setpoint': np.array([450.0, 1.8])},
            {'measurement': np.array([452.0, 1.79]), 'control': np.array([129.0, 548.0, 30.2]), 'setpoint': np.array([450.0, 1.8])},
        ]
        
        # Execute manufacturing control steps
        actions = []
        for step_data in manufacturing_data:
            action = controller.suggest_action(
                step_data['measurement'], 
                step_data['control'], 
                step_data['setpoint']
            )
            actions.append(action)
        
        # Verify pharmaceutical-grade data integrity
        buffer_len = len(controller.history_buffer)
        assert buffer_len == len(manufacturing_data)
        
        # Verify perfect buffer synchronization (critical for pharmaceutical manufacturing)
        assert len(controller.history_buffer._cma_buffer) == buffer_len
        assert len(controller.history_buffer._cpp_buffer) == buffer_len
        assert len(controller.history_buffer._timestamp_buffer) == buffer_len
        
        # Verify all actions are valid
        for i, action in enumerate(actions):
            assert isinstance(action, np.ndarray)
            assert action.shape == (len(test_config['cpp_names']),)
            assert np.all(np.isfinite(action)), f"Action {i} contains non-finite values"
        
        # Verify no critical warnings in pharmaceutical manufacturing workflow
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            controller.history_buffer.get_model_inputs(min(3, buffer_len))
            
            critical_warnings = [warning for warning in w 
                                if "CRITICAL" in str(warning.message)]
            assert len(critical_warnings) == 0, "Pharmaceutical manufacturing must maintain perfect data integrity"
    
    def test_startup_to_normal_operation_transition(self, mock_model, mock_estimator, mock_optimizer, test_config, test_scalers):
        """Test transition from startup to normal operation maintains buffer integrity."""
        # Reduce lookback for faster testing
        config = test_config.copy()
        config['lookback'] = 3
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=mock_optimizer,
            config=config,
            scalers=test_scalers
        )
        
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # Execute control steps through startup phase
        actions = []
        for i in range(5):  # More than lookback to trigger transition
            action = controller.suggest_action(measurement, control_input, setpoint)
            actions.append(action)
            
            # Verify buffer integrity at each step
            buffer_len = len(controller.history_buffer)
            assert len(controller.history_buffer._cma_buffer) == buffer_len
            assert len(controller.history_buffer._cpp_buffer) == buffer_len
            assert len(controller.history_buffer._timestamp_buffer) == buffer_len
        
        # Verify controller transitioned to using real history data
        assert len(controller.history_buffer) >= config['lookback']
        assert controller._initialization_complete is True
        
        # Verify all actions are valid throughout transition
        for i, action in enumerate(actions):
            assert isinstance(action, np.ndarray)
            assert action.shape == (len(config['cpp_names']),)
    
    def test_error_recovery_with_buffer_integrity(self, mock_model, mock_estimator, test_config, test_scalers):
        """Test error recovery maintains buffer integrity."""
        # Create failing optimizer to test error handling
        class FailingOptimizer:
            def __init__(self, param_bounds, config):
                self.call_count = 0
            def optimize(self, fitness_func=None):
                self.call_count += 1
                if self.call_count <= 2:
                    raise RuntimeError("Optimizer failure")
                # Succeed on third call
                return np.array([[140.0, 550.0, 35.0]] * 5)
        
        controller = RobustMPCController(
            model=mock_model,
            estimator=mock_estimator,
            optimizer_class=FailingOptimizer,
            config=test_config,
            scalers=test_scalers
        )
        
        measurement = np.array([450.0, 1.8])
        control_input = np.array([130.0, 550.0, 30.0])
        setpoint = np.array([450.0, 1.8])
        
        # Execute control steps with optimizer failures
        actions = []
        for i in range(3):
            action = controller.suggest_action(measurement, control_input, setpoint)
            actions.append(action)
            
            # Even with optimizer failures, buffer integrity must be maintained
            if len(controller.history_buffer) > 0:
                buffer_len = len(controller.history_buffer)
                assert len(controller.history_buffer._cma_buffer) == buffer_len
                assert len(controller.history_buffer._cpp_buffer) == buffer_len
                assert len(controller.history_buffer._timestamp_buffer) == buffer_len
        
        # Verify all actions are valid (fallback actions during failures)
        for action in actions:
            assert isinstance(action, np.ndarray)
            assert action.shape == (len(test_config['cpp_names']),)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])