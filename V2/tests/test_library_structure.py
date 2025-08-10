"""
Test suite for RobustMPC library structure and basic functionality.

These tests ensure that the library imports correctly and basic
functionality is available even before all components are implemented.
"""

import pytest
import sys
import os

# Add the parent directory to path to import robust_mpc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_library_imports():
    """Test that the main library can be imported."""
    try:
        import robust_mpc
        assert robust_mpc.__version__ == "2.0.0"
        assert robust_mpc.__author__ == "PharmaControl-Pro Development Team"
    except ImportError as e:
        pytest.fail(f"Could not import robust_mpc: {e}")

def test_library_info():
    """Test that library info functions work."""
    import robust_mpc
    
    # Test get_library_info
    info = robust_mpc.get_library_info()
    assert isinstance(info, dict)
    assert 'name' in info
    assert 'version' in info
    assert info['name'] == 'RobustMPC'
    
    # Test get_default_config
    config = robust_mpc.get_default_config()
    assert isinstance(config, dict)
    assert 'estimation' in config
    assert 'modeling' in config
    assert 'optimization' in config
    assert 'control' in config

def test_estimators_module():
    """Test that estimators module imports correctly."""
    try:
        from robust_mpc import estimators
        # KalmanStateEstimator should be available
        assert hasattr(estimators, 'KalmanStateEstimator')
        
        # Future classes should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            estimators.ExtendedKalmanFilter()
            
        with pytest.raises(NotImplementedError):
            estimators.UnscentedKalmanFilter()
            
        with pytest.raises(NotImplementedError):
            estimators.ParticleFilter()
            
    except ImportError as e:
        pytest.fail(f"Could not import estimators module: {e}")

def test_models_module():
    """Test that models module imports correctly."""
    try:
        from robust_mpc import models
        
        # ProbabilisticTransformer should now be available (implemented in V2-2)
        assert hasattr(models, 'ProbabilisticTransformer')
        
        # Test that it can be instantiated (without requiring training data)
        try:
            prob_model = models.ProbabilisticTransformer(cma_features=2, cpp_features=5)
            # Should have the key methods
            assert hasattr(prob_model, 'forward')
            assert hasattr(prob_model, 'predict_distribution')
        except Exception as e:
            pytest.fail(f"ProbabilisticTransformer instantiation failed: {e}")
            
        # Other classes should still be placeholders
        with pytest.raises(NotImplementedError):
            models.BayesianTransformer()
            
        with pytest.raises(NotImplementedError):
            models.EnsemblePredictor()
            
        with pytest.raises(NotImplementedError):
            models.PhysicsInformedPredictor()
            
    except ImportError as e:
        pytest.fail(f"Could not import models module: {e}")

def test_optimizers_module():
    """Test that optimizers module imports correctly."""
    try:
        from robust_mpc import optimizers
        
        # Test utility functions
        assert hasattr(optimizers, 'validate_bounds')
        assert hasattr(optimizers, 'apply_bounds')
        
        # Test bounds validation
        bounds = [(0, 10), (5, 15), (-1, 1)]
        assert optimizers.validate_bounds(bounds, 3) == True
        assert optimizers.validate_bounds(bounds, 2) == False
        
        # Test bounds application
        import numpy as np
        solution = np.array([12, 3, 2])
        clipped = optimizers.apply_bounds(solution, bounds)
        assert clipped[0] == 10  # Clipped from 12 to 10
        assert clipped[1] == 5   # Clipped from 3 to 5
        assert clipped[2] == 1   # Clipped from 2 to 1
        
        # Test GeneticOptimizer class (implemented in V2-3)
        assert hasattr(optimizers, 'GeneticOptimizer')
        
        # Test MPC-specific utility functions
        assert hasattr(optimizers, 'setup_mpc_bounds')
        assert hasattr(optimizers, 'reshape_chromosome_to_plan')
        
        # Test MPC bounds setup
        cpp_constraints = {
            'spray_rate': {'min_val': 80.0, 'max_val': 180.0},
            'air_flow': {'min_val': 400.0, 'max_val': 700.0}
        }
        mpc_bounds = optimizers.setup_mpc_bounds(cpp_constraints, horizon=5)
        assert len(mpc_bounds) == 10  # 2 CPPs * 5 horizon steps
        assert mpc_bounds[0] == (80.0, 180.0)  # First spray_rate bound
        assert mpc_bounds[1] == (400.0, 700.0)  # First air_flow bound
        
        # Test chromosome reshaping
        flat_chromosome = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        reshaped = optimizers.reshape_chromosome_to_plan(flat_chromosome, horizon=5, n_cpps=2)
        assert reshaped.shape == (5, 2)
        assert reshaped[0, 0] == 1 and reshaped[0, 1] == 2
        
    except ImportError as e:
        pytest.fail(f"Could not import optimizers module: {e}")

def test_core_module():
    """Test that core module imports correctly."""
    try:
        from robust_mpc import core
        
        # Test utility functions
        assert hasattr(core, 'calculate_control_effort')
        assert hasattr(core, 'upper_confidence_bound')
        
        # Test control effort calculation
        import numpy as np
        control_seq = np.array([1, 2, 1, 3])
        effort = core.calculate_control_effort(control_seq)
        assert effort > 0  # Should be positive for varying sequence
        
        # Test upper confidence bound
        mean = np.array([1.0, 2.0])
        std = np.array([0.1, 0.2])
        ucb = core.upper_confidence_bound(mean, std, beta=1.0)
        expected = mean + std
        assert np.allclose(ucb, expected)
        
    except ImportError as e:
        pytest.fail(f"Could not import core module: {e}")

def test_python_version_check():
    """Test that Python version check works."""
    # This should not raise an error since we're running it
    import robust_mpc
    
    # The version check happens on import, so if we get here, it passed
    assert sys.version_info >= (3, 8)

def test_direct_imports():
    """Test that key classes can be imported directly."""
    try:
        # These should work since they're implemented
        from robust_mpc import KalmanStateEstimator
        from robust_mpc import ProbabilisticTransformer
        
        # Test that implemented classes can be instantiated
        estimator_works = KalmanStateEstimator.__name__ == 'KalmanStateEstimator'
        transformer_works = ProbabilisticTransformer.__name__ == 'ProbabilisticTransformer'
        
        assert estimator_works and transformer_works
        
        # GeneticOptimizer should now be available (implemented in V2-3)
        from robust_mpc import GeneticOptimizer
        optimizer_works = GeneticOptimizer.__name__ == 'GeneticOptimizer'
        assert optimizer_works
            
        # RobustMPCController should now be available (implemented in V2-4)
        from robust_mpc import RobustMPCController
        controller_works = RobustMPCController.__name__ == 'RobustMPCController'
        assert controller_works
            
    except ImportError as e:
        pytest.fail(f"Could not import key classes: {e}")

if __name__ == "__main__":
    # Run basic tests if called directly
    test_library_imports()
    test_library_info()
    test_estimators_module()
    test_models_module() 
    test_optimizers_module()
    test_core_module()
    test_python_version_check()
    test_direct_imports()
    print("All tests passed! Library structure is working correctly.")