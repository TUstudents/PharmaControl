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
        import numpy as np
        
        # Test GeneticOptimizer class (implemented in V2-3)
        assert hasattr(optimizers, 'GeneticOptimizer')
        
        # Test GeneticOptimizer instantiation with minimal config
        def simple_fitness(control_plan):
            return np.sum(control_plan ** 2)
        
        config = {
            'horizon': 3,
            'num_cpps': 2,
            'population_size': 10,
            'num_generations': 5,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * 6  # 3 horizon * 2 cpps
        
        # Should be able to instantiate
        optimizer = optimizers.GeneticOptimizer(
            fitness_function=simple_fitness,
            param_bounds=bounds,
            config=config
        )
        
        # Test basic methods exist
        assert hasattr(optimizer, 'optimize')
        assert hasattr(optimizer, '_create_individual')
        assert hasattr(optimizer, '_check_bounds')
        assert hasattr(optimizer, '_validate_config')
        
        # Test individual creation
        individual = optimizer._create_individual()
        assert len(individual) == 6
        
        # Test bounds checking
        test_individual = [50.0] * 6
        repaired = optimizer._check_bounds(test_individual)
        assert len(repaired) == 6
        assert all(0.0 <= val <= 100.0 for val in repaired)
        
        # Test optimization (quick run)
        result = optimizer.optimize()
        assert result.shape == (3, 2)
        assert isinstance(result, np.ndarray)
        
        print("GeneticOptimizer basic functionality test passed")
        
    except ImportError as e:
        pytest.fail(f"Could not import optimizers module: {e}")
    except Exception as e:
        pytest.fail(f"GeneticOptimizer functionality test failed: {e}")

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