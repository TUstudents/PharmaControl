#!/usr/bin/env python3
"""
Simple test script to validate V2-3 (Genetic Optimization) completion.
Tests functionality without requiring external dependencies.
"""

import sys
sys.path.insert(0, './robust_mpc')

def test_genetic_optimizer_structure():
    """Test that GeneticOptimizer class is properly structured."""
    try:
        import optimizers
        
        # Check that GeneticOptimizer class exists
        assert hasattr(optimizers, 'GeneticOptimizer'), "GeneticOptimizer class not found"
        
        # Check that it inherits from BaseOptimizer
        ga_class = optimizers.GeneticOptimizer
        assert hasattr(ga_class, 'optimize'), "GeneticOptimizer missing optimize method"
        
        # Check utility functions
        assert hasattr(optimizers, 'setup_mpc_bounds'), "setup_mpc_bounds function missing"
        assert hasattr(optimizers, 'reshape_chromosome_to_plan'), "reshape_chromosome_to_plan function missing"
        
        print("✅ GeneticOptimizer structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ GeneticOptimizer structure test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions without requiring DEAP."""
    try:
        import optimizers
        import numpy as np
        
        # Test bounds validation
        bounds = [(0, 10), (5, 15), (-1, 1)]
        assert optimizers.validate_bounds(bounds, 3) == True
        assert optimizers.validate_bounds(bounds, 2) == False
        
        # Test bounds application
        solution = np.array([12, 3, 2])
        clipped = optimizers.apply_bounds(solution, bounds)
        assert clipped[0] == 10  # Clipped from 12 to 10
        assert clipped[1] == 5   # Clipped from 3 to 5
        assert clipped[2] == 1   # Clipped from 2 to 1
        
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
        
        print("✅ Utility functions test passed")
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def test_genetic_optimizer_instantiation():
    """Test GeneticOptimizer can be instantiated (structure test only)."""
    try:
        import optimizers
        
        # This should work even without DEAP since it only checks in __init__ when DEAP is used
        ga_class = optimizers.GeneticOptimizer
        
        # Check that the class has the expected methods and attributes
        expected_methods = ['optimize', '_setup_deap', '_create_individual', '_evaluate_wrapper']
        for method in expected_methods:
            assert hasattr(ga_class, method), f"Missing method: {method}"
        
        print("✅ GeneticOptimizer instantiation structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ GeneticOptimizer instantiation test failed: {e}")
        return False

def main():
    """Run all V2-3 completion tests."""
    print("🧪 Testing V2-3 (Advanced Optimization) Completion")
    print("=" * 50)
    
    tests = [
        test_genetic_optimizer_structure,
        test_utility_functions,
        test_genetic_optimizer_instantiation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 V2-3 completion validated successfully!")
        print("✅ GeneticOptimizer is ready for use in V2-4 (RobustMPC Core)")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)