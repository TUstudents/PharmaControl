"""
Unit tests for the GeneticOptimizer class.

This module provides comprehensive unit testing for the genetic algorithm
optimizer used in the robust MPC framework. Tests cover initialization,
configuration validation, genetic operations, and constraint handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Import the module under test
from robust_mpc.optimizers import GeneticOptimizer


class TestGeneticOptimizerInitialization:
    """Test class for GeneticOptimizer initialization and setup."""
    
    def test_valid_initialization(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test successful initialization with valid parameters."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        assert optimizer.fitness_function == quadratic_fitness_function
        assert optimizer.param_bounds == simple_param_bounds
        assert optimizer.config == simple_ga_config
        assert optimizer.n_params == len(simple_param_bounds)
        assert optimizer.n_params == simple_ga_config['horizon'] * simple_ga_config['num_cpps']
        
    def test_initialization_with_alternative_config_keys(self, pharmaceutical_ga_config, pharmaceutical_param_bounds, linear_fitness_function):
        """Test initialization with alternative configuration key names."""
        optimizer = GeneticOptimizer(
            fitness_function=linear_fitness_function,
            param_bounds=pharmaceutical_param_bounds,
            config=pharmaceutical_ga_config
        )
        
        # Should work with both cx_prob/crossover_prob and mut_prob/mutation_prob
        assert optimizer.config['cx_prob'] == 0.7
        assert optimizer.config['crossover_prob'] == 0.7
        
    def test_initialization_with_mock_fitness(self, simple_ga_config, simple_param_bounds, mock_fitness_function):
        """Test initialization with mock fitness function."""
        optimizer = GeneticOptimizer(
            fitness_function=mock_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        assert optimizer.fitness_function == mock_fitness_function


class TestConfigurationValidation:
    """Test class for configuration validation logic."""
    
    def test_valid_config_validation(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that valid configurations pass validation."""
        # Should not raise any exceptions
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        assert optimizer.config == simple_ga_config
        
    def test_missing_required_keys(self, invalid_configs, simple_param_bounds, quadratic_fitness_function):
        """Test validation fails with missing required configuration keys."""
        for config_name, invalid_config in invalid_configs.items():
            if config_name.startswith('missing_'):
                with pytest.raises(ValueError, match="Missing required config key"):
                    GeneticOptimizer(
                        fitness_function=quadratic_fitness_function,
                        param_bounds=simple_param_bounds,
                        config=invalid_config
                    )
                    
    def test_parameter_bounds_length_mismatch(self, mismatched_bounds_configs, quadratic_fitness_function):
        """Test validation fails when parameter bounds length doesn't match config."""
        for mismatch_case in mismatched_bounds_configs:
            config = mismatch_case['config']
            bounds = mismatch_case['bounds']
            
            with pytest.raises(ValueError, match="Parameter bounds length .* != expected"):
                GeneticOptimizer(
                    fitness_function=quadratic_fitness_function,
                    param_bounds=bounds,
                    config=config
                )
                
    def test_zero_horizon_config(self, invalid_configs, quadratic_fitness_function):
        """Test handling of zero horizon configuration."""
        config = invalid_configs['zero_horizon']
        bounds = []  # Empty bounds for zero horizon
        
        with pytest.raises(ValueError):
            GeneticOptimizer(
                fitness_function=quadratic_fitness_function,
                param_bounds=bounds,
                config=config
            )


class TestIndividualCreation:
    """Test class for individual creation and chromosome generation."""
    
    def test_create_individual_shape(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that created individuals have correct shape and structure."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        
        assert len(individual) == optimizer.n_params
        assert len(individual) == simple_ga_config['horizon'] * simple_ga_config['num_cpps']
        
    def test_create_individual_bounds_satisfaction(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that all genes in created individuals satisfy bounds."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create multiple individuals to test randomness
        for _ in range(10):
            individual = optimizer._create_individual()
            
            for i, (low, high) in enumerate(simple_param_bounds):
                assert low <= individual[i] <= high, f"Gene {i} value {individual[i]} not in bounds [{low}, {high}]"
                
    def test_individual_randomness(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that created individuals show appropriate randomness."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create multiple individuals
        individuals = [optimizer._create_individual() for _ in range(20)]
        
        # Check that not all individuals are identical
        first_individual = individuals[0]
        identical_count = sum(1 for ind in individuals if ind == first_individual)
        
        # Very unlikely that all 20 random individuals are identical
        assert identical_count < 20, "All individuals are identical - possible randomness issue"
        
    def test_individual_type_and_fitness_attribute(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that individuals have correct DEAP type and fitness attribute."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        
        # Should be a DEAP Individual (which inherits from list)
        assert isinstance(individual, list)
        assert hasattr(individual, 'fitness')


class TestBoundManagement:
    """Test class for parameter bound enforcement and repair."""
    
    def test_check_bounds_no_violations(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test bound checking with no violations."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual within bounds
        individual = [100.0, 500.0, 25.0] * simple_ga_config['horizon']
        original_individual = individual.copy()
        
        repaired = optimizer._check_bounds(individual)
        
        assert repaired == original_individual, "Valid individual was modified"
        
    def test_check_bounds_upper_violations(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test bound checking with upper bound violations."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual with upper bound violations
        individual = [200.0, 800.0, 50.0] * simple_ga_config['horizon']  # All above upper bounds
        
        repaired = optimizer._check_bounds(individual)
        
        # Check that violations are clipped to upper bounds
        expected_bounds = [(80.0, 180.0), (400.0, 700.0), (20.0, 40.0)] * simple_ga_config['horizon']
        for i, (_, high) in enumerate(expected_bounds):
            assert repaired[i] == high, f"Gene {i} not clipped to upper bound"
            
    def test_check_bounds_lower_violations(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test bound checking with lower bound violations."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual with lower bound violations
        individual = [50.0, 300.0, 10.0] * simple_ga_config['horizon']  # All below lower bounds
        
        repaired = optimizer._check_bounds(individual)
        
        # Check that violations are clipped to lower bounds
        expected_bounds = [(80.0, 180.0), (400.0, 700.0), (20.0, 40.0)] * simple_ga_config['horizon']
        for i, (low, _) in enumerate(expected_bounds):
            assert repaired[i] == low, f"Gene {i} not clipped to lower bound"
            
    def test_check_bounds_mixed_violations(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test bound checking with mixed upper and lower violations."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual with mixed violations
        individual = []
        for i in range(simple_ga_config['horizon']):
            individual.extend([50.0, 800.0, 25.0])  # low, high, valid
            
        repaired = optimizer._check_bounds(individual)
        
        # Check specific clipping
        for i in range(0, len(individual), 3):
            assert repaired[i] == 80.0,   # Clipped to spray_rate lower bound
            assert repaired[i+1] == 700.0,  # Clipped to air_flow upper bound
            assert repaired[i+2] == 25.0    # Should remain unchanged (within bounds)
            
    def test_check_bounds_boundary_values(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test bound checking with values exactly at boundaries."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual with boundary values
        individual = []
        for _ in range(simple_ga_config['horizon']):
            individual.extend([80.0, 700.0, 20.0])  # All at upper/lower bounds
            
        original_individual = individual.copy()
        repaired = optimizer._check_bounds(individual)
        
        assert repaired == original_individual, "Boundary values were modified"


class TestGeneticOperations:
    """Test class for genetic algorithm operations (crossover, mutation)."""
    
    def test_mate_with_bounds_basic(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test basic crossover operation with bound enforcement."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create two parent individuals
        parent1 = optimizer._create_individual()
        parent2 = optimizer._create_individual()
        
        # Store originals for comparison
        original_parent1 = parent1.copy()
        original_parent2 = parent2.copy()
        
        # Perform crossover
        child1, child2 = optimizer._mate_with_bounds(parent1, parent2)
        
        # Children should be different from parents (with high probability)
        # and should satisfy bounds
        for i, (low, high) in enumerate(simple_param_bounds):
            assert low <= child1[i] <= high
            assert low <= child2[i] <= high
            
    def test_mutate_with_bounds_basic(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test basic mutation operation with bound enforcement."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create individual
        individual = optimizer._create_individual()
        original_individual = individual.copy()
        
        # Perform mutation
        mutated_individual, = optimizer._mutate_with_bounds(individual)
        
        # Mutated individual should satisfy bounds
        for i, (low, high) in enumerate(simple_param_bounds):
            assert low <= mutated_individual[i] <= high
            
        # Should return tuple (DEAP requirement)
        assert isinstance(optimizer._mutate_with_bounds(individual), tuple)
        
    @patch('robust_mpc.optimizers.tools.cxTwoPoint')
    def test_mate_with_bounds_calls_crossover(self, mock_crossover, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that crossover operation calls DEAP crossover function."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        parent1 = optimizer._create_individual()
        parent2 = optimizer._create_individual()
        
        optimizer._mate_with_bounds(parent1, parent2)
        
        # Verify that DEAP crossover was called
        mock_crossover.assert_called_once_with(parent1, parent2)
        
    @patch('robust_mpc.optimizers.tools.mutGaussian')
    def test_mutate_with_bounds_calls_mutation(self, mock_mutation, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that mutation operation calls DEAP mutation function."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        
        optimizer._mutate_with_bounds(individual)
        
        # Verify that DEAP mutation was called with correct parameters
        mock_mutation.assert_called_once_with(individual, mu=0, sigma=0.2, indpb=0.1)


class TestFitnessEvaluation:
    """Test class for fitness function evaluation and chromosome reshaping."""
    
    def test_evaluate_basic_functionality(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test basic fitness evaluation functionality."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        fitness_tuple = optimizer._evaluate(individual)
        
        # Should return a tuple (DEAP requirement)
        assert isinstance(fitness_tuple, tuple)
        assert len(fitness_tuple) == 1
        
        # Fitness should be a numeric value
        fitness_value = fitness_tuple[0]
        assert isinstance(fitness_value, (int, float))
        assert not np.isnan(fitness_value)
        assert not np.isinf(fitness_value)
        
    def test_evaluate_chromosome_reshaping(self, simple_ga_config, simple_param_bounds):
        """Test that chromosomes are correctly reshaped for fitness evaluation."""
        def shape_checking_fitness(control_plan):
            # Verify that the control plan has the expected shape
            expected_shape = (simple_ga_config['horizon'], simple_ga_config['num_cpps'])
            assert control_plan.shape == expected_shape
            return np.sum(control_plan)
        
        optimizer = GeneticOptimizer(
            fitness_function=shape_checking_fitness,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        # This should not raise an assertion error if reshaping is correct
        fitness_tuple = optimizer._evaluate(individual)
        
        assert isinstance(fitness_tuple, tuple)
        
    def test_evaluate_with_mock_fitness(self, simple_ga_config, simple_param_bounds):
        """Test fitness evaluation with mock fitness function."""
        mock_fitness = Mock(return_value=42.0)
        
        optimizer = GeneticOptimizer(
            fitness_function=mock_fitness,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        fitness_tuple = optimizer._evaluate(individual)
        
        # Verify mock was called and returned expected value
        assert fitness_tuple == (42.0,)
        mock_fitness.assert_called_once()
        
        # Verify the argument passed to fitness function has correct shape
        call_args = mock_fitness.call_args[0][0]  # First positional argument
        expected_shape = (simple_ga_config['horizon'], simple_ga_config['num_cpps'])
        assert call_args.shape == expected_shape
        
    def test_evaluate_fitness_function_exception_handling(self, simple_ga_config, simple_param_bounds):
        """Test handling of exceptions in fitness function."""
        def failing_fitness(control_plan):
            raise ValueError("Simulated fitness function error")
        
        optimizer = GeneticOptimizer(
            fitness_function=failing_fitness,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        individual = optimizer._create_individual()
        
        # Should propagate the exception from fitness function
        with pytest.raises(ValueError, match="Simulated fitness function error"):
            optimizer._evaluate(individual)


class TestDEAPIntegration:
    """Test class for DEAP framework integration and creator management."""
    
    def test_deap_creator_cleanup(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that DEAP creators are properly cleaned up."""
        from deap import creator
        
        # Set up creators manually to test cleanup
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
        # Create optimizer (should clean up and recreate)
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Creators should exist after initialization
        assert hasattr(creator, 'FitnessMin')
        assert hasattr(creator, 'Individual')
        
    def test_multiple_optimizer_instances(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test creating multiple optimizer instances (DEAP creator conflicts)."""
        # Create first optimizer
        optimizer1 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Create second optimizer (should handle creator conflicts)
        optimizer2 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Both should work independently
        individual1 = optimizer1._create_individual()
        individual2 = optimizer2._create_individual()
        
        assert len(individual1) == len(simple_param_bounds)
        assert len(individual2) == len(simple_param_bounds)
        
    def test_toolbox_registration(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that DEAP toolbox is correctly configured."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        # Check that required operators are registered
        assert hasattr(optimizer.toolbox, 'individual')
        assert hasattr(optimizer.toolbox, 'population')
        assert hasattr(optimizer.toolbox, 'evaluate')
        assert hasattr(optimizer.toolbox, 'mate')
        assert hasattr(optimizer.toolbox, 'mutate')
        assert hasattr(optimizer.toolbox, 'select')
        
    def test_population_generation(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that population can be generated correctly."""
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=simple_ga_config
        )
        
        population = optimizer.toolbox.population(n=10)
        
        assert len(population) == 10
        
        for individual in population:
            assert len(individual) == len(simple_param_bounds)
            # Check bounds satisfaction
            for i, (low, high) in enumerate(simple_param_bounds):
                assert low <= individual[i] <= high


class TestConfigurationCompatibility:
    """Test class for configuration key compatibility and defaults."""
    
    def test_crossover_prob_key_compatibility(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test compatibility between crossover_prob and cx_prob keys."""
        # Test with crossover_prob
        config1 = simple_ga_config.copy()
        config1['crossover_prob'] = 0.8
        
        optimizer1 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config1
        )
        
        # Test with cx_prob
        config2 = simple_ga_config.copy()
        del config2['crossover_prob']  # Remove crossover_prob
        config2['cx_prob'] = 0.6
        
        optimizer2 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config2
        )
        
        # Both should work and access should work in optimize method
        assert optimizer1.config['crossover_prob'] == 0.8
        assert optimizer2.config['cx_prob'] == 0.6
        
    def test_mutation_prob_key_compatibility(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test compatibility between mutation_prob and mut_prob keys."""
        # Test with mutation_prob
        config1 = simple_ga_config.copy()
        config1['mutation_prob'] = 0.3
        
        optimizer1 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config1
        )
        
        # Test with mut_prob
        config2 = simple_ga_config.copy()
        del config2['mutation_prob']  # Remove mutation_prob
        config2['mut_prob'] = 0.1
        
        optimizer2 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config2
        )
        
        # Both should work
        assert optimizer1.config['mutation_prob'] == 0.3
        assert optimizer2.config['mut_prob'] == 0.1
        
    def test_default_probability_values(self, simple_ga_config, simple_param_bounds, quadratic_fitness_function):
        """Test that default probability values are used when keys are missing."""
        # Create config without probability keys
        minimal_config = {
            'horizon': simple_ga_config['horizon'],
            'num_cpps': simple_ga_config['num_cpps'],
            'population_size': simple_ga_config['population_size'],
            'num_generations': simple_ga_config['num_generations']
        }
        
        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=minimal_config
        )
        
        # Should still work with defaults
        assert optimizer.config == minimal_config


class TestErrorHandling:
    """Test class for error handling and edge cases."""
    
    def test_invalid_fitness_function_type(self, simple_ga_config, simple_param_bounds):
        """Test error handling with invalid fitness function."""
        with pytest.raises(TypeError):
            GeneticOptimizer(
                fitness_function="not_a_function",
                param_bounds=simple_param_bounds,
                config=simple_ga_config
            )
            
    def test_invalid_param_bounds_type(self, simple_ga_config, quadratic_fitness_function):
        """Test error handling with invalid parameter bounds."""
        with pytest.raises((TypeError, ValueError)):
            GeneticOptimizer(
                fitness_function=quadratic_fitness_function,
                param_bounds="not_a_list",
                config=simple_ga_config
            )
            
    def test_invalid_config_type(self, simple_param_bounds, quadratic_fitness_function):
        """Test error handling with invalid configuration."""
        with pytest.raises((TypeError, KeyError)):
            GeneticOptimizer(
                fitness_function=quadratic_fitness_function,
                param_bounds=simple_param_bounds,
                config="not_a_dict"
            )
            
    def test_empty_param_bounds(self, simple_ga_config, quadratic_fitness_function):
        """Test error handling with empty parameter bounds."""
        config = simple_ga_config.copy()
        config['horizon'] = 0
        config['num_cpps'] = 0
        
        with pytest.raises(ValueError):
            GeneticOptimizer(
                fitness_function=quadratic_fitness_function,
                param_bounds=[],
                config=config
            )


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])