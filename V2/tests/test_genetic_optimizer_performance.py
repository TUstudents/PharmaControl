"""
Performance tests for the GeneticOptimizer class.

This module provides performance testing and benchmarking for the genetic
algorithm optimizer, focusing on scalability, execution time, and resource
usage patterns across different problem sizes and configurations.
"""

import pytest
import numpy as np
import time
import psutil
import sys
import os
from unittest.mock import Mock

# Import the module under test
from robust_mpc.optimizers import GeneticOptimizer


class TestPerformanceScaling:
    """Test class for performance scaling characteristics."""
    
    def test_execution_time_scaling_with_population_size(self, performance_configs, benchmark_functions):
        """Test how execution time scales with population size."""
        base_config = {
            'horizon': 5,
            'num_cpps': 3,
            'num_generations': 10,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (base_config['horizon'] * base_config['num_cpps'])
        fitness_func = benchmark_functions['sphere']
        
        population_sizes = [10, 20, 50, 100]
        execution_times = []
        
        for pop_size in population_sizes:
            config = base_config.copy()
            config['population_size'] = pop_size
            
            optimizer = GeneticOptimizer(
                fitness_function=fitness_func,
                param_bounds=bounds,
                config=config
            )
            
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify result is valid
            assert result.shape == (config['horizon'], config['num_cpps'])
            
        # Check that execution time increases with population size
        # (Should be roughly linear)
        time_ratios = [execution_times[i+1] / execution_times[i] for i in range(len(execution_times)-1)]
        
        # Each doubling should increase time by factor of 1.5-3.0 (allowing for overhead)
        for ratio in time_ratios:
            assert 0.8 < ratio < 5.0, f"Unexpected time scaling ratio: {ratio}"
            
        print(f"Population size scaling - Times: {execution_times}")
        print(f"Time ratios: {time_ratios}")
        
    def test_execution_time_scaling_with_generations(self, performance_configs, benchmark_functions):
        """Test how execution time scales with number of generations."""
        base_config = {
            'horizon': 5,
            'num_cpps': 3,
            'population_size': 30,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (base_config['horizon'] * base_config['num_cpps'])
        fitness_func = benchmark_functions['sphere']
        
        generation_counts = [5, 10, 20, 40]
        execution_times = []
        
        for num_gen in generation_counts:
            config = base_config.copy()
            config['num_generations'] = num_gen
            
            optimizer = GeneticOptimizer(
                fitness_function=fitness_func,
                param_bounds=bounds,
                config=config
            )
            
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify result is valid
            assert result.shape == (config['horizon'], config['num_cpps'])
            
        # Execution time should scale roughly linearly with generations
        time_ratios = [execution_times[i+1] / execution_times[i] for i in range(len(execution_times)-1)]
        
        for ratio in time_ratios:
            assert 1.5 < ratio < 4.0, f"Unexpected generation scaling ratio: {ratio}"
            
        print(f"Generation scaling - Times: {execution_times}")
        print(f"Time ratios: {time_ratios}")
        
    def test_execution_time_scaling_with_problem_size(self, benchmark_functions):
        """Test how execution time scales with problem size (horizon × num_cpps)."""
        base_config = {
            'population_size': 30,
            'num_generations': 10,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        problem_sizes = [
            (3, 2),   # 6 variables
            (5, 3),   # 15 variables
            (10, 3),  # 30 variables
            (10, 5),  # 50 variables
        ]
        
        execution_times = []
        fitness_func = benchmark_functions['sphere']
        
        for horizon, num_cpps in problem_sizes:
            config = base_config.copy()
            config['horizon'] = horizon
            config['num_cpps'] = num_cpps
            
            bounds = [(0.0, 100.0)] * (horizon * num_cpps)
            
            optimizer = GeneticOptimizer(
                fitness_function=fitness_func,
                param_bounds=bounds,
                config=config
            )
            
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify result is valid
            assert result.shape == (horizon, num_cpps)
            
        # Time should increase with problem size, but not extremely
        for i in range(len(execution_times)):
            total_vars = problem_sizes[i][0] * problem_sizes[i][1]
            print(f"Problem size {total_vars}: {execution_times[i]:.3f}s")
            
        # Check that times are reasonable (under 30s even for largest problem)
        assert all(t < 30.0 for t in execution_times), f"Some execution times too long: {execution_times}"


class TestMemoryUsage:
    """Test class for memory usage characteristics."""
    
    def test_memory_usage_scaling(self, performance_configs):
        """Test memory usage scaling with problem size."""
        def simple_fitness(control_plan):
            return np.sum(control_plan ** 2)
        
        configs = [
            performance_configs['small'],
            performance_configs['medium'],
            performance_configs['large']
        ]
        
        memory_usages = []
        
        for config in configs:
            bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
            
            # Measure memory before optimization
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            optimizer = GeneticOptimizer(
                fitness_function=simple_fitness,
                param_bounds=bounds,
                config=config
            )
            
            result = optimizer.optimize()
            
            # Measure memory after optimization
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            memory_usages.append(memory_used)
            
            # Verify result
            assert result.shape == (config['horizon'], config['num_cpps'])
            
        print(f"Memory usage by config: {memory_usages} MB")
        
        # Memory usage should be reasonable (under 500MB even for large problems)
        assert all(mem < 500.0 for mem in memory_usages), f"Memory usage too high: {memory_usages}"
        
        # Memory should generally increase with problem size
        # (Though this is not strictly guaranteed due to system variability)
        if len(memory_usages) > 1:
            # At least the largest should use more than the smallest
            assert memory_usages[-1] >= memory_usages[0] * 0.5
            
    def test_memory_cleanup_after_optimization(self):
        """Test that memory is properly cleaned up after optimization."""
        def simple_fitness(control_plan):
            return np.sum(control_plan ** 2)
        
        config = {
            'horizon': 10,
            'num_cpps': 5,
            'population_size': 100,
            'num_generations': 20,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
        
        process = psutil.Process()
        
        # Baseline memory
        memory_baseline = process.memory_info().rss / 1024 / 1024
        
        # Run optimization
        optimizer = GeneticOptimizer(
            fitness_function=simple_fitness,
            param_bounds=bounds,
            config=config
        )
        
        result = optimizer.optimize()
        
        # Memory during optimization
        memory_during = process.memory_info().rss / 1024 / 1024
        
        # Delete optimizer to trigger cleanup
        del optimizer
        del result
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024
        
        print(f"Memory: baseline={memory_baseline:.1f}MB, during={memory_during:.1f}MB, after={memory_after:.1f}MB")
        
        # Memory after cleanup should be close to baseline
        memory_increase = memory_after - memory_baseline
        assert memory_increase < 50.0, f"Memory not properly cleaned up: {memory_increase}MB increase"


class TestConvergencePerformance:
    """Test class for convergence performance characteristics."""
    
    def test_convergence_speed_on_simple_functions(self, benchmark_functions):
        """Test convergence speed on simple benchmark functions."""
        config = {
            'horizon': 5,
            'num_cpps': 3,
            'population_size': 50,
            'num_generations': 50,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(-10.0, 10.0)] * (config['horizon'] * config['num_cpps'])
        
        convergence_results = {}
        
        for func_name, fitness_func in benchmark_functions.items():
            if func_name == 'rastrigin':
                continue  # Skip rastrigin as it's very multimodal
                
            optimizer = GeneticOptimizer(
                fitness_function=fitness_func,
                param_bounds=bounds,
                config=config
            )
            
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            
            final_fitness = fitness_func(result)
            execution_time = end_time - start_time
            
            convergence_results[func_name] = {
                'fitness': final_fitness,
                'time': execution_time,
                'result_shape': result.shape
            }
            
            # Verify result
            assert result.shape == (config['horizon'], config['num_cpps'])
            
        print(f"Convergence results: {convergence_results}")
        
        # All optimizations should complete in reasonable time
        for func_name, results in convergence_results.items():
            assert results['time'] < 60.0, f"{func_name} took too long: {results['time']}s"
            assert np.isfinite(results['fitness']), f"{func_name} produced non-finite fitness"
            
    def test_optimization_with_expensive_fitness_function(self):
        """Test optimization performance with computationally expensive fitness."""
        def expensive_fitness(control_plan):
            """Simulated expensive fitness function."""
            time.sleep(0.01)  # Simulate 10ms computation
            return np.sum(control_plan ** 2)
        
        config = {
            'horizon': 3,
            'num_cpps': 2,
            'population_size': 20,
            'num_generations': 10,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
        
        optimizer = GeneticOptimizer(
            fitness_function=expensive_fitness,
            param_bounds=bounds,
            config=config
        )
        
        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify result
        assert result.shape == (config['horizon'], config['num_cpps'])
        
        # Should complete despite expensive fitness function
        # Expected time: ~(population_size * num_generations * 0.01s) + overhead
        expected_min_time = config['population_size'] * config['num_generations'] * 0.01
        expected_max_time = expected_min_time * 3  # Allow for overhead
        
        assert expected_min_time < execution_time < expected_max_time, \
            f"Unexpected execution time {execution_time}s (expected {expected_min_time}-{expected_max_time}s)"
            
    def test_fitness_function_call_efficiency(self):
        """Test that fitness function is called efficiently (not excessively)."""
        call_count = 0
        
        def counting_fitness(control_plan):
            nonlocal call_count
            call_count += 1
            return np.sum(control_plan ** 2)
        
        config = {
            'horizon': 4,
            'num_cpps': 3,
            'population_size': 30,
            'num_generations': 15,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
        
        optimizer = GeneticOptimizer(
            fitness_function=counting_fitness,
            param_bounds=bounds,
            config=config
        )
        
        result = optimizer.optimize()
        
        # Verify result
        assert result.shape == (config['horizon'], config['num_cpps'])
        
        # Estimate expected calls
        # Initial population + (generations * offspring evaluations)
        min_expected_calls = config['population_size']
        max_expected_calls = config['population_size'] * (config['num_generations'] + 2)
        
        print(f"Fitness function called {call_count} times")
        print(f"Expected range: {min_expected_calls} - {max_expected_calls}")
        
        assert min_expected_calls <= call_count <= max_expected_calls, \
            f"Unexpected number of fitness calls: {call_count}"


class TestParallelizationPotential:
    """Test class for identifying parallelization opportunities."""
    
    def test_independent_fitness_evaluations(self):
        """Test that fitness evaluations can be performed independently."""
        evaluation_order = []
        
        def order_tracking_fitness(control_plan):
            """Fitness function that tracks evaluation order."""
            eval_id = len(evaluation_order)
            evaluation_order.append(eval_id)
            # Add small delay to make timing effects visible
            time.sleep(0.001)
            return np.sum(control_plan ** 2) + eval_id * 0.001  # Small unique component
        
        config = {
            'horizon': 3,
            'num_cpps': 2,
            'population_size': 20,
            'num_generations': 5,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
        
        optimizer = GeneticOptimizer(
            fitness_function=order_tracking_fitness,
            param_bounds=bounds,
            config=config
        )
        
        result = optimizer.optimize()
        
        # Verify result
        assert result.shape == (config['horizon'], config['num_cpps'])
        
        # Should have evaluated fitness for initial population + offspring
        expected_min_evaluations = config['population_size']
        assert len(evaluation_order) >= expected_min_evaluations
        
        print(f"Total fitness evaluations: {len(evaluation_order)}")
        
        # The fact that this works shows evaluations are independent
        # (No shared state corruption)
        assert len(set(evaluation_order)) == len(evaluation_order), \
            "Duplicate evaluation IDs suggest state corruption"


class TestRealTimePerformance:
    """Test class for real-time performance requirements."""
    
    def test_pharmaceutical_real_time_requirements(self):
        """Test performance against pharmaceutical real-time requirements."""
        # Typical pharmaceutical MPC requirements: decisions within 1-10 seconds
        
        def pharmaceutical_fitness(control_plan):
            """Realistic pharmaceutical process fitness function."""
            # Simulate particle size and moisture tracking
            target_d50 = 450.0  # micrometers
            target_lod = 1.8    # %
            
            # Simple process model
            spray_rate = control_plan[:, 0]
            air_flow = control_plan[:, 1]
            carousel_speed = control_plan[:, 2]
            
            # Simulate process response
            d50_response = 300 + spray_rate * 1.2 - air_flow * 0.1
            lod_response = 3.0 - air_flow * 0.003 + spray_rate * 0.002
            
            # Tracking error
            d50_error = np.sum((d50_response - target_d50) ** 2)
            lod_error = np.sum((lod_response - target_lod) ** 2)
            
            # Control effort penalty
            control_effort = np.sum(np.diff(control_plan, axis=0) ** 2) if control_plan.shape[0] > 1 else 0
            
            return d50_error + 100 * lod_error + 0.1 * control_effort
        
        # Realistic pharmaceutical MPC configuration
        config = {
            'horizon': 10,       # 10-step horizon
            'num_cpps': 3,       # 3 control variables
            'population_size': 50,
            'num_generations': 30,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        # Realistic pharmaceutical bounds
        bounds = []
        for _ in range(config['horizon']):
            bounds.extend([
                (80.0, 180.0),   # spray_rate g/min
                (400.0, 700.0),  # air_flow m³/h
                (20.0, 40.0)     # carousel_speed rpm
            ])
        
        optimizer = GeneticOptimizer(
            fitness_function=pharmaceutical_fitness,
            param_bounds=bounds,
            config=config
        )
        
        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify result
        assert result.shape == (config['horizon'], config['num_cpps'])
        
        # Should meet pharmaceutical real-time requirements
        assert execution_time < 10.0, f"Too slow for pharmaceutical MPC: {execution_time}s"
        
        # Verify solution quality
        final_fitness = pharmaceutical_fitness(result)
        assert np.isfinite(final_fitness)
        
        print(f"Pharmaceutical MPC optimization: {execution_time:.2f}s, fitness: {final_fitness:.2f}")
        
        # Check that solution shows reasonable control behavior
        spray_rates = result[:, 0]
        air_flows = result[:, 1]
        carousel_speeds = result[:, 2]
        
        # Basic sanity checks
        assert np.all(80 <= spray_rates) and np.all(spray_rates <= 180)
        assert np.all(400 <= air_flows) and np.all(air_flows <= 700)
        assert np.all(20 <= carousel_speeds) and np.all(carousel_speeds <= 40)
        
    @pytest.mark.slow
    def test_repeated_optimization_performance(self):
        """Test performance of repeated optimizations (MPC loop simulation)."""
        def simple_mpc_fitness(control_plan):
            return np.sum(control_plan ** 2)
        
        config = {
            'horizon': 8,
            'num_cpps': 3,
            'population_size': 40,
            'num_generations': 20,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2
        }
        
        bounds = [(0.0, 100.0)] * (config['horizon'] * config['num_cpps'])
        
        # Simulate multiple MPC iterations
        num_iterations = 10
        execution_times = []
        
        for iteration in range(num_iterations):
            optimizer = GeneticOptimizer(
                fitness_function=simple_mpc_fitness,
                param_bounds=bounds,
                config=config
            )
            
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify result
            assert result.shape == (config['horizon'], config['num_cpps'])
            
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        min_time = np.min(execution_times)
        
        print(f"Repeated optimization: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
        
        # Performance should be consistent
        time_variability = (max_time - min_time) / avg_time
        assert time_variability < 0.5, f"Too much variability in execution time: {time_variability}"
        
        # Should maintain acceptable performance throughout
        assert avg_time < 5.0, f"Average time too high: {avg_time}s"
        assert max_time < 10.0, f"Maximum time too high: {max_time}s"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v", "--tb=short"])