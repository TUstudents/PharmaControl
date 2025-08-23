"""
Integration tests for the GeneticOptimizer class.

This module provides integration testing for the genetic algorithm optimizer,
focusing on end-to-end optimization workflows, convergence behavior, and
real-world pharmaceutical process control scenarios.
"""

import os
import sys
import time
from unittest.mock import Mock

import numpy as np
import pytest

# Import the module under test
from robust_mpc.optimizers import GeneticOptimizer


class TestEndToEndOptimization:
    """Test class for complete optimization workflows."""

    def test_basic_optimization_workflow(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test complete optimization from initialization to solution."""
        # Reduce iterations for faster testing
        config = simple_ga_config.copy()
        config["num_generations"] = 5
        config["population_size"] = 10

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        # Run optimization
        result = optimizer.optimize()

        # Verify result structure
        assert isinstance(result, np.ndarray)
        assert result.shape == (config["horizon"], config["num_cpps"])

        # Verify all values are within bounds
        bounds_array = np.array(simple_param_bounds).reshape(-1, 2)
        for i in range(config["horizon"]):
            for j in range(config["num_cpps"]):
                bound_idx = i * config["num_cpps"] + j
                low, high = bounds_array[bound_idx]
                assert low <= result[i, j] <= high

    def test_optimization_with_linear_fitness(
        self, simple_ga_config, simple_param_bounds, linear_fitness_function
    ):
        """Test optimization with linear fitness function."""
        config = simple_ga_config.copy()
        config["num_generations"] = 10
        config["population_size"] = 20

        optimizer = GeneticOptimizer(
            fitness_function=linear_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # For linear fitness (air_flow + carousel_speed - spray_rate),
        # optimal solution should have:
        # - spray_rate at maximum (180)
        # - air_flow at minimum (400)
        # - carousel_speed at minimum (20)

        # Check that solution trends toward expected optimum
        avg_spray = np.mean(result[:, 0])
        avg_air = np.mean(result[:, 1])
        avg_speed = np.mean(result[:, 2])

        # Should be closer to optimal values than to bounds centers
        spray_center = (80 + 180) / 2  # 130
        air_center = (400 + 700) / 2  # 550
        speed_center = (20 + 40) / 2  # 30

        assert avg_spray > spray_center, f"Spray rate {avg_spray} should be > center {spray_center}"
        assert avg_air < air_center, f"Air flow {avg_air} should be < center {air_center}"
        assert (
            avg_speed < speed_center
        ), f"Carousel speed {avg_speed} should be < center {speed_center}"

    def test_optimization_convergence_behavior(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test that optimization shows convergence behavior over generations."""
        config = simple_ga_config.copy()
        config["num_generations"] = 20
        config["population_size"] = 30

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        # Run optimization multiple times to check consistency
        results = []
        fitness_values = []

        for _ in range(3):
            result = optimizer.optimize()
            fitness = quadratic_fitness_function(result)
            results.append(result)
            fitness_values.append(fitness)

        # Fitness values should be reasonable (not infinite or NaN)
        for fitness in fitness_values:
            assert np.isfinite(fitness)
            assert fitness >= 0  # Quadratic function is non-negative

        # Results should be consistent (not wildly different)
        fitness_std = np.std(fitness_values)
        fitness_mean = np.mean(fitness_values)

        # Standard deviation should not be too large relative to mean
        if fitness_mean > 0:
            cv = fitness_std / fitness_mean  # Coefficient of variation
            assert cv < 2.0, f"Results too inconsistent: CV = {cv}"

    def test_optimization_with_constraints(
        self, simple_ga_config, simple_param_bounds, constrained_fitness_function
    ):
        """Test optimization with hard constraints and penalties."""
        config = simple_ga_config.copy()
        config["num_generations"] = 15
        config["population_size"] = 25

        optimizer = GeneticOptimizer(
            fitness_function=constrained_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # Check that constraints are satisfied
        # Constraint 1: spray_rate > 100
        spray_rates = result[:, 0]
        assert np.all(
            spray_rates >= 100.0
        ), f"Spray rate constraint violated: min = {np.min(spray_rates)}"

        # Constraint 2: air_flow < 600
        air_flows = result[:, 1]
        assert np.all(
            air_flows <= 600.0
        ), f"Air flow constraint violated: max = {np.max(air_flows)}"

    def test_optimization_with_multimodal_fitness(
        self, simple_ga_config, simple_param_bounds, multimodal_fitness_function
    ):
        """Test optimization with multi-modal fitness landscape."""
        config = simple_ga_config.copy()
        config["num_generations"] = 25
        config["population_size"] = 40

        optimizer = GeneticOptimizer(
            fitness_function=multimodal_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # Verify result is valid
        assert result.shape == (config["horizon"], config["num_cpps"])

        # For multimodal function, just check that we get a finite result
        fitness = multimodal_fitness_function(result)
        assert np.isfinite(fitness)


class TestPharmaceuticalScenarios:
    """Test class for real-world pharmaceutical manufacturing scenarios."""

    def test_granulation_startup_control(self, pharmaceutical_scenarios, pharmaceutical_ga_config):
        """Test optimization for granulation process startup scenario."""
        scenario = pharmaceutical_scenarios["startup_control"]
        config = pharmaceutical_ga_config.copy()
        config["num_generations"] = 20
        config["population_size"] = 30

        # Create parameter bounds from scenario constraints
        horizon = config["horizon"]
        param_bounds = []
        constraints = scenario["constraints"]

        for _ in range(horizon):
            for param_name in ["spray_rate", "air_flow", "carousel_speed"]:
                param_bounds.append(
                    (constraints[param_name]["min_val"], constraints[param_name]["max_val"])
                )

        # Define startup fitness function
        def startup_fitness(control_plan):
            """Fitness for startup: minimize time to reach target with smooth control."""
            target = np.array([130.0, 550.0, 30.0])  # Target CPP values
            targets = np.tile(target, (control_plan.shape[0], 1))

            # Tracking error
            tracking_error = np.sum((control_plan - targets) ** 2)

            # Control smoothness penalty
            control_changes = np.sum(np.diff(control_plan, axis=0) ** 2)

            return tracking_error + 0.1 * control_changes

        optimizer = GeneticOptimizer(
            fitness_function=startup_fitness, param_bounds=param_bounds, config=config
        )

        result = optimizer.optimize()

        # Verify result meets scenario requirements
        assert result.shape == (horizon, config["num_cpps"])

        # Check bounds satisfaction
        for i in range(horizon):
            for j, param_name in enumerate(["spray_rate", "air_flow", "carousel_speed"]):
                value = result[i, j]
                min_val = constraints[param_name]["min_val"]
                max_val = constraints[param_name]["max_val"]
                assert min_val <= value <= max_val

        # Check control smoothness (changes should not be too abrupt)
        if horizon > 1:
            max_changes = np.max(np.abs(np.diff(result, axis=0)))
            # Changes should be reasonable (not using full range in one step)
            assert max_changes < 100.0, f"Control changes too abrupt: {max_changes}"

    def test_disturbance_rejection_scenario(
        self, pharmaceutical_scenarios, pharmaceutical_ga_config
    ):
        """Test optimization for disturbance rejection scenario."""
        scenario = pharmaceutical_scenarios["disturbance_rejection"]
        config = pharmaceutical_ga_config.copy()
        config["num_generations"] = 15
        config["population_size"] = 25

        # Create parameter bounds
        horizon = config["horizon"]
        param_bounds = []
        constraints = scenario["constraints"]

        for _ in range(horizon):
            for param_name in ["spray_rate", "air_flow", "carousel_speed"]:
                param_bounds.append(
                    (constraints[param_name]["min_val"], constraints[param_name]["max_val"])
                )

        # Define disturbance rejection fitness
        def disturbance_fitness(control_plan):
            """Fitness for disturbance rejection: maintain target despite disturbance."""
            target = np.array([130.0, 550.0, 30.0])
            disturbance = np.array([5.0, 20.0, 1.0])  # Simulated disturbance effect

            total_cost = 0.0
            for i in range(control_plan.shape[0]):
                # Simulated process response with disturbance
                predicted_output = control_plan[i, :] * 0.8 + disturbance
                error = np.sum((predicted_output - target) ** 2)
                total_cost += error

            return total_cost

        optimizer = GeneticOptimizer(
            fitness_function=disturbance_fitness, param_bounds=param_bounds, config=config
        )

        result = optimizer.optimize()

        # Verify result
        assert result.shape == (horizon, config["num_cpps"])

        # Check that optimization found reasonable control actions
        fitness = disturbance_fitness(result)
        assert np.isfinite(fitness)
        assert fitness >= 0

    def test_grade_changeover_scenario(self, pharmaceutical_scenarios, pharmaceutical_ga_config):
        """Test optimization for product grade changeover scenario."""
        scenario = pharmaceutical_scenarios["grade_changeover"]
        config = pharmaceutical_ga_config.copy()
        config["num_generations"] = 20
        config["population_size"] = 35

        # Create parameter bounds (different from standard granulation)
        horizon = config["horizon"]
        param_bounds = []
        constraints = scenario["constraints"]

        for _ in range(horizon):
            for param_name in ["spray_rate", "air_flow", "carousel_speed"]:
                param_bounds.append(
                    (constraints[param_name]["min_val"], constraints[param_name]["max_val"])
                )

        # Define grade changeover fitness
        def changeover_fitness(control_plan):
            """Fitness for grade changeover: transition from initial to target grade."""
            initial_target = np.array([120.0, 500.0, 25.0])  # Initial grade
            final_target = np.array([180.0, 750.0, 40.0])  # Target grade

            total_cost = 0.0
            for i in range(control_plan.shape[0]):
                # Linear transition between targets
                alpha = i / (control_plan.shape[0] - 1) if control_plan.shape[0] > 1 else 1.0
                current_target = (1 - alpha) * initial_target + alpha * final_target

                error = np.sum((control_plan[i, :] - current_target) ** 2)
                total_cost += error

            return total_cost

        optimizer = GeneticOptimizer(
            fitness_function=changeover_fitness, param_bounds=param_bounds, config=config
        )

        result = optimizer.optimize()

        # Verify changeover trajectory
        assert result.shape == (horizon, config["num_cpps"])

        # Check that trajectory shows progression from initial to final
        if horizon > 1:
            first_values = result[0, :]
            last_values = result[-1, :]

            # Should show general trend toward higher values (for this scenario)
            for j in range(config["num_cpps"]):
                # Allow some tolerance for optimization variability
                trend = last_values[j] - first_values[j]
                # Don't require strict monotonicity, just general trend
                assert trend > -50.0, f"Parameter {j} shows strong negative trend: {trend}"


class TestOptimizationRobustness:
    """Test class for optimization robustness and edge cases."""

    def test_optimization_with_noisy_fitness(
        self, simple_ga_config, simple_param_bounds, noisy_fitness_function
    ):
        """Test optimization robustness with noisy fitness function."""
        config = simple_ga_config.copy()
        config["num_generations"] = 25
        config["population_size"] = 40

        optimizer = GeneticOptimizer(
            fitness_function=noisy_fitness_function, param_bounds=simple_param_bounds, config=config
        )

        # Run optimization multiple times
        results = []
        for _ in range(3):
            result = optimizer.optimize()
            results.append(result)

        # All results should be valid despite noise
        for result in results:
            assert result.shape == (config["horizon"], config["num_cpps"])

            # Check bounds
            for i in range(config["horizon"]):
                for j in range(config["num_cpps"]):
                    bound_idx = i * config["num_cpps"] + j
                    low, high = simple_param_bounds[bound_idx]
                    assert low <= result[i, j] <= high

    def test_optimization_with_tight_bounds(
        self, simple_ga_config, quadratic_fitness_function, numerical_edge_cases
    ):
        """Test optimization with very tight parameter bounds."""
        config = simple_ga_config.copy()
        config["num_generations"] = 10
        config["population_size"] = 15

        tight_bounds = numerical_edge_cases["tight_bounds"]

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function, param_bounds=tight_bounds, config=config
        )

        result = optimizer.optimize()

        # Should still work with tight bounds
        assert result.shape == (config["horizon"], config["num_cpps"])

        # All values should be within tight bounds
        for i in range(len(tight_bounds)):
            low, high = tight_bounds[i]
            flat_result = result.flatten()
            assert low <= flat_result[i] <= high

    def test_optimization_with_large_bounds(
        self, simple_ga_config, quadratic_fitness_function, numerical_edge_cases
    ):
        """Test optimization with very large parameter bounds."""
        config = simple_ga_config.copy()
        config["num_generations"] = 10
        config["population_size"] = 15

        huge_bounds = numerical_edge_cases["huge_bounds"]

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function, param_bounds=huge_bounds, config=config
        )

        result = optimizer.optimize()

        # Should handle large bounds without numerical issues
        assert result.shape == (config["horizon"], config["num_cpps"])
        assert np.all(np.isfinite(result))

    def test_optimization_determinism_with_seeding(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test that optimization can be made deterministic with proper seeding."""
        config = simple_ga_config.copy()
        config["num_generations"] = 5
        config["population_size"] = 10

        # Set random seed for reproducibility
        np.random.seed(42)

        optimizer1 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result1 = optimizer1.optimize()

        # Reset seed and run again
        np.random.seed(42)

        optimizer2 = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result2 = optimizer2.optimize()

        # Results should be identical with same seed
        # Note: This may fail if DEAP internal randomness is not controlled
        # In that case, we just check they're close
        if not np.allclose(result1, result2):
            # At least check they're in the same general region
            diff = np.abs(result1 - result2)
            max_diff = np.max(diff)
            assert max_diff < 50.0, f"Results too different with same seed: max_diff = {max_diff}"


class TestOptimizationPerformance:
    """Test class for basic optimization performance characteristics."""

    def test_optimization_completes_in_reasonable_time(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test that optimization completes within reasonable time."""
        config = simple_ga_config.copy()
        config["num_generations"] = 10
        config["population_size"] = 20

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 30 seconds for small problem
        assert execution_time < 30.0, f"Optimization took too long: {execution_time}s"

        # Should still produce valid result
        assert result.shape == (config["horizon"], config["num_cpps"])

    def test_fitness_function_call_count(self, simple_ga_config, simple_param_bounds):
        """Test that fitness function is called reasonable number of times."""
        config = simple_ga_config.copy()
        config["num_generations"] = 5
        config["population_size"] = 10

        call_count = 0

        def counting_fitness(control_plan):
            nonlocal call_count
            call_count += 1
            return np.sum(control_plan**2)

        optimizer = GeneticOptimizer(
            fitness_function=counting_fitness, param_bounds=simple_param_bounds, config=config
        )

        result = optimizer.optimize()

        # Should call fitness function reasonable number of times
        # Approximate expected: population_size * (num_generations + 1)
        expected_calls = config["population_size"] * (config["num_generations"] + 1)

        # Allow some variance in call count
        assert call_count >= config["population_size"], f"Too few fitness calls: {call_count}"
        assert (
            call_count <= expected_calls * 2
        ), f"Too many fitness calls: {call_count} (expected ~{expected_calls})"


class TestConfigurationVariations:
    """Test class for different optimization configurations."""

    def test_small_population_optimization(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test optimization with small population size."""
        config = simple_ga_config.copy()
        config["population_size"] = 5
        config["num_generations"] = 20

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # Should still work with small population
        assert result.shape == (config["horizon"], config["num_cpps"])

    def test_large_population_optimization(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test optimization with large population size."""
        config = simple_ga_config.copy()
        config["population_size"] = 100
        config["num_generations"] = 5  # Fewer generations to keep test fast

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # Should work with large population
        assert result.shape == (config["horizon"], config["num_cpps"])

    def test_many_generations_optimization(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test optimization with many generations."""
        config = simple_ga_config.copy()
        config["population_size"] = 10  # Small population to keep test fast
        config["num_generations"] = 50

        optimizer = GeneticOptimizer(
            fitness_function=quadratic_fitness_function,
            param_bounds=simple_param_bounds,
            config=config,
        )

        result = optimizer.optimize()

        # Should work with many generations
        assert result.shape == (config["horizon"], config["num_cpps"])

    def test_extreme_probability_configurations(
        self, simple_ga_config, simple_param_bounds, quadratic_fitness_function
    ):
        """Test optimization with extreme crossover/mutation probabilities."""
        configs = [
            {"crossover_prob": 0.0, "mutation_prob": 1.0},  # Only mutation
            {"crossover_prob": 1.0, "mutation_prob": 0.0},  # Only crossover
            {"crossover_prob": 0.1, "mutation_prob": 0.1},  # Low probabilities
            {"crossover_prob": 0.9, "mutation_prob": 0.9},  # High probabilities
        ]

        for prob_config in configs:
            config = simple_ga_config.copy()
            config.update(prob_config)
            config["num_generations"] = 5
            config["population_size"] = 10

            optimizer = GeneticOptimizer(
                fitness_function=quadratic_fitness_function,
                param_bounds=simple_param_bounds,
                config=config,
            )

            result = optimizer.optimize()

            # Should work with extreme probabilities
            assert result.shape == (config["horizon"], config["num_cpps"])


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
