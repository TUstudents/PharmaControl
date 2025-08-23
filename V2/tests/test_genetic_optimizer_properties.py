"""
Property-based tests for the GeneticOptimizer class.

This module uses property-based testing with Hypothesis to verify
algorithmic properties and invariants of the genetic algorithm optimizer.
Tests focus on mathematical properties that should hold regardless of
specific parameter values.
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Import the module under test
from robust_mpc.optimizers import GeneticOptimizer


# Custom strategies for generating test data
@st.composite
def valid_ga_config(draw):
    """Generate valid GA configuration dictionaries."""
    horizon = draw(st.integers(min_value=1, max_value=20))
    num_cpps = draw(st.integers(min_value=1, max_value=10))
    population_size = draw(st.integers(min_value=5, max_value=100))
    num_generations = draw(st.integers(min_value=1, max_value=50))
    crossover_prob = draw(st.floats(min_value=0.0, max_value=1.0))
    mutation_prob = draw(st.floats(min_value=0.0, max_value=1.0))

    return {
        "horizon": horizon,
        "num_cpps": num_cpps,
        "population_size": population_size,
        "num_generations": num_generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
    }


@st.composite
def parameter_bounds(draw, config):
    """Generate parameter bounds matching a given configuration."""
    horizon = config["horizon"]
    num_cpps = config["num_cpps"]
    total_params = horizon * num_cpps

    bounds = []
    for _ in range(total_params):
        low = draw(st.floats(min_value=-1000.0, max_value=1000.0))
        high = draw(st.floats(min_value=low + 0.1, max_value=low + 2000.0))
        bounds.append((low, high))

    return bounds


@st.composite
def simple_fitness_function(draw):
    """Generate simple fitness functions for testing."""
    function_type = draw(st.sampled_from(["quadratic", "linear", "constant"]))

    if function_type == "quadratic":

        def fitness(control_plan):
            return np.sum(control_plan**2)

    elif function_type == "linear":
        weights = draw(
            arrays(
                np.float64,
                shape=control_plan.shape[1],
                elements=st.floats(min_value=-10.0, max_value=10.0),
            )
        )

        def fitness(control_plan):
            return np.sum(control_plan @ weights)

    else:  # constant
        constant = draw(st.floats(min_value=0.0, max_value=100.0))

        def fitness(control_plan):
            return constant

    return fitness


class TestOptimizationInvariants:
    """Test class for optimization invariants that should always hold."""

    @given(config=valid_ga_config())
    @settings(max_examples=20, deadline=10000)  # Limit examples for performance
    def test_optimization_returns_valid_shape(self, config):
        """Test that optimization always returns correct shape."""
        assume(config["horizon"] * config["num_cpps"] <= 100)  # Keep problem size reasonable

        bounds = []
        for _ in range(config["horizon"] * config["num_cpps"]):
            bounds.append((0.0, 100.0))  # Simple uniform bounds

        def simple_fitness(control_plan):
            return np.sum(control_plan**2)

        optimizer = GeneticOptimizer(
            fitness_function=simple_fitness, param_bounds=bounds, config=config
        )

        result = optimizer.optimize()

        # Property: Result should always have expected shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (config["horizon"], config["num_cpps"])

    @given(config=valid_ga_config())
    @settings(max_examples=15, deadline=10000)
    def test_optimization_respects_bounds(self, config):
        """Test that optimization results always respect parameter bounds."""
        assume(config["horizon"] * config["num_cpps"] <= 50)

        # Generate random but valid bounds
        bounds = []
        bound_values = []
        for _ in range(config["horizon"] * config["num_cpps"]):
            low = np.random.uniform(-100, 100)
            high = low + np.random.uniform(1, 200)
            bounds.append((low, high))
            bound_values.append((low, high))

        def simple_fitness(control_plan):
            return np.sum(np.abs(control_plan))

        optimizer = GeneticOptimizer(
            fitness_function=simple_fitness, param_bounds=bounds, config=config
        )

        result = optimizer.optimize()

        # Property: All values should be within bounds
        flat_result = result.flatten()
        for i, (low, high) in enumerate(bound_values):
            assert (
                low <= flat_result[i] <= high + 1e-10
            ), f"Value {flat_result[i]} not in bounds [{low}, {high}]"

    @given(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=10, deadline=8000)
    def test_individual_creation_properties(self, horizon, num_cpps, population_size):
        """Test properties of individual creation."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": population_size,
            "num_generations": 5,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 100.0)] * (horizon * num_cpps)

        def dummy_fitness(control_plan):
            return 0.0

        optimizer = GeneticOptimizer(
            fitness_function=dummy_fitness, param_bounds=bounds, config=config
        )

        # Create multiple individuals
        individuals = [optimizer._create_individual() for _ in range(10)]

        # Property: All individuals should have correct length
        for individual in individuals:
            assert len(individual) == horizon * num_cpps

        # Property: All individuals should satisfy bounds
        for individual in individuals:
            for i, (low, high) in enumerate(bounds):
                assert low <= individual[i] <= high

        # Property: Individuals should show some diversity (not all identical)
        if len(individuals) > 1:
            all_identical = all(ind == individuals[0] for ind in individuals[1:])
            # With random generation, very unlikely all are identical
            assert not all_identical or (horizon * num_cpps == 1)

    @given(st.integers(min_value=2, max_value=8), st.integers(min_value=2, max_value=5))
    @settings(max_examples=10, deadline=6000)
    def test_bound_repair_properties(self, horizon, num_cpps):
        """Test properties of bound repair mechanism."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 10,
            "num_generations": 3,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        # Create bounds with specific ranges
        bounds = []
        for i in range(horizon * num_cpps):
            low = i * 10.0
            high = low + 50.0
            bounds.append((low, high))

        def dummy_fitness(control_plan):
            return 0.0

        optimizer = GeneticOptimizer(
            fitness_function=dummy_fitness, param_bounds=bounds, config=config
        )

        # Create individual with violations
        violating_individual = []
        for i, (low, high) in enumerate(bounds):
            if i % 3 == 0:
                violating_individual.append(low - 10.0)  # Below lower bound
            elif i % 3 == 1:
                violating_individual.append(high + 10.0)  # Above upper bound
            else:
                violating_individual.append((low + high) / 2)  # Within bounds

        repaired = optimizer._check_bounds(violating_individual)

        # Property: Repaired individual should satisfy all bounds
        for i, (low, high) in enumerate(bounds):
            assert low <= repaired[i] <= high

        # Property: Values within bounds should be unchanged
        for i, (low, high) in enumerate(bounds):
            if low <= violating_individual[i] <= high:
                assert repaired[i] == violating_individual[i]

        # Property: Violations should be clipped to nearest bound
        for i, (low, high) in enumerate(bounds):
            if violating_individual[i] < low:
                assert repaired[i] == low
            elif violating_individual[i] > high:
                assert repaired[i] == high


class TestGeneticOperationProperties:
    """Test properties of genetic operations (crossover, mutation)."""

    @given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=5))
    @settings(max_examples=8, deadline=5000)
    def test_crossover_properties(self, horizon, num_cpps):
        """Test properties of crossover operation."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 10,
            "num_generations": 3,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 100.0)] * (horizon * num_cpps)

        def dummy_fitness(control_plan):
            return 0.0

        optimizer = GeneticOptimizer(
            fitness_function=dummy_fitness, param_bounds=bounds, config=config
        )

        # Create two parent individuals
        parent1 = optimizer._create_individual()
        parent2 = optimizer._create_individual()

        # Perform crossover
        child1, child2 = optimizer._mate_with_bounds(parent1.copy(), parent2.copy())

        # Property: Children should have same length as parents
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)

        # Property: Children should satisfy bounds
        for i, (low, high) in enumerate(bounds):
            assert low <= child1[i] <= high
            assert low <= child2[i] <= high

        # Property: Children should be different from parents (with high probability)
        # This test is probabilistic and may occasionally fail
        if len(parent1) > 2:  # Only test for non-trivial cases
            child1_different = child1 != parent1
            child2_different = child2 != parent2
            # At least one child should be different (very high probability)
            assert child1_different or child2_different

    @given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=5))
    @settings(max_examples=8, deadline=5000)
    def test_mutation_properties(self, horizon, num_cpps):
        """Test properties of mutation operation."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 10,
            "num_generations": 3,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 100.0)] * (horizon * num_cpps)

        def dummy_fitness(control_plan):
            return 0.0

        optimizer = GeneticOptimizer(
            fitness_function=dummy_fitness, param_bounds=bounds, config=config
        )

        # Create individual
        original = optimizer._create_individual()
        individual = original.copy()

        # Perform mutation
        mutated_tuple = optimizer._mutate_with_bounds(individual)

        # Property: Should return tuple
        assert isinstance(mutated_tuple, tuple)
        assert len(mutated_tuple) == 1

        mutated = mutated_tuple[0]

        # Property: Mutated individual should have same length
        assert len(mutated) == len(original)

        # Property: Mutated individual should satisfy bounds
        for i, (low, high) in enumerate(bounds):
            assert low <= mutated[i] <= high

        # Property: Mutation may or may not change individual (probabilistic)
        # We don't assert difference since mutation probability < 1


class TestFitnessEvaluationProperties:
    """Test properties of fitness evaluation."""

    @given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=5000)
    def test_fitness_evaluation_consistency(self, horizon, num_cpps):
        """Test that fitness evaluation is consistent."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 10,
            "num_generations": 3,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 100.0)] * (horizon * num_cpps)

        def deterministic_fitness(control_plan):
            """Deterministic fitness function for consistency testing."""
            return np.sum(control_plan**2) + np.prod(control_plan.shape)

        optimizer = GeneticOptimizer(
            fitness_function=deterministic_fitness, param_bounds=bounds, config=config
        )

        # Create individual
        individual = optimizer._create_individual()

        # Evaluate multiple times
        fitness1 = optimizer._evaluate(individual.copy())
        fitness2 = optimizer._evaluate(individual.copy())
        fitness3 = optimizer._evaluate(individual.copy())

        # Property: Fitness should be consistent for same individual
        assert fitness1 == fitness2 == fitness3

        # Property: Fitness should be tuple with one element
        assert isinstance(fitness1, tuple)
        assert len(fitness1) == 1

        # Property: Fitness value should be finite
        assert np.isfinite(fitness1[0])

    @given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=4))
    @settings(max_examples=8, deadline=4000)
    def test_fitness_evaluation_shape_handling(self, horizon, num_cpps):
        """Test that fitness evaluation correctly handles array shapes."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 10,
            "num_generations": 3,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 10.0)] * (horizon * num_cpps)

        def shape_sensitive_fitness(control_plan):
            """Fitness that depends on array shape."""
            # Verify shape is as expected
            assert control_plan.shape == (horizon, num_cpps)
            return control_plan.shape[0] * control_plan.shape[1] + np.sum(control_plan)

        optimizer = GeneticOptimizer(
            fitness_function=shape_sensitive_fitness, param_bounds=bounds, config=config
        )

        # Create and evaluate individual
        individual = optimizer._create_individual()
        fitness = optimizer._evaluate(individual)

        # Property: Should successfully evaluate without shape errors
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert np.isfinite(fitness[0])


class TestOptimizationConvergenceProperties:
    """Test properties related to optimization convergence."""

    @given(st.integers(min_value=2, max_value=6), st.integers(min_value=2, max_value=4))
    @settings(max_examples=5, deadline=15000)  # Longer timeout for convergence tests
    def test_optimization_progress_property(self, horizon, num_cpps):
        """Test that optimization shows progress over generations."""
        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": 20,
            "num_generations": 10,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
        }

        bounds = [(0.0, 10.0)] * (horizon * num_cpps)

        # Use simple convex function with known minimum
        target = np.ones((horizon, num_cpps)) * 5.0  # Center of bounds

        def convex_fitness(control_plan):
            return np.sum((control_plan - target) ** 2)

        optimizer = GeneticOptimizer(
            fitness_function=convex_fitness, param_bounds=bounds, config=config
        )

        # Run optimization
        result = optimizer.optimize()
        final_fitness = convex_fitness(result)

        # Property: Final fitness should be reasonable for convex function
        # (Should be much better than worst case)
        worst_case_fitness = np.sum((np.zeros((horizon, num_cpps)) - target) ** 2)

        # Should achieve at least some improvement
        assert final_fitness < worst_case_fitness * 0.8

        # Property: Result should be finite and within bounds
        assert np.all(np.isfinite(result))
        for i in range(horizon):
            for j in range(num_cpps):
                bound_idx = i * num_cpps + j
                low, high = bounds[bound_idx]
                assert low <= result[i, j] <= high

    @given(st.integers(min_value=10, max_value=50), st.integers(min_value=20, max_value=100))
    @settings(max_examples=3, deadline=20000)  # Very limited for performance
    def test_population_diversity_property(self, population_size, num_generations):
        """Test that population maintains some diversity during optimization."""
        # Keep problem small for performance
        horizon, num_cpps = 3, 2

        config = {
            "horizon": horizon,
            "num_cpps": num_cpps,
            "population_size": population_size,
            "num_generations": min(num_generations, 20),  # Cap generations
            "crossover_prob": 0.7,
            "mutation_prob": 0.3,  # Higher mutation to maintain diversity
        }

        bounds = [(0.0, 100.0)] * (horizon * num_cpps)

        def simple_fitness(control_plan):
            return np.sum(control_plan**2)

        optimizer = GeneticOptimizer(
            fitness_function=simple_fitness, param_bounds=bounds, config=config
        )

        # Create initial population
        population = optimizer.toolbox.population(n=population_size)

        # Property: Initial population should show diversity
        if population_size > 1:
            # Convert to numpy for easier analysis
            pop_array = np.array([list(ind) for ind in population])

            # Check that not all individuals are identical
            first_individual = pop_array[0]
            all_identical = np.all(pop_array == first_individual, axis=0)

            # With random initialization, very unlikely all are identical
            # (unless problem is very constrained)
            total_identical_genes = np.sum(all_identical)
            assert total_identical_genes < len(first_individual)


class TestBoundaryConditions:
    """Test properties at boundary conditions."""

    @given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=8, deadline=8000)
    def test_extreme_probability_properties(self, crossover_prob, mutation_prob):
        """Test properties with extreme crossover and mutation probabilities."""
        config = {
            "horizon": 3,
            "num_cpps": 2,
            "population_size": 10,
            "num_generations": 5,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
        }

        bounds = [(0.0, 10.0)] * 6

        def simple_fitness(control_plan):
            return np.sum(control_plan)

        optimizer = GeneticOptimizer(
            fitness_function=simple_fitness, param_bounds=bounds, config=config
        )

        # Property: Should work with any valid probabilities
        result = optimizer.optimize()

        assert result.shape == (3, 2)
        assert np.all(np.isfinite(result))

        # Check bounds
        flat_result = result.flatten()
        for i, (low, high) in enumerate(bounds):
            assert low <= flat_result[i] <= high


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
