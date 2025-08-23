"""
Shared pytest fixtures and configurations for V2 robust_mpc testing.

This module provides common test fixtures, mock objects, and configurations
used across multiple test files for the robust_mpc library components.
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Callable
from unittest.mock import Mock, MagicMock

# Add the parent directory to path to import robust_mpc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def simple_ga_config():
    """Basic GA configuration for unit testing."""
    return {
        "horizon": 5,
        "num_cpps": 3,
        "population_size": 20,
        "num_generations": 10,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
    }


@pytest.fixture
def pharmaceutical_ga_config():
    """Realistic GA configuration for pharmaceutical processes."""
    return {
        "horizon": 10,
        "num_cpps": 3,
        "population_size": 50,
        "num_generations": 50,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "cx_prob": 0.7,  # Alternative key name
        "mut_prob": 0.2,  # Alternative key name
    }


@pytest.fixture
def large_scale_ga_config():
    """Large-scale GA configuration for performance testing."""
    return {
        "horizon": 20,
        "num_cpps": 5,
        "population_size": 100,
        "num_generations": 100,
        "crossover_prob": 0.8,
        "mutation_prob": 0.15,
    }


@pytest.fixture
def granulation_constraints():
    """Standard granulation process parameter constraints."""
    return {
        "spray_rate": {"min_val": 80.0, "max_val": 180.0},
        "air_flow": {"min_val": 400.0, "max_val": 700.0},
        "carousel_speed": {"min_val": 20.0, "max_val": 40.0},
    }


@pytest.fixture
def coating_constraints():
    """Coating process parameter constraints with tighter bounds."""
    return {
        "spray_rate": {"min_val": 50.0, "max_val": 120.0},
        "air_flow": {"min_val": 300.0, "max_val": 500.0},
        "pan_speed": {"min_val": 5.0, "max_val": 15.0},
        "inlet_temp": {"min_val": 40.0, "max_val": 80.0},
        "coating_time": {"min_val": 60.0, "max_val": 180.0},
    }


@pytest.fixture
def simple_param_bounds(simple_ga_config):
    """Simple parameter bounds matching simple_ga_config."""
    horizon = simple_ga_config["horizon"]
    num_cpps = simple_ga_config["num_cpps"]

    # Create bounds for 3 CPPs: spray_rate, air_flow, carousel_speed
    bounds = []
    for _ in range(horizon):
        bounds.extend(
            [
                (80.0, 180.0),  # spray_rate
                (400.0, 700.0),  # air_flow
                (20.0, 40.0),  # carousel_speed
            ]
        )

    return bounds


@pytest.fixture
def pharmaceutical_param_bounds(pharmaceutical_ga_config):
    """Pharmaceutical parameter bounds matching pharmaceutical_ga_config."""
    horizon = pharmaceutical_ga_config["horizon"]
    num_cpps = pharmaceutical_ga_config["num_cpps"]

    bounds = []
    for _ in range(horizon):
        bounds.extend(
            [
                (80.0, 180.0),  # spray_rate
                (400.0, 700.0),  # air_flow
                (20.0, 40.0),  # carousel_speed
            ]
        )

    return bounds


@pytest.fixture
def quadratic_fitness_function():
    """Simple quadratic fitness function with known minimum."""

    def fitness(control_plan):
        """Quadratic function: sum((x - target)^2) with target at center of bounds."""
        target = np.array([130.0, 550.0, 30.0])  # Center of typical bounds
        targets = np.tile(target, (control_plan.shape[0], 1))
        return np.sum((control_plan - targets) ** 2)

    return fitness


@pytest.fixture
def linear_fitness_function():
    """Linear fitness function favoring high spray_rate, low others."""

    def fitness(control_plan):
        """Linear combination: air_flow + carousel_speed - spray_rate."""
        spray_sum = np.sum(control_plan[:, 0])
        air_sum = np.sum(control_plan[:, 1])
        speed_sum = np.sum(control_plan[:, 2])
        return air_sum + speed_sum - spray_sum

    return fitness


@pytest.fixture
def constrained_fitness_function():
    """Fitness function with hard constraints and penalties."""

    def fitness(control_plan):
        """Fitness with constraint violations heavily penalized."""
        # Base quadratic cost
        target = np.array([130.0, 550.0, 30.0])
        targets = np.tile(target, (control_plan.shape[0], 1))
        base_cost = np.sum((control_plan - targets) ** 2)

        # Add constraint penalties
        penalty = 0.0

        # Spray rate constraint: must be > 100
        spray_violations = np.sum(np.maximum(0, 100.0 - control_plan[:, 0]))
        penalty += 1000.0 * spray_violations

        # Air flow constraint: must be < 600
        air_violations = np.sum(np.maximum(0, control_plan[:, 1] - 600.0))
        penalty += 1000.0 * air_violations

        return base_cost + penalty

    return fitness


@pytest.fixture
def multimodal_fitness_function():
    """Multi-modal fitness function with multiple local optima."""

    def fitness(control_plan):
        """Fitness with multiple peaks and valleys."""
        cost = 0.0
        for i in range(control_plan.shape[0]):
            # Create multiple peaks
            x, y, z = control_plan[i, :]
            cost += np.sin(x / 30) * np.cos(y / 100) * np.sin(z / 10) + 0.01 * (x**2 + y**2 + z**2)

        return -cost  # Negative for minimization

    return fitness


@pytest.fixture
def noisy_fitness_function():
    """Fitness function with random noise for robustness testing."""

    def fitness(control_plan):
        """Quadratic fitness with Gaussian noise."""
        target = np.array([130.0, 550.0, 30.0])
        targets = np.tile(target, (control_plan.shape[0], 1))
        base_cost = np.sum((control_plan - targets) ** 2)

        # Add noise with fixed seed for reproducibility
        np.random.seed(42)
        noise = np.random.normal(0, base_cost * 0.1)

        return base_cost + noise

    return fitness


@pytest.fixture
def mock_fitness_function():
    """Mock fitness function for controlled testing."""
    mock = Mock()
    mock.return_value = 10.0
    return mock


@pytest.fixture
def invalid_configs():
    """Collection of invalid configurations for error testing."""
    return {
        "missing_horizon": {"num_cpps": 3, "population_size": 20, "num_generations": 10},
        "missing_num_cpps": {"horizon": 5, "population_size": 20, "num_generations": 10},
        "missing_population_size": {"horizon": 5, "num_cpps": 3, "num_generations": 10},
        "missing_num_generations": {"horizon": 5, "num_cpps": 3, "population_size": 20},
        "zero_horizon": {"horizon": 0, "num_cpps": 3, "population_size": 20, "num_generations": 10},
        "negative_population": {
            "horizon": 5,
            "num_cpps": 3,
            "population_size": -10,
            "num_generations": 10,
        },
    }


@pytest.fixture
def mismatched_bounds_configs():
    """Configs with mismatched parameter bounds for error testing."""
    return [
        {
            "config": {"horizon": 5, "num_cpps": 3, "population_size": 20, "num_generations": 10},
            "bounds": [(0, 1), (0, 1)],  # Only 2 bounds, need 15
        },
        {
            "config": {"horizon": 2, "num_cpps": 2, "population_size": 20, "num_generations": 10},
            "bounds": [(0, 1)] * 10,  # 10 bounds, need 4
        },
    ]


@pytest.fixture
def benchmark_functions():
    """Collection of standard optimization benchmark functions."""

    def sphere_function(control_plan):
        """Sphere function: sum(x_i^2)."""
        return np.sum(control_plan**2)

    def rosenbrock_function(control_plan):
        """Rosenbrock function for 2D slices."""
        total = 0.0
        for i in range(control_plan.shape[0]):
            for j in range(control_plan.shape[1] - 1):
                x, y = control_plan[i, j], control_plan[i, j + 1]
                total += 100 * (y - x**2) ** 2 + (1 - x) ** 2
        return total

    def rastrigin_function(control_plan):
        """Rastrigin function with many local minima."""
        A = 10
        n = control_plan.size
        return A * n + np.sum(control_plan**2 - A * np.cos(2 * np.pi * control_plan))

    return {
        "sphere": sphere_function,
        "rosenbrock": rosenbrock_function,
        "rastrigin": rastrigin_function,
    }


@pytest.fixture
def pharmaceutical_scenarios():
    """Real-world pharmaceutical manufacturing scenarios."""
    return {
        "startup_control": {
            "description": "Process startup from cold conditions",
            "initial_state": np.array([300.0, 3.0]),  # d50=300μm, LOD=3.0%
            "target_state": np.array([450.0, 1.8]),  # d50=450μm, LOD=1.8%
            "constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 180.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 40.0},
            },
        },
        "disturbance_rejection": {
            "description": "Rejection of feed moisture disturbance",
            "initial_state": np.array([450.0, 1.8]),
            "target_state": np.array([450.0, 1.8]),
            "disturbance": np.array([0.0, 0.5]),  # +0.5% moisture disturbance
            "constraints": {
                "spray_rate": {"min_val": 80.0, "max_val": 180.0},
                "air_flow": {"min_val": 400.0, "max_val": 700.0},
                "carousel_speed": {"min_val": 20.0, "max_val": 40.0},
            },
        },
        "grade_changeover": {
            "description": "Product grade changeover operation",
            "initial_state": np.array([450.0, 1.8]),
            "target_state": np.array([600.0, 1.2]),  # Different grade
            "constraints": {
                "spray_rate": {"min_val": 100.0, "max_val": 200.0},  # Higher range
                "air_flow": {"min_val": 500.0, "max_val": 800.0},
                "carousel_speed": {"min_val": 25.0, "max_val": 45.0},
            },
        },
    }


@pytest.fixture
def optimization_tolerances():
    """Standard tolerances for optimization testing."""
    return {
        "fitness_tolerance": 1e-3,
        "constraint_tolerance": 1e-6,
        "convergence_tolerance": 1e-4,
        "relative_tolerance": 1e-2,
        "absolute_tolerance": 1e-6,
    }


def create_bounds_from_constraints(constraints: Dict, horizon: int) -> List[Tuple[float, float]]:
    """Helper function to create parameter bounds from constraint dict."""
    bounds = []
    cpp_names = list(constraints.keys())

    for _ in range(horizon):
        for name in cpp_names:
            bounds.append((constraints[name]["min_val"], constraints[name]["max_val"]))

    return bounds


@pytest.fixture
def bounds_helper():
    """Helper function fixture for creating bounds."""
    return create_bounds_from_constraints


# Performance testing fixtures
@pytest.fixture
def performance_configs():
    """Configurations for performance testing."""
    return {
        "small": {"horizon": 3, "num_cpps": 2, "population_size": 10, "num_generations": 5},
        "medium": {"horizon": 10, "num_cpps": 3, "population_size": 50, "num_generations": 25},
        "large": {"horizon": 20, "num_cpps": 5, "population_size": 100, "num_generations": 50},
        "xlarge": {"horizon": 50, "num_cpps": 8, "population_size": 200, "num_generations": 100},
    }


# Numerical precision fixtures
@pytest.fixture
def numerical_edge_cases():
    """Edge cases for numerical testing."""
    return {
        "tiny_bounds": [(1e-10, 1e-9)] * 15,
        "huge_bounds": [(1e6, 1e9)] * 15,
        "tight_bounds": [(0.0, 1e-6)] * 15,
        "asymmetric_bounds": [(i, i + 0.1) for i in range(15)],
    }
