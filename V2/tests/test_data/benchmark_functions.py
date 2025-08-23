"""
Standard optimization benchmark functions for testing genetic algorithms.

This module provides a collection of well-known optimization benchmark
functions used for testing and validating genetic algorithm performance.
Each function has known characteristics (unimodal/multimodal, separable/
non-separable, global optimum location) useful for systematic testing.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np


def get_sphere_function() -> Callable[[np.ndarray], float]:
    """
    Sphere function: f(x) = sum(x_i^2)

    Properties:
    - Unimodal (single global minimum)
    - Separable
    - Global minimum: f(0,...,0) = 0
    - Search domain: [-5.12, 5.12]^n
    - Easy optimization problem

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def sphere(control_plan: np.ndarray) -> float:
        return np.sum(control_plan**2)

    sphere.properties = {
        "name": "Sphere",
        "global_minimum": 0.0,
        "global_optimum": "zeros",
        "modality": "unimodal",
        "separable": True,
        "search_domain": (-5.12, 5.12),
        "difficulty": "easy",
    }

    return sphere


def get_rosenbrock_function() -> Callable[[np.ndarray], float]:
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Properties:
    - Unimodal with narrow curved valley
    - Non-separable
    - Global minimum: f(1,...,1) = 0
    - Search domain: [-2.048, 2.048]^n
    - Medium difficulty (narrow valley makes convergence slow)

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def rosenbrock(control_plan: np.ndarray) -> float:
        total = 0.0
        flat_plan = control_plan.flatten()

        for i in range(len(flat_plan) - 1):
            x_i = flat_plan[i]
            x_i1 = flat_plan[i + 1]
            total += 100 * (x_i1 - x_i**2) ** 2 + (1 - x_i) ** 2

        return total

    rosenbrock.properties = {
        "name": "Rosenbrock",
        "global_minimum": 0.0,
        "global_optimum": "ones",
        "modality": "unimodal",
        "separable": False,
        "search_domain": (-2.048, 2.048),
        "difficulty": "medium",
    }

    return rosenbrock


def get_rastrigin_function() -> Callable[[np.ndarray], float]:
    """
    Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))

    Properties:
    - Highly multimodal (many local minima)
    - Separable
    - Global minimum: f(0,...,0) = 0
    - Search domain: [-5.12, 5.12]^n
    - Hard optimization problem due to many local optima

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def rastrigin(control_plan: np.ndarray) -> float:
        A = 10
        flat_plan = control_plan.flatten()
        n = len(flat_plan)

        return A * n + np.sum(flat_plan**2 - A * np.cos(2 * np.pi * flat_plan))

    rastrigin.properties = {
        "name": "Rastrigin",
        "global_minimum": 0.0,
        "global_optimum": "zeros",
        "modality": "multimodal",
        "separable": True,
        "search_domain": (-5.12, 5.12),
        "difficulty": "hard",
    }

    return rastrigin


def get_ackley_function() -> Callable[[np.ndarray], float]:
    """
    Ackley function: Complex multimodal function with exponential terms

    Properties:
    - Multimodal with many local minima
    - Non-separable
    - Global minimum: f(0,...,0) = 0
    - Search domain: [-32, 32]^n
    - Hard optimization problem

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def ackley(control_plan: np.ndarray) -> float:
        flat_plan = control_plan.flatten()
        n = len(flat_plan)

        sum_sq = np.sum(flat_plan**2)
        sum_cos = np.sum(np.cos(2 * np.pi * flat_plan))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)

        return term1 + term2 + 20 + np.e

    ackley.properties = {
        "name": "Ackley",
        "global_minimum": 0.0,
        "global_optimum": "zeros",
        "modality": "multimodal",
        "separable": False,
        "search_domain": (-32, 32),
        "difficulty": "hard",
    }

    return ackley


def get_booth_function() -> Callable[[np.ndarray], float]:
    """
    Booth function: f(x,y) = (x + 2*y - 7)^2 + (2*x + y - 5)^2

    Properties:
    - Unimodal
    - 2D function (uses first two variables of control plan)
    - Global minimum: f(1, 3) = 0
    - Search domain: [-10, 10]^2
    - Easy optimization problem

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def booth(control_plan: np.ndarray) -> float:
        flat_plan = control_plan.flatten()
        if len(flat_plan) < 2:
            # Pad with zeros if needed
            x = flat_plan[0] if len(flat_plan) > 0 else 0
            y = 0
        else:
            x, y = flat_plan[0], flat_plan[1]

        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    booth.properties = {
        "name": "Booth",
        "global_minimum": 0.0,
        "global_optimum": (1, 3),
        "modality": "unimodal",
        "separable": False,
        "search_domain": (-10, 10),
        "difficulty": "easy",
        "dimensions": 2,
    }

    return booth


def get_beale_function() -> Callable[[np.ndarray], float]:
    """
    Beale function: Complex 2D function with multiple local minima

    Properties:
    - Multimodal
    - 2D function (uses first two variables)
    - Global minimum: f(3, 0.5) = 0
    - Search domain: [-4.5, 4.5]^2
    - Medium difficulty

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def beale(control_plan: np.ndarray) -> float:
        flat_plan = control_plan.flatten()
        if len(flat_plan) < 2:
            x = flat_plan[0] if len(flat_plan) > 0 else 0
            y = 0
        else:
            x, y = flat_plan[0], flat_plan[1]

        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2

        return term1 + term2 + term3

    beale.properties = {
        "name": "Beale",
        "global_minimum": 0.0,
        "global_optimum": (3, 0.5),
        "modality": "multimodal",
        "separable": False,
        "search_domain": (-4.5, 4.5),
        "difficulty": "medium",
        "dimensions": 2,
    }

    return beale


def get_pharmaceutical_mpc_function() -> Callable[[np.ndarray], float]:
    """
    Pharmaceutical MPC-specific fitness function for realistic testing.

    Models typical pharmaceutical granulation process with:
    - Particle size (d50) tracking
    - Moisture content (LOD) tracking
    - Control effort penalties
    - Process constraints

    Returns:
        Callable that accepts control_plan array and returns scalar fitness
    """

    def pharmaceutical_mpc(control_plan: np.ndarray) -> float:
        # Assume control_plan shape: (horizon, 3) for [spray_rate, air_flow, carousel_speed]
        if control_plan.ndim == 1:
            # Reshape flat array to (horizon, 3)
            horizon = len(control_plan) // 3
            control_plan = control_plan[: horizon * 3].reshape(horizon, 3)

        # Target process outputs
        target_d50 = 450.0  # micrometers
        target_lod = 1.8  # percent

        # Extract control variables
        spray_rate = control_plan[:, 0]
        air_flow = control_plan[:, 1]
        carousel_speed = control_plan[:, 2]

        # Simple process model (realistic pharmaceutical relationships)
        d50_response = 300 + spray_rate * 1.2 - air_flow * 0.15 + carousel_speed * 2.0
        lod_response = 3.5 - air_flow * 0.004 + spray_rate * 0.003 - carousel_speed * 0.02

        # Tracking errors
        d50_error = np.sum((d50_response - target_d50) ** 2)
        lod_error = np.sum((lod_response - target_lod) ** 2)

        # Control effort (smoothness penalty)
        control_effort = 0.0
        if len(control_plan) > 1:
            control_changes = np.diff(control_plan, axis=0)
            control_effort = np.sum(control_changes**2)

        # Constraint penalties
        constraint_penalty = 0.0

        # Spray rate constraints (80-180 g/min)
        constraint_penalty += 1000 * np.sum(np.maximum(0, 80 - spray_rate))
        constraint_penalty += 1000 * np.sum(np.maximum(0, spray_rate - 180))

        # Air flow constraints (400-700 m³/h)
        constraint_penalty += 1000 * np.sum(np.maximum(0, 400 - air_flow))
        constraint_penalty += 1000 * np.sum(np.maximum(0, air_flow - 700))

        # Carousel speed constraints (20-40 rpm)
        constraint_penalty += 1000 * np.sum(np.maximum(0, 20 - carousel_speed))
        constraint_penalty += 1000 * np.sum(np.maximum(0, carousel_speed - 40))

        # Combined objective
        total_cost = (
            d50_error
            + 100 * lod_error  # Tracking (LOD more critical)
            + 0.1 * control_effort  # Smoothness
            + constraint_penalty
        )  # Constraints

        return total_cost

    pharmaceutical_mpc.properties = {
        "name": "Pharmaceutical MPC",
        "global_minimum": "unknown",
        "global_optimum": "problem_dependent",
        "modality": "multimodal",
        "separable": False,
        "search_domain": "pharmaceutical_bounds",
        "difficulty": "medium",
        "realistic": True,
        "constraints": True,
    }

    return pharmaceutical_mpc


# Registry of all benchmark functions
BENCHMARK_FUNCTIONS: Dict[str, Callable[[], Callable[[np.ndarray], float]]] = {
    "sphere": get_sphere_function,
    "rosenbrock": get_rosenbrock_function,
    "rastrigin": get_rastrigin_function,
    "ackley": get_ackley_function,
    "booth": get_booth_function,
    "beale": get_beale_function,
    "pharmaceutical_mpc": get_pharmaceutical_mpc_function,
}


def get_benchmark_function_properties(function_name: str) -> Dict[str, Any]:
    """Get properties of a benchmark function."""
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {function_name}")

    func = BENCHMARK_FUNCTIONS[function_name]()
    return getattr(func, "properties", {})


def evaluate_on_benchmark_suite(
    control_plan: np.ndarray, function_names: list = None
) -> Dict[str, float]:
    """
    Evaluate a control plan on multiple benchmark functions.

    Args:
        control_plan: Control sequence to evaluate
        function_names: List of function names to use (default: all)

    Returns:
        Dictionary mapping function names to fitness values
    """
    if function_names is None:
        function_names = list(BENCHMARK_FUNCTIONS.keys())

    results = {}
    for name in function_names:
        if name in BENCHMARK_FUNCTIONS:
            func = BENCHMARK_FUNCTIONS[name]()
            results[name] = func(control_plan)
        else:
            raise ValueError(f"Unknown benchmark function: {name}")

    return results


def get_optimal_bounds_for_function(function_name: str, horizon: int, num_cpps: int) -> list:
    """
    Get appropriate parameter bounds for a benchmark function.

    Args:
        function_name: Name of benchmark function
        horizon: Control horizon length
        num_cpps: Number of control variables

    Returns:
        List of (min, max) tuples for parameter bounds
    """
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {function_name}")

    func = BENCHMARK_FUNCTIONS[function_name]()
    properties = getattr(func, "properties", {})

    if function_name == "pharmaceutical_mpc":
        # Use realistic pharmaceutical bounds
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

    # Use function's search domain
    domain = properties.get("search_domain", (-10, 10))
    if isinstance(domain, tuple):
        low, high = domain
    else:
        low, high = -10, 10

    total_params = horizon * num_cpps
    return [(low, high)] * total_params


def create_test_problem(
    function_name: str, horizon: int = 5, num_cpps: int = 3
) -> Tuple[Callable, list, Dict]:
    """
    Create a complete test problem with function, bounds, and metadata.

    Args:
        function_name: Name of benchmark function
        horizon: Control horizon length
        num_cpps: Number of control variables

    Returns:
        Tuple of (fitness_function, parameter_bounds, problem_info)
    """
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {function_name}")

    fitness_function = BENCHMARK_FUNCTIONS[function_name]()
    parameter_bounds = get_optimal_bounds_for_function(function_name, horizon, num_cpps)

    properties = getattr(fitness_function, "properties", {})
    problem_info = {
        "function_name": function_name,
        "horizon": horizon,
        "num_cpps": num_cpps,
        "total_variables": horizon * num_cpps,
        "properties": properties,
    }

    return fitness_function, parameter_bounds, problem_info
