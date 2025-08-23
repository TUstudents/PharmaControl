"""
Test data package for robust_mpc testing.

This package provides standardized test data, benchmark functions, and
pharmaceutical scenarios used across multiple test modules.
"""

from .benchmark_functions import *
from .pharmaceutical_scenarios import *

__all__ = [
    # Benchmark functions
    "get_sphere_function",
    "get_rosenbrock_function",
    "get_rastrigin_function",
    "get_ackley_function",
    "get_booth_function",
    "get_beale_function",
    "BENCHMARK_FUNCTIONS",
    # Pharmaceutical scenarios
    "get_granulation_scenario",
    "get_coating_scenario",
    "get_tableting_scenario",
    "get_startup_scenario",
    "get_disturbance_rejection_scenario",
    "get_grade_changeover_scenario",
    "PHARMACEUTICAL_SCENARIOS",
    # Utility functions
    "create_bounds_from_scenario",
    "generate_test_control_sequence",
    "validate_pharmaceutical_constraints",
]
