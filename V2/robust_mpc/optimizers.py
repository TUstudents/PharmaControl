"""
Advanced Optimization Module

This module provides sophisticated optimization algorithms for finding
optimal control sequences in complex, constrained spaces. These optimizers
are essential for MPC systems that need to handle multi-objective optimization
and large search spaces efficiently.

Key Classes:
- GeneticOptimizer: Genetic Algorithm wrapper around DEAP  ✅ Available as of V2-3
- BayesianOptimizer: Gaussian Process-based optimization (future)
- ParticleSwarmOptimizer: PSO for continuous optimization (future)
- MultiObjectiveOptimizer: Multi-objective optimization (future)

Dependencies:
- deap: Distributed Evolutionary Algorithms in Python
- numpy: Numerical computations
- scipy: Scientific computing utilities
"""

import numpy as np
import random
from typing import Callable, List, Optional, Tuple, Dict, Any
import warnings
from abc import ABC, abstractmethod

# DEAP imports
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    warnings.warn("DEAP not available. Install with: pip install deap")


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""
    
    @abstractmethod
    def optimize(self, fitness_function: Callable, constraints: Optional[Dict] = None):
        """Find optimal solution given fitness function and constraints."""
        pass


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer wrapper around DEAP.
    
    Uses evolutionary computation to find optimal control sequences
    in complex, multi-dimensional search spaces with constraints.
    
    This implementation is now complete as of V2-3.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 elite_size: int = 2,
                 mutation_std: float = 0.1):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
            tournament_size (int): Size of tournament for selection
            elite_size (int): Number of elite individuals to preserve
            mutation_std (float): Standard deviation for Gaussian mutation
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP is required for GeneticOptimizer. Install with: pip install deap")
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.mutation_std = mutation_std
        
        # Statistics tracking
        self.stats = None
        self.logbook = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.hall_of_fame = None
        
        # DEAP setup (will be initialized when optimize is called)
        self.toolbox = None
        self._setup_complete = False
    
    def _setup_deap(self, bounds: List[Tuple[float, float]]):
        """Initialize DEAP toolbox with problem-specific parameters."""
        # Clear any existing creator classes to avoid conflicts
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        # Define fitness and individual structure
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.bounds = bounds
        self.n_params = len(bounds)
        
        # Register attribute generator for each parameter
        def create_random_param(bounds_list):
            return [random.uniform(low, high) for low, high in bounds_list]
        
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_wrapper)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_gaussian)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        self._setup_complete = True
    
    def _create_individual(self):
        """Create a random individual respecting bounds."""
        individual = creator.Individual()
        for low, high in self.bounds:
            individual.append(random.uniform(low, high))
        return individual
    
    def _evaluate_wrapper(self, individual):
        """Wrapper to connect DEAP evaluation to fitness function."""
        # Apply bounds clipping
        clipped_individual = self._apply_bounds(individual)
        fitness_value = self.fitness_function(np.array(clipped_individual))
        return (fitness_value,)
    
    def _mutate_gaussian(self, individual, mu=0, sigma=None, indpb=0.1):
        """Gaussian mutation with bounds checking."""
        if sigma is None:
            sigma = self.mutation_std
            
        for i in range(len(individual)):
            if random.random() < indpb:
                # Apply Gaussian noise
                individual[i] += random.gauss(mu, sigma)
                
        # Apply bounds
        individual[:] = self._apply_bounds(individual)
        return individual,
    
    def _apply_bounds(self, individual):
        """Ensure individual stays within parameter bounds."""
        clipped = []
        for i, (low, high) in enumerate(self.bounds):
            clipped.append(max(low, min(high, individual[i])))
        return clipped
    
    def optimize(self, 
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Find optimal solution using genetic algorithm.
        
        Args:
            fitness_function: Function to minimize (lower is better)
            bounds: List of (min, max) bounds for each variable
            constraints: Optional constraint functions
            
        Returns:
            np.ndarray: Best solution found
        """
        # Store fitness function for use in wrapper
        self.fitness_function = fitness_function
        
        # Setup DEAP if not already done
        if not self._setup_complete:
            self._setup_deap(bounds)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics tracking
        self.hall_of_fame = tools.HallOfFame(maxsize=self.elite_size)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # Run evolution
        population, self.logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=self.stats,
            halloffame=self.hall_of_fame,
            verbose=False
        )
        
        # Extract best solution
        self.best_individual = self.hall_of_fame[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        return np.array(self.best_individual)
    
    def optimize_pareto(self, 
                       multi_objective_fitness: Callable,
                       bounds: List[Tuple[float, float]],
                       n_objectives: int,
                       constraints: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Multi-objective optimization returning Pareto front.
        
        Args:
            multi_objective_fitness: Function returning tuple of objectives
            bounds: Variable bounds
            n_objectives: Number of objectives
            constraints: Optional constraints
            
        Returns:
            List[np.ndarray]: Pareto-optimal solutions
        """
        # This is a placeholder for future implementation
        raise NotImplementedError("Multi-objective optimization will be implemented in a future version")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if self.logbook is None:
            return {"message": "No optimization run yet"}
            
        return {
            "best_fitness": self.best_fitness,
            "generations_run": len(self.logbook),
            "final_population_stats": {
                "mean_fitness": self.logbook[-1]["avg"],
                "std_fitness": self.logbook[-1]["std"],
                "min_fitness": self.logbook[-1]["min"],
                "max_fitness": self.logbook[-1]["max"]
            },
            "convergence_history": [record["min"] for record in self.logbook]
        }


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Processes.
    
    Efficiently explores expensive-to-evaluate functions by maintaining
    a probabilistic model of the objective function.
    
    Note: This is a placeholder for future implementation (V2.2+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "BayesianOptimizer is planned for future implementation (V2.2+). "
            "Use GeneticOptimizer for current optimization needs."
        )
    
    def optimize(self, fitness_function: Callable, constraints: Optional[Dict] = None):
        raise NotImplementedError("Planned for V2.2+")


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization for continuous problems.
    
    Uses swarm intelligence principles where particles explore the search
    space by following their own best position and the global best.
    
    Note: This is a placeholder for future implementation (V2.2+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "ParticleSwarmOptimizer is planned for future implementation (V2.2+). "
            "Use GeneticOptimizer for current optimization needs."
        )
    
    def optimize(self, fitness_function: Callable, constraints: Optional[Dict] = None):
        raise NotImplementedError("Planned for V2.2+")


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Differential Evolution optimization algorithm.
    
    Similar to genetic algorithms but uses vector differences for mutation
    and is often effective for continuous optimization problems.
    
    Note: This is a placeholder for future implementation (V2.2+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "DifferentialEvolutionOptimizer is planned for future implementation (V2.2+). "
            "Use GeneticOptimizer for current optimization needs."
        )
    
    def optimize(self, fitness_function: Callable, constraints: Optional[Dict] = None):
        raise NotImplementedError("Planned for V2.2+")


# Utility functions for optimization
def validate_bounds(bounds: List[Tuple[float, float]], n_vars: int) -> bool:
    """
    Validate that bounds are properly specified.
    
    Args:
        bounds: List of (min, max) tuples
        n_vars: Expected number of variables
        
    Returns:
        bool: True if bounds are valid
    """
    if len(bounds) != n_vars:
        return False
    
    for min_val, max_val in bounds:
        if min_val >= max_val:
            return False
    
    return True


def apply_bounds(solution: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip solution to stay within bounds.
    
    Args:
        solution: Solution vector
        bounds: List of (min, max) bounds
        
    Returns:
        np.ndarray: Clipped solution
    """
    clipped = solution.copy()
    for i, (min_val, max_val) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], min_val, max_val)
    return clipped


def evaluate_constraints(solution: np.ndarray, 
                        constraint_functions: List[Callable]) -> bool:
    """
    Evaluate whether solution satisfies all constraints.
    
    Args:
        solution: Solution to evaluate
        constraint_functions: List of constraint functions (should return True if satisfied)
        
    Returns:
        bool: True if all constraints are satisfied
    """
    return all(constraint(solution) for constraint in constraint_functions)


def penalty_method(fitness: float, 
                  constraint_violations: List[float],
                  penalty_weight: float = 1000.0) -> float:
    """
    Apply penalty method for constraint violations.
    
    Args:
        fitness: Original fitness value
        constraint_violations: List of constraint violation magnitudes
        penalty_weight: Weight for penalty term
        
    Returns:
        float: Penalized fitness value
    """
    total_violation = sum(max(0, violation) for violation in constraint_violations)
    return fitness + penalty_weight * total_violation


# Helper functions for MPC-specific optimization
def setup_mpc_bounds(cpp_constraints: Dict[str, Dict[str, float]], 
                    horizon: int) -> List[Tuple[float, float]]:
    """
    Create bounds list for MPC optimization over a horizon.
    
    Args:
        cpp_constraints: Dict of constraints for each CPP
        horizon: Planning horizon length
        
    Returns:
        List of (min, max) bounds for flattened chromosome
    """
    bounds = []
    cpp_names = list(cpp_constraints.keys())
    
    for _ in range(horizon):
        for name in cpp_names:
            bounds.append((
                cpp_constraints[name]['min_val'],
                cpp_constraints[name]['max_val']
            ))
    
    return bounds


def reshape_chromosome_to_plan(chromosome: np.ndarray, 
                              horizon: int, 
                              n_cpps: int) -> np.ndarray:
    """
    Reshape flat chromosome into control plan matrix.
    
    Args:
        chromosome: Flat array of control values
        horizon: Planning horizon
        n_cpps: Number of control variables
        
    Returns:
        np.ndarray: Control plan (horizon x n_cpps)
    """
    return chromosome.reshape(horizon, n_cpps)