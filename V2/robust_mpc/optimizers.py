import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Callable

class GeneticOptimizer:
    """Genetic Algorithm optimizer for complex pharmaceutical process control optimization.
    
    This class implements a sophisticated evolutionary optimization approach using the
    DEAP (Distributed Evolutionary Algorithms in Python) framework. Designed specifically
    for solving challenging Model Predictive Control optimization problems where traditional
    gradient-based methods may fail due to non-convex objective functions, complex constraints,
    or discontinuous fitness landscapes common in pharmaceutical manufacturing.
    
    Key Advantages:
        - Global optimization capability for non-convex problems
        - Robust handling of complex constraint spaces
        - Parallelizable population-based search
        - Effective exploration of multimodal fitness landscapes
        - No gradient information required
    
    Algorithm Overview:
        1. **Population Initialization**: Random sampling within constraint bounds
        2. **Fitness Evaluation**: Objective function assessment for each individual
        3. **Selection**: Tournament selection of high-fitness individuals
        4. **Crossover**: Two-point crossover to create offspring solutions
        5. **Mutation**: Gaussian mutation for solution diversity
        6. **Bound Enforcement**: Constraint satisfaction through repair mechanisms
    
    Optimization Framework:
        - Decision Variables: Control action sequences over prediction horizon
        - Objective Function: Weighted combination of tracking error and control effort
        - Constraints: Process variable bounds and rate-of-change limitations
        - Search Strategy: Population-based evolutionary exploration
    
    Performance Characteristics:
        - Convergence: Typically 50-200 generations for pharmaceutical applications
        - Population Size: Recommended 50-100 individuals for control problems
        - Computational Cost: O(population_size × generations × fitness_evaluations)
        - Success Rate: High reliability for global optima in complex landscapes
    
    Args:
        fitness_function (callable): Objective function to minimize accepting control_plan
            of shape (horizon, num_cpps) and returning scalar cost value.
            Should incorporate tracking error, control effort, and constraint penalties.
        param_bounds (list): Parameter bounds as [(min₁, max₁), (min₂, max₂), ...] for each
            control variable at each time step. Length = horizon × num_cpps.
            Bounds should reflect physical actuator limits and safety constraints.
        config (dict): Optimizer configuration containing:
            - 'horizon': Control horizon length (typically 5-10 steps)
            - 'num_cpps': Number of Critical Process Parameters
            - 'population_size': GA population size (default: 50)
            - 'n_generations': Maximum generations (default: 100)
            - 'cx_prob': Crossover probability (default: 0.7)
            - 'mut_prob': Mutation probability (default: 0.2)
            - 'tournament_size': Selection tournament size (default: 3)
    
    Attributes:
        fitness_function (callable): Stored objective function
        param_bounds (list): Stored parameter constraints
        config (dict): Stored optimizer configuration
        n_params (int): Total number of decision variables (horizon × num_cpps)
        toolbox (deap.base.Toolbox): DEAP framework toolbox with registered operators
    
    Example:
        >>> # Configure for granulation process control
        >>> def mpc_fitness(control_plan):
        ...     # Evaluate control sequence quality
        ...     tracking_error = calculate_tracking_error(control_plan)
        ...     control_effort = calculate_control_effort(control_plan)
        ...     return tracking_error + 0.1 * control_effort
        >>> 
        >>> # Define parameter bounds for 5-step horizon, 3 CPPs
        >>> bounds = [(80, 180), (400, 700), (20, 40)] * 5  # spray, air, speed
        >>> config = {'horizon': 5, 'num_cpps': 3, 'population_size': 60}
        >>> optimizer = GeneticOptimizer(mpc_fitness, bounds, config)
        >>> 
        >>> # Find optimal control sequence
        >>> best_solution, best_cost = optimizer.optimize()
        >>> optimal_action = best_solution[0, :]  # First control action
    
    Notes:
        - Uses DEAP framework for robust evolutionary algorithm implementation
        - Tournament selection balances exploration and exploitation
        - Gaussian mutation provides local search capability
        - Two-point crossover maintains solution structure
        - Automatic constraint handling through bound repair mechanisms
        - Suitable for non-differentiable and multi-modal objective functions
    """
    def __init__(self, fitness_function, param_bounds, config):
        # Validate input types
        if not callable(fitness_function):
            raise TypeError("fitness_function must be callable")
        if not isinstance(param_bounds, list):
            raise TypeError("param_bounds must be a list")
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
            
        self.fitness_function = fitness_function
        self.param_bounds = param_bounds # List of (min, max) for each gene
        self.config = config
        self.n_params = len(param_bounds) # Number of genes in an individual
        
        # Validate configuration
        self._validate_config()
        
        # Clean up any existing DEAP creators to prevent conflicts
        self._cleanup_deap_creators()

        # Synchronize RNGs: if the user seeded numpy beforehand (np.random.seed(...)),
        # derive a seed for the Python `random` module so DEAP and other code using
        # the stdlib random become deterministic with the same numpy seed.
        # If an explicit seed is provided in config under 'seed', use that for both.
        self._sync_rngs()

        # --- DEAP Toolbox Setup ---
        # 1. Define the fitness objective (minimizing a single value)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # 2. Define the structure of an individual (a list of floats with a fitness attribute)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # 3. Individual generator: Create a complete individual with position-specific bounds
        self.toolbox.register("individual", self._create_individual)

        # 4. Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 5. Register the evolutionary operators
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._mate_with_bounds)
        self.toolbox.register("mutate", self._mutate_with_bounds)
        tournsize = self.config.get('tournament_size', 3)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)

    def _sync_rngs(self):
        """Ensure Python's `random` is seeded consistently with NumPy or explicit config.

        Behavior:
        - If config contains 'seed', seed both numpy and random with it.
        - Otherwise, draw a reproducible integer from numpy's RNG and seed random
          with that value. This makes code that calls `np.random.seed(...)` before
          constructing the optimizer also control the stdlib random (and therefore
          DEAP which uses it), improving reproducibility for tests.
        """
        if 'seed' in self.config and self.config['seed'] is not None:
            seed = int(self.config['seed'])
            np.random.seed(seed)
            random.seed(seed)
        else:
            # Draw from numpy's RNG to produce a seed for the stdlib random.
            # Use a 32-bit unsigned range to be safe for random.seed
            try:
                seed = int(np.random.randint(0, 2**31 - 1))
            except Exception:
                # Fallback: use a fixed seed if numpy RNG is not available for any reason
                seed = 0
            random.seed(seed)

    def _create_individual(self):
        """Create a complete individual with position-specific parameter bounds.
        
        Generates a control sequence individual where each gene respects its specific
        parameter bounds. This ensures proper constraint handling during initialization
        and maintains the relationship between gene position and parameter type.
        
        Returns:
            creator.Individual: Complete individual with all genes within their bounds
        """
        individual = creator.Individual()
        for i in range(self.n_params):
            low, high = self.param_bounds[i]
            individual.append(random.uniform(low, high))
        return individual
    
    def _validate_config(self):
        """Validate configuration parameters for consistency."""
        required_keys = ['horizon', 'num_cpps', 'population_size', 'num_generations']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate non-zero positive values
        if self.config['horizon'] <= 0:
            raise ValueError("horizon must be greater than 0")
        if self.config['num_cpps'] <= 0:
            raise ValueError("num_cpps must be greater than 0")
        if self.config['population_size'] <= 0:
            raise ValueError("population_size must be greater than 0")
        if self.config['num_generations'] <= 0:
            raise ValueError("num_generations must be greater than 0")
            
        expected_params = self.config['horizon'] * self.config['num_cpps']
        if len(self.param_bounds) != expected_params:
            raise ValueError(f"Parameter bounds length {len(self.param_bounds)} != expected {expected_params}")
        
        # Validate param_bounds is not empty when expected_params > 0
        if expected_params > 0 and not self.param_bounds:
            raise ValueError("param_bounds cannot be empty when horizon and num_cpps are greater than 0")
    
    def _cleanup_deap_creators(self):
        """Clean up existing DEAP creator classes to prevent conflicts."""
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual
    
    def _mate_with_bounds(self, ind1, ind2):
        """Crossover with bound repair."""
        tools.cxTwoPoint(ind1, ind2)
        self._check_bounds(ind1)
        self._check_bounds(ind2)
        return ind1, ind2
    
    def _mutate_with_bounds(self, individual):
        """Mutation with bound repair."""
        tools.mutGaussian(individual, mu=0, sigma=0.2, indpb=0.1)
        self._check_bounds(individual)
        return individual,

    def _evaluate(self, individual):
        """Evaluate individual fitness using MPC objective function.
        
        Transforms the DEAP individual representation (flat list) into the expected
        control plan format and computes the fitness value using the provided
        objective function. The fitness incorporates tracking error, control effort,
        and constraint violations for comprehensive solution assessment.
        
        Args:
            individual (list): Flat list of decision variables representing
                control actions over the horizon (length = horizon × num_cpps)
        
        Returns:
            tuple: Single-element tuple containing fitness value (DEAP requirement)
                Lower values indicate better solutions for minimization problems
        
        Notes:
            - Reshapes flat individual into (horizon, num_cpps) control plan matrix
            - Fitness function should handle constraint violations internally
            - Returned tuple format required by DEAP framework conventions
        """
        # DEAP works with a flat list, we need to reshape it into a control plan
        control_plan = np.array(individual).reshape(self.config['horizon'], self.config['num_cpps'])
        cost = self.fitness_function(control_plan)
        return (cost,) # DEAP expects a tuple

    def _check_bounds(self, individual):
        """Repair operator to enforce parameter bounds after genetic operations.
        
        Ensures that all genes remain within their specified parameter bounds after
        crossover and mutation operations. Uses simple clipping to the nearest
        boundary value, which maintains feasibility while minimizing disruption
        to the genetic algorithm's search process.
        
        Args:
            individual (list): Individual potentially violating bounds after mutation/crossover
        
        Returns:
            list: Repaired individual with all genes within their respective bounds
        
        Notes:
            - Applies hard constraints through boundary clipping
            - Maintains individual structure and gene relationships
            - Called automatically after genetic operations
            - Alternative repair strategies could include reflection or random replacement
        """
        for i, (low, high) in enumerate(self.param_bounds):
            if individual[i] > high:
                individual[i] = high
            elif individual[i] < low:
                individual[i] = low
        return individual

    def optimize(self):
        """Execute genetic algorithm optimization to find optimal control sequence.
        
        Performs evolutionary optimization using the configured genetic algorithm
        parameters to search for the control sequence that minimizes the MPC
        objective function. Uses Hall of Fame tracking to preserve the best
        solution found during the evolutionary process.
        
        Algorithm Flow:
            1. Initialize random population within parameter bounds
            2. Evaluate fitness for all individuals
            3. Execute evolutionary loop:
               - Selection: Tournament selection of parents
               - Crossover: Two-point recombination of parent solutions  
               - Mutation: Gaussian perturbation for diversity
               - Constraint repair: Bound enforcement
               - Replacement: Generational replacement strategy
            4. Return best solution from Hall of Fame
        
        Returns:
            np.ndarray: Optimal control sequence of shape (horizon, num_cpps) 
                representing the best control plan found during optimization.
                First row contains the immediate control action to apply.
        
        Raises:
            RuntimeError: If optimization fails to find valid solutions
            ValueError: If configuration parameters are invalid
        
        Notes:
            - Population diversity maintained through tournament selection
            - Convergence typically achieved within 50-200 generations
            - Best solution preserved in Hall of Fame regardless of final population
            - Automatic constraint handling through repair operator
        """
        pop = self.toolbox.population(n=self.config['population_size'])

        # Use a hall of fame to keep track of the best individual found so far
        hof = tools.HallOfFame(1)

        # Run the evolution
        # Handle different config key names for compatibility
        crossover_prob = self.config.get('crossover_prob', self.config.get('cx_prob', 0.7))
        mutation_prob = self.config.get('mutation_prob', self.config.get('mut_prob', 0.2))
        
        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=self.config['num_generations'],
            halloffame=hof,
            verbose=False
        )

        # Reshape the best individual back into a plan
        best_individual = hof[0]
        best_plan = np.array(best_individual).reshape(self.config['horizon'], self.config['num_cpps'])

        return best_plan
