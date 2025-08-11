import random
import numpy as np
from deap import base, creator, tools, algorithms

class GeneticOptimizer:
    """
    A wrapper for the DEAP library to find optimal control sequences using a
    Genetic Algorithm.
    """
    def __init__(self, fitness_function, param_bounds, config):
        self.fitness_function = fitness_function
        self.param_bounds = param_bounds # List of (min, max) for each gene
        self.config = config
        self.n_params = len(param_bounds) # Number of genes in an individual

        # --- DEAP Toolbox Setup ---
        # 1. Define the fitness objective (minimizing a single value)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # 2. Define the structure of an individual (a list of floats with a fitness attribute)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # 3. Gene generator: How to create a single gene (a random float within bounds)
        self.toolbox.register("attr_float", self._rand_float_in_bounds)

        # 4. Individual and Population generators
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n_params)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 5. Register the evolutionary operators
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _rand_float_in_bounds(self):
        """Helper to create a gene by respecting the parameter bounds."""
        return [random.uniform(low, high) for low, high in self.param_bounds]

    def _evaluate(self, individual):
        """Wrapper to connect DEAP's evaluation to our MPC fitness function."""
        # DEAP works with a flat list, we need to reshape it into a control plan
        control_plan = np.array(individual).reshape(self.config['horizon'], self.config['num_cpps'])
        cost = self.fitness_function(control_plan)
        return (cost,) # DEAP expects a tuple

    def _check_bounds(self, individual):
        """Ensures that mutated/crossed-over individuals stay within bounds."""
        for i, (low, high) in enumerate(self.param_bounds):
            if individual[i] > high:
                individual[i] = high
            elif individual[i] < low:
                individual[i] = low
        return individual

    def optimize(self):
        """Runs the genetic algorithm to find the best control plan."""
        pop = self.toolbox.population(n=self.config['population_size'])

        # Use a hall of fame to keep track of the best individual found so far
        hof = tools.HallOfFame(1)

        # Run the evolution
        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.config['crossover_prob'],
            mutpb=self.config['mutation_prob'],
            ngen=self.config['num_generations'],
            halloffame=hof,
            verbose=False
        )

        # Reshape the best individual back into a plan
        best_individual = hof[0]
        best_plan = np.array(best_individual).reshape(self.config['horizon'], self.config['num_cpps'])

        return best_plan
