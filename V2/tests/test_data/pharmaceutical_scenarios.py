"""
Pharmaceutical manufacturing scenarios for testing genetic algorithm optimization.

This module provides realistic pharmaceutical process control scenarios
including granulation, coating, tableting, and various operational
conditions like startup, disturbance rejection, and grade changeover.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass


@dataclass
class PharmaceuticalScenario:
    """Data class representing a pharmaceutical manufacturing scenario."""

    name: str
    description: str
    process_type: str
    initial_state: np.ndarray
    target_state: np.ndarray
    constraints: Dict[str, Dict[str, float]]
    disturbances: Dict[str, np.ndarray] = None
    time_horizon: int = 10
    control_variables: List[str] = None
    output_variables: List[str] = None
    fitness_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.control_variables is None:
            self.control_variables = ["spray_rate", "air_flow", "carousel_speed"]
        if self.output_variables is None:
            self.output_variables = ["d50", "lod"]
        if self.fitness_weights is None:
            self.fitness_weights = {"tracking": 1.0, "control_effort": 0.1, "constraints": 1000.0}


def get_granulation_scenario() -> PharmaceuticalScenario:
    """
    Standard continuous granulation process scenario.

    Process: Continuous granulation for tablet manufacturing
    Objective: Control particle size (d50) and moisture content (LOD)
    Control variables: spray_rate, air_flow, carousel_speed

    Returns:
        PharmaceuticalScenario for granulation process
    """
    return PharmaceuticalScenario(
        name="Standard Granulation",
        description="Continuous granulation process for tablet manufacturing",
        process_type="granulation",
        initial_state=np.array([400.0, 2.0]),  # d50=400μm, LOD=2.0%
        target_state=np.array([450.0, 1.8]),  # d50=450μm, LOD=1.8%
        constraints={
            "spray_rate": {"min_val": 80.0, "max_val": 180.0, "units": "g/min"},
            "air_flow": {"min_val": 400.0, "max_val": 700.0, "units": "m³/h"},
            "carousel_speed": {"min_val": 20.0, "max_val": 40.0, "units": "rpm"},
        },
        time_horizon=10,
        control_variables=["spray_rate", "air_flow", "carousel_speed"],
        output_variables=["d50", "lod"],
        fitness_weights={"tracking": 1.0, "control_effort": 0.1, "constraints": 1000.0},
    )


def get_coating_scenario() -> PharmaceuticalScenario:
    """
    Tablet coating process scenario.

    Process: Aqueous film coating of tablets
    Objective: Control coating thickness and uniformity
    Control variables: spray_rate, air_flow, pan_speed, inlet_temp, coating_time

    Returns:
        PharmaceuticalScenario for coating process
    """
    return PharmaceuticalScenario(
        name="Tablet Coating",
        description="Aqueous film coating process for tablets",
        process_type="coating",
        initial_state=np.array([0.0, 85.0]),  # thickness=0μm, uniformity=85%
        target_state=np.array([50.0, 95.0]),  # thickness=50μm, uniformity=95%
        constraints={
            "spray_rate": {"min_val": 50.0, "max_val": 120.0, "units": "g/min"},
            "air_flow": {"min_val": 300.0, "max_val": 500.0, "units": "m³/h"},
            "pan_speed": {"min_val": 5.0, "max_val": 15.0, "units": "rpm"},
            "inlet_temp": {"min_val": 40.0, "max_val": 80.0, "units": "°C"},
            "coating_time": {"min_val": 60.0, "max_val": 180.0, "units": "min"},
        },
        time_horizon=12,
        control_variables=["spray_rate", "air_flow", "pan_speed", "inlet_temp", "coating_time"],
        output_variables=["coating_thickness", "uniformity"],
        fitness_weights={"tracking": 1.0, "control_effort": 0.05, "constraints": 2000.0},
    )


def get_tableting_scenario() -> PharmaceuticalScenario:
    """
    Tablet compression process scenario.

    Process: Direct compression tableting
    Objective: Control tablet weight and hardness
    Control variables: compression_force, turret_speed, fill_depth

    Returns:
        PharmaceuticalScenario for tableting process
    """
    return PharmaceuticalScenario(
        name="Tablet Compression",
        description="Direct compression tableting process",
        process_type="tableting",
        initial_state=np.array([200.0, 50.0]),  # weight=200mg, hardness=50N
        target_state=np.array([250.0, 80.0]),  # weight=250mg, hardness=80N
        constraints={
            "compression_force": {"min_val": 5.0, "max_val": 25.0, "units": "kN"},
            "turret_speed": {"min_val": 10.0, "max_val": 50.0, "units": "rpm"},
            "fill_depth": {"min_val": 5.0, "max_val": 15.0, "units": "mm"},
        },
        time_horizon=8,
        control_variables=["compression_force", "turret_speed", "fill_depth"],
        output_variables=["tablet_weight", "hardness"],
        fitness_weights={"tracking": 1.0, "control_effort": 0.2, "constraints": 1500.0},
    )


def get_startup_scenario() -> PharmaceuticalScenario:
    """
    Process startup scenario.

    Scenario: Starting granulation process from cold conditions
    Challenge: Large initial deviation from target, potential overshoot

    Returns:
        PharmaceuticalScenario for startup conditions
    """
    return PharmaceuticalScenario(
        name="Process Startup",
        description="Starting granulation process from cold conditions",
        process_type="granulation",
        initial_state=np.array([300.0, 3.0]),  # d50=300μm, LOD=3.0%
        target_state=np.array([450.0, 1.8]),  # d50=450μm, LOD=1.8%
        constraints={
            "spray_rate": {"min_val": 80.0, "max_val": 180.0, "units": "g/min"},
            "air_flow": {"min_val": 400.0, "max_val": 700.0, "units": "m³/h"},
            "carousel_speed": {"min_val": 20.0, "max_val": 40.0, "units": "rpm"},
        },
        time_horizon=15,  # Longer horizon for startup
        fitness_weights={"tracking": 1.0, "control_effort": 0.05, "constraints": 1000.0},
    )


def get_disturbance_rejection_scenario() -> PharmaceuticalScenario:
    """
    Disturbance rejection scenario.

    Scenario: Feed moisture content disturbance affects process
    Challenge: Maintain targets despite unmeasured disturbance

    Returns:
        PharmaceuticalScenario with disturbance effects
    """
    return PharmaceuticalScenario(
        name="Disturbance Rejection",
        description="Rejection of feed moisture disturbance",
        process_type="granulation",
        initial_state=np.array([450.0, 1.8]),  # At target initially
        target_state=np.array([450.0, 1.8]),  # Maintain target
        constraints={
            "spray_rate": {"min_val": 80.0, "max_val": 180.0, "units": "g/min"},
            "air_flow": {"min_val": 400.0, "max_val": 700.0, "units": "m³/h"},
            "carousel_speed": {"min_val": 20.0, "max_val": 40.0, "units": "rpm"},
        },
        disturbances={
            "feed_moisture": np.array([0.0, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0, 0.0])
        },
        time_horizon=10,
        fitness_weights={"tracking": 2.0, "control_effort": 0.1, "constraints": 1000.0},
    )


def get_grade_changeover_scenario() -> PharmaceuticalScenario:
    """
    Product grade changeover scenario.

    Scenario: Transition from one product grade to another
    Challenge: Smooth transition while meeting different specifications

    Returns:
        PharmaceuticalScenario for grade changeover
    """
    return PharmaceuticalScenario(
        name="Grade Changeover",
        description="Product grade changeover operation",
        process_type="granulation",
        initial_state=np.array([450.0, 1.8]),  # Grade A
        target_state=np.array([600.0, 1.2]),  # Grade B
        constraints={
            "spray_rate": {"min_val": 100.0, "max_val": 200.0, "units": "g/min"},  # Wider range
            "air_flow": {"min_val": 500.0, "max_val": 800.0, "units": "m³/h"},
            "carousel_speed": {"min_val": 25.0, "max_val": 45.0, "units": "rpm"},
        },
        time_horizon=20,  # Longer transition
        fitness_weights={"tracking": 1.0, "control_effort": 0.2, "constraints": 1000.0},
    )


def get_high_variability_scenario() -> PharmaceuticalScenario:
    """
    High process variability scenario.

    Scenario: Process with high natural variability requiring tight control
    Challenge: Achieve consistent quality despite process variations

    Returns:
        PharmaceuticalScenario with tight control requirements
    """
    return PharmaceuticalScenario(
        name="High Variability Process",
        description="Process with high variability requiring tight control",
        process_type="granulation",
        initial_state=np.array([450.0, 1.8]),
        target_state=np.array([450.0, 1.8]),
        constraints={
            "spray_rate": {"min_val": 90.0, "max_val": 170.0, "units": "g/min"},
            "air_flow": {"min_val": 450.0, "max_val": 650.0, "units": "m³/h"},
            "carousel_speed": {"min_val": 22.0, "max_val": 38.0, "units": "rpm"},
        },
        time_horizon=10,
        fitness_weights={
            "tracking": 5.0,
            "control_effort": 0.3,
            "constraints": 1000.0,
        },  # High tracking weight
    )


def get_constrained_operation_scenario() -> PharmaceuticalScenario:
    """
    Constrained operation scenario.

    Scenario: Operation with very tight equipment constraints
    Challenge: Achieve control objectives within narrow operating windows

    Returns:
        PharmaceuticalScenario with tight constraints
    """
    return PharmaceuticalScenario(
        name="Constrained Operation",
        description="Operation with tight equipment constraints",
        process_type="granulation",
        initial_state=np.array([420.0, 2.0]),
        target_state=np.array([450.0, 1.8]),
        constraints={
            "spray_rate": {"min_val": 110.0, "max_val": 140.0, "units": "g/min"},  # Very narrow
            "air_flow": {"min_val": 480.0, "max_val": 520.0, "units": "m³/h"},  # Very narrow
            "carousel_speed": {"min_val": 28.0, "max_val": 32.0, "units": "rpm"},  # Very narrow
        },
        time_horizon=10,
        fitness_weights={
            "tracking": 1.0,
            "control_effort": 0.1,
            "constraints": 5000.0,
        },  # High constraint penalty
    )


# Registry of all pharmaceutical scenarios
PHARMACEUTICAL_SCENARIOS: Dict[str, Callable[[], PharmaceuticalScenario]] = {
    "granulation": get_granulation_scenario,
    "coating": get_coating_scenario,
    "tableting": get_tableting_scenario,
    "startup": get_startup_scenario,
    "disturbance_rejection": get_disturbance_rejection_scenario,
    "grade_changeover": get_grade_changeover_scenario,
    "high_variability": get_high_variability_scenario,
    "constrained_operation": get_constrained_operation_scenario,
}


def create_bounds_from_scenario(scenario: PharmaceuticalScenario) -> List[Tuple[float, float]]:
    """
    Create parameter bounds list from pharmaceutical scenario.

    Args:
        scenario: PharmaceuticalScenario object

    Returns:
        List of (min, max) tuples for optimization bounds
    """
    bounds = []

    for _ in range(scenario.time_horizon):
        for control_var in scenario.control_variables:
            if control_var in scenario.constraints:
                min_val = scenario.constraints[control_var]["min_val"]
                max_val = scenario.constraints[control_var]["max_val"]
                bounds.append((min_val, max_val))
            else:
                # Default bounds if not specified
                bounds.append((0.0, 100.0))

    return bounds


def create_fitness_function_from_scenario(
    scenario: PharmaceuticalScenario,
) -> Callable[[np.ndarray], float]:
    """
    Create fitness function from pharmaceutical scenario.

    Args:
        scenario: PharmaceuticalScenario object

    Returns:
        Fitness function that accepts control_plan and returns scalar cost
    """

    def scenario_fitness(control_plan: np.ndarray) -> float:
        # Reshape if needed
        if control_plan.ndim == 1:
            horizon = scenario.time_horizon
            num_controls = len(scenario.control_variables)
            if len(control_plan) >= horizon * num_controls:
                control_plan = control_plan[: horizon * num_controls].reshape(horizon, num_controls)
            else:
                # Pad if too short
                padded = np.zeros(horizon * num_controls)
                padded[: len(control_plan)] = control_plan
                control_plan = padded.reshape(horizon, num_controls)

        # Simple process model based on scenario type
        if scenario.process_type == "granulation":
            predicted_outputs = _granulation_process_model(control_plan, scenario)
        elif scenario.process_type == "coating":
            predicted_outputs = _coating_process_model(control_plan, scenario)
        elif scenario.process_type == "tableting":
            predicted_outputs = _tableting_process_model(control_plan, scenario)
        else:
            # Generic model
            predicted_outputs = _generic_process_model(control_plan, scenario)

        # Calculate costs
        tracking_cost = _calculate_tracking_cost(predicted_outputs, scenario)
        control_effort_cost = _calculate_control_effort_cost(control_plan, scenario)
        constraint_penalty = _calculate_constraint_penalty(control_plan, scenario)

        # Weighted combination
        weights = scenario.fitness_weights
        total_cost = (
            weights["tracking"] * tracking_cost
            + weights["control_effort"] * control_effort_cost
            + weights["constraints"] * constraint_penalty
        )

        return total_cost

    return scenario_fitness


def _granulation_process_model(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> np.ndarray:
    """Simple granulation process model."""
    # Assume control variables: [spray_rate, air_flow, carousel_speed]
    spray_rate = control_plan[:, 0]
    air_flow = control_plan[:, 1]
    carousel_speed = control_plan[:, 2]

    # Simple empirical model
    d50 = 300 + spray_rate * 1.2 - air_flow * 0.15 + carousel_speed * 2.0
    lod = 3.5 - air_flow * 0.004 + spray_rate * 0.003 - carousel_speed * 0.02

    # Apply disturbances if present
    if scenario.disturbances:
        if "feed_moisture" in scenario.disturbances:
            moisture_dist = scenario.disturbances["feed_moisture"]
            if len(moisture_dist) >= len(lod):
                lod += moisture_dist[: len(lod)]

    return np.column_stack([d50, lod])


def _coating_process_model(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> np.ndarray:
    """Simple coating process model."""
    # Assume control variables: [spray_rate, air_flow, pan_speed, inlet_temp, coating_time]
    spray_rate = control_plan[:, 0]
    air_flow = control_plan[:, 1]
    pan_speed = control_plan[:, 2]
    inlet_temp = control_plan[:, 3] if control_plan.shape[1] > 3 else np.ones_like(spray_rate) * 60
    coating_time = (
        control_plan[:, 4] if control_plan.shape[1] > 4 else np.ones_like(spray_rate) * 120
    )

    # Simple coating model
    thickness = spray_rate * coating_time * 0.4 - air_flow * 0.05 + inlet_temp * 0.3
    uniformity = 70 + pan_speed * 2.0 - np.abs(inlet_temp - 60) * 0.3 + spray_rate * 0.1

    return np.column_stack([thickness, uniformity])


def _tableting_process_model(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> np.ndarray:
    """Simple tableting process model."""
    # Assume control variables: [compression_force, turret_speed, fill_depth]
    compression_force = control_plan[:, 0]
    turret_speed = control_plan[:, 1]
    fill_depth = control_plan[:, 2]

    # Simple tableting model
    weight = fill_depth * 20 - turret_speed * 0.5 + compression_force * 2.0
    hardness = compression_force * 4.0 - turret_speed * 0.3 + fill_depth * 1.5

    return np.column_stack([weight, hardness])


def _generic_process_model(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> np.ndarray:
    """Generic process model for unknown process types."""
    # Simple linear model
    num_outputs = len(scenario.output_variables)
    outputs = np.zeros((control_plan.shape[0], num_outputs))

    for i in range(num_outputs):
        outputs[:, i] = np.sum(control_plan * (i + 1), axis=1)

    return outputs


def _calculate_tracking_cost(
    predicted_outputs: np.ndarray, scenario: PharmaceuticalScenario
) -> float:
    """Calculate tracking error cost."""
    target = np.tile(scenario.target_state, (predicted_outputs.shape[0], 1))
    tracking_errors = (predicted_outputs - target) ** 2
    return np.sum(tracking_errors)


def _calculate_control_effort_cost(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> float:
    """Calculate control effort (smoothness) cost."""
    if control_plan.shape[0] <= 1:
        return 0.0

    control_changes = np.diff(control_plan, axis=0)
    return np.sum(control_changes**2)


def _calculate_constraint_penalty(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> float:
    """Calculate constraint violation penalties."""
    penalty = 0.0

    for i, control_var in enumerate(scenario.control_variables):
        if i >= control_plan.shape[1]:
            break

        if control_var in scenario.constraints:
            min_val = scenario.constraints[control_var]["min_val"]
            max_val = scenario.constraints[control_var]["max_val"]

            values = control_plan[:, i]

            # Penalty for violations
            penalty += np.sum(np.maximum(0, min_val - values))
            penalty += np.sum(np.maximum(0, values - max_val))

    return penalty


def generate_test_control_sequence(
    scenario: PharmaceuticalScenario, sequence_type: str = "random"
) -> np.ndarray:
    """
    Generate test control sequence for a scenario.

    Args:
        scenario: PharmaceuticalScenario object
        sequence_type: Type of sequence ('random', 'step', 'ramp', 'optimal_guess')

    Returns:
        Control sequence array of shape (horizon, num_controls)
    """
    horizon = scenario.time_horizon
    num_controls = len(scenario.control_variables)

    if sequence_type == "random":
        control_seq = np.zeros((horizon, num_controls))
        for i, control_var in enumerate(scenario.control_variables):
            if control_var in scenario.constraints:
                min_val = scenario.constraints[control_var]["min_val"]
                max_val = scenario.constraints[control_var]["max_val"]
                control_seq[:, i] = np.random.uniform(min_val, max_val, horizon)
            else:
                control_seq[:, i] = np.random.uniform(0, 100, horizon)

    elif sequence_type == "step":
        control_seq = np.zeros((horizon, num_controls))
        for i, control_var in enumerate(scenario.control_variables):
            if control_var in scenario.constraints:
                min_val = scenario.constraints[control_var]["min_val"]
                max_val = scenario.constraints[control_var]["max_val"]
                mid_val = (min_val + max_val) / 2
                control_seq[: horizon // 2, i] = min_val + (mid_val - min_val) * 0.3
                control_seq[horizon // 2 :, i] = min_val + (mid_val - min_val) * 0.7
            else:
                control_seq[: horizon // 2, i] = 30
                control_seq[horizon // 2 :, i] = 70

    elif sequence_type == "ramp":
        control_seq = np.zeros((horizon, num_controls))
        for i, control_var in enumerate(scenario.control_variables):
            if control_var in scenario.constraints:
                min_val = scenario.constraints[control_var]["min_val"]
                max_val = scenario.constraints[control_var]["max_val"]
                control_seq[:, i] = np.linspace(min_val * 1.1, max_val * 0.9, horizon)
            else:
                control_seq[:, i] = np.linspace(10, 90, horizon)

    elif sequence_type == "optimal_guess":
        # Simple heuristic for "good" control sequence
        control_seq = np.zeros((horizon, num_controls))
        for i, control_var in enumerate(scenario.control_variables):
            if control_var in scenario.constraints:
                min_val = scenario.constraints[control_var]["min_val"]
                max_val = scenario.constraints[control_var]["max_val"]
                # Use middle of range as guess
                mid_val = (min_val + max_val) / 2
                control_seq[:, i] = mid_val
            else:
                control_seq[:, i] = 50
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    return control_seq


def validate_pharmaceutical_constraints(
    control_plan: np.ndarray, scenario: PharmaceuticalScenario
) -> Dict[str, bool]:
    """
    Validate that control plan satisfies pharmaceutical constraints.

    Args:
        control_plan: Control sequence to validate
        scenario: PharmaceuticalScenario with constraints

    Returns:
        Dictionary mapping constraint names to satisfaction status
    """
    results = {}

    for i, control_var in enumerate(scenario.control_variables):
        if i >= control_plan.shape[1]:
            results[control_var] = False
            continue

        if control_var in scenario.constraints:
            min_val = scenario.constraints[control_var]["min_val"]
            max_val = scenario.constraints[control_var]["max_val"]

            values = control_plan[:, i]
            satisfies_min = np.all(values >= min_val)
            satisfies_max = np.all(values <= max_val)

            results[control_var] = satisfies_min and satisfies_max
        else:
            results[control_var] = True  # No constraints

    return results
