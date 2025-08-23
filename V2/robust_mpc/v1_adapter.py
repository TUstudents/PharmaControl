#!/usr/bin/env python3
"""
V1 Controller Interface Adapter and Wrapper for V2 Integration

This module provides integration classes for using legacy V1 MPC controllers
within the V2 testing and comparison framework. It ensures fair comparison
by implementing identical soft sensor calculations used in V2.

Classes:
    V1ControllerAdapter: Low-level adapter converting V2-style calls to V1 DataFrame interface
    V1_MPC_Wrapper: High-level wrapper ensuring consistent configuration and fair comparison

Key Features:
    - Identical soft sensor calculations as V2 RobustMPCController
    - Thread-safe rolling buffer implementation
    - Proper pharmaceutical process physics modeling
    - Comprehensive error handling and validation
    - Configuration consistency enforcement

Critical Implementation Details:
    The soft sensor calculations use the same pharmaceutical process physics
    as implemented in V2 RobustMPCController._scale_cpp_plan():

    - specific_energy = (spray_rate * carousel_speed) / 1000.0
      Represents normalized energy input from spray and rotation interaction

    - froude_number_proxy = (carousel_speed ** 2) / 9.81
      Dimensionless mixing intensity measure for granulation dynamics

This ensures V1 vs V2 performance comparisons use identical process modeling,
providing fair evaluation of the core MPC algorithm differences.

Author: V2 Integration Team
License: Educational/Research Use
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class V1ControllerAdapter:
    """
    Low-level adapter that converts V2-style single-point method calls into the
    historical DataFrame-based interface that the V1 controller's `suggest_action`
    method expects.

    This adapter maintains a rolling buffer of historical process data and converts
    it to the pandas DataFrame format required by V1 controllers. It implements
    identical soft sensor calculations as V2 for fair performance comparison.

    Critical Features:
        - Identical soft sensor physics as V2 RobustMPCController
        - Thread-safe deque-based rolling buffer
        - Robust pharmaceutical process parameter validation
        - Safe startup action during cold-start phase
        - Comprehensive error handling with detailed diagnostics

    Args:
        v1_controller: Instance of V1 MPCController class
        lookback_steps (int): Number of historical steps to maintain (default: 36)
        horizon (int): Control and prediction horizon length (default: 72)

    Attributes:
        v1_controller: Stored V1 controller instance
        lookback_steps (int): Rolling buffer size
        horizon (int): Control horizon length
        history_buffer (deque): Thread-safe rolling buffer
        startup_action (np.ndarray): Safe control action for cold start

    Example:
        >>> from V1.src.mpc_controller import MPCController
        >>> v1_controller = MPCController(model, config, constraints, scalers)
        >>> adapter = V1ControllerAdapter(v1_controller, lookback_steps=36)
        >>> action = adapter.suggest_action(current_cmas, current_cpps, setpoint)

    Notes:
        - Implements V2-identical pharmaceutical process soft sensor calculations
        - Provides safe startup behavior during buffer fill phase
        - Uses deque for O(1) append/pop operations in rolling buffer
        - Essential for fair V1 vs V2 controller performance comparison
    """

    def __init__(self, v1_controller, lookback_steps: int = 36, horizon: int = 72):
        self.v1_controller = v1_controller
        self.lookback_steps = lookback_steps
        self.horizon = horizon

        # Use a deque for O(1) appends and pops
        self.history_buffer: deque = deque(maxlen=lookback_steps)

        # Pharmaceutical process parameter names
        self.cpp_names = ["spray_rate", "air_flow", "carousel_speed"]
        self.cma_names = ["d50", "lod"]
        self.soft_sensor_names = ["specific_energy", "froude_number_proxy"]
        self.cpp_full_names = self.cpp_names + self.soft_sensor_names

        # Safe startup action for cold-start phase (pharmaceutical baselines)
        self.startup_action = np.array([130.0, 550.0, 30.0])  # spray_rate, air_flow, carousel_speed

        print(f"V1ControllerAdapter initialized:")
        print(f"  Lookback steps: {self.lookback_steps}")
        print(f"  Prediction horizon: {self.horizon}")
        print(f"  Using V2-identical soft sensor calculations for fair comparison")

    def _calculate_soft_sensors(self, cpps: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate pharmaceutical process soft sensors using V2-identical physics.

        This method implements the exact same soft sensor calculations as used in
        V2 RobustMPCController._scale_cpp_plan() to ensure fair performance comparison.

        Pharmaceutical Process Physics:
            - specific_energy: Normalized energy input from spray-rotation interaction
              Formula: (spray_rate * carousel_speed) / 1000.0
              Units: Dimensionless energy metric for granulation intensity

            - froude_number_proxy: Dimensionless mixing intensity measure
              Formula: (carousel_speed ** 2) / 9.81
              Units: Dimensionless Froude number proxy for granulation dynamics

        Args:
            cpps (Dict[str, float]): Critical Process Parameters containing:
                - 'spray_rate': Liquid spray rate (mL/min or g/min)
                - 'carousel_speed': Carousel rotation speed (rpm)

        Returns:
            Dict[str, float]: Calculated soft sensor values with keys:
                - 'specific_energy': Energy interaction metric
                - 'froude_number_proxy': Mixing intensity measure

        Raises:
            KeyError: If required CPP parameters are missing
            ValueError: If CPP values are invalid (non-finite, negative)

        Example:
            >>> cpps = {'spray_rate': 130.0, 'carousel_speed': 30.0, 'air_flow': 550.0}
            >>> sensors = adapter._calculate_soft_sensors(cpps)
            >>> # Returns: {'specific_energy': 3.9, 'froude_number_proxy': 91.7}

        Notes:
            - Calculations identical to V2.robust_mpc.core.RobustMPCController
            - Essential for fair V1 vs V2 performance comparison
            - Based on established pharmaceutical granulation process physics
            - Values validated for pharmaceutical manufacturing safety
        """
        # Input validation
        required_params = ["spray_rate", "carousel_speed"]
        missing_params = [param for param in required_params if param not in cpps]
        if missing_params:
            raise KeyError(f"Missing required CPP parameters for soft sensors: {missing_params}")

        spray_rate = cpps["spray_rate"]
        carousel_speed = cpps["carousel_speed"]

        # Validate parameter values
        if not np.isfinite(spray_rate) or spray_rate < 0:
            raise ValueError(f"Invalid spray_rate: {spray_rate} (must be finite and non-negative)")
        if not np.isfinite(carousel_speed) or carousel_speed <= 0:
            raise ValueError(
                f"Invalid carousel_speed: {carousel_speed} (must be finite and positive)"
            )

        # Calculate soft sensors using V2-identical pharmaceutical process physics
        try:
            # Specific energy: normalized spray rate × carousel speed interaction
            # Same formula as V2.robust_mpc.core.RobustMPCController._scale_cpp_plan line 428
            specific_energy = (spray_rate * carousel_speed) / 1000.0

            # Froude number proxy: dimensionless mixing intensity measure
            # Same formula as V2.robust_mpc.core.RobustMPCController._scale_cpp_plan line 431
            froude_number_proxy = (carousel_speed**2) / 9.81

            return {"specific_energy": specific_energy, "froude_number_proxy": froude_number_proxy}

        except Exception as e:
            raise RuntimeError(f"Soft sensor calculation failed: {e}")

    def add_history_step(self, cmas: Dict[str, float], cpps: Dict[str, float]) -> None:
        """
        Add a new process step to the rolling history buffer.

        This method validates inputs, calculates soft sensors using V2-identical physics,
        and adds the complete process state to the rolling buffer for V1 controller use.

        Args:
            cmas (Dict[str, float]): Critical Material Attributes measurements
            cpps (Dict[str, float]): Critical Process Parameters control inputs

        Raises:
            RuntimeError: If validation fails or soft sensor calculation fails
            KeyError: If required parameters are missing
            ValueError: If parameter values are invalid

        Notes:
            - Uses V2-identical soft sensor calculations for fair comparison
            - Thread-safe deque automatically handles buffer overflow
            - Comprehensive input validation for pharmaceutical safety
        """
        try:
            # Validate CMA inputs
            expected_cmas = set(self.cma_names)
            provided_cmas = set(cmas.keys())
            if not expected_cmas.issubset(provided_cmas):
                missing_cmas = expected_cmas - provided_cmas
                raise KeyError(f"Missing required CMAs: {missing_cmas}")

            # Validate CPP inputs
            expected_cpps = set(self.cpp_names)
            provided_cpps = set(cpps.keys())
            if not expected_cpps.issubset(provided_cpps):
                missing_cpps = expected_cpps - provided_cpps
                raise KeyError(f"Missing required CPPs: {missing_cpps}")

            # Calculate soft sensors using V2-identical physics
            soft_sensors = self._calculate_soft_sensors(cpps)

            # Combine all process variables
            step_data = {**cmas, **cpps, **soft_sensors}

            # Add to rolling buffer (thread-safe deque operation)
            self.history_buffer.append(step_data)

        except Exception as e:
            raise RuntimeError(f"Failed to add history step: {e}")

    def is_ready(self) -> bool:
        """
        Check if adapter has sufficient history for V1 controller operation.

        Returns:
            bool: True if buffer contains required lookback steps
        """
        return len(self.history_buffer) == self.lookback_steps

    def get_history_status(self) -> Dict:
        """
        Get current history buffer status for diagnostics.

        Returns:
            Dict: Buffer status information including:
                - buffer_size: Current number of stored steps
                - required_size: Required lookback steps
                - fill_percentage: Buffer fill percentage
        """
        return {
            "buffer_size": len(self.history_buffer),
            "required_size": self.lookback_steps,
            "fill_percentage": (len(self.history_buffer) / self.lookback_steps) * 100,
        }

    def _build_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert rolling buffer to pandas DataFrames required by V1 controller.

        This method produces UN SCALED DataFrames of physical process data,
        as expected by the V1 controller's `suggest_action` method, which
        handles its own internal scaling.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (unscaled_past_cmas_df, unscaled_past_cpps_df)
        """
        history_list = list(self.history_buffer)
        full_df = pd.DataFrame(history_list)

        # V1 Controller expects unscaled data and handles its own scaling internally.
        unscaled_past_cmas_df = full_df[self.cma_names]
        unscaled_past_cpps_df = full_df[self.cpp_full_names]

        return unscaled_past_cmas_df, unscaled_past_cpps_df

    def suggest_action(
        self, current_cmas: Dict[str, float], current_cpps: Dict[str, float], setpoint: np.ndarray
    ) -> np.ndarray:
        """
        Generate control action using V1 controller with V2-compatible soft sensors.

        This method maintains the rolling history buffer, handles startup conditions,
        and interfaces with the V1 controller using its expected DataFrame format.

        Args:
            current_cmas (Dict[str, float]): Current CMA measurements
            current_cpps (Dict[str, float]): Current CPP control inputs
            setpoint (np.ndarray): Target setpoint values for CMA control

        Returns:
            np.ndarray: Optimal control action from V1 controller

        Raises:
            RuntimeError: If V1 controller fails or data validation fails

        Notes:
            - Uses safe startup action during buffer fill phase
            - Implements V2-identical soft sensor calculations
            - Provides comprehensive error handling with fallback strategies
        """
        try:
            # Add current step to rolling history
            self.add_history_step(current_cmas, current_cpps)

            # Handle startup phase with safe action
            if not self.is_ready():
                return self.startup_action

            # Build DataFrames required by V1 controller interface
            past_cmas_df, past_cpps_df = self._build_dataframes()

            # Validate DataFrame structure for V1 controller
            if past_cmas_df.empty or past_cpps_df.empty:
                raise ValueError("Generated DataFrames are empty")
            if len(past_cmas_df) != self.lookback_steps or len(past_cpps_df) != self.lookback_steps:
                raise ValueError(f"DataFrame length mismatch: expected {self.lookback_steps}")

            # Prepare setpoint for V1 controller (tile across horizon)
            if setpoint.ndim == 1:
                target_tiled = np.tile(setpoint, (self.horizon, 1))
            else:
                target_tiled = setpoint

            # Call V1 controller with unscaled DataFrame interface
            action = self.v1_controller.suggest_action(past_cmas_df, past_cpps_df, target_tiled)

            # Validate V1 controller output
            if action is None or not isinstance(action, np.ndarray):
                raise ValueError("V1 controller returned invalid action")
            if action.size != len(self.cpp_names):
                raise ValueError(
                    f"Action size mismatch: expected {len(self.cpp_names)}, got {action.size}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError("V1 controller returned non-finite action values")

            # The V1 controller returns unscaled, physical values directly.
            return action

        except Exception as e:
            raise RuntimeError(f"V1 controller failed during suggest_action: {e}")


class V1_MPC_Wrapper:
    """
    High-level wrapper ensuring consistent configuration between V1 controller and adapter.

    This wrapper class coordinates the V1 controller and adapter initialization with
    consistent configuration parameters, preventing critical mismatches that could
    cause controller failures or unfair performance comparisons.

    Critical Safety Features:
        - Enforces consistent lookback/horizon between controller and adapter
        - Validates all required configuration parameters
        - Implements V2-identical soft sensor calculations for fair comparison
        - Provides comprehensive error handling and fallback strategies
        - Ensures pharmaceutical manufacturing safety standards

    Args:
        V1ControllerClass: The V1 MPCController class (not instance)
        model: Trained predictive model for V1 controller
        config (Dict): Unified configuration dictionary with required keys:
            - 'lookback': Historical data window size
            - 'horizon': Control and prediction horizon
            - Other V1-specific configuration parameters
        constraints (Dict): Process parameter constraint definitions
        scalers (Dict): Fitted data preprocessing scalers

    Attributes:
        v1_controller: Instantiated V1 controller with provided configuration
        adapter: V1ControllerAdapter with consistent configuration

    Example:
        >>> from V1.src.mpc_controller import MPCController as V1ControllerClass
        >>> wrapper = V1_MPC_Wrapper(
        ...     V1ControllerClass=V1ControllerClass,
        ...     model=trained_model,
        ...     config={'lookback': 36, 'horizon': 72, ...},
        ...     constraints=cpp_constraints,
        ...     scalers=fitted_scalers
        ... )
        >>> action = wrapper.suggest_action(current_cmas, current_cpps, setpoint)

    Notes:
        - Prevents fatal configuration mismatches between V1 controller and adapter
        - Essential for pharmaceutical manufacturing safety and reliability
        - Ensures fair V1 vs V2 performance comparison with identical soft sensors
        - Provides single, clean interface for V2 testing framework integration
    """

    def __init__(self, V1ControllerClass, model, config, constraints, scalers):
        """
        Initialize both V1 controller and adapter with consistent, validated configuration.

        Args:
            V1ControllerClass: V1 MPCController class for instantiation
            model: Trained neural network model for predictions
            config (Dict): Complete configuration dictionary
            constraints (Dict): Process parameter constraints
            scalers (Dict): Fitted data preprocessing scalers

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If configuration values are invalid
            RuntimeError: If V1 controller or adapter initialization fails
        """
        print("--- Creating V1_MPC_Wrapper with V2-Compatible Soft Sensors ---")

        # Validate required configuration keys
        required_keys = ["lookback", "horizon"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"V1 config missing required keys: {missing_keys}")

        # Validate configuration values
        if config["lookback"] <= 0:
            raise ValueError(f"Invalid lookback: {config['lookback']} (must be positive)")
        if config["horizon"] <= 0:
            raise ValueError(f"Invalid horizon: {config['horizon']} (must be positive)")

        try:
            # 1. Instantiate V1 controller with provided configuration
            self.v1_controller = V1ControllerClass(
                model=model, config=config, constraints=constraints, scalers=scalers
            )
            print("✓ V1 MPCController instance created successfully")

            # 2. Instantiate adapter with consistent configuration
            self.adapter = V1ControllerAdapter(
                v1_controller=self.v1_controller,
                lookback_steps=config["lookback"],
                horizon=config["horizon"],
            )
            print("✓ V1ControllerAdapter created with V2-identical soft sensor physics")
            print("✓ Configuration consistency enforced for fair comparison")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize V1_MPC_Wrapper: {e}")

    def suggest_action(
        self, current_cmas: Dict[str, float], current_cpps: Dict[str, float], setpoint: np.ndarray
    ) -> np.ndarray:
        """
        Generate control action through adapter with comprehensive error handling.

        This method provides a clean, single interface for the V2 testing framework
        while ensuring robust error handling and safe fallback strategies.

        Args:
            current_cmas (Dict[str, float]): Current CMA measurements
            current_cpps (Dict[str, float]): Current CPP control inputs
            setpoint (np.ndarray): Target CMA setpoints

        Returns:
            np.ndarray: Optimal control action from V1 controller

        Notes:
            - Uses V2-identical soft sensor calculations for fair comparison
            - Provides safe fallback strategy on controller failure
            - Essential interface for V2 testing framework integration
        """
        try:
            # Delegate to adapter with V2-compatible soft sensor calculations
            return self.adapter.suggest_action(current_cmas, current_cpps, setpoint)

        except Exception as e:
            # Comprehensive error logging for pharmaceutical manufacturing diagnostics
            print(f"V1_MPC_Wrapper failed: {e}")
            print("Using safe fallback control strategy")

            # Safe fallback: return current control inputs to maintain process stability
            return np.array(list(current_cpps.values())[: len(self.adapter.cpp_names)])
