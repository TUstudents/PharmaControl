#!/usr/bin/env python3
"""
V1 Controller Interface Adapter

This adapter allows V1 MPC controller to be used with the same interface as V2,
while maintaining the proper historical data requirements and DataFrame formats
that V1 expects.

The V1 controller expects:
- past_cmas_unscaled: DataFrame with lookback_steps of CMA history
- past_cpps_unscaled: DataFrame with lookback_steps of CPP history (including soft sensors)
- target_unscaled: ndarray tiled over prediction horizon

This adapter converts from V2-style single-point calls to V1's historical DataFrame interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class V1ControllerAdapter:
    """
    Interface adapter for V1 MPC Controller.
    
    Converts V2-style interface calls to V1's DataFrame-based historical interface.
    Maintains rolling history buffer and calculates required soft sensors.
    """
    
    def __init__(self, v1_controller, lookback_steps: int = 36, horizon: int = 72):
        """
        Initialize V1 controller adapter.
        
        Args:
            v1_controller: V1 MPCController instance
            lookback_steps: Historical window size (default 36 from V1 config)
            horizon: Prediction horizon (default 72 from V1 config)
        """
        self.v1_controller = v1_controller
        self.lookback_steps = lookback_steps
        self.horizon = horizon
        self.history_buffer: List[Dict] = []
        self.cpp_names = ['spray_rate', 'air_flow', 'carousel_speed']
        self.cma_names = ['d50', 'lod']
        self.soft_sensor_names = ['specific_energy', 'froude_number_proxy']
        self.cpp_full_names = self.cpp_names + self.soft_sensor_names
        
        print(f"V1ControllerAdapter initialized:")
        print(f"  Lookback steps: {self.lookback_steps}")
        print(f"  Prediction horizon: {self.horizon}")
        print(f"  CMA names: {self.cma_names}")
        print(f"  CPP names: {self.cpp_names}")
        print(f"  Soft sensors: {self.soft_sensor_names}")
    
    def _calculate_soft_sensors(self, cpps: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate soft sensor values from CPPs.
        
        Based on V1 implementation from notebook 05:
        - specific_energy = (spray_rate * carousel_speed) / 1000.0
        - froude_number_proxy = (carousel_speed^2) / 9.81
        
        Args:
            cpps: Dictionary of CPP values
            
        Returns:
            Dictionary of soft sensor values
        """
        soft_sensors = {}
        
        try:
            # Calculate specific energy
            soft_sensors['specific_energy'] = (
                cpps['spray_rate'] * cpps['carousel_speed']
            ) / 1000.0
            
            # Calculate Froude number proxy
            soft_sensors['froude_number_proxy'] = (
                cpps['carousel_speed'] ** 2
            ) / 9.81
            
        except KeyError as e:
            raise ValueError(f"Missing required CPP for soft sensor calculation: {e}")
        except Exception as e:
            raise ValueError(f"Soft sensor calculation failed: {e}")
        
        return soft_sensors
    
    def add_history_step(self, cmas: Dict[str, float], cpps: Dict[str, float]) -> None:
        """
        Add a time step to the rolling history buffer.
        
        Args:
            cmas: Dictionary of CMA values (d50, lod)
            cpps: Dictionary of CPP values (spray_rate, air_flow, carousel_speed)
        """
        try:
            # Validate input data
            for cma_name in self.cma_names:
                if cma_name not in cmas:
                    raise ValueError(f"Missing CMA: {cma_name}")
                if not isinstance(cmas[cma_name], (int, float)):
                    raise ValueError(f"CMA {cma_name} must be numeric, got {type(cmas[cma_name])}")
            
            for cpp_name in self.cpp_names:
                if cpp_name not in cpps:
                    raise ValueError(f"Missing CPP: {cpp_name}")
                if not isinstance(cpps[cpp_name], (int, float)):
                    raise ValueError(f"CPP {cpp_name} must be numeric, got {type(cpps[cpp_name])}")
            
            # Calculate soft sensors
            soft_sensors = self._calculate_soft_sensors(cpps)
            
            # Create complete step data
            step_data = {
                **cmas,
                **cpps,
                **soft_sensors
            }
            
            # Add to history buffer
            self.history_buffer.append(step_data)
            
            # Maintain rolling window
            if len(self.history_buffer) > self.lookback_steps:
                self.history_buffer.pop(0)
                
        except Exception as e:
            raise RuntimeError(f"Failed to add history step: {e}")
    
    def is_ready(self) -> bool:
        """
        Check if adapter has sufficient history for V1 controller.
        
        Returns:
            True if sufficient history available, False otherwise
        """
        return len(self.history_buffer) >= self.lookback_steps
    
    def get_history_status(self) -> Dict[str, any]:
        """
        Get current history buffer status.
        
        Returns:
            Dictionary with buffer status information
        """
        return {
            'buffer_size': len(self.history_buffer),
            'required_size': self.lookback_steps,
            'is_ready': self.is_ready(),
            'fill_percentage': (len(self.history_buffer) / self.lookback_steps) * 100
        }
    
    def _build_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build DataFrames from history buffer for V1 controller.
        
        Returns:
            Tuple of (past_cmas_df, past_cpps_df)
        """
        if not self.is_ready():
            raise RuntimeError(f"Insufficient history: {len(self.history_buffer)}/{self.lookback_steps}")
        
        # Get recent history
        recent_history = self.history_buffer[-self.lookback_steps:]
        
        # Convert to DataFrame
        history_df = pd.DataFrame(recent_history)
        
        # Validate all required columns are present
        missing_cmas = [col for col in self.cma_names if col not in history_df.columns]
        missing_cpps = [col for col in self.cpp_full_names if col not in history_df.columns]
        
        if missing_cmas:
            raise ValueError(f"Missing CMA columns in history: {missing_cmas}")
        if missing_cpps:
            raise ValueError(f"Missing CPP columns in history: {missing_cpps}")
        
        # Extract required DataFrames
        past_cmas_df = history_df[self.cma_names].copy()
        past_cpps_df = history_df[self.cpp_full_names].copy()
        
        # Validate DataFrame shapes
        if len(past_cmas_df) != self.lookback_steps:
            raise ValueError(f"CMA DataFrame wrong size: {len(past_cmas_df)} != {self.lookback_steps}")
        if len(past_cpps_df) != self.lookback_steps:
            raise ValueError(f"CPP DataFrame wrong size: {len(past_cpps_df)} != {self.lookback_steps}")
        
        return past_cmas_df, past_cpps_df
    
    def suggest_action(self, current_cmas: Dict[str, float], current_cpps: Dict[str, float], 
                      setpoint: np.ndarray) -> np.ndarray:
        """
        Interface adapter method that converts V2-style calls to V1 format.
        
        Args:
            current_cmas: Current CMA measurements
            current_cpps: Current CPP control values
            setpoint: Target setpoint array [d50_target, lod_target]
            
        Returns:
            Control action array [spray_rate, air_flow, carousel_speed]
        """
        try:
            # Add current step to history buffer
            self.add_history_step(current_cmas, current_cpps)
            
            # Check if we have sufficient history
            if not self.is_ready():
                print(f"V1 adapter not ready: {len(self.history_buffer)}/{self.lookback_steps} steps")
                # Return current control as fallback
                return np.array([
                    current_cpps['spray_rate'],
                    current_cpps['air_flow'], 
                    current_cpps['carousel_speed']
                ])
            
            # Build DataFrames for V1 controller
            past_cmas_df, past_cpps_df = self._build_dataframes()
            
            # Validate setpoint format
            if not isinstance(setpoint, np.ndarray) or setpoint.shape != (2,):
                raise ValueError(f"Setpoint must be ndarray shape (2,), got {type(setpoint)} shape {getattr(setpoint, 'shape', 'N/A')}")
            
            # Tile setpoint over prediction horizon (V1 requirement)
            target_tiled = np.tile(setpoint, (self.horizon, 1))
            
            # Call V1 controller with proper interface
            action = self.v1_controller.suggest_action(
                past_cmas_df, 
                past_cpps_df, 
                target_tiled
            )
            
            # Validate action format
            if not isinstance(action, np.ndarray) or len(action) != 3:
                raise ValueError(f"V1 controller returned invalid action: {type(action)} length {len(action) if hasattr(action, '__len__') else 'N/A'}")
            
            return action
            
        except Exception as e:
            print(f"V1 adapter suggest_action failed: {e}")
            # Return current control as safe fallback
            return np.array([
                current_cpps['spray_rate'],
                current_cpps['air_flow'],
                current_cpps['carousel_speed']
            ])


def create_v1_adapter(v1_controller, config: Optional[Dict] = None) -> V1ControllerAdapter:
    """
    Factory function to create V1 controller adapter with proper configuration.
    
    Args:
        v1_controller: V1 MPCController instance
        config: Optional configuration dictionary
        
    Returns:
        Configured V1ControllerAdapter instance
    """
    # Default configuration from V1 notebook 05
    default_config = {
        'lookback_steps': 36,   # LOOKBACK from V1 config
        'horizon': 72           # HORIZON from V1 config
    }
    
    if config:
        default_config.update(config)
    
    return V1ControllerAdapter(
        v1_controller,
        lookback_steps=default_config['lookback_steps'],
        horizon=default_config['horizon']
    )


def validate_v1_adapter(adapter: V1ControllerAdapter, test_data: Dict) -> List[str]:
    """
    Validate V1 adapter functionality with test data.
    
    Args:
        adapter: V1ControllerAdapter instance to test
        test_data: Test data dictionary with 'cmas', 'cpps', 'setpoint'
        
    Returns:
        List of error messages (empty if all tests pass)
    """
    errors = []
    
    try:
        # Test history addition
        adapter.add_history_step(test_data['cmas'], test_data['cpps'])
        
        # Test ready status
        status = adapter.get_history_status()
        if 'buffer_size' not in status:
            errors.append("History status missing buffer_size")
        
        # Test action suggestion (may fail if not enough history)
        try:
            action = adapter.suggest_action(
                test_data['cmas'],
                test_data['cpps'], 
                test_data['setpoint']
            )
            if not isinstance(action, np.ndarray) or len(action) != 3:
                errors.append(f"Invalid action format: {type(action)} length {len(action) if hasattr(action, '__len__') else 'N/A'}")
        except Exception as e:
            # This may be expected if not enough history
            if adapter.is_ready():
                errors.append(f"Action suggestion failed when adapter ready: {e}")
        
    except Exception as e:
        errors.append(f"Validation test failed: {e}")
    
    return errors


if __name__ == "__main__":
    # Basic test/demo
    print("V1 Controller Adapter - Basic Test")
    print("=" * 40)
    
    # Mock V1 controller for testing
    class MockV1Controller:
        def suggest_action(self, past_cmas_df, past_cpps_df, target_tiled):
            return np.array([100.0, 500.0, 25.0])
    
    mock_controller = MockV1Controller()
    adapter = create_v1_adapter(mock_controller)
    
    # Test data
    test_cmas = {'d50': 400.0, 'lod': 1.5}
    test_cpps = {'spray_rate': 120.0, 'air_flow': 550.0, 'carousel_speed': 30.0}
    test_setpoint = np.array([450.0, 1.3])
    
    # Run validation
    test_data = {'cmas': test_cmas, 'cpps': test_cpps, 'setpoint': test_setpoint}
    errors = validate_v1_adapter(adapter, test_data)
    
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Basic validation passed")
    
    # Show status
    status = adapter.get_history_status()
    print(f"\nAdapter Status: {status}")