"""
Rolling data buffer for maintaining historical time series data in MPC applications.

This module provides thread-safe circular buffers for storing historical measurements
and control actions, essential for accurate model predictions in Model Predictive Control.
"""

import numpy as np
import threading
from typing import Optional, Tuple, Union
from collections import deque
import time


class DataBuffer:
    """Thread-safe rolling buffer for historical CMA and CPP data.
    
    Maintains fixed-size circular buffers for Critical Material Attributes (CMAs)
    and Critical Process Parameters (CPPs) to provide accurate historical context
    for model predictions in pharmaceutical process control.
    
    Features:
        - Thread-safe operations for real-time control applications
        - Circular buffer implementation for memory efficiency
        - Automatic timestamping and sequence validation
        - Configurable buffer sizes and data validation
        - Integration with scaling and preprocessing pipelines
    
    Args:
        cma_features (int): Number of CMA variables (e.g., d50, LOD)
        cpp_features (int): Number of CPP variables (e.g., spray_rate, air_flow, carousel_speed)
        buffer_size (int): Maximum number of historical samples to retain
        validate_sequence (bool): Whether to validate timestamp ordering
        
    Example:
        >>> buffer = DataBuffer(cma_features=2, cpp_features=3, buffer_size=50)
        >>> buffer.add_measurement(np.array([450.0, 1.8]), timestamp=time.time())
        >>> buffer.add_control_action(np.array([130.0, 550.0, 30.0]), timestamp=time.time())
        >>> past_cmas, past_cpps = buffer.get_model_inputs(lookback=20)
    """
    
    def __init__(self, cma_features: int, cpp_features: int, buffer_size: int, 
                 validate_sequence: bool = True):
        self.cma_features = cma_features
        self.cpp_features = cpp_features
        self.buffer_size = buffer_size
        self.validate_sequence = validate_sequence
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Circular buffers for time series data
        self._cma_buffer = deque(maxlen=buffer_size)
        self._cpp_buffer = deque(maxlen=buffer_size)
        self._timestamp_buffer = deque(maxlen=buffer_size)
        
        # State tracking
        self._sample_count = 0
        self._last_timestamp = None
        
        # Validation counters
        self._validation_errors = 0
        self._sequence_errors = 0
        
    def add_measurement(self, measurement: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """Add CMA measurement to the buffer.
        
        Args:
            measurement (np.ndarray): CMA values, shape (cma_features,)
            timestamp (float, optional): Unix timestamp. If None, uses current time.
            
        Returns:
            bool: True if measurement was added successfully, False if validation failed
            
        Raises:
            ValueError: If measurement shape is incorrect or contains invalid values
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Input validation
        if not isinstance(measurement, np.ndarray):
            raise ValueError(f"Measurement must be numpy array, got {type(measurement)}")
            
        if measurement.shape != (self.cma_features,):
            raise ValueError(f"Expected measurement shape ({self.cma_features},), got {measurement.shape}")
            
        if not np.all(np.isfinite(measurement)):
            self._validation_errors += 1
            raise ValueError("Measurement contains non-finite values (NaN/inf)")
            
        with self._lock:
            # Sequence validation
            if self.validate_sequence and self._last_timestamp is not None:
                if timestamp < self._last_timestamp:
                    self._sequence_errors += 1
                    raise ValueError(f"Timestamp {timestamp} is before last timestamp {self._last_timestamp}")
                    
            # Add to buffers
            self._cma_buffer.append(measurement.copy())
            self._timestamp_buffer.append(timestamp)
            self._last_timestamp = timestamp
            self._sample_count += 1
            
            return True
            
    def add_control_action(self, control_action: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """Add CPP control action to the buffer.
        
        Args:
            control_action (np.ndarray): CPP values, shape (cpp_features,)
            timestamp (float, optional): Unix timestamp. If None, uses current time.
            
        Returns:
            bool: True if control action was added successfully, False if validation failed
            
        Raises:
            ValueError: If control action shape is incorrect or contains invalid values
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Input validation
        if not isinstance(control_action, np.ndarray):
            raise ValueError(f"Control action must be numpy array, got {type(control_action)}")
            
        if control_action.shape != (self.cpp_features,):
            raise ValueError(f"Expected control action shape ({self.cpp_features},), got {control_action.shape}")
            
        if not np.all(np.isfinite(control_action)):
            self._validation_errors += 1
            raise ValueError("Control action contains non-finite values (NaN/inf)")
            
        with self._lock:
            # Sequence validation
            if self.validate_sequence and self._last_timestamp is not None:
                if timestamp < self._last_timestamp:
                    self._sequence_errors += 1
                    raise ValueError(f"Timestamp {timestamp} is before last timestamp {self._last_timestamp}")
                    
            # Add to buffer
            self._cpp_buffer.append(control_action.copy())
            
            # Update timestamp tracking (if this is the latest addition)
            if not self._timestamp_buffer or timestamp >= self._timestamp_buffer[-1]:
                self._last_timestamp = timestamp
                
            return True
            
    def get_model_inputs(self, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get historical data formatted for model input.
        
        Args:
            lookback (int): Number of historical samples to retrieve
            
        Returns:
            tuple: (past_cmas, past_cpps) both with shape (lookback, features)
            
        Raises:
            ValueError: If insufficient data available or invalid lookback value
        """
        if lookback <= 0:
            raise ValueError(f"Lookback must be positive, got {lookback}")
            
        with self._lock:
            available_samples = min(len(self._cma_buffer), len(self._cpp_buffer))
            
            if available_samples < lookback:
                raise ValueError(f"Insufficient data: need {lookback} samples, have {available_samples}")
                
            # Extract the most recent lookback samples
            cma_data = np.array(list(self._cma_buffer)[-lookback:])
            cpp_data = np.array(list(self._cpp_buffer)[-lookback:])
            
            return cma_data, cpp_data
            
    def is_ready(self, min_samples: int) -> bool:
        """Check if buffer has sufficient data for model predictions.
        
        Args:
            min_samples (int): Minimum number of samples required
            
        Returns:
            bool: True if buffer has at least min_samples in both CMA and CPP buffers
        """
        with self._lock:
            return (len(self._cma_buffer) >= min_samples and 
                   len(self._cpp_buffer) >= min_samples)
                   
    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Get the most recent measurement and control action.
        
        Returns:
            tuple: (latest_cma, latest_cpp, latest_timestamp) or (None, None, None) if empty
        """
        with self._lock:
            latest_cma = self._cma_buffer[-1].copy() if self._cma_buffer else None
            latest_cpp = self._cpp_buffer[-1].copy() if self._cpp_buffer else None
            latest_timestamp = self._timestamp_buffer[-1] if self._timestamp_buffer else None
            
            return latest_cma, latest_cpp, latest_timestamp
            
    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            self._cma_buffer.clear()
            self._cpp_buffer.clear()
            self._timestamp_buffer.clear()
            self._sample_count = 0
            self._last_timestamp = None
            self._validation_errors = 0
            self._sequence_errors = 0
            
    def get_statistics(self) -> dict:
        """Get buffer statistics and health metrics.
        
        Returns:
            dict: Buffer statistics including size, errors, and data quality metrics
        """
        with self._lock:
            return {
                'buffer_size': self.buffer_size,
                'cma_samples': len(self._cma_buffer),
                'cpp_samples': len(self._cpp_buffer),
                'timestamp_samples': len(self._timestamp_buffer),
                'total_samples_added': self._sample_count,
                'validation_errors': self._validation_errors,
                'sequence_errors': self._sequence_errors,
                'last_timestamp': self._last_timestamp,
                'is_full': len(self._cma_buffer) == self.buffer_size,
                'utilization_percent': 100.0 * len(self._cma_buffer) / self.buffer_size
            }
            
    def __len__(self) -> int:
        """Return the number of complete sample pairs in the buffer."""
        with self._lock:
            return min(len(self._cma_buffer), len(self._cpp_buffer))
            
    def __repr__(self) -> str:
        """String representation of the buffer state."""
        stats = self.get_statistics()
        return (f"DataBuffer(cma_features={self.cma_features}, "
                f"cpp_features={self.cpp_features}, "
                f"samples={len(self)}/{self.buffer_size}, "
                f"utilization={stats['utilization_percent']:.1f}%)")


class StartupHistoryGenerator:
    """Generate safe startup history during initial MPC operation.
    
    Provides realistic historical data during the initial lookback period
    when insufficient real data is available for model predictions.
    """
    
    def __init__(self, cma_features: int, cpp_features: int, 
                 initial_cma_state: np.ndarray, initial_cpp_state: np.ndarray):
        self.cma_features = cma_features
        self.cpp_features = cpp_features
        self.initial_cma_state = initial_cma_state.copy()
        self.initial_cpp_state = initial_cpp_state.copy()
        
    def generate_startup_history(self, lookback: int, noise_scale: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic startup history around initial conditions.
        
        Args:
            lookback (int): Number of historical samples to generate
            noise_scale (float): Standard deviation of process noise (relative to state)
            
        Returns:
            tuple: (past_cmas, past_cpps) with gradual convergence to initial state
        """
        # Generate gradual convergence to initial state
        past_cmas = np.zeros((lookback, self.cma_features))
        past_cpps = np.zeros((lookback, self.cpp_features))
        
        for i in range(lookback):
            # Exponential convergence to initial state
            convergence_factor = np.exp(-3.0 * (lookback - i) / lookback)
            
            # Add realistic process variations
            cma_noise = np.random.normal(0, noise_scale, self.cma_features)
            cpp_noise = np.random.normal(0, noise_scale, self.cpp_features)
            
            past_cmas[i] = self.initial_cma_state * (0.8 + 0.2 * convergence_factor) + cma_noise
            past_cpps[i] = self.initial_cpp_state * (0.9 + 0.1 * convergence_factor) + cpp_noise
            
        return past_cmas, past_cpps