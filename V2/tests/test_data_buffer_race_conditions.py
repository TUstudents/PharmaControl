"""
Test suite for DataBuffer race condition vulnerability fixes.

This module validates the critical atomic sample operations that prevent
race condition data misalignment in pharmaceutical manufacturing environments.
"""

import pytest
import numpy as np
import threading
import time
import warnings
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from V2.robust_mpc.data_buffer import DataBuffer


class TestDataBufferRaceConditions:
    """Test suite for DataBuffer race condition fixes."""
    
    @pytest.fixture
    def test_buffer(self):
        """Create test DataBuffer for race condition testing."""
        return DataBuffer(cma_features=2, cpp_features=3, buffer_size=50)
    
    def test_atomic_add_sample_basic(self, test_buffer):
        """Test basic atomic add_sample functionality."""
        measurement = np.array([450.0, 1.8])
        control_action = np.array([130.0, 550.0, 30.0])
        timestamp = time.time()
        
        result = test_buffer.add_sample(measurement, control_action, timestamp)
        
        assert result is True
        assert len(test_buffer) == 1
        assert len(test_buffer._cma_buffer) == 1
        assert len(test_buffer._cpp_buffer) == 1
        assert len(test_buffer._timestamp_buffer) == 1
        
        # Verify data integrity
        retrieved_cmas, retrieved_cpps = test_buffer.get_model_inputs(1)
        np.testing.assert_array_equal(retrieved_cmas[0], measurement)
        np.testing.assert_array_equal(retrieved_cpps[0], control_action)
    
    def test_atomic_add_sample_validation(self, test_buffer):
        """Test atomic add_sample input validation."""
        valid_measurement = np.array([450.0, 1.8])
        valid_control = np.array([130.0, 550.0, 30.0])
        
        # Test invalid measurement types
        with pytest.raises(ValueError, match="Measurement must be numpy array"):
            test_buffer.add_sample([450.0, 1.8], valid_control)
            
        # Test invalid measurement shapes
        with pytest.raises(ValueError, match="Expected measurement shape"):
            test_buffer.add_sample(np.array([450.0]), valid_control)
            
        # Test invalid measurement values
        with pytest.raises(ValueError, match="Measurement contains non-finite values"):
            test_buffer.add_sample(np.array([np.nan, 1.8]), valid_control)
            
        # Test invalid control action types
        with pytest.raises(ValueError, match="Control action must be numpy array"):
            test_buffer.add_sample(valid_measurement, [130.0, 550.0, 30.0])
            
        # Test invalid control action shapes
        with pytest.raises(ValueError, match="Expected control action shape"):
            test_buffer.add_sample(valid_measurement, np.array([130.0, 550.0]))
            
        # Test invalid control action values
        with pytest.raises(ValueError, match="Control action contains non-finite values"):
            test_buffer.add_sample(valid_measurement, np.array([130.0, np.inf, 30.0]))
    
    def test_atomic_add_sample_rollback_safety(self, test_buffer):
        """Test that validation failures don't partially modify buffers."""
        valid_measurement = np.array([450.0, 1.8])
        invalid_control = np.array([130.0, np.nan, 30.0])  # Contains NaN
        
        # Buffer should be empty initially
        assert len(test_buffer) == 0
        
        # Attempt to add invalid sample
        with pytest.raises(ValueError, match="Control action contains non-finite values"):
            test_buffer.add_sample(valid_measurement, invalid_control)
        
        # Verify no partial buffer modifications occurred (rollback safety)
        assert len(test_buffer) == 0
        assert len(test_buffer._cma_buffer) == 0
        assert len(test_buffer._cpp_buffer) == 0
        assert len(test_buffer._timestamp_buffer) == 0
    
    def test_deprecated_methods_warnings(self, test_buffer):
        """Test that deprecated methods emit appropriate warnings."""
        measurement = np.array([450.0, 1.8])
        control_action = np.array([130.0, 550.0, 30.0])
        
        # Test add_measurement deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_buffer.add_measurement(measurement)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "add_measurement() is deprecated due to race condition vulnerability" in str(w[0].message)
            assert "add_sample" in str(w[0].message)
        
        # Test add_control_action deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_buffer.add_control_action(control_action)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "add_control_action() is deprecated due to race condition vulnerability" in str(w[0].message)
            assert "add_sample" in str(w[0].message)
    
    def test_buffer_misalignment_detection(self, test_buffer):
        """Test detection of buffer misalignment (race condition aftermath)."""
        measurement = np.array([450.0, 1.8])
        control_action = np.array([130.0, 550.0, 30.0])
        
        # Deliberately create misalignment using deprecated methods (simulating race condition)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            test_buffer.add_measurement(measurement)
            test_buffer.add_measurement(measurement)  # Add second measurement without control
            test_buffer.add_control_action(control_action)  # Now buffers are misaligned: 2 vs 1
        
        # Verify misalignment exists
        assert len(test_buffer._cma_buffer) == 2
        assert len(test_buffer._cpp_buffer) == 1
        
        # Test that get_model_inputs detects and warns about misalignment
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cmas, cpps = test_buffer.get_model_inputs(1)
            
            # Should emit critical buffer misalignment warning
            misalignment_warnings = [warning for warning in w 
                                   if "CRITICAL: Buffer misalignment detected" in str(warning.message)]
            assert len(misalignment_warnings) >= 1
            assert "race condition occurred" in str(misalignment_warnings[0].message)
    
    def test_simulated_race_condition_scenario(self, test_buffer):
        """Test the exact race condition scenario described in the bug report."""
        measurement1 = np.array([450.0, 1.8])
        measurement2 = np.array([460.0, 1.9])
        control_action = np.array([130.0, 550.0, 30.0])
        
        # Simulate the race condition scenario from the bug report:
        # Thread A: add_measurement(m1), add_measurement(m2)
        # Thread B: add_control_action(c3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            test_buffer.add_measurement(measurement1, 1.0)  # t1
            test_buffer.add_measurement(measurement2, 2.0)  # t2  
            test_buffer.add_control_action(control_action, 3.0)  # t3
        
        # Verify problematic state: cma=[m1, m2], cpp=[c3], time=[t1, t2]
        assert len(test_buffer._cma_buffer) == 2
        assert len(test_buffer._cpp_buffer) == 1
        assert len(test_buffer._timestamp_buffer) == 2
        
        # This creates the critical misalignment where get_model_inputs(1) would
        # incorrectly pair m2 with c3, even though they're from different time steps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cmas, cpps = test_buffer.get_model_inputs(1)
            
            # Should detect buffer misalignment (CMA != CPP)
            assert any("CRITICAL: Buffer misalignment detected" in str(warning.message) for warning in w)
            # Note: No timestamp warning expected here since timestamp buffer follows CMA buffer
    
    def test_multi_threaded_race_condition_stress(self, test_buffer):
        """Stress test to detect race conditions in multi-threaded environment."""
        # Disable timestamp validation for this test
        test_buffer.validate_sequence = False
        
        num_samples = 50  # Reduced for faster testing
        measurement_errors = []
        control_errors = []
        
        def add_measurements():
            """Thread function to add measurements."""
            try:
                for i in range(num_samples):
                    measurement = np.array([450.0 + i, 1.8 + i * 0.01])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        test_buffer.add_measurement(measurement, float(i))
                    time.sleep(0.001)  # Small delay to increase race condition chance
            except Exception as e:
                measurement_errors.append(e)
        
        def add_controls():
            """Thread function to add control actions."""
            try:
                for i in range(num_samples):
                    control = np.array([130.0 + i, 550.0 + i, 30.0 + i])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        test_buffer.add_control_action(control, float(i) + 0.5)
                    time.sleep(0.001)  # Small delay to increase race condition chance
            except Exception as e:
                control_errors.append(e)
        
        # Start threads to simulate concurrent access
        thread1 = threading.Thread(target=add_measurements)
        thread2 = threading.Thread(target=add_controls)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Check for any threading errors
        assert len(measurement_errors) == 0, f"Measurement errors: {measurement_errors}"
        assert len(control_errors) == 0, f"Control errors: {control_errors}"
        
        # Verify that race condition detection works
        cma_len = len(test_buffer._cma_buffer)
        cpp_len = len(test_buffer._cpp_buffer)
        
        if cma_len != cpp_len:
            # If misalignment occurred, verify it's detected
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    test_buffer.get_model_inputs(min(cma_len, cpp_len))
                    assert any("CRITICAL: Buffer misalignment detected" in str(warning.message) for warning in w)
                except ValueError:
                    pass  # Expected if insufficient data
    
    def test_atomic_operations_prevent_race_conditions(self, test_buffer):
        """Test that atomic add_sample operations prevent race conditions."""
        # Disable timestamp validation for this threading test
        test_buffer.validate_sequence = False
        
        num_samples = 30  # Reduced for faster testing
        sample_errors = []
        
        def add_atomic_samples(start_idx, base_time):
            """Thread function to add atomic samples."""
            try:
                for i in range(num_samples):
                    measurement = np.array([450.0 + start_idx + i, 1.8 + i * 0.01])
                    control = np.array([130.0 + start_idx + i, 550.0 + start_idx + i, 30.0 + i])
                    test_buffer.add_sample(measurement, control, base_time + i)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                sample_errors.append(e)
        
        # Start multiple threads using atomic operations with different time bases
        threads = []
        for thread_id, base_time in enumerate([0.0, 1000.0, 2000.0]):
            thread = threading.Thread(target=add_atomic_samples, args=(thread_id * 100, base_time))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check for any threading errors
        assert len(sample_errors) == 0, f"Sample errors: {sample_errors}"
        
        # Verify perfect buffer synchronization (no race condition)
        cma_len = len(test_buffer._cma_buffer)
        cpp_len = len(test_buffer._cpp_buffer)
        timestamp_len = len(test_buffer._timestamp_buffer)
        
        assert cma_len == cpp_len == timestamp_len, \
            f"Atomic operations failed: CMA={cma_len}, CPP={cpp_len}, Time={timestamp_len}"
        
        # Verify no misalignment warnings when using atomic operations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if len(test_buffer) > 0:
                test_buffer.get_model_inputs(min(10, len(test_buffer)))
            
            # Should have no critical misalignment warnings
            misalignment_warnings = [warning for warning in w 
                                   if "CRITICAL: Buffer misalignment detected" in str(warning.message)]
            assert len(misalignment_warnings) == 0, "Atomic operations should prevent misalignment"
    
    def test_backwards_compatibility(self, test_buffer):
        """Test that existing deprecated methods still work (with warnings)."""
        measurement = np.array([450.0, 1.8])
        control_action = np.array([130.0, 550.0, 30.0])
        
        # Test that deprecated methods still function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            result1 = test_buffer.add_measurement(measurement)
            result2 = test_buffer.add_control_action(control_action)
            
            assert result1 is True
            assert result2 is True
            assert len(test_buffer) == 1
    
    def test_pharmaceutical_manufacturing_scenario(self, test_buffer):
        """Test realistic pharmaceutical manufacturing data flow scenario."""
        # Simulate realistic pharmaceutical control loop data
        measurements = [
            np.array([450.0, 1.8]),  # d50, LOD
            np.array([455.0, 1.75]),
            np.array([460.0, 1.82]),
            np.array([458.0, 1.78])
        ]
        
        control_actions = [
            np.array([130.0, 550.0, 30.0]),  # spray_rate, air_flow, carousel_speed
            np.array([132.0, 555.0, 31.0]),
            np.array([128.0, 545.0, 29.0]),
            np.array([131.0, 552.0, 30.5])
        ]
        
        timestamps = [time.time() + i for i in range(4)]
        
        # Add samples atomically (safe pharmaceutical manufacturing)
        for i in range(4):
            result = test_buffer.add_sample(measurements[i], control_actions[i], timestamps[i])
            assert result is True
        
        # Verify pharmaceutical-grade data integrity
        assert len(test_buffer) == 4
        retrieved_cmas, retrieved_cpps = test_buffer.get_model_inputs(4)
        
        # Verify exact measurement-control pairing integrity
        for i in range(4):
            np.testing.assert_array_equal(retrieved_cmas[i], measurements[i])
            np.testing.assert_array_equal(retrieved_cpps[i], control_actions[i])
        
        # Verify no integrity warnings in pharmaceutical manufacturing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_buffer.get_model_inputs(3)
            
            critical_warnings = [warning for warning in w 
                                if "CRITICAL" in str(warning.message)]
            assert len(critical_warnings) == 0, "Pharmaceutical manufacturing must have perfect data integrity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])