import pytest
import pandas as pd
import numpy as np
import torch
import joblib
from unittest.mock import MagicMock, patch

# Add project root to path to allow V1 and V2 imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# V1 and V2 components
from V1.src.mpc_controller import MPCController as V1Controller
from V2.robust_mpc.v1_adapter import V1_MPC_Wrapper
from V2.robust_mpc.models import load_trained_model

# Constants
V1_DATA_PATH = Path(__file__).parent.parent.parent / "V1/data"
DEVICE = 'cpu'

@pytest.fixture(scope="module")
def v1_components():
    """Loads all necessary V1 components for testing."""
    scalers = joblib.load(V1_DATA_PATH / "scalers.joblib")
    model = load_trained_model(
        V1_DATA_PATH / "best_predictor_model.pth",
        device=DEVICE,
        validate=True
    )
    config = {
        'lookback': 36,
        'horizon': 72,
        'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
        'cma_names': ['d50', 'lod'],
        'cpp_names_and_soft_sensors': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
        'control_effort_lambda': 0.05,
        'discretization_steps': 3,
    }
    constraints = {
        'spray_rate': {'min_val': 80.0, 'max_val': 180.0, 'max_change_per_step': 10.0},
        'air_flow': {'min_val': 400.0, 'max_val': 700.0, 'max_change_per_step': 25.0},
        'carousel_speed': {'min_val': 20.0, 'max_val': 40.0, 'max_change_per_step': 2.0}
    }
    return {
        "model": model,
        "config": config,
        "constraints": constraints,
        "scalers": scalers
    }

def test_v1_adapter_passes_unscaled_data(v1_components):
    """
    Tests the fixed V1 adapter to ensure it passes UN SCALED data to the V1 controller.
    This test verifies the fix for the double-scaling bug.
    """
    # 1. Instantiate the real V1 controller and the wrapper
    v1_controller_instance = V1Controller(
        model=v1_components["model"],
        config=v1_components["config"],
        constraints=v1_components["constraints"],
        scalers=v1_components["scalers"]
    )

    # 2. Create a mock of the suggest_action method to spy on its inputs
    # We are testing the *input* to the V1 controller, not its output.
    v1_controller_instance.suggest_action = MagicMock(
        return_value=np.array([130.0, 550.0, 30.0]) # Return a valid action
    )

    # 3. Instantiate the adapter, passing the controller with the mocked method
    adapter = V1_MPC_Wrapper(
        V1ControllerClass=lambda **kwargs: v1_controller_instance, # Use a lambda to pass the already created instance
        model=v1_components["model"],
        config=v1_components["config"],
        constraints=v1_components["constraints"],
        scalers=v1_components["scalers"]
    )

    # 4. Fill the adapter's history buffer until it's ready
    lookback = v1_components["config"]["lookback"]
    for i in range(lookback):
        # Use realistic, UN SCALED physical data
        cmas_unscaled = {'d50': 420.0 + i*0.1, 'lod': 1.6 - i*0.01}
        cpps_unscaled = {'spray_rate': 130.0, 'air_flow': 550.0, 'carousel_speed': 30.0}
        adapter.adapter.add_history_step(cmas_unscaled, cpps_unscaled)

    assert adapter.adapter.is_ready()

    # 5. Define the final step of unscaled data and the setpoint
    final_cmas_unscaled = {'d50': 430.0, 'lod': 1.5}
    final_cpps_unscaled = {'spray_rate': 135.0, 'air_flow': 555.0, 'carousel_speed': 32.0}
    setpoint = np.array([450.0, 1.4])

    # 6. Call the wrapper's suggest_action method
    adapter.suggest_action(final_cmas_unscaled, final_cpps_unscaled, setpoint)

    # 7. THE CRITICAL ASSERTION
    # Check that the mocked suggest_action was called exactly once
    v1_controller_instance.suggest_action.assert_called_once()

    # Get the arguments that were passed to the mocked method
    args, kwargs = v1_controller_instance.suggest_action.call_args
    
    # The arguments are passed positionally, so they will be in 'args'
    # In the original V1 controller, the signature is (self, past_cmas, past_cpps, target_cmas)
    # The mock captures all args, so we check args[0] and args[1]
    passed_past_cmas = kwargs.get('past_cmas') or args[0]
    passed_past_cpps = kwargs.get('past_cpps') or args[1]

    # Assert that the received data is a pandas DataFrame
    assert isinstance(passed_past_cmas, pd.DataFrame), "past_cmas should be a DataFrame"
    assert isinstance(passed_past_cpps, pd.DataFrame), "past_cpps should be a DataFrame"

    # Get the last row of data passed to the controller
    # This should correspond to the `final_cmas_unscaled` and `final_cpps_unscaled` data
    last_cmas_passed = passed_past_cmas.iloc[-1]
    last_cpps_passed = passed_past_cpps.iloc[-1]

    # Verify that the data was NOT scaled
    # The values should be close to the physical values we passed in.
    assert np.isclose(last_cmas_passed['d50'], final_cmas_unscaled['d50'])
    assert np.isclose(last_cmas_passed['lod'], final_cmas_unscaled['lod'])
    assert np.isclose(last_cpps_passed['spray_rate'], final_cpps_unscaled['spray_rate'])
    assert np.isclose(last_cpps_passed['air_flow'], final_cpps_unscaled['air_flow'])
    assert np.isclose(last_cpps_passed['carousel_speed'], final_cpps_unscaled['carousel_speed'])

    print("\n✅ Test Passed: V1ControllerAdapter correctly passed UN SCALED data to the V1Controller.")
