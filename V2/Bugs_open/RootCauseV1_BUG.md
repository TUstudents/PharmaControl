Root Cause Analysis: Why is the Adapter Generating Garbage Data?

The evidence from the debugging notebook points to a clear and now obvious bug in the V1ControllerAdapter.

The Bug: The adapter was designed to work with unscaled, physical units. The add_history_step method accepts dictionaries of physical values (e.g., {'d50': 420.0}). However, in the debugging notebook, we fed it dictionaries derived from already scaled data from the V1 training set.

Let's trace the error for spray_rate:

    Input to Adapter: The perfect_past_cpps_df has a spray_rate value of 0.09027. This is the scaled value. The adapter receives {'spray_rate': 0.09027}.

    Adapter's Soft Sensor Calculation: The _calculate_soft_sensors method in the adapter takes this 0.09027 and treats it as a physical value, feeding it into its formulas. This produces a near-zero soft sensor value, which is wrong.

    The V1 Controller's Internal Scaling: The V1 controller's suggest_action method expects unscaled DataFrames, because its internal logic is responsible for scaling the data before prediction. When it receives the already-scaled-but-treated-as-unscaled data from the adapter, it tries to scale it again.

        Let's assume the spray_rate scaler has a min of 80 and a max of 180.

        When the V1 controller receives 0.09027, it calculates: (0.09027 - 80) / (180 - 80) = -0.799. This produces the large negative numbers we see in the output, which are nonsensical inputs for the model.

The original design intent of the adapter was correct, but the debugging test itself was flawed because it fed the adapter the wrong type of data (scaled instead of unscaled). This is a fantastic example of how a well-designed test can reveal a latent logical inconsistency.

The bug that caused the failure in the V2-7 analysis notebook was that the v1_adapter.py's suggest_action method was also likely being called with scaled data instead of the unscaled physical values it was designed for.
The Final Fix: Enforcing the Unscaled Data Contract

The fix involves correcting the v1_adapter.py to be more robust and ensuring it always works with unscaled data, and then correcting the calling code to respect this contract.

Step 1: Fix the v1_adapter.py to use UN SCALED data for its internal logic.

The adapter's primary role is to buffer physical data and calculate soft sensors from that physical data. It should never handle scaled data internally. Its job is to produce an unscaled DataFrame that the V1 controller can then process. The current v1_adapter.py code is actually almost correct in its intent; the problem was how it was being used in the tests.

Step 2: Correct the V1_MPC_Wrapper in v1_adapter.py (if it has the same flaw).

The wrapper's suggest_action method must ensure it passes unscaled current_cmas and current_cpps dictionaries to the adapter.

Step 3: Modify the V1 Controller (V1/src/mpc_controller.py) for clarity.

The root of the confusion is that the V1 suggest_action method accepted arguments named past_cmas_unscaled but the debugging test passed it scaled data. The method signature should be strictly enforced.


