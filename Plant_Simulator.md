
---

### **A Dynamic Process Simulator for Model Predictive Control Development in Continuous Pharmaceutical Granulation**

**Asset:** `AdvancedPlantSimulator`  
**Version:** 1.0  
**Authors:** Johannes Poms  
**Date:** 2025-07-10

---

### **1. Abstract**

This report details the design, implementation, and validation of the `AdvancedPlantSimulator`, a dynamic, discrete-time digital twin of a continuous pharmaceutical granulation and drying line. Developed in Python, the simulator serves as the primary environment for the offline development, tuning, and testing of Model Predictive Control (MPC) strategies. It is designed to be computationally efficient for rapid iteration while capturing key realistic process characteristics, including **nonlinear dynamics**, **CMA interactions**, **transport delays (time lags)**, and **un-modeled disturbances**. Validation against expected process behavior confirms its suitability as a challenging and representative testbed for advanced process control research, forming the bedrock upon which the project's control frameworks were built and benchmarked.

### **2. Introduction**

The development of effective Model Predictive Control (MPC) controllers is an iterative process that requires extensive testing and tuning. Performing these iterations on a live manufacturing line is impractical, costly, and potentially unsafe. A high-fidelity process simulator is therefore an essential tool, providing a safe and cost-effective environment for controller development.

While first-principles models offer the highest fidelity, they are often complex to develop and too computationally intensive for the thousands of simulations required in a typical MPC project. The `AdvancedPlantSimulator` was designed to bridge this gap. It is an empirical, state-space model whose dynamics are based on established process heuristics and mathematical functions chosen to replicate known nonlinear behaviors. Its primary purpose is not to provide a perfect physicochemical representation, but to create a realistic and challenging control problem for benchmarking APC strategies like the `RobustMPCController`. This document serves as a complete technical reference for the simulator's architecture and capabilities.

### **3. Process Description**

The simulator models a continuous manufacturing line consisting of two main unit operations that are common in modern pharmaceutical production:

1.  **Twin-Screw Wet Granulation:** Raw powder is fed into a granulator where it is mixed with a liquid binder, causing particles to agglomerate into larger, more uniform granules.
2.  **Fluid-Bed Drying:** The wet granules are then conveyed into a semi-continuous dryer where heated air removes excess moisture to a precise target level.

The key variables that govern this process are defined as:

-   **Critical Process Parameters (CPPs) - Inputs:** These are the manipulated variables available to the control system.
    -   `spray_rate` (g/min): The mass flow rate of the liquid binder. This is a primary driver of granule growth.
    -   `air_flow` (m³/h): The volumetric flow rate of heated air in the dryer. This primarily affects the drying rate and can also lead to granule breakage (attrition).
    -   `carousel_speed` (rph): The speed of the rotating carousel in the semi-continuous dryer. This directly controls the mean residence time of particles in the drying chamber.

-   **Critical Material Attributes (CMAs) - Outputs:** These are the key quality attributes of the final product that must be controlled.
    -   `d50` (μm): The median particle size of the final granules, a critical determinant of flowability and compression behavior.
    -   `LOD` (%): The residual moisture content (Loss on Drying), which must be maintained within a tight specification to ensure product stability and efficacy.

### **4. Mathematical Model & Implementation**

The simulator operates in discrete time steps. At each step `k`, it takes the current CPP setpoints $\mathbf{u}_k$ as input and updates its internal state $\mathbf{x}_k$ to produce the new state $\mathbf{x}_{k+1}$.

#### **4.1. State Vector**

The internal state of the simulator consists of the true (noise-free) values of the CMAs, plus internal variables used to model disturbances and time lags:
-   $\mathbf{x}_k = [d_{50,k}, \text{LOD}_k]^T$
-   Internal states: `d50_lag_buffer`, `lod_lag_buffer`, `filter_blockage`

#### **4.2. Dynamic Equations**

The core of the simulator is the `step(cpps)` method, which implements the following logic to calculate the target steady-state values for the CMAs based on the current inputs.

##### **d50 (Granule Size) Dynamics:**
The target `d50` value is modeled as a function of `spray_rate` and `carousel_speed`, incorporating known process nonlinearities.

-   **Nonlinearity:** The effect of `spray_rate` on particle growth is not linear; it exhibits diminishing returns. This is modeled using a hyperbolic tangent (`tanh`) function, which provides a smooth saturation curve.
    $$ \text{spray\_effect} = G_{sr} \cdot \tanh\left(\frac{\text{spray\_rate} - U_{sr,nom}}{S_{sr}}\right) $$
    where $G_{sr}$ is the gain and $S_{sr}$ is a scaling factor determining the sharpness of the saturation.

-   **Linear Effect:** `carousel_speed` is assumed to have a linear effect on the final particle size. Higher speeds reduce the residence time for particle agglomeration in the granulator, leading to smaller particles.
    $$ \text{speed\_effect} = G_{cs} \cdot (\text{carousel\_speed} - U_{cs,nom}) $$

-   **Target Calculation:** The steady-state target is the sum of these effects around a base value.
    $$ d_{50, \text{target}} = \text{d50}_{\text{base}} + \text{spray\_effect} + \text{speed\_effect} $$

-   **Time Lag:** The physical transport delay and mixing dynamics are modeled using a simple but effective moving average filter, implemented as a circular buffer of size `N_d50`. At each time step, the calculated target value is pushed into the buffer, and the output `d50` is the mean of the buffer's contents.

##### **LOD (Moisture) Dynamics:**
The target `LOD` is modeled as a function of the drying parameters and, crucially, the current `d50` state, introducing a key process interaction.

-   **Linear Effects:** `air_flow` and `carousel_speed` are modeled with opposing linear effects on the target LOD. Higher air flow increases drying, while higher carousel speed reduces drying time.

-   **State Interaction:** A critical feature of the simulator is the coupling between CMAs. Larger granules (higher `d50`) are inherently more difficult to dry due to a lower surface-area-to-volume ratio. This is modeled by adding a term proportional to the current `d50` value, making the drying dynamics dependent on the granulation state.
    $$ \text{granule\_size\_effect} = G_{d50} \cdot (d_{50,k} - d_{50,\text{base}}) $$

-   **Un-modeled Disturbance:** To test the controller's robustness and adaptive capabilities, a slow-acting, unmeasured disturbance is included. The internal state variable `filter_blockage` increases linearly over time, representing the gradual clogging of a dryer exhaust filter. This reduces overall drying efficiency and provides a persistent offset that a well-designed controller must overcome.
    $$ \text{disturbance\_effect} = \text{filter\_blockage}_k $$

-   **Target & Lag:** The final `LOD_target` is calculated by summing all contributing effects. A separate moving average buffer (of size `N_lod`) is used to model the distinct dynamics and transport lag of the drying unit.

#### **4.3. Measurement Noise**
The final output of the `step` method is the true internal state plus a small amount of randomly generated Gaussian noise, simulating the inherent variability and imprecision of real-world PAT sensor measurements.

### **5. Dynamic Behavior Validation**

To confirm the simulator's ability to produce realistic dynamic responses, a step-test was performed. The process was allowed to reach a steady state, after which a simultaneous step change was applied to all CPPs.

![Step Response Graph](placeholder_step_response.png)
*Figure 1: Simulated process response to a step change in CPPs at t=75. The plot demonstrates the simulator's nonlinear, time-delayed, and interactive dynamic behavior.*

The results in Figure 1 confirm the expected dynamic behaviors:
-   **Time Lags:** The CMAs do not respond instantly to the CPP change, exhibiting a clear dead time followed by a gradual transition.
-   **Nonlinearity:** The response curve is not a simple first-order exponential, reflecting the `tanh` saturation and interactive dynamics.
-   **Interactions:** The final `LOD` value is a complex result of the competing effects of increased `air_flow` (which acts to decrease LOD) and the simultaneously increasing `d50` (which acts to increase LOD), accurately mimicking real process coupling.

### **6. Usage Guide for Control Development**

The `AdvancedPlantSimulator` is designed for straightforward integration into a Python-based control development workflow.

**Initialization:**
```python
from plant_simulator import AdvancedPlantSimulator
# Initialize with a specific starting state
plant = AdvancedPlantSimulator(initial_state={'d50': 400.0, 'lod': 1.5})
```

**Running a Simulation Step:**
```python
# Define the control actions for the current time step
cpp_inputs = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}

# Get the measured state for the next time step
measured_state = plant.step(cpp_inputs) 
# Example output: {'d50': 401.2, 'lod': 1.51}
```

**Generating a Training Dataset:**
A rich dataset for training a predictive model can be generated by running the simulator in a long loop while applying randomized CPPs at regular intervals. The inputs (CPPs) and outputs (CMAs) should be logged at each step and saved to a file. This "system identification" procedure is detailed in the project's V2 Notebook `02_Data_Wrangling`.

### **7. Conclusion & Limitations**

The `AdvancedPlantSimulator` is a powerful and efficient tool for the development and testing of advanced process controllers. It successfully captures the key dynamic complexities of a continuous granulation process—including nonlinearities, interactions, and disturbances—providing a realistic testbed for benchmarking algorithms like the V1 and V2 MPC frameworks.

It is important to acknowledge the simulator's limitations to ensure its appropriate use:
-   **Empirical Nature:** The model is not based on first principles. Its parameters (`G_{sr}`, `S_{sr}`, etc.) are heuristics and are not directly tied to physical properties of the materials or equipment. It is therefore not suitable for process design, scale-up, or predicting behavior far outside the trained operating range.
-   **Simplified Dynamics:** The time lags and dynamic responses are modeled with simple moving average filters. Real process dynamics are more complex and may exhibit higher-order or non-stationary behavior.
-   **Limited Scope:** The simulator models the granulation and drying units in isolation. It does not include upstream (feeding) or downstream (milling, tableting) operations, nor does it model the effect of raw material property variations.
