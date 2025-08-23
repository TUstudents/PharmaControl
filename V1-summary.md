---

### **A Data-Driven Model Predictive Control Framework for Continuous Pharmaceutical Granulation (V1)**

**Version:** 1.0  
**Authors:** Johannes Poms  
**Date:** 2025-07-01

---

### **1. Abstract**

This report documents the design and performance of the first-generation (V1) Model Predictive Control (MPC) prototype for a continuous pharmaceutical granulation line. The objective was to demonstrate the feasibility of using a data-driven, deep learning model as the predictive kernel within an MPC framework. The V1 controller integrates a **deterministic Transformer-based model** trained on simulated process data and employs an **exhaustive search** over a discretized action space for optimization. Performance was validated in a simulated standard production scenario, where the controller successfully steered the process to a new setpoint. While the prototype proves the viability of the approach, this report also identifies key limitations, including susceptibility to noise and a lack of robustness to un-modeled disturbances, which will be addressed in future development cycles.

---

### **2. Introduction**

Continuous Manufacturing (CM) in the pharmaceutical sector requires sophisticated control strategies to manage the complex, multivariate dynamics of interconnected unit operations. This project targeted the control of a twin-screw granulation and fluid-bed drying process, where the primary goal is to maintain the Critical Material Attributes (CMAs) of particle size (`d50`) and moisture content (`LOD`) at desired targets by manipulating Critical Process Parameters (CPPs) like spray rate and air flow.

The V1 framework was developed as a proof-of-concept to replace traditional empirical models with a more flexible, data-driven deep learning model. The hypothesis was that a sequence-to-sequence neural network could effectively capture the nonlinear, time-delayed process dynamics and serve as the predictive engine for an MPC controller. Unlike first-principles models, which can be difficult to develop and computationally prohibitive for real-time use, this data-driven approach aims to learn the system's behavior directly from operational data. This report details the architecture, implementation, and performance of this initial prototype.

---

### **3. System Formulation**

The control problem is defined as a discrete-time, multivariate, deterministic system for the purpose of this V1 implementation. The controller acts directly on sensor measurements, assuming them to be the true state of the process.

-   **State Vector ($\mathbf{x}_k \in \mathbb{R}^{n_x}$):** The vector of measured CMA values at time step `k`.
    -   $n_x=2$: The states are the median particle size and the loss on drying.
    -   $\mathbf{x}_k = [d_{50,k}, \text{LOD}_k]^T$

-   **Control Input Vector ($\mathbf{u}_k \in \mathbb{R}^{n_u}$):** The vector of manipulated setpoints for the CPPs at time step `k`. This vector is augmented with calculated "soft sensor" values derived from the base CPPs to provide richer input to the predictive model.
    -   $n_u=3$: The base manipulated variables are the spray rate, dryer air flow, and carousel speed.
    -   $\mathbf{u}_k^{\text{base}} = [\text{spray\_rate}_k, \text{air\_flow}_k, \text{carousel\_speed}_k]^T$
    -   $n_{u'}=5$: The augmented input vector $\mathbf{u}_k$ includes two soft sensors (`specific_energy`, `froude_number_proxy`), which are algebraic functions of the base CPPs.

-   **System Dynamics:** The process is governed by an unknown, nonlinear discrete-time function `f`:
    $$
    \mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{x}_{k-1}, ..., \mathbf{x}_{k-L+1}, \mathbf{u}_k, \mathbf{u}_{k-1}, ..., \mathbf{u}_{k-L})
    $$
    where `L` is the lookback window length. The objective of the machine learning model is to provide a computationally tractable approximation of this function, $\mathbf{\hat{f}}$.

-   **Control Objective:** The primary objective is to compute a sequence of future control inputs $\{\mathbf{u}_k, ..., \mathbf{u}_{k+H-1}\}$ that drives the predicted future state trajectory $\{\mathbf{\hat{x}}_{k+1}, ..., \mathbf{\hat{x}}_{k+H}\}$ as close as possible to a desired setpoint trajectory $\{\mathbf{r}_{k+1}, ..., \mathbf{r}_{k+H}\}$.

-   **Constraints:** The controller must operate within predefined operational limits for the base CPPs. These are implemented as simple box constraints:
    $$
    \mathbf{u}_{\text{min}} \le \mathbf{u}_k^{\text{base}} \le \mathbf{u}_{\text{max}}
    $$
    The V1 framework does not explicitly handle rate-of-change constraints within its optimization algorithm; however, the discretized nature of its action space inherently limits the magnitude of control moves between steps.

---

### **4. The Predictive Model (Deterministic Transformer)**

The success of the data-driven MPC framework hinges on the predictive model's ability to accurately forecast the process's future behavior. For the V1 prototype, a **deterministic sequence-to-sequence model** based on the Transformer architecture was developed. This model provides a single, best-guess (point) forecast for a given input, with no explicit quantification of predictive uncertainty.

#### **4.1. Model Input-Output Specification**

The model is designed as a direct, multi-step-ahead predictor. It takes a history of process variables and a future plan of control actions as input and directly outputs the entire predicted trajectory of the CMAs. This approach was chosen to avoid the error accumulation that can occur with recursive one-step-ahead prediction models.

At each decision time step `k`, the model approximates the process dynamics as a function $\mathbf{\hat{f}}$:

$\mathbf{\hat{X}}_{k+1:k+H} = \mathbf{\hat{f}}(\mathbf{X}_{k-L+1:k}, \mathbf{U}_{k-L:k-1}, \mathbf{U}_{k:k+H-1})$

Where:

-   **Model Inputs:**
    -   **Past CMAs ($\mathbf{X}_{k-L+1:k}$):** A sequence (tensor) of the last `L` measured CMA states: $\{\mathbf{x}_{k-L+1}, ..., \mathbf{x}_k\}$. For V1, `L=36`.
    -   **Past CPPs ($\mathbf{U}_{k-L:k-1}$):** A sequence (tensor) of the last `L` augmented control inputs, including soft sensors: $\{\mathbf{u}_{k-L}, ..., \mathbf{u}_{k-1}\}$.
    -   **Future CPPs (Candidate Plan) ($\mathbf{U}_{k:k+H-1}$):** A proposed sequence (tensor) of `H` future augmented control inputs being evaluated by the optimizer: $\{\mathbf{u}_k, ..., \mathbf{u}_{k+H-1}\}$. For V1, `H=72`.

-   **Model Outputs:**
    -   A single, deterministic sequence (tensor) of `H` predicted future CMA states: $\mathbf{\hat{X}}_{k+1:k+H} = \{\mathbf{\hat{x}}_{k+1}, ..., \mathbf{\hat{x}}_{k+H}\}$.

#### **4.2. Architecture**

The model is a **Transformer Encoder-Decoder** implemented in PyTorch, chosen for its proven success in sequence modeling tasks. The `batch_first=True` convention is used for all tensor manipulations.

![Model Architecture Diagram](placeholder_model_architecture.png)
*Figure 1: High-level schematic of the Transformer Encoder-Decoder architecture. The encoder creates a contextualized memory of the past, which the decoder uses along with the future control plan to generate predictions.*

-   **Input Embeddings:** Separate linear layers project the input CMA and CPP features into the model's hidden dimension (`d_model`).
-   **Positional Encoding:** Standard sinusoidal positional encodings are added to the input embeddings to provide the model with temporal context.
-   **Encoder:** Processes the sequence of historical data. Its self-attention mechanism allows it to identify and weigh the importance of different past events when creating its final contextual representation of the process history.
-   **Decoder:** Takes the sequence of planned future CPPs as input and uses cross-attention to query the encoder's memory. This allows the decoder to predict the likely outcome of a future control plan based on how similar plans have affected the process in the past.
-   **Output Layer:** A final linear layer projects the decoder's output back to the dimension of the CMAs ($n_x=2$).

The model was trained via supervised learning on a large dataset generated from a high-fidelity process simulator, using Mean Squared Error (MSE) as the loss function and the Adam optimizer.

---

### **5. The Controller Architecture (V1)**

The V1 `MPCController` implements a straightforward receding horizon control algorithm based on a discrete action space. The design prioritizes simplicity and guaranteed convergence to the best *available* action within the predefined search space.

#### **5.1. Algorithm (at decision time `k`)**

1.  **State Measurement:** Receive the current measurement vector $\mathbf{x}_k$ from the process sensors.
2.  **History Buffering:** Append the latest state $\mathbf{x}_k$ and the previously applied control action $\mathbf{u}_{k-1}$ to an internal history buffer.
3.  **Candidate Action Generation:** Create a discrete "lattice" of all possible control plans for the next time step. This is achieved by defining a small, fixed number of potential moves for each base CPP (e.g., `spray_rate`: {110, 120, 130} g/min). The Cartesian product of these sets forms the complete set of candidate actions.
4.  **Plan Formulation:** For each candidate action in the lattice, a full future control plan $\mathbf{U}_{k:k+H-1}$ is formulated by assuming the chosen action is held constant for the entire prediction horizon `H`. The corresponding augmented input vectors (including soft sensors) are then calculated.
5.  **Prediction and Evaluation (Exhaustive Search):** The controller iterates through every candidate plan in the lattice:
    a.  The Transformer model is called with the current history and the candidate plan to generate a predicted future CMA trajectory, $\mathbf{\hat{X}}$.
    b.  The cost `J` of this trajectory is calculated using the cost function.
6.  **Optimization:** The control plan that resulted in the minimum cost `J` is selected as the optimal plan. This brute-force, exhaustive search guarantees that the best option within the discrete lattice is always found.
7.  **Implementation:** The first step of the winning control plan, $\mathbf{u}_k^*$, is sent to the process actuators. The rest of the plan is discarded, and the cycle repeats at the next time step.

#### **5.2. The Cost Function (`J`)**

The cost function for the V1 controller is a standard quadratic formulation that balances the competing objectives of setpoint tracking and control effort.

$$
J = \sum_{j=1}^{H} \left( (\mathbf{\hat{x}}_{k+j} - \mathbf{r}_{k+j})^T Q (\mathbf{\hat{x}}_{k+j} - \mathbf{r}_{k+j}) + (\Delta \mathbf{u}_{k+j-1})^T R (\Delta \mathbf{u}_{k+j-1}) \right)
$$

-   **Tracking Error Term:** The first term penalizes the squared deviation of the predicted state $\mathbf{\hat{x}}$ from the setpoint $\mathbf{r}$. The diagonal matrix `Q` allows for different weights to be placed on controlling `d50` versus `LOD`.
-   **Control Effort Term:** The second term penalizes the magnitude of the change in control action, $\Delta \mathbf{u} = \mathbf{u}_j - \mathbf{u}_{j-1}$. The diagonal matrix `R` is used to tune the aggressiveness of the controller, with larger `R` values promoting smoother, less frequent control moves.

---

### **6. Simulation & Performance Analysis**

The V_1_ controller's performance was validated in a simulated environment designed to mimic a standard operational task. The objective of the test was to assess the controller's ability to perform a routine grade transition under nominal conditions, without the presence of significant process noise or unmodeled disturbances.

#### **6.1. Test Scenario: Standard Grade Transition**

The simulation scenario was defined as follows:

1.  **Initial Steady-State:** The process simulator was initialized and run at a stable steady-state condition (Setpoint A) to allow all process variables to stabilize.
2.  **Setpoint Change:** After a period of stable operation, a command was issued to the MPC controller to drive the process to a new target grade (Setpoint B). This involved a simultaneous change in the desired setpoints for both median particle size (_d_~50~) and loss on drying (LOD).
3.  **Observation:** The simulation was continued until the process reached a new steady state at Setpoint B, with the controller autonomously managing the transition.

#### **6.2. Results: Setpoint Tracking**

Figure 2 illustrates the closed-loop trajectories of the Critical Material Attributes (_d_~50~ and LOD) as the V_1_ controller manages the transition from Setpoint A to Setpoint B.


*Figure 2: CMA trajectories for the V_1_ MPC controller during a standard grade transition. The controller successfully steers both _d_~50~ (blue) and LOD (green) to their new target setpoints (dashed lines).*

The results demonstrate that the V_1_ prototype successfully achieves its primary control objective. The controller effectively utilizes the predictions from its internal Transformer model to compute a sequence of control actions that drives the process to the new target. The transition is stable, with a well-damped response and minimal overshoot, indicating that the controller's tuning (via the _Q_ and _R_ cost matrices) is appropriate for this task.

#### **6.3. Results: Control Actions**

Figure 3 shows the corresponding sequence of control actions (CPPs) computed and applied by the MPC controller during the simulation.


*Figure 3: CPP control actions implemented by the V_1_ MPC controller. The step-wise changes are a direct result of the exhaustive search over a discrete action space.*

The control action profile is characterized by distinct, step-wise changes. This behavior is a direct consequence of the **exhaustive search** optimization method, where the controller selects the best action from a pre-defined, discrete lattice of options at each time step. While not necessarily smooth, the resulting control policy is effective and computationally guaranteed to be the optimal choice within the defined search space.

---

### **7. Conclusion & V1 Limitations**

The V_1_ Model Predictive Control framework successfully validates the core hypothesis of this project: a data-driven, deep learning model can serve as an effective predictive engine for the control of a complex, nonlinear manufacturing process. The prototype demonstrated stable, closed-loop setpoint tracking in a simulated environment, proving the viability of the overall architectural approach.

However, this initial development phase also served to highlight several critical limitations inherent in the V_1_ design. These weaknesses, while acceptable for a proof-of-concept, must be addressed to create a truly robust and industrially-viable system. The key limitations identified are:

1.  **Susceptibility to Noise:** The controller acts directly on raw measurements and lacks a state estimator (e.g., a Kalman Filter). In a real plant, this would make it vulnerable to sensor noise, potentially leading to excessive and unstable control actions as it attempts to "chase" random fluctuations.

2.  **Lack of Robustness to Disturbances:** The controller is purely reactive and has no mechanism (such as integral action) to handle un-modeled, persistent disturbances or model-plant mismatch. This was a known design trade-off for simplicity in V_1_, but it means the controller would likely exhibit significant steady-state error if the real process behavior deviates from the training data.

3.  **Deterministic Nature:** The predictive model provides only a single point forecast and cannot communicate its own uncertainty. Consequently, the controller makes decisions with an assumed—but unjustified—level of confidence. This is a potential risk when operating in novel process regions where the model's predictions may be less reliable.

4.  **Unscalable Optimization:** The exhaustive search optimization method, while simple and effective for this prototype, is computationally expensive. Its complexity scales exponentially with the number of control variables and the granularity of the action space. This approach is not feasible for more complex problems or for controllers requiring more sophisticated, multi-step action plans.

These identified weaknesses provide a clear and direct motivation for the architectural enhancements planned for the V_2_ framework. The next development cycle will focus on integrating components such as a Kalman Filter, a probabilistic predictive model, and a more advanced optimization algorithm to address each of these limitations and build a more powerful and reliable control system.