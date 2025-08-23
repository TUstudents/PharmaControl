"""
Reinforcement Learning Environment for AutoPharm V3.

This module provides a Gymnasium-compatible environment wrapper
for the pharmaceutical granulation process, enabling RL-based control.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Mock plant simulator for now - in practice would import actual simulator
class MockPlantSimulator:
    """Mock plant simulator for RL training."""

    def __init__(self):
        self.state = {"d50": 300.0, "lod": 2.5}
        self.noise_std = {"d50": 5.0, "lod": 0.1}

    def step(self, cpps: Dict[str, float]) -> Dict[str, float]:
        """Simplified plant dynamics for RL training."""
        # Simple linear model + noise for demonstration
        d50_response = (cpps["spray_rate"] - 100) * 2.0 + (cpps["air_flow"] - 500) * 0.3 + 350
        lod_response = (
            -(cpps["spray_rate"] - 120) * 0.01 + (cpps["carousel_speed"] - 30) * 0.05 + 2.0
        )

        # Add process noise
        d50_response += np.random.normal(0, self.noise_std["d50"])
        lod_response += np.random.normal(0, self.noise_std["lod"])

        # Update internal state
        self.state["d50"] = max(50, min(800, d50_response))
        self.state["lod"] = max(0.1, min(5.0, lod_response))

        return self.state.copy()


class GranulationEnv(gym.Env):
    """
    A custom Gymnasium environment for pharmaceutical granulation control.

    This environment wraps the plant simulator and provides the standard
    RL interface for training control policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any]):
        super(GranulationEnv, self).__init__()
        self.config = config
        self.plant = MockPlantSimulator()

        # Episode management
        self.max_episode_steps = config.get("episode_length", 500)
        self.current_step = 0

        # --- Define Action and Observation Space ---
        # Actions are the deltas (changes) to the CPPs
        action_low = np.array([-10.0, -25.0, -2.0])  # Max change down
        action_high = np.array([10.0, 25.0, 2.0])  # Max change up
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observations: [d50, lod, d50_target, lod_target, spray_rate, air_flow, carousel_speed]
        obs_low = np.array([50, 0.1, 300, 1.0, 80, 400, 20])
        obs_high = np.array([800, 5.0, 500, 3.0, 180, 700, 40])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Process constraints
        self.cpp_bounds = {
            "spray_rate": (80, 180),
            "air_flow": (400, 700),
            "carousel_speed": (20, 40),
        }

        # Initialize state
        self.setpoint = np.array([config["target_d50"], config["target_lod"]])
        self.current_cpps = np.array(
            [
                config["initial_cpps"]["spray_rate"],
                config["initial_cpps"]["air_flow"],
                config["initial_cpps"]["carousel_speed"],
            ]
        )

    def _get_obs(self) -> np.ndarray:
        """Get current observation vector."""
        plant_state = self.plant.state
        return np.array(
            [
                plant_state["d50"],
                plant_state["lod"],
                self.setpoint[0],
                self.setpoint[1],
                self.current_cpps[0],  # spray_rate
                self.current_cpps[1],  # air_flow
                self.current_cpps[2],  # carousel_speed
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward based on tracking performance and control effort."""
        # Extract current CMAs and targets
        current_d50, current_lod = obs[0], obs[1]
        target_d50, target_lod = obs[2], obs[3]

        # Tracking errors
        d50_error = abs(current_d50 - target_d50)
        lod_error = abs(current_lod - target_lod)

        # Reward for being close to targets (higher reward for smaller errors)
        d50_reward = np.exp(-0.01 * d50_error)  # Scale factor for d50 error
        lod_reward = np.exp(-2.0 * lod_error)  # Higher penalty for LOD error

        tracking_reward = d50_reward + lod_reward

        # Penalty for large control actions (encourage smoothness)
        action_penalty = -0.01 * np.sum(np.square(action))

        # Bonus for being within tight tolerance
        tolerance_bonus = 0.0
        if d50_error < 10 and lod_error < 0.1:
            tolerance_bonus = 2.0

        total_reward = tracking_reward + action_penalty + tolerance_bonus
        return total_reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset plant simulator
        self.plant = MockPlantSimulator()

        # Reset episode step counter
        self.current_step = 0

        # Reset CPPs to initial values with small random variation
        initial_cpps = self.config["initial_cpps"]
        self.current_cpps = np.array(
            [
                initial_cpps["spray_rate"] + self.np_random.normal(0, 2),
                initial_cpps["air_flow"] + self.np_random.normal(0, 10),
                initial_cpps["carousel_speed"] + self.np_random.normal(0, 1),
            ]
        )

        # Clip to bounds
        self.current_cpps[0] = np.clip(self.current_cpps[0], *self.cpp_bounds["spray_rate"])
        self.current_cpps[1] = np.clip(self.current_cpps[1], *self.cpp_bounds["air_flow"])
        self.current_cpps[2] = np.clip(self.current_cpps[2], *self.cpp_bounds["carousel_speed"])

        # Randomize setpoint slightly for robustness
        base_d50 = self.config["target_d50"]
        base_lod = self.config["target_lod"]
        self.setpoint = np.array(
            [
                base_d50 + self.np_random.uniform(-20, 20),
                base_lod + self.np_random.uniform(-0.2, 0.2),
            ]
        )

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # 1. Apply action (as change to current CPPs)
        self.current_cpps += action

        # Clip to operational bounds
        self.current_cpps[0] = np.clip(self.current_cpps[0], *self.cpp_bounds["spray_rate"])
        self.current_cpps[1] = np.clip(self.current_cpps[1], *self.cpp_bounds["air_flow"])
        self.current_cpps[2] = np.clip(self.current_cpps[2], *self.cpp_bounds["carousel_speed"])

        # 2. Step the plant simulator
        cpp_dict = {
            "spray_rate": self.current_cpps[0],
            "air_flow": self.current_cpps[1],
            "carousel_speed": self.current_cpps[2],
        }
        self.plant.step(cpp_dict)

        # 3. Get new observation
        obs = self._get_obs()

        # 4. Calculate reward
        reward = self._calculate_reward(obs, action)

        # 5. Check termination conditions
        self.current_step += 1
        terminated = False  # Process control is typically continuous
        truncated = self.current_step >= self.max_episode_steps

        # Additional termination for safety violations
        if (
            obs[0] < 50 or obs[0] > 800 or obs[1] < 0.1 or obs[1] > 5.0  # d50 out of safe range
        ):  # LOD out of safe range
            terminated = True
            reward -= 100  # Large penalty for unsafe operation

        info = {
            "tracking_error_d50": abs(obs[0] - obs[2]),
            "tracking_error_lod": abs(obs[1] - obs[3]),
            "current_cpps": self.current_cpps.copy(),
            "episode_step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment (optional for RL training)."""
        if mode == "human":
            obs = self._get_obs()
            print(
                f"Step {self.current_step}: d50={obs[0]:.1f} (target: {obs[2]:.1f}), "
                f"LOD={obs[1]:.2f} (target: {obs[3]:.2f})"
            )

    def close(self):
        """Clean up environment resources."""
        pass
