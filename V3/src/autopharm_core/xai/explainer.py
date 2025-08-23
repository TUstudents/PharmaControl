from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn


# Simplified types for demo (would import from ..common.types in full implementation)
class StateVector:
    def __init__(self, timestamp: float, cmas: Dict[str, float], cpps: Dict[str, float]):
        self.timestamp = timestamp
        self.cmas = cmas
        self.cpps = cpps


class ControlAction:
    def __init__(
        self, timestamp: float, cpp_setpoints: Dict[str, float], action_id: str, confidence: float
    ):
        self.timestamp = timestamp
        self.cpp_setpoints = cpp_setpoints
        self.action_id = action_id
        self.confidence = confidence


class DecisionExplanation:
    def __init__(
        self,
        decision_id: str,
        control_action: ControlAction,
        narrative: str,
        feature_attributions: Dict[str, float],
        confidence_factors: Dict[str, float],
        alternatives_considered: int,
    ):
        self.decision_id = decision_id
        self.control_action = control_action
        self.narrative = narrative
        self.feature_attributions = feature_attributions
        self.confidence_factors = confidence_factors
        self.alternatives_considered = alternatives_considered


# Simplified model for demonstration (would use ProbabilisticTransformer in full implementation)
class SimpleProcessModel(nn.Module):
    """Simplified neural network model that mimics our transformer's input structure."""

    def __init__(self, input_features: int = 15, output_features: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_features),
        )

    def forward(self, x):
        return self.network(x)


class ShapExplainer:
    """
    Provides SHAP-based explanations for model predictions and control decisions.
    Generates human-interpretable explanations for autonomous control actions.
    """

    def __init__(
        self,
        model: nn.Module,
        training_data_summary: np.ndarray,
        feature_names: List[str],
        config: Dict[str, Any],
    ):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained neural network model
            training_data_summary: Representative background dataset for SHAP
            feature_names: Names of input features
            config: Explainer configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Prepare background data for SHAP
        self.background_data = torch.tensor(training_data_summary, dtype=torch.float32).to(
            self.device
        )

        # Initialize SHAP explainer
        self._initialize_shap_explainer()

        # Explanation templates for different scenarios
        self.explanation_templates = self._load_explanation_templates()

    def _initialize_shap_explainer(self):
        """Initialize SHAP DeepExplainer for the model."""

        def model_wrapper(inputs):
            """Wrapper function that SHAP can call."""
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs_tensor)

            return outputs.cpu().numpy()

        self.model_wrapper = model_wrapper

        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(model_wrapper, self.background_data.cpu().numpy())

    def explain_prediction(self, model_input: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single model prediction.

        Args:
            model_input: Model input array, shape (n_features,)

        Returns:
            Dict[str, Any]: SHAP explanation results
        """
        # Ensure input is 2D for SHAP
        if model_input.ndim == 1:
            model_input = model_input.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explainer.shap_values(model_input)

        # Handle multi-output case
        if isinstance(shap_values, list):
            # For multi-output, we'll focus on the first output (d50)
            shap_values = shap_values[0]

        # Map SHAP values to feature names
        feature_attributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(shap_values[0]):
                feature_attributions[name] = float(shap_values[0][i])

        # Get model prediction for context
        with torch.no_grad():
            model_input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)
            prediction = self.model(model_input_tensor)
            prediction_np = prediction.squeeze().cpu().numpy()

        explanation = {
            "feature_attributions": feature_attributions,
            "prediction": prediction_np,
            "top_positive_features": self._get_top_features(feature_attributions, positive=True),
            "top_negative_features": self._get_top_features(feature_attributions, positive=False),
            "explanation_quality": self._assess_explanation_quality(feature_attributions),
        }

        return explanation

    def generate_decision_narrative(
        self,
        history: List[StateVector],
        action: ControlAction,
        prediction_explanation: Optional[Dict[str, Any]] = None,
    ) -> DecisionExplanation:
        """
        Generate human-readable explanation for a control decision.

        Args:
            history: Recent process history
            action: Control action taken
            prediction_explanation: Optional pre-computed SHAP explanation

        Returns:
            DecisionExplanation: Complete decision explanation
        """
        # Generate prediction explanation if not provided
        if prediction_explanation is None:
            # Convert history to model input format
            model_input = self._convert_history_to_input(history, action)
            prediction_explanation = self.explain_prediction(model_input)

        # Generate narrative explanation
        narrative = self._create_narrative_explanation(history, action, prediction_explanation)

        # Calculate confidence factors
        confidence_factors = self._analyze_confidence_factors(prediction_explanation, action)

        decision_explanation = DecisionExplanation(
            decision_id=action.action_id,
            control_action=action,
            narrative=narrative,
            feature_attributions=prediction_explanation["feature_attributions"],
            confidence_factors=confidence_factors,
            alternatives_considered=self._estimate_alternatives_considered(prediction_explanation),
        )

        return decision_explanation

    def _get_top_features(
        self, attributions: Dict[str, float], positive: bool = True, n_top: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top contributing features from SHAP attributions."""
        sorted_features = sorted(
            attributions.items(), key=lambda x: x[1] if positive else -x[1], reverse=True
        )

        if positive:
            return [(name, value) for name, value in sorted_features[:n_top] if value > 0]
        else:
            return [(name, abs(value)) for name, value in sorted_features[:n_top] if value < 0]

    def _assess_explanation_quality(self, attributions: Dict[str, float]) -> Dict[str, float]:
        """Assess the quality and reliability of the explanation."""
        values = list(attributions.values())

        quality_metrics = {
            "attribution_magnitude": np.sum(np.abs(values)),
            "attribution_concentration": np.std(values) / (np.mean(np.abs(values)) + 1e-8),
            "n_significant_features": sum(1 for v in values if abs(v) > 0.01),
            "explanation_clarity": min(
                1.0, np.max(np.abs(values)) / (np.mean(np.abs(values)) + 1e-8)
            ),
        }

        return quality_metrics

    def _convert_history_to_input(
        self, history: List[StateVector], action: ControlAction
    ) -> np.ndarray:
        """Convert StateVector history and action to model input format."""
        # Extract recent state (simplified for demo)
        recent_state = (
            history[-1]
            if history
            else StateVector(
                0,
                {"d50": 400, "lod": 1.5},
                {"spray_rate": 120, "air_flow": 500, "carousel_speed": 30},
            )
        )

        # Create input features combining state and planned action
        input_features = [
            recent_state.cmas.get("d50", 400),
            recent_state.cmas.get("lod", 1.5),
            recent_state.cpps.get("spray_rate", 120),
            recent_state.cpps.get("air_flow", 500),
            recent_state.cpps.get("carousel_speed", 30),
            action.cpp_setpoints.get("spray_rate", 120),
            action.cpp_setpoints.get("air_flow", 500),
            action.cpp_setpoints.get("carousel_speed", 30),
            # Add soft sensor calculations
            (
                action.cpp_setpoints.get("spray_rate", 120)
                * action.cpp_setpoints.get("carousel_speed", 30)
            )
            / 1000.0,  # specific energy
            (action.cpp_setpoints.get("carousel_speed", 30) ** 2) / 9.81,  # froude number proxy
            # Add trend indicators (simplified)
            np.random.randn(),  # d50 trend
            np.random.randn(),  # lod trend
            np.random.randn(),  # spray rate trend
            np.random.randn(),  # air flow trend
            np.random.randn(),  # carousel speed trend
        ]

        return np.array(input_features, dtype=np.float32)

    def _create_narrative_explanation(
        self, history: List[StateVector], action: ControlAction, explanation: Dict[str, Any]
    ) -> str:
        """Create human-readable narrative explanation."""
        # Get current process state
        current_state = (
            history[-1]
            if history
            else StateVector(
                0,
                {"d50": 400, "lod": 1.5},
                {"spray_rate": 120, "air_flow": 500, "carousel_speed": 30},
            )
        )

        # Identify primary control objective
        primary_objective = self._identify_primary_objective(current_state, action)

        # Get top influencing factors
        top_positive = explanation["top_positive_features"][:3]
        top_negative = explanation["top_negative_features"][:3]

        # Build narrative
        narrative_parts = []

        # Opening statement
        narrative_parts.append(
            f"Control action taken at {datetime.fromtimestamp(action.timestamp).strftime('%H:%M:%S')}:"
        )
        narrative_parts.append(f"Primary objective: {primary_objective}")

        # Control actions
        actions_text = []
        for cpp_name, value in action.cpp_setpoints.items():
            current_val = current_state.cpps.get(cpp_name, 0.0)
            change = value - current_val
            direction = "increase" if change > 0 else "decrease" if change < 0 else "maintain"
            actions_text.append(f"{direction} {cpp_name} to {value:.1f}")

        narrative_parts.append(f"Actions: {', '.join(actions_text)}")

        # Key reasoning
        if top_positive:
            positive_factors = [f"{name} (impact: {value:.3f})" for name, value in top_positive]
            narrative_parts.append(f"Key supporting factors: {', '.join(positive_factors)}")

        if top_negative:
            negative_factors = [f"{name} (concern: {value:.3f})" for name, value in top_negative]
            narrative_parts.append(f"Key constraints considered: {', '.join(negative_factors)}")

        # Confidence statement
        confidence_pct = int(action.confidence * 100)
        narrative_parts.append(f"Decision confidence: {confidence_pct}%")

        return " | ".join(narrative_parts)

    def _identify_primary_objective(self, current_state: StateVector, action: ControlAction) -> str:
        """Identify the primary control objective based on state and action."""
        # Identify based on largest action change
        max_change = 0
        primary_cpp = ""

        for cpp_name, new_value in action.cpp_setpoints.items():
            current_val = current_state.cpps.get(cpp_name, 0.0)
            change = abs(new_value - current_val)

            if change > max_change:
                max_change = change
                primary_cpp = cpp_name

        # Map CPP to likely objective
        objective_map = {
            "spray_rate": "particle size control",
            "air_flow": "moisture content adjustment",
            "carousel_speed": "residence time optimization",
        }

        return objective_map.get(primary_cpp, "process optimization")

    def _analyze_confidence_factors(
        self, explanation: Dict[str, Any], action: ControlAction
    ) -> Dict[str, float]:
        """Analyze factors contributing to decision confidence."""
        attribution_magnitude = explanation["explanation_quality"]["attribution_magnitude"]
        explanation_clarity = explanation["explanation_quality"]["explanation_clarity"]

        confidence_factors = {
            "model_certainty": min(1.0, attribution_magnitude / 10.0),  # Normalize
            "explanation_clarity": explanation_clarity,
            "feature_consensus": len(explanation["top_positive_features"]) / 10.0,
            "action_magnitude": min(
                1.0, sum(abs(v) for v in action.cpp_setpoints.values()) / 100.0
            ),
        }

        return confidence_factors

    def _estimate_alternatives_considered(self, explanation: Dict[str, Any]) -> int:
        """Estimate number of alternatives considered based on explanation analysis."""
        significant_features = explanation["explanation_quality"]["n_significant_features"]
        return max(3, significant_features * 2)  # Rough estimate

    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different scenarios."""
        return {
            "tracking_control": "Adjusting {cpp} to {direction} {cma} towards target of {target}",
            "disturbance_rejection": "Countering process disturbance by {action}",
            "optimization": "Optimizing process efficiency through {strategy}",
            "safety_action": "Taking precautionary action to maintain safe operation",
        }

    def visualize_explanation(self, explanation: Dict[str, Any], save_path: Optional[str] = None):
        """Create visualization of SHAP explanation."""
        feature_attributions = explanation["feature_attributions"]

        # Create bar plot of feature attributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Positive contributions
        positive_attrs = {k: v for k, v in feature_attributions.items() if v > 0}
        if positive_attrs:
            sorted_positive = sorted(positive_attrs.items(), key=lambda x: x[1], reverse=True)[:8]
            names, values = zip(*sorted_positive)
            ax1.barh(names, values, color="green", alpha=0.7)
            ax1.set_title(
                "Positive Feature Contributions (Supporting the Decision)", fontweight="bold"
            )
            ax1.set_xlabel("SHAP Value")

        # Negative contributions
        negative_attrs = {k: abs(v) for k, v in feature_attributions.items() if v < 0}
        if negative_attrs:
            sorted_negative = sorted(negative_attrs.items(), key=lambda x: x[1], reverse=True)[:8]
            names, values = zip(*sorted_negative)
            ax2.barh(names, values, color="red", alpha=0.7)
            ax2.set_title(
                "Negative Feature Contributions (Constraints Considered)", fontweight="bold"
            )
            ax2.set_xlabel("|SHAP Value|")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def get_explanation_quality_metrics(self) -> Dict[str, Any]:
        """Get metrics about explanation system performance."""
        return {
            "explainer_type": "SHAP DeepExplainer",
            "feature_count": len(self.feature_names),
            "background_samples": self.background_data.shape[0],
            "explanation_templates": len(self.explanation_templates),
        }
