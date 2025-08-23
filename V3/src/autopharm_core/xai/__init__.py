"""
Explainable AI components for AutoPharm V3.

This module provides SHAP-based explanations and decision transparency
for building trust in autonomous control systems.
"""

# Progressive imports as components become available
try:
    from .explainer import ShapExplainer

    __all__ = ["ShapExplainer"]
except ImportError:
    __all__ = []
