"""
Explainable AI components for AutoPharm V3.

This module provides explanation and trust-building capabilities
for autonomous control decisions using SHAP-based techniques.
"""

# Progressive imports as components become available
try:
    from .explainer import ShapExplainer
    __all__ = ['ShapExplainer']
except ImportError:
    __all__ = []