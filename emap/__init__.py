"""EMAP: Empirical Multimodally-Additive Projection for two-modality models.

See README.md for usage and adaptation notes.
"""
from .projection import project_logits, appendix_g_test
from .evaluator import EMAPEvaluator

__all__ = ["project_logits", "appendix_g_test", "EMAPEvaluator"]
