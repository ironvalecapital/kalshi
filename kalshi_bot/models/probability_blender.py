from __future__ import annotations


def blend_probabilities(p_fundamental: float, delta_flow: float, flow_lambda: float) -> float:
    """
    Hybrid model:
      p_model = p_fundamental + lambda * delta_flow
    """
    p_model = float(p_fundamental) + float(flow_lambda) * float(delta_flow)
    return max(0.01, min(0.99, p_model))
