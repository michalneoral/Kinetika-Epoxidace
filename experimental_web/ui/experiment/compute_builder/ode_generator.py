from __future__ import annotations

from typing import List, Tuple

from experimental_web.domain.ode_generation import ODEModel, generate_ode_model
from experimental_web.ui.experiment.compute_builder.graph import GraphState

Edge = Tuple[str, str]


def generate_odes_model(graph_state: GraphState) -> ODEModel:
    """Generate an ODE model (text + state/param names) from the current graph state.

    The output text is compatible with `load_odes_from_txt` / `load_odes_from_text`
    from `experimental_web.domain.ode_compiler`.
    """
    return generate_ode_model(graph_state.nodes, graph_state.edge_modes)


def generate_odes_text(graph_state: GraphState) -> str:
    """Backward compatible helper returning only the text."""
    return generate_odes_model(graph_state).ode_text


def get_state_names(graph_state: GraphState) -> List[str]:
    return generate_odes_model(graph_state).state_names


def get_param_names(graph_state: GraphState) -> List[str]:
    return generate_odes_model(graph_state).param_names
