"""
Graph Builder
=============

Utilities for constructing LangGraph workflows.
"""

from typing import Callable, Type
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from point9_platform.graph.nodes import should_continue


def build_default_graph(
    agent,
    planner: Callable,
    executor: Callable,
    reflector: Callable,
    responder: Callable,
    state_class: Type = None
):
    """
    Build the default LangGraph workflow.
    
    Flow: Planner → Executor ↔ Reflector → Responder
    
    Args:
        agent: BaseAgent instance (for state class extraction)
        planner: Planner node function
        executor: Executor node function
        reflector: Reflector node function
        responder: Responder node function
        state_class: Optional TypedDict class for state (recommended)
        
    Returns:
        Compiled LangGraph application
    """
    # Get state class - prefer explicit class over type inference
    # type(dict_instance) returns 'dict', not the TypedDict!
    if state_class is None:
        # Try to get from agent's type hints
        from typing import get_type_hints
        from point9_platform.agent.state import BaseAgentState
        try:
            hints = get_type_hints(agent.create_initial_state)
            state_class = hints.get('return', BaseAgentState)
        except Exception:
            state_class = BaseAgentState
    
    # Create the state graph
    workflow = StateGraph(state_class)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("reflector", reflector)
    workflow.add_node("responder", responder)
    
    # Define edges (using START instead of deprecated set_entry_point)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")
    workflow.add_conditional_edges("reflector", should_continue)
    workflow.add_edge("responder", END)
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def build_simple_graph(
    state_class: Type,
    processor: Callable
):
    """
    Build a simple single-node graph.
    
    Useful for simple agents that don't need planning/reflection.
    
    Args:
        state_class: TypedDict class for state
        processor: Single node that processes the request
        
    Returns:
        Compiled LangGraph application
    """
    workflow = StateGraph(state_class)
    
    workflow.add_node("processor", processor)
    workflow.add_edge(START, "processor")
    workflow.add_edge("processor", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
