"""
Graph Builder
=============

Utilities for constructing LangGraph workflows.
"""

from typing import Callable, Type, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from point9_platform.graph.nodes import should_continue


def build_default_graph(
    agent,
    planner: Callable,
    executor: Callable,
    reflector: Callable,
    responder: Callable
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
        
    Returns:
        Compiled LangGraph application
    """
    # Get state class from agent
    sample_state = agent.create_initial_state(agent.session_id)
    state_class = type(sample_state)
    
    # Create the state graph
    workflow = StateGraph(state_class)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("reflector", reflector)
    workflow.add_node("responder", responder)
    
    # Define edges
    workflow.set_entry_point("planner")
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
    workflow.set_entry_point("processor")
    workflow.add_edge("processor", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
