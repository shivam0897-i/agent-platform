"""
Default Graph Nodes
===================

Default implementations for planner, executor, reflector, responder nodes.
"""

import json
import logging
import re
from typing import Dict, Any, Literal, Callable

from point9_platform.observability.emitter import get_session_emitter, StepType, StepStatus

logger = logging.getLogger(__name__)


def create_default_planner(llm, planner_prompt: str = None) -> Callable:
    """
    Creates the planning node.
    
    The planner:
    1. Analyzes user request
    2. Creates execution plan
    3. Validates domain relevance
    """
    
    def planner(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        session_id = state.get("session_id", "")
        
        # Build planning prompt
        system_prompt = planner_prompt or _get_default_planner_prompt()
        
        # Prepare messages for LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": _get_role(m), "content": m.get("content", "")} 
              for m in messages[-5:]]
        ]
        
        # Emit planning step
        emitter = get_session_emitter(session_id)
        if emitter:
            emitter.emit_blocking(
                StepType.AGENT_PLANNING,
                "Analyzing request and creating plan...",
                status=StepStatus.IN_PROGRESS,
                progress=10
            )
        
        try:
            response = llm.completion(
                messages=llm_messages,
                model=state.get("model")
            )
            content = response.choices[0].message.content
            
            # Parse plan from response
            plan_data = _parse_json_from_response(content)
            plan = plan_data.get("plan", [])
            thoughts = [
                plan_data.get("task_understanding", ""),
                plan_data.get("reasoning", "")
            ]
            
            # Emit plan created
            if emitter:
                plan_summary = f"Plan: {', '.join(plan[:3])}" + ("..." if len(plan) > 3 else "")
                emitter.emit_blocking(
                    StepType.AGENT_PLANNING,
                    plan_summary,
                    status=StepStatus.COMPLETED,
                    progress=15,
                    details={"plan": plan}
                )
                
        except Exception as e:
            logger.error(f"Planning error: {e}")
            plan = ["Process request", "Validate results", "Provide summary"]
            thoughts = [f"Default plan due to error: {e}"]
        
        return {
            "plan": plan,
            "current_step": 0,
            "thoughts": state.get("thoughts", []) + thoughts
        }
    
    return planner


def create_default_executor(llm, tool_registry) -> Callable:
    """
    Creates the execution node.
    
    The executor:
    1. Takes current step from plan
    2. Selects appropriate tool via LLM
    3. Executes tool with retry logic
    """
    from point9_platform.tools.executor import ToolExecutor
    
    def executor(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        session_id = state.get("session_id", "")
        
        # Check if plan is complete
        if current_step >= len(plan):
            return {"should_continue": False}
        
        current_task = plan[current_step]
        
        # Get tool definitions
        tool_executor = ToolExecutor(state, tool_registry)
        tools = tool_executor.get_tool_definitions()
        
        exec_prompt = _get_default_executor_prompt(current_task, current_step + 1, len(plan))
        
        exec_messages = [
            {"role": "system", "content": exec_prompt},
            {"role": "user", "content": f"Execute: {current_task}"}
        ]
        
        emitter = get_session_emitter(session_id)
        
        try:
            response = llm.completion(
                messages=exec_messages,
                model=state.get("model"),
                tools=tools if tools else None,
                tool_choice="auto" if tools else None
            )
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError(f"LLM returned invalid response: {response}")
            
            assistant_msg = response.choices[0].message
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            
            results = dict(state.get("results", {}))
            new_messages = [{"role": "assistant", "content": assistant_msg.content or ""}]
            
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    
                    logger.info(f"Executing tool: {tool_name}")
                    
                    tool_result = tool_executor.execute(
                        tool_name=tool_name,
                        args=tool_args,
                        emitter=emitter
                    )
                    
                    # Store result by document_id if present, otherwise by tool_name
                    if "document_id" in tool_result:
                        results[tool_result["document_id"]] = tool_result
                    else:
                        # Fallback: store by tool name
                        results[tool_name] = tool_result
                    
                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result)
                    })
            
            return {
                "messages": new_messages,
                "results": results,
                "current_step": current_step + 1,
                "current_task": current_task,
                "iteration": state.get("iteration", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            if emitter:
                emitter.emit_blocking(
                    StepType.ERROR,
                    f"Execution failed: {str(e)[:100]}",
                    status=StepStatus.FAILED
                )
            return {
                "error": str(e),
                "current_step": current_step + 1,
                "iteration": state.get("iteration", 0) + 1
            }
    
    return executor


def create_default_reflector(llm) -> Callable:
    """
    Creates the reflection node.
    
    The reflector:
    1. Checks termination conditions
    2. Handles errors
    3. Decides whether to continue or respond
    """
    
    def reflector(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        error = state.get("error")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        
        # Check termination conditions
        if iteration >= max_iterations:
            return {
                "should_continue": False,
                "thoughts": state.get("thoughts", []) + ["Max iterations reached"]
            }
        
        if current_step >= len(plan):
            return {
                "should_continue": False,
                "thoughts": state.get("thoughts", []) + ["Plan completed successfully"]
            }
        
        if error:
            return {
                "should_continue": True,
                "error": None,
                "thoughts": state.get("thoughts", []) + [f"Recovered from error: {error}"]
            }
        
        return {"should_continue": True}
    
    return reflector


def create_default_responder(llm, responder_prompt: str = None) -> Callable:
    """
    Creates the response node.
    
    The responder:
    1. Formats results for user
    2. Generates human-readable response
    """
    
    def responder(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        results = state.get("results", {})
        session_id = state.get("session_id", "")
        
        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" or msg.get("type") == "human":
                user_message = msg.get("content", "")
                break
        
        system_prompt = responder_prompt or _get_default_responder_prompt(results)
        
        resp_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = llm.completion(
                messages=resp_messages,
                model=state.get("model")
            )
            final_response = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            final_response = _generate_fallback_response(results)
        
        # Emit completion
        emitter = get_session_emitter(session_id)
        if emitter:
            emitter.complete_blocking(
                success=True,
                final_message="Processing complete",
                result_data={"message": final_response, "results": results}
            )
        
        return {
            "messages": [{"role": "assistant", "content": final_response}]
        }
    
    return responder


def should_continue(state: Dict[str, Any]) -> Literal["executor", "responder"]:
    """Determine whether to continue execution or respond"""
    if state.get("should_continue", True) and state.get("current_step", 0) < len(state.get("plan", [])):
        return "executor"
    return "responder"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_role(msg: Dict) -> str:
    """Convert message to LLM role"""
    msg_type = msg.get("type", msg.get("role", "user"))
    if msg_type in ("human", "user"):
        return "user"
    elif msg_type in ("ai", "assistant"):
        return "assistant"
    return "user"


def _parse_json_from_response(content: str) -> Dict[str, Any]:
    """Extract JSON from LLM response"""
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {"plan": [], "task_understanding": content, "reasoning": ""}


def _get_default_planner_prompt() -> str:
    return """You are a planning agent. Analyze the user's request and create an execution plan.

Return JSON with:
{
    "task_understanding": "Brief summary of what user wants",
    "reasoning": "Why this plan makes sense",
    "plan": ["Step 1", "Step 2", "Step 3"]
}"""


def _get_default_executor_prompt(task: str, step: int, total: int) -> str:
    return f"""You are an execution agent. Execute step {step}/{total}.

Current task: {task}

Use the available tools to complete this task. Be precise and efficient."""


def _get_default_responder_prompt(results: Dict) -> str:
    result_summary = json.dumps(results, indent=2)[:1000] if results else "No results"
    return f"""Generate a clear, helpful response for the user.

Results available:
{result_summary}

Be concise and informative."""


def _generate_fallback_response(results: Dict[str, Any]) -> str:
    """Generate fallback response if LLM fails"""
    if not results:
        return "Processing completed but no results were generated."
    return f"Processed {len(results)} item(s). Results are available."
