"""
Step Emitter
============

SSE Real-Time Streaming for execution progress.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of an execution step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """
    Types of execution steps.
    
    These are generic step types. For domain-specific types,
    just use plain strings in the `message` field instead.
    """
    # Agent lifecycle
    AGENT_PLANNING = "agent_planning"
    AGENT_THINKING = "agent_thinking"
    AGENT_EXECUTING = "agent_executing"
    
    # Tool execution
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    
    # Completion
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ExecutionStep:
    """Represents a single execution step"""
    step_type: StepType
    status: StepStatus
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    details: Optional[Dict[str, Any]] = None
    progress: Optional[int] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "progress": self.progress,
            "duration_ms": self.duration_ms
        }
    
    def to_sse(self) -> str:
        """Convert to SSE format"""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class StepEmitter:
    """
    Manages execution steps for a session.
    Supports real-time streaming via SSE.
    """
    
    def __init__(self, process_id: str):
        self.process_id = process_id
        self.steps: List[ExecutionStep] = []
        self._subscribers: List[asyncio.Queue] = []
        self._completed = False
    
    def reset(self):
        """Reset for a new request"""
        self.steps = []
        self._completed = False
    
    def emit(
        self,
        step_type: StepType,
        message: str,
        status: StepStatus = StepStatus.COMPLETED,
        details: Optional[Dict[str, Any]] = None,
        progress: Optional[int] = None
    ):
        """Emit a step (async-compatible)"""
        step = ExecutionStep(
            step_type=step_type,
            status=status,
            message=message,
            details=details,
            progress=progress
        )
        
        self.steps.append(step)
        
        # Notify subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(step)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {self.process_id}")
    
    def emit_blocking(
        self,
        step_type: StepType,
        message: str,
        status: StepStatus = StepStatus.COMPLETED,
        details: Optional[Dict[str, Any]] = None,
        progress: Optional[int] = None
    ):
        """Emit from synchronous code (thread-safe)"""
        step = ExecutionStep(
            step_type=step_type,
            status=status,
            message=message,
            details=details,
            progress=progress
        )
        
        self.steps.append(step)
        logger.info(f"[{self.process_id}] {step_type.value}: {message}")
        
        # Notify subscribers in thread-safe way using call_soon_threadsafe
        self._notify_subscribers_threadsafe(step)
    
    def _notify_subscribers_threadsafe(self, step, end_signal: bool = False):
        """Thread-safe notification of subscribers"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - try get_event_loop for backwards compat
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop = None
            except:
                loop = None
        
        for queue in self._subscribers:
            try:
                if loop and loop.is_running():
                    # Thread-safe: schedule on event loop
                    loop.call_soon_threadsafe(queue.put_nowait, step)
                    if end_signal:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                else:
                    # Fallback for same-thread calls
                    queue.put_nowait(step)
                    if end_signal:
                        queue.put_nowait(None)
            except Exception as e:
                logger.debug(f"Failed to notify subscriber: {e}")
    
    def complete_blocking(
        self,
        success: bool = True,
        final_message: str = None,
        result_data: Dict[str, Any] = None
    ):
        """Mark process as complete (thread-safe)"""
        self._completed = True
        
        step_type = StepType.COMPLETE if success else StepType.ERROR
        message = final_message or ("Processing complete" if success else "Processing failed")
        
        step = ExecutionStep(
            step_type=step_type,
            status=StepStatus.COMPLETED if success else StepStatus.FAILED,
            message=message,
            details={"result_data": result_data} if result_data else None,
            progress=100 if success else None
        )
        
        self.steps.append(step)
        
        # Thread-safe notification with end signal
        self._notify_subscribers_threadsafe(step, end_signal=True)
    
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to step updates"""
        queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        
        # Send existing steps
        for step in self.steps:
            try:
                queue.put_nowait(step)
            except asyncio.QueueFull:
                break
        
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from updates"""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
    
    async def stream(self):
        """Async generator for SSE streaming"""
        queue = self.subscribe()
        try:
            while True:
                step = await queue.get()
                if step is None:
                    break
                yield step.to_sse()
        finally:
            self.unsubscribe(queue)
    
    def get_all_steps(self) -> List[Dict]:
        """Get all steps as dictionaries"""
        return [s.to_dict() for s in self.steps]
    
    def is_completed(self) -> bool:
        return self._completed


# =============================================================================
# Global Registry
# =============================================================================

_emitters: Dict[str, StepEmitter] = {}


def get_or_create_emitter(process_id: str) -> StepEmitter:
    """Get existing emitter or create new one"""
    if process_id not in _emitters:
        _emitters[process_id] = StepEmitter(process_id)
    return _emitters[process_id]


def get_emitter(process_id: str) -> Optional[StepEmitter]:
    """Get emitter if exists"""
    return _emitters.get(process_id)


def set_session_emitter(session_id: str, emitter: StepEmitter):
    """Store emitter for a session"""
    _emitters[session_id] = emitter


def get_session_emitter(session_id: str) -> Optional[StepEmitter]:
    """Get emitter for a session"""
    return _emitters.get(session_id)


def remove_emitter(process_id: str):
    """Remove emitter when done"""
    if process_id in _emitters:
        del _emitters[process_id]


def list_active_processes() -> List[str]:
    """List all active process IDs"""
    return list(_emitters.keys())
