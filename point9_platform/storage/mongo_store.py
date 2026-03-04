"""
MongoDB Session Store
=====================

Handles session CRUD, processing logs, and chat history for Point9 agents.

Usage:
    from point9_platform.storage import MongoStore, get_mongo_store
    
    # Using singleton
    store = get_mongo_store()
    store.create_session("session-123")
    store.add_log("session-123", "processing", "Started analysis")
    
    # Or create custom instance
    store = MongoStore(database="my_agent")

Environment Variables:
    MONGODB_URI: Connection string (default: mongodb://localhost:27017)
    MONGODB_DB: Database name (default: point9_agent)
    MONGODB_COLLECTION: Collection name (default: sessions)
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class MongoStore:
    """
    MongoDB storage client for Point9 agent sessions.
    
    Handles:
    - Session CRUD operations
    - Processing logs
    - Intermediate tool results
    - Chat history
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: Optional[str] = None
    ):
        """
        Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI (or from env MONGODB_URI)
            database: Database name (or from env MONGODB_DB)
            collection: Collection name (or from env MONGODB_COLLECTION)
        """
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = database or os.getenv("MONGODB_DB", "point9_agent")
        self.collection_name = collection or os.getenv("MONGODB_COLLECTION", "sessions")
        
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.sessions: Collection = self.db[self.collection_name]
        
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create indexes for efficient queries."""
        try:
            self.sessions.create_index("session_id", unique=True)
            self.sessions.create_index([("created_at", DESCENDING)])
            self.sessions.create_index([("status", 1), ("created_at", DESCENDING)])
            logger.debug("MongoDB indexes ensured")
        except Exception as e:
            logger.warning("Index creation warning: %s", e)
    
    # ==================== Session CRUD ====================
    
    def create_session(
        self,
        session_id: str,
        input_files: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            input_files: List of input file metadata
            metadata: Additional session metadata
            
        Returns:
            Created session document
        """
        now = datetime.now(timezone.utc)
        
        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": "created",
            "input_files": input_files or [],
            "output_s3_key": None,
            "intermediate_results": {},
            "metadata": metadata or {},
            "logs": [
                {"ts": now, "level": "info", "step": "init", "msg": "Session created"}
            ],
            "chat_history": [],
            "error": None
        }
        
        try:
            result = self.sessions.insert_one(session)
            session["_id"] = str(result.inserted_id)
            logger.info("[%s] Session created", session_id)
            return session
            
        except Exception as e:
            logger.error("[%s] Failed to create session: %s", session_id, e)
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by session_id."""
        session = self.sessions.find_one({"session_id": session_id})
        if session:
            session["_id"] = str(session["_id"])
        return session
    
    def update_session(self, session_id: str, update: Dict[str, Any]) -> bool:
        """
        Update session fields.
        
        Args:
            session_id: Session identifier
            update: Fields to update
            
        Returns:
            True if updated, False otherwise
        """
        update["updated_at"] = datetime.now(timezone.utc)
        
        result = self.sessions.update_one(
            {"session_id": session_id},
            {"$set": update}
        )
        
        return result.modified_count > 0
    
    def update_status(self, session_id: str, status: str) -> bool:
        """Update session status."""
        return self.update_session(session_id, {"status": status})
    
    def set_output(self, session_id: str, s3_key: str) -> bool:
        """Set output S3 key after results are uploaded."""
        return self.update_session(session_id, {"output_s3_key": s3_key})
    
    def set_error(self, session_id: str, error: str) -> bool:
        """Set session error and mark as failed."""
        return self.update_session(session_id, {
            "status": "failed",
            "error": error
        })
    
    def list_sessions(
        self,
        limit: int = 10,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent sessions.
        
        Args:
            limit: Maximum number of sessions
            status: Optional status filter
            
        Returns:
            List of session documents
        """
        query = {}
        if status:
            query["status"] = status
        
        sessions = list(
            self.sessions.find(query)
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        
        for s in sessions:
            s["_id"] = str(s["_id"])
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        result = self.sessions.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    
    # ==================== Logging ====================
    
    def add_log(
        self,
        session_id: str,
        step: str,
        msg: str,
        level: str = "info"
    ) -> bool:
        """
        Add a log entry to the session.
        
        Args:
            session_id: Session identifier
            step: Processing step (e.g., 'analyze_image', 'upload')
            msg: Log message
            level: Log level ('info', 'warn', 'error')
            
        Returns:
            True if added, False otherwise
        """
        log_entry = {
            "ts": datetime.now(timezone.utc),
            "level": level,
            "step": step,
            "msg": msg
        }
        
        result = self.sessions.update_one(
            {"session_id": session_id},
            {"$push": {"logs": log_entry}}
        )
        
        return result.modified_count > 0
    
    def get_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a session."""
        session = self.get_session(session_id)
        return session.get("logs", []) if session else []
    
    # ==================== Intermediate Results ====================
    
    def store_result(
        self,
        session_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Store intermediate result from a tool execution.
        
        Args:
            session_id: Session identifier
            tool_name: Tool name (e.g., 'analyze_image')
            result: Tool execution result
            
        Returns:
            True if stored, False otherwise
        """
        result_with_meta = {
            "status": result.get("status", "success"),
            "executed_at": datetime.now(timezone.utc),
            **result
        }
        
        update_result = self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {f"intermediate_results.{tool_name}": result_with_meta}}
        )
        
        if update_result.modified_count > 0:
            logger.debug("[%s] Stored %s result", session_id, tool_name)
        
        return update_result.modified_count > 0
    
    def get_results(self, session_id: str) -> Dict[str, Any]:
        """Get all intermediate results for a session."""
        session = self.get_session(session_id)
        return session.get("intermediate_results", {}) if session else {}
    
    def get_result(self, session_id: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get specific tool result."""
        results = self.get_results(session_id)
        return results.get(tool_name)
    
    # ==================== Chat History ====================
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Add a chat message to session history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            True if added, False otherwise
        """
        message = {
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc)
        }
        
        result = self.sessions.update_one(
            {"session_id": session_id},
            {"$push": {"chat_history": message}}
        )
        
        return result.modified_count > 0
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        session = self.get_session(session_id)
        return session.get("chat_history", []) if session else []
    
    # ==================== Cleanup ====================
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


# Singleton instance
_mongo_instance: Optional[MongoStore] = None


def get_mongo_store() -> MongoStore:
    """Get or create MongoDB store singleton instance."""
    global _mongo_instance
    if _mongo_instance is None:
        _mongo_instance = MongoStore()
    return _mongo_instance
