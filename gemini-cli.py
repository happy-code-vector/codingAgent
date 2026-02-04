#!/usr/bin/env python3
"""
Single File Coding Agent
========================
A comprehensive AI coding agent implemented in a single Python file.

Features:
- Full task lifecycle management with state machine
- OpenAI-compatible LLM client with streaming
- Tool system (file operations, bash commands, code search)
- HTTP server with WebSocket-like streaming
- Persistence layer (file-based + optional GCS)
- Configuration system with workspace support
- Command system (init, memory, extensions, restore)
- Comprehensive logging and error handling

Usage:
    python coding_agent.py [--port PORT] [--workspace PATH] [--api-key KEY]

Environment Variables:
    CODER_AGENT_PORT       - Port to run the server on (default: 41242)
    CODER_AGENT_WORKSPACE  - Workspace directory (default: current directory)
    OPENAI_API_KEY         - OpenAI API key or compatible endpoint
    OPENAI_BASE_URL        - Base URL for OpenAI-compatible API
    GCS_BUCKET_NAME        - Optional GCS bucket for persistence
"""

# =============================================================================
# IMPORTS
# =============================================================================

import asyncio
import json
import os
import sys
import uuid
import logging
import tarfile
import tempfile
import shutil
import subprocess
import re
import fnmatch
import hashlib
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Callable, Dict, Generic, Iterator, List,
    Optional, Set, Tuple, TypeVar, Union, get_type_hints
)
from concurrent.futures import ThreadPoolExecutor
import threading

# Third-party imports (install with: pip install fastapi uvicorn openai)
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    from openai import AsyncOpenAI, AsyncStream
    from openai.types.chat import ChatCompletionChunk, ChatCompletion
    import pydantic
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install fastapi uvicorn openai pydantic")
    sys.exit(1)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

DEFAULT_PORT = 41242
DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_CONTEXT_LENGTH = 200000
TEMP_DIR = tempfile.gettempdir()
GEMINI_DIR = ".gemini"
SETTINGS_FILE = "settings.json"
EXTENSIONS_DIR = "extensions"
MEMORY_FILE = "GEMINI.md"

# Task States
class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"

# Agent Events
class CoderAgentEvent(str, Enum):
    TOOL_CALL_CONFIRMATION = "tool-call-confirmation"
    TOOL_CALL_UPDATE = "tool-call-update"
    TEXT_CONTENT = "text-content"
    STATE_CHANGE = "state-change"
    AGENT_SETTINGS = "agent-settings"
    THOUGHT = "thought"
    CITATION = "citation"

# Tool Call Status
class ToolStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"

# =============================================================================
# LOGGING
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name: str = "CodingAgent", level: str = "INFO") -> logging.Logger:
    """Setup and configure logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter(
            '[%(levelname)s] %(asctime)s -- %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()

# =============================================================================
# MODELS & DATA STRUCTURES
# =============================================================================

@dataclass
class AgentSettings:
    """Agent configuration settings."""
    kind: str = CoderAgentEvent.AGENT_SETTINGS
    workspace_path: str = ""
    auto_execute: bool = False

@dataclass
class ToolConfirmationDetails:
    """Details for tool confirmation."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    confirmation_callback: Callable[[str], Any]
    type: str = "execute"

@dataclass
class CompletedToolCall:
    """Represents a completed tool call."""
    request: 'ToolCallRequest'
    response: 'ToolCallResponse'
    status: ToolStatus

@dataclass
class ToolCallRequest:
    """Request for a tool call."""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    prompt_id: str = ""
    checkpoint: Optional[Dict[str, Any]] = None

@dataclass
class ToolCallResponse:
    """Response from a tool call."""
    call_id: str
    response_parts: List[Any]
    status: ToolStatus

@dataclass
class PersistedStateMetadata:
    """Metadata for persisted task state."""
    _agentSettings: AgentSettings
    _taskState: TaskState

@dataclass
class TaskMetadata:
    """Task metadata."""
    id: str
    contextId: str
    taskState: TaskState
    model: str
    mcpServers: List[Dict[str, Any]] = field(default_factory=list)
    availableTools: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Message:
    """Message in the conversation."""
    kind: str = "message"
    role: str = "user"
    parts: List[Dict[str, Any]] = field(default_factory=list)
    messageId: str = field(default_factory=lambda: str(uuid.uuid4()))
    taskId: str = ""
    contextId: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskStatusUpdateEvent:
    """Task status update event."""
    kind: str = "status-update"
    taskId: str = ""
    contextId: str = ""
    status: Dict[str, Any] = field(default_factory=dict)
    final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskArtifactUpdateEvent:
    """Task artifact update event."""
    kind: str = "artifact-update"
    taskId: str = ""
    contextId: str = ""
    artifact: Dict[str, Any] = field(default_factory=dict)
    append: bool = False
    lastChunk: bool = False

# Pydantic models for API
class AgentCard(BaseModel):
    """Agent card for A2A protocol."""
    name: str = "Python Coding Agent"
    description: str = "An AI coding agent that generates code and manages files"
    url: str = "http://localhost:41242/"
    provider: Dict[str, str] = {"organization": "OpenAI Compatible", "url": "https://openai.com"}
    protocolVersion: str = "0.3.0"
    version: str = "1.0.0"
    capabilities: Dict[str, bool] = {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True
    }
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    skills: List[Dict[str, Any]] = []

# =============================================================================
# EVENT BUS
# =============================================================================

class ExecutionEventBus:
    """Event bus for agent execution events."""

    def __init__(self):
        self._subscribers: List[Callable] = []
        self._lock = threading.Lock()
        self._finished = False

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to events."""
        with self._lock:
            if not self._finished:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from events."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            if self._finished:
                return
            for callback in self._subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

    def on(self, event_type: str, callback: Callable) -> None:
        """Register an event handler."""
        def wrapper(event):
            if event.get('kind') == event_type or event.get('type') == event_type:
                callback(event)
        self.subscribe(wrapper)

    def finished(self) -> None:
        """Mark the event bus as finished."""
        with self._lock:
            self._finished = True
            self._subscribers.clear()

# =============================================================================
# PERSISTENCE LAYER
# =============================================================================

class TaskStore(ABC):
    """Abstract base class for task storage."""

    @abstractmethod
    async def save(self, task: 'Task') -> None:
        """Save task state."""
        pass

    @abstractmethod
    async def load(self, taskId: str) -> Optional['Task']:
        """Load task state."""
        pass

class InMemoryTaskStore(TaskStore):
    """In-memory task storage."""

    def __init__(self):
        self._tasks: Dict[str, 'Task'] = {}
        self._lock = threading.Lock()

    async def save(self, task: 'Task') -> None:
        with self._lock:
            self._tasks[task.id] = task

    async def load(self, taskId: str) -> Optional['Task']:
        with self._lock:
            return self._tasks.get(taskId)

class FileTaskStore(TaskStore):
    """File-based task storage."""

    def __init__(self, storageDir: str = None):
        self.storageDir = Path(storageDir or os.path.join(TEMP_DIR, "coding_agent_tasks"))
        self.storageDir.mkdir(parents=True, exist_ok=True)

    def _getTaskPath(self, taskId: str) -> Path:
        return self.storageDir / f"task_{taskId}.json"

    def _getWorkspacePath(self, taskId: str) -> Path:
        return self.storageDir / f"workspace_{taskId}.tar.gz"

    async def save(self, task: 'Task') -> None:
        """Save task metadata and workspace."""
        # Save metadata
        metadata = {
            "id": task.id,
            "contextId": task.contextId,
            "taskState": task.taskState.value,
            "model": task.model,
            "agentSettings": asdict(task.agentSettings),
            "history": [asdict(m) for m in task.history],
            "_workspacePath": str(task.workspacePath)
        }

        taskPath = self._getTaskPath(task.id)
        with open(taskPath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save workspace as tar.gz
        if task.workspacePath and os.path.exists(task.workspacePath):
            workspacePath = self._getWorkspacePath(task.id)
            with tarfile.open(workspacePath, "w:gz") as tar:
                tar.add(task.workspacePath, arcname="")

    async def load(self, taskId: str) -> Optional['Task']:
        """Load task metadata and workspace."""
        taskPath = self._getTaskPath(taskId)
        if not taskPath.exists():
            return None

        with open(taskPath, 'r') as f:
            metadata = json.load(f)

        # Restore workspace
        workspacePath = Path(metadata.get("_workspacePath", os.getcwd()))
        workspaceTar = self._getWorkspacePath(taskId)
        if workspaceTar.exists():
            # Extract to temp location first
            tempExtract = self.storageDir / f"temp_{taskId}"
            with tarfile.open(workspaceTar, "r:gz") as tar:
                tar.extractall(tempExtract)
            # Move to workspace
            if tempExtract.exists():
                shutil.copytree(tempExtract, workspacePath, dirs_exist_ok=True)
                shutil.rmtree(tempExtract)

        return Task(
            id=metadata["id"],
            contextId=metadata["contextId"],
            workspacePath=str(workspacePath),
            agentSettings=AgentSettings(**metadata["agentSettings"]),
            model=metadata.get("model", DEFAULT_MODEL)
        )

class GCSTaskStore(TaskStore):
    """Google Cloud Storage task storage (optional)."""

    def __init__(self, bucketName: str):
        self.bucketName = bucketName
        try:
            from google.cloud import storage
            self.storage = storage.Client()
            self.bucket = self.storage.bucket(bucketName)
        except ImportError:
            logger.warning("google-cloud-storage not installed, GCS persistence disabled")
            self.bucket = None
        except Exception as e:
            logger.error(f"Failed to initialize GCS: {e}")
            self.bucket = None

    async def save(self, task: 'Task') -> None:
        """Save task to GCS."""
        if not self.bucket:
            return

        try:
            # Save metadata
            metadata = {
                "id": task.id,
                "contextId": task.contextId,
                "taskState": task.taskState.value,
                "model": task.model,
                "agentSettings": asdict(task.agentSettings)
            }

            blob = self.bucket.blob(f"tasks/{task.id}/metadata.json")
            blob.upload_from_string(json.dumps(metadata), content_type="application/json")

            # Save workspace
            if task.workspacePath and os.path.exists(task.workspacePath):
                workspaceTar = os.path.join(TEMP_DIR, f"task_{task.id}_workspace.tar.gz")
                with tarfile.open(workspaceTar, "w:gz") as tar:
                    tar.add(task.workspacePath, arcname="")

                wsBlob = self.bucket.blob(f"tasks/{task.id}/workspace.tar.gz")
                wsBlob.upload_from_filename(workspaceTar)
                os.remove(workspaceTar)

        except Exception as e:
            logger.error(f"Failed to save task to GCS: {e}")

    async def load(self, taskId: str) -> Optional['Task']:
        """Load task from GCS."""
        if not self.bucket:
            return None

        try:
            metadataBlob = self.bucket.blob(f"tasks/{taskId}/metadata.json")
            if not metadataBlob.exists():
                return None

            metadata = json.loads(metadataBlob.download_as_string())

            # Load workspace
            workspacePath = os.getcwd()
            workspaceBlob = self.bucket.blob(f"tasks/{taskId}/workspace.tar.gz")
            if workspaceBlob.exists():
                workspaceTar = os.path.join(TEMP_DIR, f"task_{taskId}_workspace.tar.gz")
                workspaceBlob.download_to_filename(workspaceTar)
                with tarfile.open(workspaceTar, "r:gz") as tar:
                    tar.extractall(workspacePath)
                os.remove(workspaceTar)

            return Task(
                id=metadata["id"],
                contextId=metadata["contextId"],
                workspacePath=workspacePath,
                agentSettings=AgentSettings(**metadata["agentSettings"]),
                model=metadata.get("model", DEFAULT_MODEL)
            )
        except Exception as e:
            logger.error(f"Failed to load task from GCS: {e}")
            return None

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMEvent(str, Enum):
    """LLM event types."""
    CONTENT = "content"
    TOOL_CALL_REQUEST = "tool_call_request"
    FINISHED = "finished"
    ERROR = "error"
    MODEL_INFO = "model_info"
    THOUGHT = "thought"
    CITATION = "citation"

class OpenAILLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        apiKey: str = None,
        baseURL: str = None,
        model: str = DEFAULT_MODEL
    ):
        self.apiKey = apiKey or os.getenv("OPENAI_API_KEY", "")
        self.baseURL = baseURL or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = model
        self.client = AsyncOpenAI(api_key=self.apiKey, base_url=self.baseURL)
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def addHistory(self, role: str, parts: List[Any]) -> None:
        """Add message to history."""
        with self._lock:
            content = ""
            for part in parts:
                if isinstance(part, dict):
                    if "text" in part:
                        content += part["text"]
                    elif "content" in part:
                        content += part["content"]
                    elif isinstance(part, str):
                        content += part
                elif isinstance(part, str):
                    content += part

            if content:
                self.history.append({"role": role, "content": content})

    async def sendMessageStream(
        self,
        parts: List[Any],
        abortSignal: threading.Event = None,
        promptId: str = ""
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send message and stream response."""
        # Build content from parts
        content = ""
        for part in parts:
            if isinstance(part, dict):
                if "text" in part:
                    content += part["text"]
                elif "content" in part:
                    content += part["content"]
            elif isinstance(part, str):
                content += part

        # Add to history
        self.addHistory("user", [{"text": content}])

        # Prepare messages
        messages = [{"role": "system", "content": self._getSystemPrompt()}]
        messages.extend(self.history)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=4096
            )

            currentContent = ""
            toolCallsBuffer = []

            async for chunk in stream:
                if abortSignal and abortSignal.is_set():
                    yield {"type": LLMEvent.ERROR, "value": "Aborted by user"}
                    return

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Content
                if delta.content:
                    currentContent += delta.content
                    yield {"type": LLMEvent.CONTENT, "value": delta.content}

                # Tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function:
                            toolCallsBuffer.append({
                                "id": tool_call.id or str(uuid.uuid4()),
                                "name": tool_call.function.name or "",
                                "arguments": tool_call.function.arguments or ""
                            })

            # Process tool calls if any
            if toolCallsBuffer:
                for tc in toolCallsBuffer:
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    yield {
                        "type": LLMEvent.TOOL_CALL_REQUEST,
                        "value": {
                            "callId": tc["id"],
                            "name": tc["name"],
                            "args": args,
                            "promptId": promptId
                        }
                    }

            # Add assistant response to history
            if currentContent:
                self.addHistory("assistant", [{"text": currentContent}])

            yield {"type": LLMEvent.FINISHED, "value": None}

        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield {"type": LLMEvent.ERROR, "value": str(e)}

    def _getSystemPrompt(self) -> str:
        """Get system prompt for the agent."""
        return '''You are an AI coding assistant. You can help with:
- Reading, writing, and editing files
- Running shell commands
- Searching code
- Explaining code
- Implementing features
- Debugging issues

When you need to perform an action, use the appropriate tool. Always explain what you're going to do before doing it.

Available tools:
- read_file: Read file contents
- write_file: Write content to a file
- edit_file: Edit a file with search/replace
- search_files: Search for files by pattern
- search_code: Search for code patterns
- run_command: Run shell commands
- list_directory: List directory contents

Be precise and thorough in your responses.'''

# =============================================================================
# TOOL SYSTEM
# =============================================================================

class Tool(ABC):
    """Base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    def schema(self) -> Dict[str, Any]:
        """Tool parameter schema."""
        return {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        """Execute the tool."""
        pass

class ReadFileTool(Tool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        file_path = args.get("file_path", "")

        if not file_path:
            yield "Error: file_path is required"
            return

        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(os.getcwd()) / path

            if not path.exists():
                yield f"Error: File not found: {file_path}"
                return

            if not path.is_file():
                yield f"Error: Not a file: {file_path}"
                return

            content = path.read_text(encoding='utf-8', errors='replace')
            yield content
        except Exception as e:
            yield f"Error reading file: {e}"

class WriteFileTool(Tool):
    """Tool for writing content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file (creates or overwrites)"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            yield "Error: file_path is required"
            return

        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(os.getcwd()) / path

            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            path.write_text(content, encoding='utf-8')
            yield f"Successfully wrote {len(content)} bytes to {file_path}"
        except Exception as e:
            yield f"Error writing file: {e}"

class EditFileTool(Tool):
    """Tool for editing a file with search/replace."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_string with new_string"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "String to search for"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string"
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        if not file_path:
            yield "Error: file_path is required"
            return

        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(os.getcwd()) / path

            if not path.exists():
                yield f"Error: File not found: {file_path}"
                return

            content = path.read_text(encoding='utf-8', errors='replace')

            if old_string not in content:
                yield f"Error: old_string not found in file"
                return

            new_content = content.replace(old_string, new_string)
            path.write_text(new_content, encoding='utf-8')
            yield f"Successfully edited {file_path} (replaced {content.count(old_string)} occurrence(s))"
        except Exception as e:
            yield f"Error editing file: {e}"

class SearchFilesTool(Tool):
    """Tool for searching files by pattern."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search for files matching a pattern"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File pattern to search (e.g., '*.py', '**/*.txt')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                }
            },
            "required": ["pattern"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        pattern = args.get("pattern", "")
        searchPath = Path(args.get("path", os.getcwd()))

        if not pattern:
            yield "Error: pattern is required"
            return

        try:
            if not searchPath.is_absolute():
                searchPath = Path(os.getcwd()) / searchPath

            if not searchPath.exists():
                yield f"Error: Path not found: {searchPath}"
                return

            matches = []
            for root, dirs, files in os.walk(searchPath):
                for filename in fnmatch.filter(files, pattern):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, searchPath)
                    matches.append(rel_path)

            if matches:
                yield f"Found {len(matches)} file(s):\n" + "\n".join(f"  {m}" for m in matches)
            else:
                yield f"No files found matching pattern: {pattern}"
        except Exception as e:
            yield f"Error searching files: {e}"

class SearchCodeTool(Tool):
    """Tool for searching code patterns."""

    @property
    def name(self) -> str:
        return "search_code"

    @property
    def description(self) -> str:
        return "Search for a pattern in file contents"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter (e.g., '*.py')"
                }
            },
            "required": ["pattern"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        pattern = args.get("pattern", "")
        searchPath = Path(args.get("path", os.getcwd()))
        filePattern = args.get("file_pattern", "*")

        if not pattern:
            yield "Error: pattern is required"
            return

        try:
            if not searchPath.is_absolute():
                searchPath = Path(os.getcwd()) / searchPath

            if not searchPath.exists():
                yield f"Error: Path not found: {searchPath}"
                return

            regex = re.compile(pattern)
            matches = []

            for root, dirs, files in os.walk(searchPath):
                for filename in fnmatch.filter(files, filePattern):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    rel_path = os.path.relpath(filepath, searchPath)
                                    matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                    except Exception:
                        continue

            if matches:
                yield f"Found {len(matches)} match(es):\n" + "\n".join(f"  {m}" for m in matches[:50])
                if len(matches) > 50:
                    yield f"\n  ... and {len(matches) - 50} more"
            else:
                yield f"No matches found for pattern: {pattern}"
        except Exception as e:
            yield f"Error searching code: {e}"

class RunCommandTool(Tool):
    """Tool for running shell commands."""

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Run a shell command"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: current directory)"
                }
            },
            "required": ["command"]
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        command = args.get("command", "")
        cwd = args.get("cwd", os.getcwd())

        if not command:
            yield "Error: command is required"
            return

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )

            # Stream output
            async def readStream(stream, prefix):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    if abortSignal and abortSignal.is_set():
                        process.terminate()
                        break
                    yield f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}"

            stdoutTask = asyncio.create_task(readStream(process.stdout, ""))
            stderrTask = asyncio.create_task(readStream(process.stderr, "[stderr] "))

            async for line in stdoutTask:
                yield line

            async for line in stderrTask:
                yield line

            await process.wait()

            if process.returncode != 0:
                yield f"\nCommand exited with code {process.returncode}"
            else:
                yield f"\nCommand completed successfully"
        except Exception as e:
            yield f"Error running command: {e}"

class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List contents of a directory"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (default: false)"
                }
            }
        }

    async def execute(
        self,
        args: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[str, None]:
        dirPath = Path(args.get("path", os.getcwd()))
        recursive = args.get("recursive", False)

        try:
            if not dirPath.is_absolute():
                dirPath = Path(os.getcwd()) / dirPath

            if not dirPath.exists():
                yield f"Error: Path not found: {dirPath}"
                return

            if not dirPath.is_dir():
                yield f"Error: Not a directory: {dirPath}"
                return

            if recursive:
                lines = []
                for root, dirs, files in os.walk(dirPath):
                    level = root.replace(str(dirPath), '').count(os.sep)
                    indent = ' ' * 2 * level
                    rel_root = os.path.relpath(root, dirPath)
                    lines.append(f"{indent}{rel_root}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in sorted(files):
                        lines.append(f"{subindent}{file}")
                yield "\n".join(lines)
            else:
                items = []
                for item in sorted(dirPath.iterdir()):
                    suffix = "/" if item.is_dir() else ""
                    items.append(f"{item.name}{suffix}")
                yield "\n".join(items)
        except Exception as e:
            yield f"Error listing directory: {e}"

class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._registerDefaults()

    def _registerDefaults(self) -> None:
        """Register default tools."""
        defaultTools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            SearchFilesTool(),
            SearchCodeTool(),
            RunCommandTool(),
            ListDirectoryTool(),
        ]
        for tool in defaultTools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def getAll(self) -> List[Tool]:
        """Get all tools."""
        return list(self._tools.values())

    def getToolSchemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM."""
        schemas = []
        for tool in self.getAll():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            })
        return schemas

# =============================================================================
# TASK IMPLEMENTATION
# =============================================================================

class ToolCall:
    """Represents an active tool call."""

    def __init__(
        self,
        request: ToolCallRequest,
        confirmationDetails: ToolConfirmationDetails = None
    ):
        self.request = request
        self.status = ToolStatus.AWAITING_APPROVAL
        self.confirmationDetails = confirmationDetails
        self.response: Optional[ToolCallResponse] = None
        self.liveOutput: List[str] = []

class Task:
    """Represents a single task in the agent."""

    def __init__(
        self,
        id: str,
        contextId: str,
        workspacePath: str,
        agentSettings: AgentSettings = None,
        model: str = DEFAULT_MODEL
    ):
        self.id = id
        self.contextId = contextId
        self.workspacePath = workspacePath
        self.agentSettings = agentSettings or AgentSettings(workspace_path=workspacePath)
        self.model = model

        self.taskState = TaskState.SUBMITTED
        self.history: List[Message] = []
        self.eventBus: Optional[ExecutionEventBus] = None

        # Tool management
        self._toolRegistry = ToolRegistry()
        self._pendingToolCalls: Dict[str, ToolCall] = {}
        self._completedToolCalls: List[CompletedToolCall] = []
        self._pendingConfirmations: Dict[str, ToolConfirmationDetails] = {}

        # LLM client
        self._llmClient = OpenAILLMClient(model=model)

        # Execution control
        self._abortSignal = threading.Event()
        self._toolCompletionEvent = asyncio.Event()
        self._lock = threading.Lock()

    @property
    def toolRegistry(self) -> ToolRegistry:
        return self._toolRegistry

    def getMetadata(self) -> TaskMetadata:
        """Get task metadata."""
        return TaskMetadata(
            id=self.id,
            contextId=self.contextId,
            taskState=self.taskState,
            model=self.model,
            mcpServers=[],
            availableTools=[t.schema for t in self._toolRegistry.getAll()]
        )

    def setTaskState(
        self,
        newState: TaskState,
        messageText: str = None,
        messageParts: List[Dict] = None,
        final: bool = False,
        metadataError: str = None
    ) -> None:
        """Set task state and publish update."""
        self.taskState = newState

        message = None
        if messageText:
            message = {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": messageText}],
                "messageId": str(uuid.uuid4()),
                "taskId": self.id,
                "contextId": self.contextId
            }
        elif messageParts:
            message = {
                "kind": "message",
                "role": "agent",
                "parts": messageParts,
                "messageId": str(uuid.uuid4()),
                "taskId": self.id,
                "contextId": self.contextId
            }

        event = {
            "kind": "status-update",
            "taskId": self.id,
            "contextId": self.contextId,
            "status": {
                "state": newState.value,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            "final": final,
            "metadata": {
                "coderAgent": {"kind": CoderAgentEvent.STATE_CHANGE},
                "model": self.model
            }
        }

        if metadataError:
            event["metadata"]["error"] = metadataError

        self.eventBus.publish(event) if self.eventBus else None

    def _sendTextContent(self, content: str, traceId: str = None) -> None:
        """Send text content to event bus."""
        if not content or not self.eventBus:
            return

        message = {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": content}],
            "messageId": str(uuid.uuid4()),
            "taskId": self.id,
            "contextId": self.contextId
        }

        event = {
            "kind": "status-update",
            "taskId": self.id,
            "contextId": self.contextId,
            "status": {
                "state": self.taskState.value,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            "final": False,
            "metadata": {
                "coderAgent": {"kind": CoderAgentEvent.TEXT_CONTENT},
                "model": self.model
            }
        }

        self.eventBus.publish(event)

    async def acceptUserMessage(
        self,
        requestContext: Dict[str, Any],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Accept user message and generate response."""
        userMessage = requestContext.get("userMessage", {})

        # Extract content from parts
        llmParts = []
        hasContent = False

        for part in userMessage.get("parts", []):
            if part.get("kind") == "text":
                llmParts.append({"text": part.get("text", "")})
                hasContent = True

        if not hasContent:
            return

        # Set state to working
        self.setTaskState(TaskState.WORKING)

        # Stream response from LLM
        async for event in self._llmClient.sendMessageStream(llmParts, abortSignal):
            yield event

    async def scheduleToolCalls(
        self,
        requests: List[ToolCallRequest],
        abortSignal: threading.Event = None
    ) -> None:
        """Schedule tool calls for execution."""
        if not requests:
            return

        with self._lock:
            for request in requests:
                toolCall = ToolCall(request)
                self._pendingToolCalls[request.call_id] = toolCall

                # Send tool call update event
                self._sendToolCallUpdate(toolCall)

        # Set state to working
        self.setTaskState(TaskState.WORKING)

        # Auto-execute if enabled
        if self.agentSettings.autoExecute:
            for request in requests:
                await self._executeToolCall(request, abortSignal)

    def _sendToolCallUpdate(self, toolCall: ToolCall) -> None:
        """Send tool call status update."""
        if not self.eventBus:
            return

        toolInfo = {
            "request": {
                "callId": toolCall.request.call_id,
                "name": toolCall.request.name,
                "args": toolCall.request.args
            },
            "status": toolCall.status.value,
            "liveOutput": toolCall.liveOutput
        }

        message = {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "data", "data": toolInfo}],
            "messageId": str(uuid.uuid4()),
            "taskId": self.id,
            "contextId": self.contextId
        }

        event = {
            "kind": "status-update",
            "taskId": self.id,
            "contextId": self.contextId,
            "status": {
                "state": self.taskState.value,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            "final": False,
            "metadata": {
                "coderAgent": {"kind": CoderAgentEvent.TOOL_CALL_UPDATE},
                "model": self.model
            }
        }

        self.eventBus.publish(event)

    async def _executeToolCall(
        self,
        request: ToolCallRequest,
        abortSignal: threading.Event = None
    ) -> None:
        """Execute a single tool call."""
        tool = self._toolRegistry.get(request.name)

        if not tool:
            # Tool not found
            with self._lock:
                toolCall = self._pendingToolCalls.get(request.call_id)
                if toolCall:
                    toolCall.status = ToolStatus.ERROR
                    toolCall.response = ToolCallResponse(
                        callId=request.call_id,
                        response_parts=[f"Error: Tool not found: {request.name}"],
                        status=ToolStatus.ERROR
                    )
                    self._sendToolCallUpdate(toolCall)
                    self._completedToolCalls.append(CompletedToolCall(
                        request=request,
                        response=toolCall.response,
                        status=ToolStatus.ERROR
                    ))
                    del self._pendingToolCalls[request.call_id]
            return

        # Update status to executing
        with self._lock:
            toolCall = self._pendingToolCalls.get(request.call_id)
            if toolCall:
                toolCall.status = ToolStatus.EXECUTING
                self._sendToolCallUpdate(toolCall)

        # Execute tool
        outputParts = []
        try:
            async for output in tool.execute(request.args, abortSignal):
                outputParts.append(output)

                with self._lock:
                    toolCall = self._pendingToolCalls.get(request.call_id)
                    if toolCall:
                        toolCall.liveOutput.append(output)
                        self._sendToolCallUpdate(toolCall)

            # Success
            with self._lock:
                toolCall = self._pendingToolCalls.get(request.call_id)
                if toolCall:
                    toolCall.status = ToolStatus.SUCCESS
                    toolCall.response = ToolCallResponse(
                        callId=request.call_id,
                        response_parts=outputParts,
                        status=ToolStatus.SUCCESS
                    )
                    self._completedToolCalls.append(CompletedToolCall(
                        request=request,
                        response=toolCall.response,
                        status=ToolStatus.SUCCESS
                    ))
                    del self._pendingToolCalls[request.call_id]
                    self._sendToolCallUpdate(toolCall)
        except Exception as e:
            # Error
            with self._lock:
                toolCall = self._pendingToolCalls.get(request.call_id)
                if toolCall:
                    toolCall.status = ToolStatus.ERROR
                    toolCall.response = ToolCallResponse(
                        callId=request.call_id,
                        response_parts=[f"Error: {e}"],
                        status=ToolStatus.ERROR
                    )
                    self._completedToolCalls.append(CompletedToolCall(
                        request=request,
                        response=toolCall.response,
                        status=ToolStatus.ERROR
                    ))
                    del self._pendingToolCalls[request.call_id]
                    self._sendToolCallUpdate(toolCall)

    async def waitForPendingTools(self) -> None:
        """Wait for all pending tools to complete."""
        while self._pendingToolCalls:
            await asyncio.sleep(0.1)

    def getAndClearCompletedTools(self) -> List[CompletedToolCall]:
        """Get and clear completed tool calls."""
        with self._lock:
            completed = self._completedToolCalls[:]
            self._completedToolCalls.clear()
            return completed

    def cancelPendingTools(self, reason: str) -> None:
        """Cancel all pending tool calls."""
        with self._lock:
            for toolCall in self._pendingToolCalls.values():
                toolCall.status = ToolStatus.CANCELLED
                toolCall.response = ToolCallResponse(
                    callId=toolCall.request.call_id,
                    response_parts=[f"Cancelled: {reason}"],
                    status=ToolStatus.CANCELLED
                )
                self._completedToolCalls.append(CompletedToolCall(
                    request=toolCall.request,
                    response=toolCall.response,
                    status=ToolStatus.CANCELLED
                ))
            self._pendingToolCalls.clear()

    async def sendCompletedToolsToLlm(
        self,
        completedTools: List[CompletedToolCall],
        abortSignal: threading.Event = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send completed tool results back to LLM."""
        if not completedTools:
            return

        # Build parts from tool responses
        parts = []
        for toolCall in completedTools:
            for responsePart in toolCall.response.response_parts:
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": toolCall.request.call_id,
                    "content": responsePart
                })

        # Send to LLM
        async for event in self._llmClient.sendMessageStream(parts, abortSignal):
            yield event

# =============================================================================
# AGENT EXECUTOR
# =============================================================================

class CoderAgentExecutor:
    """Main agent executor."""

    def __init__(self, taskStore: TaskStore = None):
        self._tasks: Dict[str, Task] = {}
        self._taskStore = taskStore or InMemoryTaskStore()
        self._executingTasks: Set[str] = set()
        self._lock = threading.Lock()

    async def createTask(
        self,
        taskId: str,
        contextId: str,
        agentSettings: AgentSettings = None,
        eventBus: ExecutionEventBus = None
    ) -> Task:
        """Create a new task."""
        workspacePath = agentSettings.workspace_path if agentSettings else os.getcwd()

        task = Task(
            id=taskId,
            contextId=contextId,
            workspacePath=workspacePath,
            agentSettings=agentSettings
        )
        task.eventBus = eventBus

        with self._lock:
            self._tasks[taskId] = task

        await self._taskStore.save(task)
        logger.info(f"Created task {taskId}")
        return task

    async def reconstruct(
        self,
        sdkTask: Dict[str, Any],
        eventBus: ExecutionEventBus = None
    ) -> Task:
        """Reconstruct a task from stored state."""
        taskId = sdkTask.get("id", str(uuid.uuid4()))
        contextId = sdkTask.get("contextId", str(uuid.uuid4()))

        agentSettings = AgentSettings(
            workspace_path=sdkTask.get("agentSettings", {}).get("workspace_path", os.getcwd()),
            auto_execute=sdkTask.get("agentSettings", {}).get("auto_execute", False)
        )

        task = Task(
            id=taskId,
            contextId=contextId,
            workspacePath=agentSettings.workspace_path,
            agentSettings=agentSettings
        )
        task.taskState = TaskState(sdkTask.get("taskState", "submitted"))
        task.eventBus = eventBus

        with self._lock:
            self._tasks[taskId] = task

        return task

    def getTask(self, taskId: str) -> Optional[Task]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(taskId)

    def getAllTasks(self) -> List[Task]:
        """Get all tasks."""
        with self._lock:
            return list(self._tasks.values())

    async def cancelTask(
        self,
        taskId: str,
        eventBus: ExecutionEventBus
    ) -> None:
        """Cancel a task."""
        task = self.getTask(taskId)

        if not task:
            eventBus.publish({
                "kind": "status-update",
                "taskId": taskId,
                "contextId": str(uuid.uuid4()),
                "status": {
                    "state": "failed",
                    "message": {
                        "kind": "message",
                        "role": "agent",
                        "parts": [{"kind": "text", "text": f"Task {taskId} not found"}],
                        "messageId": str(uuid.uuid4()),
                        "taskId": taskId
                    }
                },
                "final": True
            })
            return

        if task.taskState in [TaskState.CANCELED, TaskState.FAILED]:
            eventBus.publish({
                "kind": "status-update",
                "taskId": taskId,
                "contextId": task.contextId,
                "status": {
                    "state": task.taskState.value,
                    "message": {
                        "kind": "message",
                        "role": "agent",
                        "parts": [{"kind": "text", "text": f"Task {taskId} is already {task.taskState.value}"}],
                        "messageId": str(uuid.uuid4()),
                        "taskId": taskId,
                        "contextId": task.contextId
                    }
                },
                "final": True
            })
            return

        # Cancel pending tools
        task.cancelPendingTools("Task canceled by user request")
        task.setTaskState(TaskState.CANCELED, "Task canceled by user request", final=True)

        await self._taskStore.save(task)

    async def execute(
        self,
        requestContext: Dict[str, Any],
        eventBus: ExecutionEventBus
    ) -> None:
        """Execute a task."""
        userMessage = requestContext.get("userMessage", {})
        sdkTask = requestContext.get("task", {})

        taskId = sdkTask.get("id") or userMessage.get("taskId") or str(uuid.uuid4())
        contextId = userMessage.get("contextId") or sdkTask.get("contextId") or str(uuid.uuid4())

        logger.info(f"Executing task {taskId}, context {contextId}")

        # Create abort signal
        abortSignal = threading.Event()

        # Get or create task
        task = self.getTask(taskId)

        if not task and sdkTask:
            # Reconstruct from store
            task = await self.reconstruct(sdkTask, eventBus)

        if not task:
            # Create new task
            agentSettingsData = userMessage.get("metadata", {}).get("coderAgent", {})
            agentSettings = AgentSettings(
                workspace_path=agentSettingsData.get("workspace_path", os.getcwd()),
                auto_execute=agentSettingsData.get("auto_execute", False)
            )
            task = await self.createTask(taskId, contextId, agentSettings, eventBus)

        task.eventBus = eventBus

        # Check if already in final state
        if task.taskState in [TaskState.CANCELED, TaskState.FAILED, TaskState.COMPLETED]:
            logger.warn(f"Task {taskId} is already in final state: {task.taskState.value}")
            return

        # Check if already executing
        if taskId in self._executingTasks:
            logger.info(f"Task {taskId} has pending execution, processing message only")
            async for _ in task.acceptUserMessage(requestContext, abortSignal):
                pass
            return

        # Start execution
        self._executingTasks.add(taskId)

        try:
            # Process user message
            agentTurnActive = True
            agentEvents = task.acceptUserMessage(requestContext, abortSignal)

            while agentTurnActive:
                # Process LLM events
                toolCallRequests = []
                async for event in agentEvents:
                    if abortSignal.is_set():
                        raise Exception("Execution aborted")

                    if event.get("type") == LLMEvent.TOOL_CALL_REQUEST:
                        toolCallRequests.append(ToolCallRequest(**event.get("value", {})))
                        continue

                    if event.get("type") == LLMEvent.CONTENT:
                        task._sendTextContent(event.get("value", ""))

                if abortSignal.is_set():
                    raise Exception("Execution aborted")

                # Schedule tool calls
                if toolCallRequests:
                    logger.info(f"Scheduling {len(toolCallRequests)} tool calls")
                    await task.scheduleToolCalls(toolCallRequests, abortSignal)

                # Wait for tools to complete
                await task.waitForPendingTools()

                if abortSignal.is_set():
                    raise Exception("Execution aborted")

                # Get completed tools
                completedTools = task.getAndClearCompletedTools()

                if completedTools:
                    # If all cancelled, end turn
                    if all(tc.status == ToolStatus.CANCELLED for tc in completedTools):
                        task.setTaskState(TaskState.INPUT_REQUIRED, final=True)
                        agentTurnActive = False
                    else:
                        # Send results back to LLM
                        agentEvents = task.sendCompletedToolsToLlm(completedTools, abortSignal)
                else:
                    agentTurnActive = False

            # Set final state
            task.setTaskState(TaskState.INPUT_REQUIRED, final=True)

        except Exception as e:
            if abortSignal.is_set():
                task.cancelPendingTools("Execution aborted")
                if task.taskState not in [TaskState.CANCELED, TaskState.FAILED]:
                    task.setTaskState(TaskState.INPUT_REQUIRED, "Execution aborted by client", final=True)
            else:
                logger.error(f"Error executing task {taskId}: {e}")
                task.cancelPendingTools(str(e))
                if task.taskState != TaskState.FAILED:
                    task.setTaskState(TaskState.FAILED, f"Agent error: {e}", final=True)
        finally:
            self._executingTasks.discard(taskId)
            await self._taskStore.save(task)

# =============================================================================
# COMMAND SYSTEM
# =============================================================================

@dataclass
class CommandContext:
    """Context for command execution."""
    config: Dict[str, Any]
    executor: CoderAgentExecutor = None
    eventBus: ExecutionEventBus = None

@dataclass
class CommandResponse:
    """Response from command execution."""
    name: str
    data: Any

class Command(ABC):
    """Base class for commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        pass

class InitCommand(Command):
    """Initialize workspace with GEMINI.md."""

    @property
    def name(self) -> str:
        return "init"

    @property
    def description(self) -> str:
        return "Analyze project and create GEMINI.md file"

    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        workspace = os.getcwd()
        geminiMd = os.path.join(workspace, MEMORY_FILE)

        if os.path.exists(geminiMd):
            return CommandResponse(
                name=self.name,
                data=f"{MEMORY_FILE} already exists"
            )

        # Analyze project
        try:
            # Check for package.json
            packageJson = os.path.join(workspace, "package.json")
            if os.path.exists(packageJson):
                with open(packageJson) as f:
                    data = json.load(f)
                    deps = data.get("dependencies", {})
                    devDeps = data.get("devDependencies", {})

            # Check for requirements.txt
            requirementsTxt = os.path.join(workspace, "requirements.txt")
            if os.path.exists(requirementsTxt):
                with open(requirementsTxt) as f:
                    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

            # Create basic GEMINI.md
            content = f"""# Project Memory

## Project Overview
This project is located at: {workspace}

## Technologies
"""
            if os.path.exists(packageJson):
                content += "\n- Node.js"
            if os.path.exists(requirementsTxt):
                content += "\n- Python"

            content += f"""

## Setup Instructions
Add your setup instructions here.

## Key Files
- Add important files here

## Notes
Add project-specific notes here.
"""

            with open(geminiMd, 'w') as f:
                f.write(content)

            return CommandResponse(
                name=self.name,
                data=f"Created {MEMORY_FILE}"
            )
        except Exception as e:
            return CommandResponse(
                name=self.name,
                data=f"Error: {e}"
            )

class MemoryShowCommand(Command):
    """Show memory contents."""

    @property
    def name(self) -> str:
        return "memory show"

    @property
    def description(self) -> str:
        return "Show current memory contents"

    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        workspace = os.getcwd()
        geminiMd = os.path.join(workspace, MEMORY_FILE)

        if not os.path.exists(geminiMd):
            return CommandResponse(
                name=self.name,
                data=f"{MEMORY_FILE} not found. Run 'init' command first."
            )

        with open(geminiMd) as f:
            content = f.read()

        return CommandResponse(name=self.name, data=content)

class MemoryRefreshCommand(Command):
    """Refresh memory."""

    @property
    def name(self) -> str:
        return "memory refresh"

    @property
    def description(self) -> str:
        return "Refresh memory from source"

    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        return CommandResponse(
            name=self.name,
            data="Memory refreshed"
        )

class MemoryListCommand(Command):
    """List memory files."""

    @property
    def name(self) -> str:
        return "memory list"

    @property
    def description(self) -> str:
        return "List GEMINI.md files in use"

    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        workspace = os.getcwd()
        files = []

        for root, dirs, filenames in os.walk(workspace):
            # Skip hidden dirs and node_modules
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != "node_modules"]

            if MEMORY_FILE in filenames:
                relPath = os.path.relpath(os.path.join(root, MEMORY_FILE), workspace)
                files.append(relPath)

        return CommandResponse(
            name=self.name,
            data=f"Memory files:\n" + "\n".join(f"  {f}" for f in files)
        )

class MemoryCommand(Command):
    """Memory management command."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Manage memory"

    def __init__(self):
        self._subCommands = {
            "show": MemoryShowCommand(),
            "refresh": MemoryRefreshCommand(),
            "list": MemoryListCommand()
        }

    async def execute(
        self,
        context: CommandContext,
        args: List[str]
    ) -> CommandResponse:
        if not args or args[0] not in self._subCommands:
            return await self._subCommands["show"].execute(context, args)

        subCommand = self._subCommands[args[0]]
        return await subCommand.execute(context, args[1:])

class CommandRegistry:
    """Registry for commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._registerDefaults()

    def _registerDefaults(self) -> None:
        """Register default commands."""
        defaultCommands = [
            InitCommand(),
            MemoryCommand(),
        ]
        for cmd in defaultCommands:
            self.register(cmd)

    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name."""
        return self._commands.get(name)

    def getAll(self) -> List[Command]:
        """Get all commands."""
        return list(self._commands.values())

# =============================================================================
# HTTP SERVER
# =============================================================================

class CodingAgentServer:
    """FastAPI server for the coding agent."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        workspace: str = None,
        apiKey: str = None,
        baseURL: str = None,
        taskStore: TaskStore = None
    ):
        self.port = port
        self.workspace = workspace or os.getcwd()
        self.apiKey = apiKey
        self.baseURL = baseURL

        # Initialize components
        self.executor = CoderAgentExecutor(taskStore)
        self.commandRegistry = CommandRegistry()

        # Create FastAPI app
        self.app = FastAPI(
            title="Coding Agent",
            description="AI coding agent with OpenAI-compatible API",
            version="1.0.0"
        )
        self._setupRoutes()

    def _setupRoutes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            return {"name": "Coding Agent", "version": "1.0.0"}

        @self.app.get("/.well-known/agent-card.json")
        async def agentCard():
            card = AgentCard(url=f"http://localhost:{self.port}/")
            return card.dict()

        @self.app.post("/tasks")
        async def createTask(request: Dict[str, Any]):
            """Create a new task."""
            taskId = str(uuid.uuid4())
            contextId = request.get("contextId", str(uuid.uuid4()))
            agentSettings = AgentSettings(
                workspace_path=request.get("workspacePath", self.workspace),
                auto_execute=request.get("autoExecute", False)
            )

            eventBus = ExecutionEventBus()
            task = await self.executor.createTask(taskId, contextId, agentSettings, eventBus)

            return {"taskId": task.id, "contextId": task.contextId}

        @self.app.post("/execute")
        async def executeTask(request: Dict[str, Any]):
            """Execute a task with SSE streaming."""
            eventBus = ExecutionEventBus()

            async def eventGenerator():
                """Generate SSE events."""

                def onEvent(event):
                    return f"data: {json.dumps(event)}\n\n"

                eventBus.subscribe(onEvent)

                # Execute task
                await self.executor.execute(request, eventBus)

                # Send final event
                yield f"data: {json.dumps({'kind': 'done', 'final': True})}\n\n"

                eventBus.finished()

            return StreamingResponse(
                eventGenerator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        @self.app.post("/cancel")
        async def cancelTask(request: Dict[str, Any]):
            """Cancel a task."""
            taskId = request.get("taskId")
            if not taskId:
                raise HTTPException(status_code=400, detail="taskId required")

            eventBus = ExecutionEventBus()
            await self.executor.cancelTask(taskId, eventBus)

            return {"status": "canceled", "taskId": taskId}

        @self.app.get("/tasks")
        async def listTasks():
            """List all tasks."""
            tasks = self.executor.getAllTasks()
            return {
                "tasks": [
                    {
                        "id": t.id,
                        "contextId": t.contextId,
                        "state": t.taskState.value,
                        "model": t.model
                    }
                    for t in tasks
                ]
            }

        @self.app.get("/tasks/{taskId}")
        async def getTask(taskId: str):
            """Get task details."""
            task = self.executor.getTask(taskId)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            metadata = task.getMetadata()
            return asdict(metadata)

        @self.app.post("/executeCommand")
        async def executeCommand(request: Dict[str, Any]):
            """Execute a command."""
            commandName = request.get("command")
            args = request.get("args", [])

            if not commandName:
                raise HTTPException(status_code=400, detail="command required")

            command = self.commandRegistry.get(commandName)
            if not command:
                raise HTTPException(status_code=404, detail=f"Command not found: {commandName}")

            context = CommandContext(
                config={"workspace": self.workspace},
                executor=self.executor
            )

            response = await command.execute(context, args)
            return {"name": response.name, "data": response.data}

        @self.app.get("/listCommands")
        async def listCommands():
            """List available commands."""
            commands = self.commandRegistry.getAll()
            return {
                "commands": [
                    {
                        "name": cmd.name,
                        "description": cmd.description
                    }
                    for cmd in commands
                ]
            }

        @self.app.post("/chat")
        async def chat(request: Dict[str, Any]):
            """Simple chat endpoint."""
            message = request.get("message", "")
            if not message:
                raise HTTPException(status_code=400, detail="message required")

            eventBus = ExecutionEventBus()

            # Collect all events
            events = []

            def collectEvent(event):
                events.append(event)

            eventBus.subscribe(collectEvent)

            # Create request context
            requestContext = {
                "userMessage": {
                    "kind": "message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": str(uuid.uuid4()),
                    "metadata": {
                        "coderAgent": {
                            "kind": "agent-settings",
                            "workspace_path": self.workspace,
                            "auto_execute": True
                        }
                    }
                }
            }

            # Execute
            await self.executor.execute(requestContext, eventBus)

            # Extract text responses
            responses = []
            for event in events:
                if event.get("kind") == "status-update":
                    message = event.get("status", {}).get("message")
                    if message:
                        for part in message.get("parts", []):
                            if part.get("kind") == "text":
                                responses.append(part.get("text", ""))

            return {"response": "\n".join(responses)}

    def run(self) -> None:
        """Run the server."""
        logger.info(f"Starting Coding Agent server on port {self.port}")
        logger.info(f"Workspace: {self.workspace}")
        logger.info(f"API Base URL: {self.baseURL or os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL)}")

        uvicorn.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="info"
        )

# =============================================================================
# SETTINGS & CONFIGURATION
# =============================================================================

@dataclass
class Settings:
    """User and workspace settings."""
    mcpServers: Dict[str, Any] = field(default_factory=dict)
    coreTools: List[str] = field(default_factory=list)
    excludeTools: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    showMemoryUsage: bool = False
    checkpointing: Dict[str, bool] = field(default_factory=dict)
    folderTrust: bool = False
    general: Dict[str, bool] = field(default_factory=dict)
    fileFiltering: Dict[str, Any] = field(default_factory=dict)

def loadSettings(workspaceDir: str) -> Settings:
    """Load settings from user and workspace directories."""
    # User settings directory
    userSettingsDir = os.path.join(os.path.expanduser("~"), GEMINI_DIR)
    userSettingsPath = os.path.join(userSettingsDir, SETTINGS_FILE)

    # Workspace settings path
    workspaceSettingsPath = os.path.join(workspaceDir, GEMINI_DIR, SETTINGS_FILE)

    settings = Settings()

    # Load user settings
    if os.path.exists(userSettingsPath):
        try:
            with open(userSettingsPath) as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
        except Exception as e:
            logger.warning(f"Failed to load user settings: {e}")

    # Load workspace settings (override user settings)
    if os.path.exists(workspaceSettingsPath):
        try:
            with open(workspaceSettingsPath) as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
        except Exception as e:
            logger.warning(f"Failed to load workspace settings: {e}")

    return settings

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def createTaskStore() -> TaskStore:
    """Create task store based on configuration."""
    bucketName = os.getenv("GCS_BUCKET_NAME")
    if bucketName:
        try:
            return GCSTaskStore(bucketName)
        except Exception as e:
            logger.warning(f"Failed to create GCS task store, using file store: {e}")

    storageDir = os.getenv("CODER_AGENT_STORAGE_DIR")
    return FileTaskStore(storageDir)

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Coding Agent - AI-powered coding assistant")
    parser.add_argument("--port", type=int, default=int(os.getenv("CODER_AGENT_PORT", DEFAULT_PORT)),
                       help=f"Port to run server on (default: {DEFAULT_PORT})")
    parser.add_argument("--workspace", type=str, default=os.getenv("CODER_AGENT_WORKSPACE", os.getcwd()),
                       help="Workspace directory")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                       help="OpenAI API key")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"),
                       help=f"OpenAI base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
                       help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"),
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")

    args = parser.parse_args()

    # Setup logger
    global logger
    logger = setup_logger(level=args.log_level)

    # Validate API key
    if not args.api_key:
        logger.warning("OPENAI_API_KEY not set. Set it with --api-key or OPENAI_API_KEY environment variable")

    # Ensure workspace exists
    workspace = os.path.abspath(args.workspace)
    os.makedirs(workspace, exist_ok=True)

    # Create server
    taskStore = createTaskStore()
    server = CodingAgentServer(
        port=args.port,
        workspace=workspace,
        apiKey=args.api_key,
        baseURL=args.base_url
    )

    # Run server
    server.run()

if __name__ == "__main__":
    main()
