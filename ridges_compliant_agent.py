#!/usr/bin/env python3
"""
Ridges-Compliant Coding Agent
==============================

A Ridges.ai platform-compliant coding agent that:
- Uses only built-in Python packages (no external dependencies)
- Follows Ridges entry point conventions
- Uses Ridges proxy endpoints
- Generates proper git diff output
- Respects sandbox environment constraints

Based on next_gen_agent.py architecture.

Author: Ridges-Compliant Agent Team
Version: 1.0.0
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_colored_logging(level: str = "INFO") -> logging.Logger:
    """Setup colored logging for better terminal output."""
    logger = logging.getLogger("RidgesAgent")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with colors
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',
            'INFO': '\033[32m',
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[35m',
        }
        RESET = '\033[0m'

        def format(self, record):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)

    return logger

logger = setup_colored_logging()

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class RidgesConfig:
    """
    Ridges platform-specific configuration.

    Environment variables:
    - SANDBOX_PROXY_URL: Ridges proxy URL (provided by platform)
    - AGENT_TIMEOUT: Time limit for execution (default ~120 seconds)
    - REPO_PATH: Repository mount point (default /repo)
    """

    # Ridges Environment
    SANDBOX_PROXY_URL: str = os.getenv("SANDBOX_PROXY_URL", "http://localhost:8000")
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "120"))

    # Model Configuration
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "claude-sonnet-4")
    FALLBACK_MODELS: List[str] = [
        "claude-3-5-sonnet-20241022",
        "gpt-4o",
        "gemini-2.0-flash-exp",
    ]

    # API Configuration (using Ridges proxy)
    API_URL: str = SANDBOX_PROXY_URL
    API_TIMEOUT: int = AGENT_TIMEOUT
    MAX_RETRIES: int = 3

    # Execution Limits
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "50"))  # Reduced for 2-min limit
    MAX_DURATION: int = int(os.getenv("MAX_DURATION", str(AGENT_TIMEOUT - 20)))  # Buffer for git diff
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))  # Reduced for cost limits

    # Context Management
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "100000"))
    SUMMARY_THRESHOLD: int = int(os.getenv("SUMMARY_THRESHOLD", "50000"))
    RECENT_HISTORY: int = int(os.getenv("RECENT_HISTORY", "10"))

    # Repository Path (Ridges mounts at /repo)
    REPO_PATH: str = os.getenv("REPO_PATH", "/repo")

    # Feature Flags (disabled for Ridges to reduce complexity/cost)
    ENABLE_MCTS: bool = False  # Disable MCTS for simpler execution
    ENABLE_SELF_REFLECTION: bool = False  # Disable reflection for speed
    ENABLE_VECTOR_SEARCH: bool = False  # Disable vector search (no numpy)

    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if cls.MAX_STEPS <= 0:
            raise ValueError("MAX_STEPS must be positive")
        if cls.MAX_DURATION <= 0:
            raise ValueError("MAX_DURATION must be positive")
        if not cls.REPO_PATH:
            raise ValueError("REPO_PATH must be set")

# Validate config on import
try:
    RidgesConfig.validate()
except Exception as e:
    logger.warning(f"Configuration validation failed: {e}")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

T = TypeVar('T')

@dataclass
class Message:
    """A message in the conversation."""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_id:
            result["tool_call_id"] = self.tool_id
        return result

    def truncate(self, max_chars: int = 50000) -> 'Message':
        """Return a truncated version of this message."""
        if len(self.content) <= max_chars:
            return self
        return Message(
            role=self.role,
            content=self.content[:max_chars] + "\n...[truncated]",
            tool_calls=self.tool_calls,
            tool_id=self.tool_id,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }

@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    content: str
    is_error: bool = False
    error_type: Optional[str] = None
    execution_time: float = 0.0

    def to_message(self) -> Message:
        """Convert to a tool result message."""
        return Message(
            role="tool",
            content=self.content,
            tool_id=self.tool_call_id,
            metadata={"is_error": self.is_error, "error_type": self.error_type},
        )

@dataclass
class Thought:
    """A single thought in the reasoning chain."""
    reasoning: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reasoning": self.reasoning,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

# =============================================================================
# ENUMS
# =============================================================================

class ProblemType(Enum):
    """Type of coding problem."""
    CREATE = "create"
    FIX = "fix"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    TEST = "test"
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ToolErrorType(Enum):
    """Types of tool errors."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_INPUT = "invalid_input"
    NETWORK_ERROR = "network_error"
    PARSE_ERROR = "parse_error"
    UNKNOWN = "unknown"

# =============================================================================
# EXCEPTIONS
# =============================================================================

class AgentException(Exception):
    """Base exception for agent errors."""
    pass

class ToolException(AgentException):
    """Exception raised when a tool fails."""
    def __init__(self, message: str, error_type: ToolErrorType = ToolErrorType.UNKNOWN):
        super().__init__(message)
        self.error_type = error_type

class NetworkException(AgentException):
    """Exception raised when network request fails."""
    pass

class ValidationException(AgentException):
    """Exception raised when validation fails."""
    pass

class TimeoutException(AgentException):
    """Exception raised when operation times out."""
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for retrying with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def count_tokens(text: Union[str, List[Dict]]) -> int:
    """Estimate token count (rough approximation)."""
    if isinstance(text, list):
        text = " ".join(str(m.get("content", "")) for m in text)
    return len(text) // 4

def truncate_text(text: str, max_tokens: int = 50000) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def sanitize_json(text: str) -> str:
    """Sanitize and fix common JSON issues."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    return text.strip()

def calculate_hash(content: str) -> str:
    """Calculate SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def safe_execute(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """Safely execute a function and return default on exception."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Safe execute failed: {e}")
        return default

# =============================================================================
# RIDGES LLM CLIENT (Using urllib.request)
# =============================================================================

class RidgesLLMClient:
    """
    LLM client for Ridges platform using urllib.request (standard library).

    Uses Ridges proxy endpoints:
    - {proxy_url}/agents/inference for LLM calls
    - {proxy_url}/agents/embedding for embeddings (optional)
    """

    def __init__(
        self,
        api_url: str = RidgesConfig.API_URL,
        primary_model: str = RidgesConfig.PRIMARY_MODEL,
        fallback_models: List[str] = None,
        timeout: int = RidgesConfig.API_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.primary_model = primary_model
        self.fallback_models = fallback_models or RidgesConfig.FALLBACK_MODELS
        self.timeout = timeout
        self.cache: Dict[str, Any] = {}
        self.request_count = 0
        self.error_counts: Dict[str, int] = defaultdict(int)

    @retry_with_backoff(max_retries=2, base_delay=0.5, max_delay=5.0)
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        **kwargs,
    ) -> str:
        """
        Generate a completion using Ridges inference endpoint.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (defaults to primary_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        model = model or self.primary_model
        self.request_count += 1

        # Check cache for deterministic requests
        cache_key = self._cache_key(messages, model, temperature)
        if cache_key in self.cache and temperature == 0.0:
            logger.debug(f"Cache hit for request {self.request_count}")
            return self.cache[cache_key]

        # Prepare request for Ridges endpoint
        url = f"{self.api_url}/agents/inference"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Make request using urllib.request
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            # Parse response (Ridges may use different format)
            if "choices" in response_data:
                content = response_data["choices"][0]["message"]["content"]
            elif "content" in response_data:
                content = response_data["content"]
            elif "output" in response_data:
                content = response_data["output"]
            else:
                content = str(response_data)

            # Cache successful responses
            if temperature == 0.0:
                self.cache[cache_key] = content

            return content

        except urllib.error.HTTPError as e:
            self.error_counts[f"http_{e.code}"] += 1
            error_body = e.read().decode('utf-8', errors='replace')

            # Try fallback model on 5xx errors
            if e.code >= 500 and self.fallback_models and model != self.fallback_models[0]:
                logger.warning(f"Primary model failed, trying fallback: {e.code}")
                return self.generate(messages, self.fallback_models[0], temperature, max_tokens)

            raise NetworkException(f"HTTP {e.code}: {error_body}")

        except urllib.error.URLError as e:
            self.error_counts["url_error"] += 1
            raise NetworkException(f"URL error: {e.reason}")

        except socket.timeout:
            self.error_counts["timeout"] += 1
            raise TimeoutException(f"Request timeout after {self.timeout}s")

        except Exception as e:
            self.error_counts["unknown"] += 1
            raise AgentException(f"LLM request failed: {e}")

    def _cache_key(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Generate cache key from request parameters."""
        content = json.dumps({"messages": messages, "model": model, "temp": temperature})
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self.request_count,
            "cache_size": len(self.cache),
            "error_counts": dict(self.error_counts),
        }

# =============================================================================
# TOOL SYSTEM
# =============================================================================

class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return result."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "calls": 0,
            "errors": 0,
            "total_time": 0.0,
        })

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ToolException(f"Tool not found: {name}", ToolErrorType.INVALID_INPUT)

        start_time = time.time()
        stats = self.execution_stats[name]

        try:
            result = tool.execute(**kwargs)
            stats["calls"] += 1
            stats["total_time"] += time.time() - start_time
            return result
        except Exception as e:
            stats["errors"] += 1
            raise ToolException(f"Tool execution failed: {e}", ToolErrorType.RUNTIME_ERROR)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return dict(self.execution_stats)

# =============================================================================
# CORE TOOLS
# =============================================================================

class ReadFileTool(Tool):
    """Read file contents."""

    name = "read_file"
    description = "Read the contents of a file. Use this to examine file contents before making changes."

    def execute(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read file content, optionally with line range."""
        try:
            # Resolve path relative to repo
            path = self._resolve_path(file_path)
            if not path.exists():
                raise ToolException(f"File not found: {file_path}", ToolErrorType.FILE_NOT_FOUND)

            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')

            # Apply line range
            if start_line is not None or end_line is not None:
                start = max(0, (start_line or 1) - 1)
                end = min(len(lines), end_line or len(lines))
                lines = lines[start:end]
                content = '\n'.join(lines)

            # Truncate if too large
            if len(content) > 50000:
                content = content[:50000] + "\n...[truncated]"

            return content
        except Exception as e:
            raise ToolException(f"Failed to read file: {e}", ToolErrorType.RUNTIME_ERROR)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to repo."""
        if os.path.isabs(file_path):
            return Path(file_path)
        return Path(RidgesConfig.REPO_PATH) / file_path

class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."

    def execute(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            path = self._resolve_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise ToolException(f"Failed to write file: {e}", ToolErrorType.RUNTIME_ERROR)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to repo."""
        if os.path.isabs(file_path):
            return Path(file_path)
        return Path(RidgesConfig.REPO_PATH) / file_path

class EditFileTool(Tool):
    """Edit a file using search and replace."""

    name = "edit_file"
    description = "Edit a file by replacing an exact string match with new content."

    def execute(
        self,
        file_path: str,
        search: str,
        replace: str,
        occurrence: int = 1,
    ) -> str:
        """Edit file with search/replace."""
        try:
            path = self._resolve_path(file_path)
            if not path.exists():
                raise ToolException(f"File not found: {file_path}", ToolErrorType.FILE_NOT_FOUND)

            content = path.read_text(encoding='utf-8')

            if search not in content:
                raise ToolException(f"Search string not found in {file_path}", ToolErrorType.INVALID_INPUT)

            # Count occurrences
            count = content.count(search)
            if count > 1:
                if occurrence > count:
                    raise ToolException(f"Only {count} occurrence(s) found", ToolErrorType.INVALID_INPUT)
                # Replace specific occurrence
                parts = content.split(search)
                if occurrence <= len(parts):
                    content = search.join(parts[:occurrence]) + replace + search.join(parts[occurrence:])
            else:
                content = content.replace(search, replace, 1)

            path.write_text(content, encoding='utf-8')
            return f"Successfully edited {file_path}"
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to edit file: {e}", ToolErrorType.RUNTIME_ERROR)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to repo."""
        if os.path.isabs(file_path):
            return Path(file_path)
        return Path(RidgesConfig.REPO_PATH) / file_path

class SearchTool(Tool):
    """Search for patterns in files."""

    name = "search"
    description = "Search for text/patterns in files using grep-like functionality."

    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_insensitive: bool = False,
        max_results: int = 100,
    ) -> str:
        """Search in files."""
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)

            results = []
            search_path = self._resolve_path(path)

            for file_path in search_path.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                        lines = content.split('\n')

                        for i, line in enumerate(lines, 1):
                            if regex.search(line):
                                results.append(f"{file_path}:{i}: {line.strip()}")
                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue

                if len(results) >= max_results:
                    break

            if not results:
                return f"No matches found for pattern: {pattern}"

            return '\n'.join(results[:max_results])
        except Exception as e:
            raise ToolException(f"Search failed: {e}", ToolErrorType.RUNTIME_ERROR)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to repo."""
        if os.path.isabs(path):
            return Path(path)
        return Path(RidgesConfig.REPO_PATH) / path

class RunShellTool(Tool):
    """Execute shell commands."""

    name = "run_shell"
    description = "Execute a shell command. Use for git, npm, build tools, etc."

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
    ) -> str:
        """Execute shell command."""
        try:
            # Safety check
            dangerous = ['rm -rf /', 'rm -rf /*', 'mkfs', 'format c:', '> /dev/sda']
            if any(d in command.lower() for d in dangerous):
                raise ToolException("Dangerous command blocked", ToolErrorType.PERMISSION_DENIED)

            work_dir = cwd if cwd else RidgesConfig.REPO_PATH

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )

            output = result.stdout or result.stderr
            if result.returncode != 0:
                output = f"Command exited with code {result.returncode}\n{output}"

            return output
        except subprocess.TimeoutExpired:
            raise ToolException(f"Command timeout after {timeout}s", ToolErrorType.TIMEOUT)
        except Exception as e:
            raise ToolException(f"Shell execution failed: {e}", ToolErrorType.RUNTIME_ERROR)

class ListFilesTool(Tool):
    """List files in a directory."""

    name = "list_files"
    description = "List files and directories in a given path."

    def execute(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> str:
        """List files."""
        try:
            search_path = self._resolve_path(path)
            if not search_path.exists():
                raise ToolException(f"Path not found: {path}", ToolErrorType.FILE_NOT_FOUND)

            if recursive:
                files = list(search_path.rglob(pattern))
            else:
                files = list(search_path.glob(pattern))

            # Format output
            result = []
            for f in sorted(files):
                prefix = "DIR " if f.is_dir() else "FILE"
                result.append(f"{prefix} {f.relative_to(RidgesConfig.REPO_PATH)}")

            return '\n'.join(result) if result else "No files found"
        except Exception as e:
            raise ToolException(f"List files failed: {e}", ToolErrorType.RUNTIME_ERROR)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to repo."""
        if os.path.isabs(path):
            return Path(path)
        return Path(RidgesConfig.REPO_PATH) / path

# =============================================================================
# GIT DIFF GENERATION
# =============================================================================

def generate_git_diff(repo_path: str, base_commit: Optional[str] = None) -> str:
    """
    Generate git diff for all changes in the repository.

    Args:
        repo_path: Path to the repository
        base_commit: Base commit to diff against (if None, uses staged changes)

    Returns:
        Git diff string
    """
    try:
        original_cwd = os.getcwd()
        os.chdir(repo_path)

        # Initialize git repo if not exists
        if not Path(".git").exists():
            subprocess.run(
                ["git", "init"],
                capture_output=True,
                check=False,
                timeout=10
            )
            subprocess.run(
                ["git", "config", "user.email", "agent@ridges.ai"],
                capture_output=True,
                check=False,
                timeout=5
            )
            subprocess.run(
                ["git", "config", "user.name", "Ridges Agent"],
                capture_output=True,
                check=False,
                timeout=5
            )

        # Stage all changes
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            check=False,
            timeout=30
        )

        # Generate diff
        if base_commit:
            # Diff against specific commit
            result = subprocess.run(
                ["git", "diff", base_commit, "--unified=5"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
        else:
            # Diff staged changes
            result = subprocess.run(
                ["git", "diff", "--cached", "--unified=5"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )

        os.chdir(original_cwd)

        # Return diff (or empty if no changes)
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.error("Git diff generation timed out")
        os.chdir(original_cwd)
        return ""
    except Exception as e:
        logger.error(f"Failed to generate git diff: {e}")
        os.chdir(original_cwd)
        return ""

# =============================================================================
# MAIN RIDGES AGENT
# =============================================================================

class RidgesAgent:
    """
    Ridges-compliant coding agent.

    Simplified version of NextGenAgent optimized for:
    - Fast execution (< 2 minutes)
    - Low cost (< $2)
    - Standard library only (no external dependencies)
    - Ridges platform compatibility
    """

    def __init__(
        self,
        api_url: str = RidgesConfig.API_URL,
        primary_model: str = RidgesConfig.PRIMARY_MODEL,
    ):
        # Initialize core components
        self.llm = RidgesLLMClient(api_url=api_url, primary_model=primary_model)
        self.tools = ToolRegistry()

        # Register tools
        self._register_tools()

        # Agent state
        self.conversation_history: List[Message] = []
        self.step_count = 0
        self.start_time: Optional[float] = None

        logger.info("RidgesAgent initialized")

    def _register_tools(self) -> None:
        """Register all available tools."""
        tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            SearchTool(),
            RunShellTool(),
            ListFilesTool(),
        ]

        for tool in tools:
            self.tools.register(tool)

    def run(
        self,
        problem_statement: str,
        max_steps: int = RidgesConfig.MAX_STEPS,
        max_duration: int = RidgesConfig.MAX_DURATION,
    ) -> str:
        """
        Run the agent to solve a problem.

        Args:
            problem_statement: The problem to solve
            max_steps: Maximum number of steps to take
            max_duration: Maximum time in seconds

        Returns:
            Final result
        """
        self.start_time = time.time()
        logger.info(f"Starting RidgesAgent for: {problem_statement[:100]}")

        # Detect problem type
        problem_type = self._detect_problem_type(problem_statement)
        logger.info(f"Detected problem type: {problem_type.value}")

        # Initialize context
        self.conversation_history = [
            Message(role="system", content=self._get_system_prompt(problem_type)),
            Message(role="user", content=problem_statement),
        ]

        # Main execution loop
        for step in range(max_steps):
            # Check timeout
            if time.time() - self.start_time > max_duration:
                logger.warning(f"Timeout after {max_duration}s")
                break

            self.step_count = step + 1

            # Prepare messages
            messages = [m.to_dict() for m in self.conversation_history]

            # Generate next action
            try:
                response = self.llm.generate(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                break

            # Add response to history
            self.conversation_history.append(
                Message(role="assistant", content=response)
            )

            # Check if done
            if self._is_complete(response):
                logger.info(f"Task completed at step {step + 1}")
                break

            # Parse and execute tool calls (simplified)
            # For Ridges, we assume the LLM outputs the final answer directly
            # rather than using function calling

        # Return final result
        final_message = self.conversation_history[-1] if self.conversation_history else None
        return final_message.content if final_message else "No result generated"

    def _detect_problem_type(self, statement: str) -> ProblemType:
        """Detect the type of problem from the statement."""
        statement_lower = statement.lower()

        if any(word in statement_lower for word in ["create", "implement", "add", "build", "write"]):
            return ProblemType.CREATE
        elif any(word in statement_lower for word in ["fix", "bug", "error", "broken", "doesn't work"]):
            return ProblemType.FIX
        elif any(word in statement_lower for word in ["refactor", "clean up", "restructure"]):
            return ProblemType.REFACTOR
        elif any(word in statement_lower for word in ["optimize", "faster", "performance", "slow"]):
            return ProblemType.OPTIMIZE
        elif any(word in statement_lower for word in ["test", "testing"]):
            return ProblemType.TEST
        elif any(word in statement_lower for word in ["explain", "understand", "how does"]):
            return ProblemType.EXPLAIN
        else:
            return ProblemType.UNKNOWN

    def _is_complete(self, response: str) -> bool:
        """Check if the response indicates completion."""
        complete_indicators = [
            "done",
            "complete",
            "finished",
            "successfully",
            "the fix is",
            "the solution is",
            "here's the",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in complete_indicators)

    def _get_system_prompt(self, problem_type: ProblemType) -> str:
        """Get the system prompt for the problem type."""
        base_prompt = f"""You are an expert coding assistant working on a task in a repository located at {RidgesConfig.REPO_PATH}.

Key principles:
- Be concise and direct (you have limited time and budget)
- Make minimal, targeted changes
- Explain your changes clearly
- Use tools to read files before editing
- Handle edge cases appropriately

You have access to tools for reading, writing, and searching files. Use them to understand the codebase and make changes."""

        if problem_type == ProblemType.FIX:
            return base_prompt + """

BUG FIXING APPROACH:
1. Read and understand the problematic code
2. Identify the root cause
3. Make minimal fixes
4. Ensure no regressions

When you've completed the fix, clearly state: "The fix is complete:" followed by a summary."""

        elif problem_type == ProblemType.CREATE:
            return base_prompt + """

CODE CREATION APPROACH:
1. Understand requirements clearly
2. Design a clean solution
3. Implement with best practices
4. Add appropriate error handling

When you've completed the implementation, clearly state: "The implementation is complete:" followed by a summary."""

        else:
            return base_prompt

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "step_count": self.step_count,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "llm_stats": self.llm.get_stats(),
            "tool_stats": self.tools.get_stats(),
        }

# =============================================================================
# RIDGES ENTRY POINT
# =============================================================================

def agent_main(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ridges.ai platform entry point.

    This is the main function called by the Ridges platform.

    Args:
        input_dict: Dictionary containing:
            - repo_path: Path to the repository (default: /repo)
            - problem: Problem statement to solve
            - Additional context provided by Ridges

    Returns:
        Dictionary with:
            - patch: Git diff of all changes made
    """
    try:
        # Extract input parameters
        repo_path = input_dict.get("repo_path", RidgesConfig.REPO_PATH)
        problem_statement = input_dict.get("problem", "")

        if not problem_statement:
            logger.warning("No problem statement provided")
            return {"patch": ""}

        logger.info(f"Ridges agent starting: {problem_statement[:100]}")

        # Validate repo path
        if not Path(repo_path).exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return {"patch": ""}

        # Update configuration
        RidgesConfig.REPO_PATH = repo_path

        # Create and run agent
        agent = RidgesAgent(api_url=RidgesConfig.API_URL)
        result = agent.run(
            problem_statement=problem_statement,
            max_steps=RidgesConfig.MAX_STEPS,
            max_duration=RidgesConfig.MAX_DURATION,
        )

        # Log result summary
        duration = time.time() - (agent.start_time or 0)
        logger.info(f"Agent completed in {duration:.1f}s with {agent.step_count} steps")

        # Generate git diff of all changes
        logger.info("Generating git diff...")
        patch = generate_git_diff(repo_path)

        if not patch:
            logger.warning("No git diff generated - no changes detected")

        logger.info(f"Patch size: {len(patch)} bytes")

        # Return required format
        return {"patch": patch}

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return {"patch": ""}


# =============================================================================
# MAIN ENTRY POINT (for local testing)
# =============================================================================

def main():
    """Main entry point for local CLI usage (not used by Ridges platform)."""
    import argparse

    parser = argparse.ArgumentParser(description="Ridges-Compliant Coding Agent")
    parser.add_argument("problem", help="Problem statement to solve")
    parser.add_argument("--repo-path", default=RidgesConfig.REPO_PATH, help="Repository path")
    parser.add_argument("--api-url", default=RidgesConfig.API_URL, help="API URL")
    parser.add_argument("--model", default=RidgesConfig.PRIMARY_MODEL, help="Primary model")
    parser.add_argument("--max-steps", type=int, default=RidgesConfig.MAX_STEPS)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    setup_colored_logging(args.log_level)

    # Prepare input dict (simulating Ridges input)
    input_dict = {
        "repo_path": args.repo_path,
        "problem": args.problem,
    }

    # Call agent_main
    result = agent_main(input_dict)

    # Print results
    print("\n" + "=" * 60)
    print("GIT DIFF PATCH:")
    print("=" * 60)
    if result["patch"]:
        print(result["patch"])
    else:
        print("(No changes made)")
    print("=" * 60)

if __name__ == "__main__":
    main()
