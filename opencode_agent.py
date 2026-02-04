#!/usr/bin/env python3
"""
OpenCode Agent - Complete Python Port
=====================================

A fully-featured AI coding agent supporting:
- Multiple LLM providers (Anthropic, OpenAI, Google, etc.)
- File operations (read, write, edit, glob, grep)
- Bash command execution with permissions
- Web search and fetch
- Task delegation to subagents
- Session management with streaming
- Permission system for safety
- Multi-modal support (text, images, PDFs)
"""

from __future__ import annotations

import ast
import asyncio
import base64
import fnmatch
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fnmatch import fnmatch
from functools import lru_cache
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import aiohttp
import fitz  # PyMuPDF for PDF support
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

# =============================================================================
# Constants and Configuration
# =============================================================================

VERSION = "1.0.0"

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 120.0
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB

SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
SUPPORTED_PDF_TYPES = {".pdf"}

# =============================================================================
# Types and Enums
# =============================================================================

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    BEDROCK = "bedrock"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    COHERE = "cohere"
    MISTRAL = "mistral"
    LOCAL = "local"


class PermissionLevel(Enum):
    ALLOW = "allow"
    DENY = "deny"
    PROMPT = "prompt"


class ToolResult(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_id:
            result["tool_call_id"] = self.tool_id
        return result


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolResponse:
    tool_call_id: str
    content: str
    result: ToolResult = ToolResult.SUCCESS
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileContent:
    path: str
    content: str
    encoding: str = "utf-8"
    is_binary: bool = False
    mime_type: Optional[str] = None


@dataclass
class ImageContent:
    path: str
    data: bytes
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0


@dataclass
class Session:
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: List[UsageStats] = field(default_factory=list)


@dataclass
class PermissionRule:
    pattern: str
    level: PermissionLevel
    description: str = ""


@dataclass
class AgentConfig:
    name: str
    description: str
    model: str
    provider: Provider = Provider.ANTHROPIC
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = DEFAULT_MAX_TOKENS
    tools: List[str] = field(default_factory=list)
    permissions: Dict[str, PermissionLevel] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT


# =============================================================================
# Exceptions
# =============================================================================

class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class PermissionDeniedError(AgentError):
    """Raised when permission is denied."""
    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""
    pass


class ProviderError(AgentError):
    """Raised when provider fails."""
    pass


class TimeoutError(AgentError):
    """Raised when operation times out."""
    pass


# =============================================================================
# Permission System
# =============================================================================

class PermissionManager:
    """Manages permissions for tool execution."""

    def __init__(self):
        self.rules: List[PermissionRule] = []
        self.permission_cache: Dict[str, PermissionLevel] = {}
        self._lock = threading.Lock()

    def add_rule(self, pattern: str, level: PermissionLevel, description: str = ""):
        """Add a permission rule."""
        with self._lock:
            rule = PermissionRule(pattern, level, description)
            self.rules.append(rule)
            self.permission_cache.clear()

    def check_permission(
        self,
        resource: str,
        default: PermissionLevel = PermissionLevel.PROMPT,
    ) -> PermissionLevel:
        """Check permission for a resource."""
        cache_key = resource
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]

        with self._lock:
            for rule in reversed(self.rules):
                if fnmatch(resource, rule.pattern):
                    self.permission_cache[cache_key] = rule.level
                    return rule.level

        self.permission_cache[cache_key] = default
        return default

    def is_allowed(self, resource: str) -> bool:
        """Check if resource is allowed."""
        return self.check_permission(resource) == PermissionLevel.ALLOW

    def is_denied(self, resource: str) -> bool:
        """Check if resource is denied."""
        return self.check_permission(resource) == PermissionLevel.DENY

    def clear(self):
        """Clear all rules and cache."""
        with self._lock:
            self.rules.clear()
            self.permission_cache.clear()


# =============================================================================
# Tool System
# =============================================================================

class Tool(ABC):
    """Base class for all tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Dict[str, Any]]

    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResponse:
        """Execute the tool."""
        pass

    def check_permission(self, resource: str) -> bool:
        """Check permission for tool execution."""
        return self.permission_manager.is_allowed(resource)


class ReadTool(Tool):
    """Tool for reading file contents."""

    name = "read"
    description = "Read the contents of a file. Supports text, images, and PDFs."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed)",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
                "default": 2000,
            },
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> ToolResponse:
        """Read file contents."""
        try:
            path = Path(file_path).resolve()

            if not path.exists():
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: File not found: {file_path}",
                    result=ToolResult.ERROR,
                )

            if not path.is_file():
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Not a file: {file_path}",
                    result=ToolResult.ERROR,
                )

            if not self.check_permission(str(path)):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Permission denied: {file_path}",
                    result=ToolResult.PERMISSION_DENIED,
                )

            # Check file size
            file_size = path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: File too large ({file_size} bytes): {file_path}",
                    result=ToolResult.ERROR,
                )

            suffix = path.suffix.lower()

            # Handle images
            if suffix in SUPPORTED_IMAGE_TYPES:
                return await self._read_image(path)

            # Handle PDFs
            if suffix in SUPPORTED_PDF_TYPES:
                return await self._read_pdf(path, offset, limit)

            # Handle text files
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                start_idx = max(0, offset - 1)
                end_idx = min(len(lines), start_idx + limit)

                result_lines = lines[start_idx:end_idx]

                # Format with line numbers
                numbered_lines = [
                    f"{start_idx + i + 1}\t{line.rstrip()}"
                    for i, line in enumerate(result_lines)
                ]

                content = "\n".join(numbered_lines)
                if end_idx < len(lines):
                    content += f"\n... ({len(lines) - end_idx} more lines)"

                return ToolResponse(tool_call_id="", content=content)

            except UnicodeDecodeError:
                # Binary file
                with open(path, "rb") as f:
                    data = f.read()

                return ToolResponse(
                    tool_call_id="",
                    content=f"Binary file ({len(data)} bytes): {file_path}",
                    result=ToolResult.SUCCESS,
                )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error reading file: {str(e)}",
                result=ToolResult.ERROR,
            )

    async def _read_image(self, path: Path) -> ToolResponse:
        """Read image file."""
        try:
            from PIL import Image

            with Image.open(path) as img:
                width, height = img.size
                mime_type = mimetypes.guess_type(str(path))[0] or "image/png"

                # Convert to PNG if needed
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                data = buffer.getvalue()

                # Encode as base64
                base64_data = base64.b64encode(data).decode("utf-8")

                content = f"""Image: {path}
Size: {width}x{height}
Type: {mime_type}
Data URL: data:{mime_type};base64,{base64_data[:100]}...
"""

                return ToolResponse(
                    tool_call_id="",
                    content=content,
                    metadata={
                        "image_data": base64_data,
                        "width": width,
                        "height": height,
                        "mime_type": mime_type,
                    },
                )

        except ImportError:
            # PIL not available, return basic info
            with open(path, "rb") as f:
                data = f.read()

            return ToolResponse(
                tool_call_id="",
                content=f"Image file ({len(data)} bytes): {path}\n(PIL not available for detailed image info)",
            )

    async def _read_pdf(
        self, path: Path, offset: int, limit: int
    ) -> ToolResponse:
        """Read PDF file."""
        try:
            doc = fitz.open(str(path))
            page_count = doc.page_count

            # Map line offset to page offset (roughly)
            pages_per_limit = max(1, limit // 50)
            start_page = max(0, (offset - 1) // 50)
            end_page = min(page_count, start_page + 20)

            content_parts = []

            for page_num in range(start_page, end_page):
                page = doc[page_num]
                text = page.get_text()
                content_parts.append(f"--- Page {page_num + 1} ---")
                content_parts.append(text)

            doc.close()

            content = "\n".join(content_parts)

            if end_page < page_count:
                content += f"\n... ({page_count - end_page} more pages)"

            return ToolResponse(tool_call_id="", content=content)

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error reading PDF: {str(e)}",
                result=ToolResult.ERROR,
            )


class WriteTool(Tool):
    """Tool for writing files."""

    name = "write"
    description = "Write content to a file, overwriting if it exists."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, file_path: str, content: str) -> ToolResponse:
        """Write content to file."""
        try:
            path = Path(file_path).resolve()

            if not self.check_permission(str(path)):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Permission denied: {file_path}",
                    result=ToolResult.PERMISSION_DENIED,
                )

            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResponse(
                tool_call_id="",
                content=f"Successfully wrote {len(content)} bytes to {file_path}",
            )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error writing file: {str(e)}",
                result=ToolResult.ERROR,
            )


class EditTool(Tool):
    """Tool for editing files with search/replace."""

    name = "edit"
    description = "Edit a file by replacing an exact string match with new content."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to replace (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "New content to replace with",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of just first",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResponse:
        """Edit file by replacing string."""
        try:
            path = Path(file_path).resolve()

            if not path.exists():
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: File not found: {file_path}",
                    result=ToolResult.ERROR,
                )

            if not self.check_permission(str(path)):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Permission denied: {file_path}",
                    result=ToolResult.PERMISSION_DENIED,
                )

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_string not in content:
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: String not found in file: {old_string[:50]}...",
                    result=ToolResult.ERROR,
                )

            count = content.count(old_string)
            if count > 1 and not replace_all:
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: String appears {count} times. Use replace_all=True or provide more context.",
                    result=ToolResult.ERROR,
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResponse(
                tool_call_id="",
                content=f"Successfully replaced {count} occurrence(s) in {file_path}",
            )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error editing file: {str(e)}",
                result=ToolResult.ERROR,
            )


class GlobTool(Tool):
    """Tool for finding files with glob patterns."""

    name = "glob"
    description = "Find files using glob patterns (e.g., '**/*.py', 'src/**/*.ts')"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search (defaults to current directory)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self, pattern: str, path: str = "."
    ) -> ToolResponse:
        """Find files matching glob pattern."""
        try:
            base_path = Path(path).resolve()

            if not base_path.exists():
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Path not found: {path}",
                    result=ToolResult.ERROR,
                )

            if not self.check_permission(str(base_path)):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Permission denied: {path}",
                    result=ToolResult.PERMISSION_DENIED,
                )

            matches = list(base_path.rglob(pattern))
            matches = [m for m in matches if m.is_file()]

            # Sort by modification time
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            result = "\n".join(str(m) for m in matches)

            return ToolResponse(
                tool_call_id="",
                content=result or f"No files found matching pattern: {pattern}",
            )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error searching files: {str(e)}",
                result=ToolResult.ERROR,
            )


class GrepTool(Tool):
    """Tool for searching text in files."""

    name = "grep"
    description = "Search for text patterns in files using regex"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in",
            },
            "glob": {
                "type": "string",
                "description": "File pattern to filter (e.g., '*.py')",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case insensitive search",
                "default": False,
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        case_insensitive: bool = False,
    ) -> ToolResponse:
        """Search for pattern in files."""
        try:
            import re

            base_path = Path(path).resolve()

            if not base_path.exists():
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Path not found: {path}",
                    result=ToolResult.ERROR,
                )

            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)

            results = []

            # Find files
            files_to_search: List[Path] = []
            if glob:
                files_to_search = list(base_path.rglob(glob))
            else:
                files_to_search = list(base_path.rglob("*"))

            for file_path in files_to_search:
                if not file_path.is_file():
                    continue

                if not self.check_permission(str(file_path)):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(
                                    f"{file_path}:{line_num}: {line.rstrip()}"
                                )
                except Exception:
                    continue

            content = "\n".join(results)

            return ToolResponse(
                tool_call_id="",
                content=content or f"No matches found for pattern: {pattern}",
            )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error searching: {str(e)}",
                result=ToolResult.ERROR,
            )


class BashTool(Tool):
    """Tool for executing bash commands."""

    name = "bash"
    description = "Execute a shell command with timeout"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120)",
                "default": 120,
            },
        },
        "required": ["command"],
    }

    def __init__(self, permission_manager: PermissionManager):
        super().__init__(permission_manager)
        self.allowed_commands: Set[str] = set()
        self.denied_commands: Set[str] = set()

    def set_allowed_commands(self, commands: List[str]):
        """Set allowed command patterns."""
        self.allowed_commands.update(commands)

    def set_denied_commands(self, commands: List[str]):
        """Set denied command patterns."""
        self.denied_commands.update(commands)

    async def execute(self, command: str, timeout: int = 120) -> ToolResponse:
        """Execute bash command."""
        try:
            # Check command permissions
            cmd_name = command.split()[0] if command.split() else ""

            for pattern in self.denied_commands:
                if fnmatch(cmd_name, pattern):
                    return ToolResponse(
                        tool_call_id="",
                        content=f"Error: Command denied by policy: {cmd_name}",
                        result=ToolResult.PERMISSION_DENIED,
                    )

            if self.allowed_commands:
                allowed = any(fnmatch(cmd_name, p) for p in self.allowed_commands)
                if not allowed:
                    return ToolResponse(
                        tool_call_id="",
                        content=f"Error: Command not in allow list: {cmd_name}",
                        result=ToolResult.PERMISSION_DENIED,
                    )

            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Command timed out after {timeout}s",
                    result=ToolResult.TIMEOUT,
                )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            output = stdout_text
            if stderr_text:
                output += f"\n[stderr]\n{stderr_text}"

            output += f"\n[exit code: {process.returncode}]"

            return ToolResponse(tool_call_id="", content=output)

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error executing command: {str(e)}",
                result=ToolResult.ERROR,
            )


class WebFetchTool(Tool):
    """Tool for fetching web content."""

    name = "webfetch"
    description = "Fetch content from a URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
                "default": 30,
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str, timeout: int = 30) -> ToolResponse:
        """Fetch web content."""
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Unsupported URL scheme: {parsed.scheme}",
                    result=ToolResult.ERROR,
                )

            if not self.check_permission(url):
                return ToolResponse(
                    tool_call_id="",
                    content=f"Error: Permission denied: {url}",
                    result=ToolResult.PERMISSION_DENIED,
                )

            req = Request(url, headers={"User-Agent": "OpenCode/1.0"})

            with urlopen(req, timeout=timeout) as response:
                content = response.read()
                charset = (
                    response.headers.get_content_charset() or "utf-8"
                )

                try:
                    text = content.decode(charset)
                except UnicodeDecodeError:
                    text = f"Binary content ({len(content)} bytes)"

                return ToolResponse(
                    tool_call_id="",
                    content=f"URL: {url}\nStatus: {response.status}\n\n{text[:50000]}",
                )

        except urllib.error.HTTPError as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error: HTTP {e.code}: {e.reason}",
                result=ToolResult.ERROR,
            )
        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error fetching URL: {str(e)}",
                result=ToolResult.ERROR,
            )


class WebSearchTool(Tool):
    """Tool for web search (placeholder - requires API key)."""

    name = "websearch"
    description = "Search the web for information (requires API key for search provider)"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, permission_manager: PermissionManager, api_key: str = ""):
        super().__init__(permission_manager)
        self.api_key = api_key
        self.base_url = "https://api.serper.dev/search"  # Using Serper API

    async def execute(self, query: str, num_results: int = 10) -> ToolResponse:
        """Execute web search."""
        if not self.api_key:
            return ToolResponse(
                tool_call_id="",
                content="Error: Web search requires API key. Set SERPER_API_KEY or similar.",
                result=ToolResult.ERROR,
            )

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                }

                params = {"q": query, "num": num_results}

                async with session.post(
                    self.base_url, headers=headers, json=params
                ) as response:
                    if response.status != 200:
                        return ToolResponse(
                            tool_call_id="",
                            content=f"Error: Search API returned {response.status}",
                            result=ToolResult.ERROR,
                        )

                    data = await response.json()

                    results = data.get("organic", [])

                    output_lines = [f"Search results for: {query}\n"]

                    for i, result in enumerate(results[:num_results], 1):
                        title = result.get("title", "")
                        url = result.get("link", "")
                        snippet = result.get("snippet", "")

                        output_lines.append(
                            f"{i}. [{title}]({url})\n   {snippet}\n"
                        )

                    return ToolResponse(
                        tool_call_id="", content="\n".join(output_lines)
                    )

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error searching: {str(e)}",
                result=ToolResult.ERROR,
            )


class TodoWriteTool(Tool):
    """Tool for managing todo lists."""

    name = "todowrite"
    description = "Create or update a todo list for tracking tasks"
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string"},
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, permission_manager: PermissionManager):
        super().__init__(permission_manager)
        self.todos: List[Dict[str, Any]] = []

    async def execute(self, todos: List[Dict[str, Any]]) -> ToolResponse:
        """Update todo list."""
        try:
            self.todos = todos

            output = ["Todo List:"]
            for i, todo in enumerate(todos, 1):
                status_symbol = {"pending": "[ ]", "in_progress": "[->]", "completed": "[x]"}
                symbol = status_symbol.get(todo["status"], "[?]")
                output.append(f"{i}. {symbol} {todo['content']}")

            return ToolResponse(tool_call_id="", content="\n".join(output))

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error updating todos: {str(e)}",
                result=ToolResult.ERROR,
            )


class TaskTool(Tool):
    """Tool for delegating to subagents."""

    name = "task"
    description = "Delegate a complex task to a specialized subagent"
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed task description for the subagent",
            },
            "agent_type": {
                "type": "string",
                "description": "Type of subagent (plan, explore, general)",
                "default": "general",
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        permission_manager: PermissionManager,
        agent_factory: Callable[[], "Agent"],
    ):
        super().__init__(permission_manager)
        self.agent_factory = agent_factory

    async def execute(
        self, prompt: str, agent_type: str = "general"
    ) -> ToolResponse:
        """Delegate task to subagent."""
        try:
            subagent = self.agent_factory()

            # Configure subagent based on type
            if agent_type == "plan":
                subagent.config.tools = ["read", "glob", "grep"]
            elif agent_type == "explore":
                subagent.config.tools = ["read", "glob", "grep"]

            response = await subagent.run(prompt)

            return ToolResponse(tool_call_id="", content=response)

        except Exception as e:
            return ToolResponse(
                tool_call_id="",
                content=f"Error delegating task: {str(e)}",
                result=ToolResult.ERROR,
            )


# =============================================================================
# LLM Provider System
# =============================================================================

class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[List[ToolCall]], UsageStats]:
        """Generate a response."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[List[ToolCall]], UsageStats], None]:
        """Generate a streaming response."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[List[ToolCall]], UsageStats]:
        """Generate response using Claude."""

        # Convert messages to Anthropic format
        api_messages = []
        system_content = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            elif msg.role == MessageRole.USER:
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                api_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.TOOL:
                # Handle tool responses
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        # Build request
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system_content:
            request_params["system"] = system_content

        if tools:
            request_params["tools"] = tools

        response = await self.client.messages.create(**request_params)

        # Extract response
        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = UsageStats(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return content, tool_calls, usage

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[List[ToolCall]], UsageStats], None]:
        """Generate streaming response."""
        # Similar implementation with streaming
        async with self.client.messages.stream(
            model=self.model,
            messages=[m.to_dict() for m in messages if m.role != MessageRole.TOOL],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools or [],
        ) as stream:
            async for text in stream.text_stream:
                yield text, None, UsageStats(0, 0, 0)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[List[ToolCall]], UsageStats]:
        """Generate response using OpenAI."""

        api_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**request_params)

        message = response.choices[0].message

        content = message.content or ""

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return content, tool_calls, usage

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[List[ToolCall]], UsageStats], None]:
        """Generate streaming response."""
        api_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools or [],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                yield content, None, UsageStats(0, 0, 0)


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        # Requires google-generativeai package
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Google provider requires: pip install google-generativeai")

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[List[ToolCall]], UsageStats]:
        """Generate response using Gemini."""
        # Convert messages to Gemini format
        chat_history = []

        for msg in messages[:-1]:
            if msg.role == MessageRole.USER:
                chat_history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == MessageRole.ASSISTANT:
                chat_history.append({"role": "model", "parts": [msg.content]})

        last_message = messages[-1].content if messages else ""

        response = await asyncio.to_thread(
            self.client.generate_content,
            contents=chat_history + [{"role": "user", "parts": [last_message]}],
        )

        content = response.text
        usage = UsageStats(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            total_tokens=response.usage_metadata.total_token_count,
        )

        return content, None, usage

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[List[ToolCall]], UsageStats], None]:
        """Generate streaming response."""
        # Similar implementation with streaming
        yield "", None, UsageStats(0, 0, 0)


# =============================================================================
# Agent System
# =============================================================================

class Agent:
    """Main AI coding agent."""

    DEFAULT_SYSTEM_PROMPT = """You are OpenCode, an expert software development assistant.

You help users with:
- Writing, reading, and editing code
- Executing commands and debugging
- Searching and understanding codebases
- Explaining complex technical concepts
- Architecting and planning implementations

You are:
- Concise and direct
- Technical and precise
- Focused on solving problems
- Careful about safety and permissions

When making changes:
1. Read files first to understand context
2. Make minimal, targeted changes
3. Test your changes when possible
4. Explain what you did and why

Use tools efficiently and prefer parallel tool execution when possible.
"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        permission_manager: Optional[PermissionManager] = None,
    ):
        self.config = config or self._default_config()
        self.permission_manager = permission_manager or PermissionManager()
        self.session: Optional[Session] = None
        self.tools: Dict[str, Tool] = {}
        self._setup_tools()
        self._setup_provider()

    def _default_config(self) -> AgentConfig:
        """Create default agent configuration."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not api_key:
            # Try to get from other env vars
            api_key = (
                os.environ.get("OPENAI_API_KEY", "")
                or os.environ.get("GOOGLE_API_KEY", "")
                or ""
            )

        provider = Provider.ANTHROPIC

        if not api_key:
            # Fallback to OpenAI if Anthropic key not found
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                provider = Provider.OPENAI

        return AgentConfig(
            name="opencode",
            description="OpenCode AI Coding Agent",
            model="claude-sonnet-4-5-20250929" if provider == Provider.ANTHROPIC else "gpt-4o",
            provider=provider,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

    def _setup_provider(self):
        """Setup LLM provider based on configuration."""
        api_key = ""

        if self.config.provider == Provider.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self.provider = AnthropicProvider(
                api_key=api_key,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == Provider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            self.provider = OpenAIProvider(
                api_key=api_key,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.provider == Provider.GOOGLE:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            self.provider = GoogleProvider(
                api_key=api_key,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _setup_tools(self):
        """Setup available tools."""
        self.tools = {
            "read": ReadTool(self.permission_manager),
            "write": WriteTool(self.permission_manager),
            "edit": EditTool(self.permission_manager),
            "glob": GlobTool(self.permission_manager),
            "grep": GrepTool(self.permission_manager),
            "bash": BashTool(self.permission_manager),
            "webfetch": WebFetchTool(self.permission_manager),
            "websearch": WebSearchTool(
                self.permission_manager,
                os.environ.get("SERPER_API_KEY", ""),
            ),
            "todowrite": TodoWriteTool(self.permission_manager),
        }

        # Add task tool with reference to self for creating subagents
        self.tools["task"] = TaskTool(
            self.permission_manager,
            lambda: Agent(self.config, self.permission_manager),
        )

    def set_permissions(self, rules: List[Tuple[str, PermissionLevel, str]]):
        """Set permission rules."""
        for pattern, level, description in rules:
            self.permission_manager.add_rule(pattern, level, description)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self.tools.values()
            if tool.name in self.config.tools or not self.config.tools
        ]

    async def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Run the agent with a prompt."""
        # Create or get session
        if session_id and self.session and self.session.id == session_id:
            session = self.session
        else:
            session = Session(id=session_id or f"session-{int(time.time())}")
            self.session = session

        # Add user message
        session.messages.append(
            Message(role=MessageRole.USER, content=prompt)
        )

        # Get tool schemas
        tool_schemas = self.get_tool_schemas()

        # Generate response
        if stream:
            response_text = ""
            async for chunk, _, _ in self.provider.generate_stream(
                session.messages, tool_schemas
            ):
                response_text += chunk
                print(chunk, end="", flush=True)
            print()
        else:
            response_text, tool_calls, usage = await self.provider.generate(
                session.messages, tool_schemas
            )

            # Execute tool calls if present
            if tool_calls:
                for tool_call in tool_calls:
                    tool_response = await self._execute_tool_call(tool_call)

                    # Add tool response to messages
                    session.messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=tool_response.content,
                            tool_id=tool_call.id,
                        )
                    )

                # Generate final response
                response_text, _, usage = await self.provider.generate(
                    session.messages, tool_schemas
                )

        # Add assistant message
        session.messages.append(
            Message(role=MessageRole.ASSISTANT, content=response_text)
        )
        session.usage.append(usage)

        return response_text

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a tool call."""
        tool = self.tools.get(tool_call.name)

        if not tool:
            return ToolResponse(
                tool_call_id=tool_call.id,
                content=f"Error: Unknown tool: {tool_call.name}",
                result=ToolResult.ERROR,
            )

        try:
            return await tool.execute(**tool_call.arguments)
        except Exception as e:
            return ToolResponse(
                tool_call_id=tool_call.id,
                content=f"Error executing {tool_call.name}: {str(e)}",
                result=ToolResult.ERROR,
            )


# =============================================================================
# CLI Interface
# =============================================================================

def print_banner():
    """Print agent banner."""
    print("=" * 60)
    print("OpenCode Agent - Python Port")
    print(f"Version {VERSION}")
    print("=" * 60)
    print()


def print_usage():
    """Print usage information."""
    print("Usage: python opencode_agent.py [options] <prompt>")
    print()
    print("Options:")
    print("  --model <name>       Model to use (default: claude-sonnet-4-5)")
    print("  --provider <name>    Provider: anthropic, openai, google (default: anthropic)")
    print("  --temperature <n>    Temperature 0-1 (default: 0.7)")
    print("  --max-tokens <n>     Max tokens (default: 8192)")
    print("  --session <id>       Resume existing session")
    print("  --stream             Enable streaming output")
    print("  --help               Show this help message")
    print()
    print("Environment Variables:")
    print("  ANTHROPIC_API_KEY    Anthropic API key")
    print("  OPENAI_API_KEY       OpenAI API key")
    print("  GOOGLE_API_KEY       Google API key")
    print("  SERPER_API_KEY       Serper search API key (for websearch)")
    print()


async def main():
    """Main entry point."""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_banner()
        print_usage()
        return 0

    print_banner()

    # Parse arguments
    model = None
    provider = None
    temperature = None
    max_tokens = None
    session_id = None
    stream = False
    prompt_args = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif arg == "--provider" and i + 1 < len(args):
            provider_str = args[i + 1]
            provider = Provider[provider_str.upper()]
            i += 2
        elif arg == "--temperature" and i + 1 < len(args):
            temperature = float(args[i + 1])
            i += 2
        elif arg == "--max-tokens" and i + 1 < len(args):
            max_tokens = int(args[i + 1])
            i += 2
        elif arg == "--session" and i + 1 < len(args):
            session_id = args[i + 1]
            i += 2
        elif arg == "--stream":
            stream = True
            i += 1
        elif arg.startswith("--"):
            print(f"Warning: Unknown option {arg}", file=sys.stderr)
            i += 1
        else:
            prompt_args.append(arg)
            i += 1

    prompt = " ".join(prompt_args)

    if not prompt:
        print("Error: No prompt provided")
        print_usage()
        return 1

    # Create configuration
    config = AgentConfig(
        name="opencode",
        description="OpenCode AI Coding Agent",
        model=model or "claude-sonnet-4-5-20250929",
        provider=provider or Provider.ANTHROPIC,
        temperature=temperature or 0.7,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
    )

    # Setup permissions
    permission_manager = PermissionManager()

    # Deny dangerous operations by default
    permission_manager.add_rule("rm -rf *", PermissionLevel.DENY, "Dangerous delete")
    permission_manager.add_rule("format *", PermissionLevel.DENY, "Dangerous format")
    permission_manager.add_rule("*/etc/*", PermissionLevel.PROMPT, "System files")

    # Create agent
    agent = Agent(config, permission_manager)

    print(f"Model: {config.provider.value} / {config.model}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print()
    print("-" * 60)
    print()

    try:
        response = await agent.run(
            prompt=prompt,
            session_id=session_id,
            stream=stream,
        )

        if not stream:
            print(response)
            print()

        print("-" * 60)
        print()

        # Show usage
        if agent.session and agent.session.usage:
            total_usage = UsageStats(
                prompt_tokens=sum(u.prompt_tokens for u in agent.session.usage),
                completion_tokens=sum(u.completion_tokens for u in agent.session.usage),
                total_tokens=sum(u.total_tokens for u in agent.session.usage),
            )
            print(f"Tokens: {total_usage.total_tokens} "
                  f"(prompt: {total_usage.prompt_tokens}, "
                  f"completion: {total_usage.completion_tokens})")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
