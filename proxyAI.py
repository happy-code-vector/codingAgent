#!/usr/bin/env python3
"""
A unified coding agent that combines tool execution, LLM interaction, and task management.

This single-file agent consolidates the multi-agent architecture into a streamlined
implementation with support for:
- File operations (read, write, edit)
- Code search and exploration
- Bash command execution
- Web search capabilities
- Task/todo tracking
- User interaction/confirmation
- Multiple LLM provider support (OpenAI, Anthropic, Google, Ollama, Custom)
"""

import abc
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from functools import wraps
from contextlib import asynccontextmanager

import aiohttp
from pydantic import BaseModel, Field


# =============================================================================
# Configuration and Constants
# =============================================================================

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    CUSTOM_OPENAI = "custom_openai"
    PROXYAI = "proxyai"


class ToolApprovalType(Enum):
    """Types of tool approval for UI."""
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    BASH = "bash"
    GENERIC = "generic"


@dataclass
class ToolApprovalRequest:
    """Request for user approval of a tool call."""
    approval_type: ToolApprovalType
    title: str
    details: str
    payload: Optional[Dict[str, Any]] = None


# =============================================================================
# Events System
# =============================================================================

class AgentEvents:
    """
    Event bus for agent execution events.
    Override methods to handle specific events.
    """

    def on_text_received(self, text: str):
        """Called when text is received from LLM."""
        pass

    def on_tool_starting(self, tool_id: str, tool_name: str, args: Dict[str, Any]):
        """Called before a tool is executed."""
        pass

    def on_tool_completed(self, tool_id: str, tool_name: str, result: Any):
        """Called after a tool completes execution."""
        pass

    def on_agent_completed(self, agent_id: str):
        """Called when agent execution completes."""
        pass

    async def approve_tool_call(self, request: ToolApprovalRequest) -> bool:
        """Request user approval for a tool call. Return True if approved."""
        return True  # Default: auto-approve

    def on_token_usage_available(self, token_count: int):
        """Called when token usage information is available."""
        pass

    def on_credits_available(self, remaining: int, monthly_remaining: int, consumed: int):
        """Called when credit information is available."""
        pass

    def on_history_compression_state_changed(self, is_compressing: bool):
        """Called when history compression state changes."""
        pass

    def on_queued_messages_resolved(self):
        """Called when queued messages are processed."""
        pass


# =============================================================================
# LLM Client Interface
# =============================================================================

@dataclass
class Message:
    """Base message class."""
    pass


@dataclass
class SystemMessage(Message):
    """System message with instructions."""
    content: str


@dataclass
class UserMessage(Message):
    """User message."""
    content: str


@dataclass
class AssistantMessage(Message):
    """Assistant message."""
    content: str


@dataclass
class ToolCallMessage(Message):
    """Tool call message."""
    tool_id: str
    tool_name: str
    args: Dict[str, Any]


@dataclass
class ToolResultMessage(Message):
    """Tool result message."""
    tool_id: str
    result: Any


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tool_calls: List[ToolCallMessage] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class LLMClient(abc.ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    @asynccontextmanager
    async def session(self):
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        try:
            yield self._session
        finally:
            pass  # Keep session alive for reuse

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abc.abstractmethod
    async def complete(
        self,
        messages: List[Message],
        tools: List['Tool'],
        **kwargs
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass

    @abc.abstractmethod
    def get_system_prompt(self, working_dir: str, current_date: str) -> str:
        """Get system prompt for this provider."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o"
    ):
        super().__init__(api_key, base_url, model)

    async def complete(
        self,
        messages: List[Message],
        tools: List['Tool'],
        **kwargs
    ) -> LLMResponse:
        api_messages = self._format_messages(messages)
        api_tools = [t.get_schema() for t in tools] if tools else None

        payload = {
            "model": self.model,
            "messages": api_messages,
        }
        if api_tools:
            payload["tools"] = api_tools

        async with self.session() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()
                return self._parse_response(data)

    def _format_messages(self, messages: List[Message]) -> List[Dict]:
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolCallMessage):
                formatted.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": msg.tool_id,
                        "type": "function",
                        "function": {
                            "name": msg.tool_name,
                            "arguments": json.dumps(msg.args)
                        }
                    }]
                })
            elif isinstance(msg, ToolResultMessage):
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_id,
                    "content": str(msg.result)
                })
        return formatted

    def _parse_response(self, data: Dict) -> LLMResponse:
        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content", "") or ""
        tool_calls = []

        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                func = tc["function"]
                tool_calls.append(ToolCallMessage(
                    tool_id=tc["id"],
                    tool_name=func["name"],
                    args=json.loads(func["arguments"])
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=data.get("usage"),
            model=data.get("model")
        )

    def get_system_prompt(self, working_dir: str, current_date: str) -> str:
        return _get_openai_system_prompt(working_dir, current_date)


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-sonnet-4-5-20250929"
    ):
        super().__init__(api_key, base_url, model)

    async def complete(
        self,
        messages: List[Message],
        tools: List['Tool'],
        **kwargs
    ) -> LLMResponse:
        # Anthropic uses a different message format
        api_messages = []
        system_prompt = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolCallMessage):
                # Anthropic uses content blocks for tool calls
                api_messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": msg.tool_id,
                        "name": msg.tool_name,
                        "input": msg.args
                    }]
                })
            elif isinstance(msg, ToolResultMessage):
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_id,
                        "content": str(msg.result)
                    }]
                })

        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": 8192,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [t.get_anthropic_schema() for t in tools]

        async with self.session() as session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()
                return self._parse_response(data)

    def _parse_response(self, data: Dict) -> LLMResponse:
        content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCallMessage(
                    tool_id=block["id"],
                    tool_name=block["name"],
                    args=block["input"]
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=data.get("usage"),
            model=data.get("model")
        )

    def get_system_prompt(self, working_dir: str, current_date: str) -> str:
        return _get_anthropic_system_prompt(working_dir, current_date)


class GoogleClient(LLMClient):
    """Google Gemini API client."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-2.0-flash-exp"
    ):
        super().__init__(api_key, base_url, model)

    async def complete(
        self,
        messages: List[Message],
        tools: List['Tool'],
        **kwargs
    ) -> LLMResponse:
        # Convert messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
            elif isinstance(msg, UserMessage):
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif isinstance(msg, AssistantMessage):
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
            elif isinstance(msg, ToolCallMessage) or isinstance(msg, ToolResultMessage):
                # Gemini has different tool call/result format
                pass  # Simplified for now

        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        async with self.session() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                # Parse Gemini response format
                text = ""
                if "candidates" in data and data["candidates"]:
                    for part in data["candidates"][0].get("content", {}).get("parts", []):
                        if "text" in part:
                            text += part["text"]
                return LLMResponse(content=text)

    def get_system_prompt(self, working_dir: str, current_date: str) -> str:
        return _get_google_system_prompt(working_dir, current_date)


class OllamaClient(LLMClient):
    """Ollama local model client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2"
    ):
        super().__init__("", base_url, model)

    async def complete(
        self,
        messages: List[Message],
        tools: List['Tool'],
        **kwargs
    ) -> LLMResponse:
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                api_messages.append({"role": "assistant", "content": msg.content})

        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": False
        }

        async with self.session() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                data = await resp.json()
                return LLMResponse(
                    content=data.get("message", {}).get("content", "")
                )

    def get_system_prompt(self, working_dir: str, current_date: str) -> str:
        return _get_openai_system_prompt(working_dir, current_date)


def create_llm_client(
    provider: LLMProvider,
    api_key: str = "",
    base_url: str = "",
    model: str = ""
) -> LLMClient:
    """Factory function to create LLM client."""
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o")
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(api_key=api_key, model=model or "claude-sonnet-4-5-20250929")
    elif provider == LLMProvider.GOOGLE:
        return GoogleClient(api_key=api_key, model=model or "gemini-2.0-flash-exp")
    elif provider == LLMProvider.OLLAMA:
        return OllamaClient(base_url=base_url or "http://localhost:11434", model=model or "llama3.2")
    elif provider == LLMProvider.CUSTOM_OPENAI:
        return OpenAIClient(api_key=api_key, base_url=base_url, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# =============================================================================
# System Prompts
# =============================================================================

def _get_openai_system_prompt(working_dir: str, current_date: str) -> str:
    """Get OpenAI-style system prompt."""
    return f"""You are a coding agent that helps with software engineering tasks.

IMPORTANT: Assist with authorized security testing, defensive security, and educational contexts. Refuse requests for destructive techniques, DoS attacks, or malicious activities.

# Tone and Style
- Keep responses short and concise
- Use GitHub-flavored markdown for formatting
- Output text directly to communicate; don't use tools for communication

# Task Management
Use the TodoWrite tool frequently to track tasks and give visibility into progress.

# Tool Usage Policy
- Use specialized tools over bash when possible
- Call multiple tools in parallel when independent
- Read files before editing them
- Never guess parameters - gather them first

# Code References
When referencing code, use the format `file_path:line_number` for easy navigation.

<env>
Working directory: {working_dir}
Current date: {current_date}
</env>
"""

def _get_anthropic_system_prompt(working_dir: str, current_date: str) -> str:
    """Get Anthropic-style system prompt."""
    return f"""You are Claude Code, a coding agent that helps with software engineering tasks.

IMPORTANT: Assist with authorized security testing, defensive security, and educational contexts. Refuse requests for destructive techniques, DoS attacks, or malicious activities.

# Tone and Style
- Keep responses short and concise
- Use GitHub-flavored markdown for formatting
- Output text directly to communicate; don't use tools for communication

# Task Management
Use the TodoWrite tool frequently to track tasks and give visibility into progress.

# Tool Usage Policy
- Use specialized tools over bash when possible
- Call multiple tools in parallel when independent
- Read files before editing them
- Never guess parameters - gather them first

# Code References
When referencing code, use the format `file_path:line_number` for easy navigation.

<env>
Working directory: {working_dir}
Current date: {current_date}
</env>
"""

def _get_google_system_prompt(working_dir: str, current_date: str) -> str:
    """Get Google-style system prompt."""
    return _get_openai_system_prompt(working_dir, current_date)


# =============================================================================
# Tool System
# =============================================================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None


class Tool(abc.ABC):
    """Abstract base class for tools."""

    def __init__(self, working_dir: str = ""):
        self.working_dir = working_dir or os.getcwd()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema (JSON Schema format)."""
        return {}

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            return await self._execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    @abc.abstractmethod
    async def _execute(self, **kwargs) -> ToolResult:
        """Actual tool implementation."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI-style tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }

    def get_anthropic_schema(self) -> Dict[str, Any]:
        """Get Anthropic-style tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }


class ReadTool(Tool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read the contents of a file"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file"
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read"
            }
        }

    async def _execute(self, file_path: str, offset: int = 0, limit: int = None) -> ToolResult:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self.working_dir) / file_path

        if not path.exists():
            return ToolResult(success=False, data=None, error=f"File not found: {file_path}")

        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            if offset > 0 or limit is not None:
                end = (offset + limit) if limit else None
                lines = lines[offset:end]

            # Format with line numbers
            result = "\n".join(f"{i + offset + 1}:    {line}" for i, line in enumerate(lines))
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WriteTool(Tool):
    """Tool for writing/creating files."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write content to a file (overwrites if exists)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file"
            },
            "content": {
                "type": "string",
                "description": "Content to write"
            }
        }

    async def _execute(self, file_path: str, content: str) -> ToolResult:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self.working_dir) / file_path

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content, encoding='utf-8')
            return ToolResult(success=True, data=f"Written to {file_path}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class EditTool(Tool):
    """Tool for editing files with string replacement."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return "Replace text in a file using exact string matching"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file"
            },
            "old_string": {
                "type": "string",
                "description": "Text to replace (must match exactly)"
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text"
            }
        }

    async def _execute(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self.working_dir) / file_path

        if not path.exists():
            return ToolResult(success=False, data=None, error=f"File not found: {file_path}")

        try:
            content = path.read_text(encoding='utf-8')
            if old_string not in content:
                return ToolResult(success=False, data=None, error="Old string not found in file")

            new_content = content.replace(old_string, new_string, 1)
            path.write_text(new_content, encoding='utf-8')
            return ToolResult(success=True, data=f"Edited {file_path}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class BashTool(Tool):
    """Tool for executing bash commands."""

    def __init__(self, working_dir: str = "", timeout: int = 120):
        super().__init__(working_dir)
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Execute a bash command in the shell"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "command": {
                "type": "string",
                "description": "Command to execute"
            },
            "description": {
                "type": "string",
                "description": "Description of what the command does"
            }
        }

    async def _execute(self, command: str, description: str = "") -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode('utf-8', errors='replace')
            error = stderr.decode('utf-8', errors='replace')

            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    data=output,
                    error=error or f"Command exited with code {proc.returncode}"
                )

            return ToolResult(success=True, data=output)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, data=None, error=f"Command timed out after {self.timeout}s")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WebSearchTool(Tool):
    """Tool for web search (requires implementation)."""

    @property
    def name(self) -> str:
        return "websearch"

    @property
    def description(self) -> str:
        return "Search the web for information"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        }

    async def _execute(self, query: str) -> ToolResult:
        # Placeholder - requires actual search API integration
        return ToolResult(
            success=True,
            data=f"Web search for: {query}\n(Note: Requires search API integration)"
        )


class TodoWriteTool(Tool):
    """Tool for managing todo lists."""

    def __init__(self, working_dir: str = ""):
        super().__init__(working_dir)
        self._todos: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "todowrite"

    @property
    def description(self) -> str:
        return "Manage todo list to track tasks"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "todos": {
                "type": "array",
                "description": "List of todo items with 'content', 'status', and 'activeForm'",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string"}
                    },
                    "required": ["content", "status", "activeForm"]
                }
            }
        }

    async def _execute(self, todos: List[Dict[str, Any]]) -> ToolResult:
        self._todos = todos
        summary = "\n".join(
            f"[{t['status'][0].upper()}] {t['content']}"
            for t in todos
        )
        return ToolResult(success=True, data=summary)

    def get_todos(self) -> List[Dict[str, Any]]:
        return self._todos


class GrepTool(Tool):
    """Tool for searching content in files."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search for patterns in files"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Path to search in (default: working directory)"
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to search (e.g., '*.py')"
            }
        }

    async def _execute(self, pattern: str, path: str = "", file_pattern: str = "") -> ToolResult:
        search_path = Path(path) if path else Path(self.working_dir)

        try:
            results = []
            regex = re.compile(pattern)

            if file_pattern:
                files = list(search_path.rglob(file_pattern))
            else:
                files = list(search_path.rglob("*"))

            for file_path in files:
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{file_path}:{i}: {line}")
                except Exception:
                    continue

            return ToolResult(success=True, data="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GlobTool(Tool):
    """Tool for finding files by pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py')"
            },
            "path": {
                "type": "string",
                "description": "Base path to search (default: working directory)"
            }
        }

    async def _execute(self, pattern: str, path: str = "") -> ToolResult:
        search_path = Path(path) if path else Path(self.working_dir)

        try:
            matches = list(search_path.rglob(pattern))
            matches = [str(m.relative_to(search_path)) for m in matches if m.is_file()]
            matches.sort()

            return ToolResult(success=True, data="\n".join(matches))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


# =============================================================================
# Main Agent
# =============================================================================

class CodingAgent:
    """
    Main coding agent that orchestrates tool use and LLM interaction.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        working_dir: str = "",
        events: Optional[AgentEvents] = None,
        max_iterations: int = 100
    ):
        self.provider = provider
        self.working_dir = working_dir or os.getcwd()
        self.events = events or AgentEvents()
        self.max_iterations = max_iterations

        # Create LLM client
        self.llm_client = create_llm_client(provider, api_key, base_url, model)

        # Register tools
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

        # Conversation history
        self.messages: List[Message] = []

    def _register_default_tools(self):
        """Register default tools."""
        tools = [
            ReadTool(self.working_dir),
            WriteTool(self.working_dir),
            EditTool(self.working_dir),
            BashTool(self.working_dir),
            WebSearchTool(self.working_dir),
            TodoWriteTool(self.working_dir),
            GrepTool(self.working_dir),
            GlobTool(self.working_dir),
        ]
        for tool in tools:
            self.tools[tool.name] = tool

    def register_tool(self, tool: Tool):
        """Register a custom tool."""
        self.tools[tool.name] = tool

    def _get_system_message(self) -> SystemMessage:
        """Get system message for current provider."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = self.llm_client.get_system_prompt(self.working_dir, current_date)
        return SystemMessage(content=prompt)

    async def run(self, user_input: str) -> str:
        """
        Run the agent with user input.

        Args:
            user_input: The user's request

        Returns:
            The agent's final response
        """
        # Initialize with system message
        if not self.messages:
            self.messages.append(self._get_system_message())

        # Add user message
        self.messages.append(UserMessage(content=user_input))

        # Main agent loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Get LLM response
            try:
                response = await self.llm_client.complete(
                    self.messages,
                    list(self.tools.values())
                )
            except Exception as e:
                error_msg = f"LLM error: {e}"
                self.events.on_text_received(error_msg)
                return error_msg

            # Stream text content
            if response.content:
                self.events.on_text_received(response.content)

            # Add assistant message to history
            self.messages.append(AssistantMessage(content=response.content))

            # Check if we have tool calls
            if not response.tool_calls:
                # No tool calls - we're done
                self.events.on_agent_completed(f"run_{iteration}")
                return response.content

            # Execute tools
            for tool_call in response.tool_calls:
                tool = self.tools.get(tool_call.tool_name)
                if not tool:
                    result = ToolResult(
                        success=False,
                        data=None,
                        error=f"Tool not found: {tool_call.tool_name}"
                    )
                else:
                    self.events.on_tool_starting(
                        tool_call.tool_id,
                        tool_call.tool_name,
                        tool_call.args
                    )

                    result = await tool.execute(**tool_call.args)

                    self.events.on_tool_completed(
                        tool_call.tool_id,
                        tool_call.tool_name,
                        result
                    )

                # Add tool call and result to history
                self.messages.append(tool_call)
                self.messages.append(ToolResultMessage(
                    tool_id=tool_call.tool_id,
                    result=result.data if result.success else result.error
                ))

        self.events.on_agent_completed("max_iterations")
        return "Reached maximum iterations"

    async def close(self):
        """Clean up resources."""
        await self.llm_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Coding Agent")
    parser.add_argument("prompt", nargs="*", help="User prompt")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "ollama"],
                        default="anthropic", help="LLM provider")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="Base URL for custom/ollama")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--working-dir", help="Working directory", default=".")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Map provider string to enum
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "google": LLMProvider.GOOGLE,
        "ollama": LLMProvider.OLLAMA,
    }
    provider = provider_map[args.provider]

    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv(
        f"{args.provider.upper()}_API_KEY"
    ) or os.getenv("LLM_API_KEY") or ""

    # Create agent
    async with CodingAgent(
        provider=provider,
        api_key=api_key,
        base_url=args.base_url or "",
        model=args.model or "",
        working_dir=args.working_dir
    ) as agent:
        if args.interactive:
            print("Coding Agent - Interactive Mode")
            print("Type 'exit' or 'quit' to exit\n")

            while True:
                try:
                    user_input = input("You> ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        break
                    if not user_input:
                        continue

                    print("\nAgent>", end=" ")
                    response = await agent.run(user_input)
                    print(f"{response}\n")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
        elif args.prompt:
            user_input = " ".join(args.prompt)
            response = await agent.run(user_input)
            print(response)
        else:
            parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
