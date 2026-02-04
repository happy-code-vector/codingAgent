#!/usr/bin/env python3
"""
Cline-inspired Coding Agent - Single File Implementation

A simplified Python version of the Cline VSCode extension that can:
- Read and write files
- Execute shell commands
- Search code
- Maintain conversation context
- Use multiple LLM providers (Anthropic, OpenAI, etc.)
"""

import os
import sys
import json
import re
import subprocess
import asyncio
import aiofiles
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import time
from datetime import datetime

# Third-party imports (can be installed via pip)
try:
    import httpx
    from pydantic import BaseModel
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install httpx pydantic")
    sys.exit(1)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class ApiProvider(Enum):
    """Supported API providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class AgentConfig:
    """Configuration for the coding agent"""
    api_provider: str = ApiProvider.ANTHROPIC.value
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192
    working_dir: str = "."
    auto_approve_commands: bool = False
    max_iterations: int = 50
    timeout: int = 120

    def __post_init__(self):
        # Set default base URLs
        if not self.base_url:
            if self.api_provider == ApiProvider.ANTHROPIC.value:
                self.base_url = "https://api.anthropic.com"
            elif self.api_provider == ApiProvider.OPENAI.value:
                self.base_url = "https://api.openai.com/v1"
            elif self.api_provider == ApiProvider.OPENROUTER.value:
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.api_provider == ApiProvider.OLLAMA.value:
                self.base_url = "http://localhost:11434"


# ============================================================================
# MESSAGE TYPES
# ============================================================================

@dataclass
class Message:
    """Base message type"""
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_call_id: str
    content: str
    is_error: bool = False


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

class Tool:
    """Base tool class"""

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


# Tool definitions
TOOLS = [
    Tool(
        name="read_file",
        description="Read the contents of a file. Use this to examine file contents before making changes.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }
    ),
    Tool(
        name="write_file",
        description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        parameters={
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
    ),
    Tool(
        name="list_files",
        description="List files and directories in a given path. Use this to explore the codebase structure.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list (default: current directory)"
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', 'src/**/*.ts')"
                }
            },
            "required": []
        }
    ),
    Tool(
        name="search_files",
        description="Search for text/patterns in files using ripgrep-like functionality.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (default: current directory)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py')"
                }
            },
            "required": ["pattern"]
        }
    ),
    Tool(
        name="execute_command",
        description="Execute a shell command. Use this for running tests, builds, git operations, etc.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        }
    ),
]


# ============================================================================
# API HANDLER
# ============================================================================

class ApiHandler:
    """Handles communication with LLM API providers"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self._setup_headers()

    def _setup_headers(self):
        """Setup API headers based on provider"""
        if self.config.api_provider == ApiProvider.ANTHROPIC.value:
            self.headers = {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        elif self.config.api_provider == ApiProvider.OPENAI.value:
            self.headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "content-type": "application/json"
            }
        elif self.config.api_provider == ApiProvider.OPENROUTER.value:
            self.headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "content-type": "application/json",
                "HTTP-Referer": "https://cline.dev",
                "X-Title": "Cline Python Agent"
            }
        else:  # Ollama
            self.headers = {"content-type": "application/json"}

    def _get_api_url(self) -> str:
        """Get the appropriate API endpoint"""
        if self.config.api_provider == ApiProvider.ANTHROPIC.value:
            return f"{self.config.base_url}/v1/messages"
        elif self.config.api_provider == ApiProvider.OPENAI.value:
            return f"{self.config.base_url}/chat/completions"
        elif self.config.api_provider == ApiProvider.OPENROUTER.value:
            return f"{self.config.base_url}/chat/completions"
        else:  # Ollama
            return f"{self.config.base_url}/api/chat"

    def _format_messages(self, messages: List[Message]) -> List[dict]:
        """Format messages for the specific API provider"""
        formatted = []
        for msg in messages:
            formatted.append(msg.to_dict())
        return formatted

    def _format_tools(self) -> List[dict]:
        """Format tools for the specific API provider"""
        return [tool.to_dict() for tool in TOOLS]

    async def send_message(
        self,
        messages: List[Message],
        stream: bool = True
    ) -> AsyncGenerator[dict, None]:
        """Send a message to the LLM and stream the response"""

        payload = self._build_payload(messages, stream)

        try:
            async with self.client.stream(
                "POST",
                self._get_api_url(),
                headers=self.headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API error {response.status_code}: {error_text.decode()}")

                if self.config.api_provider == ApiProvider.ANTHROPIC.value:
                    async for chunk in self._stream_anthropic(response):
                        yield chunk
                elif self.config.api_provider in [ApiProvider.OPENAI.value, ApiProvider.OPENROUTER.value]:
                    async for chunk in self._stream_openai(response):
                        yield chunk
                else:
                    async for chunk in self._stream_ollama(response):
                        yield chunk

        except Exception as e:
            yield {"error": str(e)}

    def _build_payload(self, messages: List[Message], stream: bool) -> dict:
        """Build the request payload"""

        formatted_messages = self._format_messages(messages)

        if self.config.api_provider == ApiProvider.ANTHROPIC.value:
            return {
                "model": self.config.model,
                "messages": formatted_messages,
                "tools": self._format_tools(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": stream
            }
        elif self.config.api_provider in [ApiProvider.OPENAI.value, ApiProvider.OPENROUTER.value]:
            return {
                "model": self.config.model,
                "messages": formatted_messages,
                "tools": self._format_tools(),
                "tool_choice": "auto",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": stream
            }
        else:  # Ollama
            return {
                "model": self.config.model,
                "messages": formatted_messages,
                "tools": self._format_tools(),
                "stream": stream
            }

    async def _stream_anthropic(self, response) -> AsyncGenerator[dict, None]:
        """Stream Anthropic responses"""
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
                if chunk.get("type") == "content_block_delta":
                    yield {"delta": chunk.get("delta", {}).get("text", "")}
                elif chunk.get("type") == "content_block_start":
                    block = chunk.get("content_block", {})
                    if block.get("type") == "tool_use":
                        yield {
                            "tool_call": {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": {}
                            }
                        }
                elif chunk.get("type") == "input_json_delta":
                    yield {"tool_arguments": chunk.get("partial_json", "")}
            except json.JSONDecodeError:
                continue

    async def _stream_openai(self, response) -> AsyncGenerator[dict, None]:
        """Stream OpenAI/OpenRouter responses"""
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Text content
                if "content" in delta:
                    yield {"delta": delta["content"]}

                # Tool call start
                if "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        if tool_call.get("function"):
                            yield {
                                "tool_call": {
                                    "id": tool_call.get("id", ""),
                                    "name": tool_call["function"].get("name", ""),
                                    "arguments": {}
                                }
                            }
                            if "arguments" in tool_call["function"]:
                                yield {"tool_arguments": tool_call["function"]["arguments"]}

            except json.JSONDecodeError:
                continue

    async def _stream_ollama(self, response) -> AsyncGenerator[dict, None]:
        """Stream Ollama responses"""
        async for line in response.aiter_lines():
            if not line:
                continue

            try:
                chunk = json.loads(line)
                if "message" in chunk:
                    msg = chunk["message"]
                    if "content" in msg:
                        yield {"delta": msg["content"]}
                    if "tool_calls" in msg:
                        for tool_call in msg["tool_calls"]:
                            yield {
                                "tool_call": {
                                    "id": tool_call.get("id", ""),
                                    "name": tool_call.get("function", {}).get("name", ""),
                                    "arguments": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                                }
                            }
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class ToolExecutor:
    """Executes tool calls"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.working_dir = Path(config.working_dir).resolve()

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call"""

        try:
            if tool_call.name == "read_file":
                return await self._read_file(tool_call)
            elif tool_call.name == "write_file":
                return await self._write_file(tool_call)
            elif tool_call.name == "list_files":
                return await self._list_files(tool_call)
            elif tool_call.name == "search_files":
                return await self._search_files(tool_call)
            elif tool_call.name == "execute_command":
                return await self._execute_command(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Unknown tool: {tool_call.name}",
                    is_error=True
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing {tool_call.name}: {str(e)}",
                is_error=True
            )

    async def _read_file(self, tool_call: ToolCall) -> ToolResult:
        """Read a file"""
        file_path = tool_call.arguments.get("file_path", "")
        full_path = self._resolve_path(file_path)

        if not full_path.exists():
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"File not found: {file_path}",
                is_error=True
            )

        if not full_path.is_file():
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Path is not a file: {file_path}",
                is_error=True
            )

        try:
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Add line numbers
            lines = content.split('\n')
            numbered = '\n'.join(f"{i+1:6d}\t{line}" for i, line in enumerate(lines))

            return ToolResult(
                tool_call_id=tool_call.id,
                content=numbered
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error reading file: {str(e)}",
                is_error=True
            )

    async def _write_file(self, tool_call: ToolCall) -> ToolResult:
        """Write to a file"""
        file_path = tool_call.arguments.get("file_path", "")
        content = tool_call.arguments.get("content", "")
        full_path = self._resolve_path(file_path)

        try:
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Successfully wrote {len(content)} bytes to {file_path}"
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error writing file: {str(e)}",
                is_error=True
            )

    async def _list_files(self, tool_call: ToolCall) -> ToolResult:
        """List files in a directory"""
        path = tool_call.arguments.get("path", ".")
        pattern = tool_call.arguments.get("pattern", "")

        full_path = self._resolve_path(path)

        if not full_path.exists():
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Path not found: {path}",
                is_error=True
            )

        if not full_path.is_dir():
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Path is not a directory: {path}",
                is_error=True
            )

        try:
            if pattern:
                # Use glob pattern
                matches = list(full_path.rglob(pattern))
                files = sorted([str(m.relative_to(full_path)) for m in matches])
            else:
                # List all files
                files = []
                for item in sorted(full_path.rglob("*")):
                    if item.is_file():
                        rel_path = str(item.relative_to(full_path))
                        files.append(rel_path)

            output = "\n".join(files) if files else "(empty directory)"

            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Files in {path}:\n{output}"
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error listing files: {str(e)}",
                is_error=True
            )

    async def _search_files(self, tool_call: ToolCall) -> ToolResult:
        """Search for patterns in files"""
        pattern = tool_call.arguments.get("pattern", "")
        path = tool_call.arguments.get("path", ".")
        file_pattern = tool_call.arguments.get("file_pattern", "")

        if not pattern:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="Pattern is required",
                is_error=True
            )

        full_path = self._resolve_path(path)

        try:
            results = []
            regex = re.compile(pattern)

            # Get files to search
            if file_pattern:
                files = list(full_path.rglob(file_pattern))
            else:
                files = [f for f in full_path.rglob("*") if f.is_file()]

            for file_path in files:
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(await f.readlines(), 1):
                            if regex.search(line):
                                rel_path = str(file_path.relative_to(self.working_dir))
                                results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                except Exception:
                    continue

            if not results:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"No matches found for pattern: {pattern}"
                )

            # Limit results
            results = results[:100]

            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Found {len(results)} matches:\n" + "\n".join(results)
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error searching files: {str(e)}",
                is_error=True
            )

    async def _execute_command(self, tool_call: ToolCall) -> ToolResult:
        """Execute a shell command"""
        command = tool_call.arguments.get("command", "")
        timeout = tool_call.arguments.get("timeout", self.config.timeout)

        if not command:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="Command is required",
                is_error=True
            )

        # Check for approval
        if not self.config.auto_approve_commands:
            print(f"\n⚠️  Command to execute: {command}")
            approve = input("Approve? [y/N]: ").strip().lower()
            if approve != 'y':
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content="Command cancelled by user",
                    is_error=True
                )

        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')

                output = stdout_text
                if stderr_text:
                    output += f"\n[stderr]\n{stderr_text}"

                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=output + f"\n[exit code: {process.returncode}]"
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Command timed out after {timeout} seconds",
                    is_error=True
                )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing command: {str(e)}",
                is_error=True
            )

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to working directory"""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.working_dir / path_obj


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class StateManager:
    """Manages conversation state and history"""

    def __init__(self, state_dir: str = None):
        if state_dir is None:
            state_dir = os.path.join(os.path.expanduser("~"), ".cline_agent")

        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.conversations: Dict[str, List[Message]] = {}
        self.current_conversation: Optional[str] = None

    def new_conversation(self) -> str:
        """Start a new conversation"""
        conv_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.conversations[conv_id] = []
        self.current_conversation = conv_id
        return conv_id

    def add_message(self, role: str, content: str):
        """Add a message to the current conversation"""
        if self.current_conversation and self.current_conversation in self.conversations:
            self.conversations[self.current_conversation].append(
                Message(role=role, content=content)
            )

    def get_messages(self) -> List[Message]:
        """Get messages from the current conversation"""
        if self.current_conversation and self.current_conversation in self.conversations:
            return self.conversations[self.current_conversation]
        return []

    def save_conversation(self, conv_id: str = None):
        """Save conversation to disk"""
        conv_id = conv_id or self.current_conversation
        if not conv_id or conv_id not in self.conversations:
            return

        state_file = self.state_dir / f"conversation_{conv_id}.json"
        with open(state_file, 'w') as f:
            json.dump([
                {"role": m.role, "content": m.content}
                for m in self.conversations[conv_id]
            ], f, indent=2)

    def load_conversation(self, conv_id: str):
        """Load conversation from disk"""
        state_file = self.state_dir / f"conversation_{conv_id}.json"
        if not state_file.exists():
            return False

        with open(state_file, 'r') as f:
            data = json.load(f)
            self.conversations[conv_id] = [
                Message(role=m["role"], content=m["content"])
                for m in data
            ]
            self.current_conversation = conv_id
            return True
        return False

    def list_conversations(self) -> List[str]:
        """List all saved conversations"""
        return [
            f.stem.replace("conversation_", "")
            for f in self.state_dir.glob("conversation_*.json")
        ]


# ============================================================================
# MAIN AGENT
# ============================================================================

class CodingAgent:
    """Main coding agent class"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.api_handler = ApiHandler(config)
        self.tool_executor = ToolExecutor(config)
        self.state_manager = StateManager()

        # System prompt
        self.system_prompt = self._build_system_prompt()

        # Start new conversation
        self.state_manager.new_conversation()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM"""
        return f"""You are Cline, an AI coding assistant powered by {self.config.model}.

You help users with software engineering tasks including:
- Reading, writing, and refactoring code
- Running commands, tests, and builds
- Searching and navigating codebases
- Debugging and fixing issues
- Explaining code and architecture

## Guidelines
- ALWAYS read files before editing them to understand the existing code
- Use execute_command for running tests, builds, git operations, etc.
- When writing code, follow existing patterns and style in the codebase
- Be concise and direct - avoid over-explaining obvious things
- If unsure, ask clarifying questions rather than making assumptions
- Handle errors gracefully and provide actionable feedback

## Tools Available
- read_file: Read file contents
- write_file: Create or modify files
- list_files: Explore directory structure
- search_files: Search for text/patterns in files
- execute_command: Run shell commands

You are working in: {os.path.abspath(self.config.working_dir)}

Start by understanding the user's request, then explore the codebase if needed, and finally implement the solution."""

    async def run(self, user_prompt: str):
        """Run the agent with a user prompt"""

        print(f"\n{'='*60}")
        print(f"Cline Coding Agent")
        print(f"Provider: {self.config.api_provider} | Model: {self.config.model}")
        print(f"Working directory: {os.path.abspath(self.config.working_dir)}")
        print(f"{'='*60}\n")

        # Add user message
        self.state_manager.add_message("user", user_prompt)

        # Main interaction loop
        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1

            # Prepare messages
            messages = [
                Message(role="user", content=self.system_prompt),
                *self.state_manager.get_messages()
            ]

            # Stream response
            print(f"\n{'─'*60}")
            print(f"[Iteration {iteration}]")
            print(f"{'─'*60}\n")

            response_text = ""
            current_tool_call = None
            tool_arguments = ""

            async for chunk in await self.api_handler.send_message(messages):
                if "error" in chunk:
                    print(f"❌ Error: {chunk['error']}")
                    return

                # Text delta
                if "delta" in chunk:
                    delta = chunk["delta"]
                    print(delta, end="", flush=True)
                    response_text += delta

                # Tool call start
                if "tool_call" in chunk:
                    current_tool_call = ToolCall(
                        id=chunk["tool_call"]["id"],
                        name=chunk["tool_call"]["name"],
                        arguments={}
                    )
                    print(f"\n\n🔧 Tool: {current_tool_call.name}", end="")

                # Tool arguments
                if "tool_arguments" in chunk:
                    tool_arguments += chunk["tool_arguments"]
                    if current_tool_call:
                        try:
                            current_tool_call.arguments = json.loads(tool_arguments)
                        except json.JSONDecodeError:
                            pass

            print()  # New line after streaming

            # Check if we got a tool call
            if current_tool_call:
                # Execute tool
                print(f"  Args: {json.dumps(current_tool_call.arguments, indent=2)}\n")

                result = await self.tool_executor.execute(current_tool_call)

                print(f"{'─'*40}")
                if result.is_error:
                    print(f"❌ {result.content}")
                else:
                    # Truncate long results
                    content = result.content
                    if len(content) > 1000:
                        content = content[:1000] + f"\n... ({len(content)-1000} more bytes)"
                    print(f"{content}")
                print(f"{'─'*40}\n")

                # Add tool call and result to conversation
                self.state_manager.add_message(
                    "assistant",
                    f"<tool_use>{current_tool_call.name}</tool_use>"
                )
                self.state_manager.add_message(
                    "user",
                    f"<tool_result>{result.content}</tool_result>"
                )

                # Continue loop for next iteration
                continue

            # No tool call - we're done
            break

        # Save conversation
        print(f"\n{'='*60}")
        self.state_manager.add_message("assistant", response_text)
        self.state_manager.save_conversation()
        print("Conversation saved.")

        # Cleanup
        await self.api_handler.close()

    @classmethod
    def from_env(cls) -> "CodingAgent":
        """Create agent from environment variables"""
        config = AgentConfig(
            api_provider=os.getenv("CLINE_API_PROVIDER", ApiProvider.ANTHROPIC.value),
            api_key=os.getenv("CLINE_API_KEY", ""),
            model=os.getenv("CLINE_MODEL", "claude-sonnet-4-5-20250929"),
            working_dir=os.getenv("CLINE_WORKING_DIR", "."),
            auto_approve_commands=os.getenv("CLINE_AUTO_APPROVE", "false").lower() == "true"
        )

        return cls(config)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_banner():
    """Print the agent banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  🤖 Cline Python Agent - AI Coding Assistant              ║
║                                                           ║
║  A simplified single-file version of Cline               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cline Python Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python coding_agent.py

  # Direct task
  python coding_agent.py "Fix the bug in src/main.py"

  # With custom directory
  python coding_agent.py "Refactor the API" --working-dir ./my-project

  # Using environment variables
  export CLINE_API_KEY=sk-ant-...
  export CLINE_API_PROVIDER=anthropic
  python coding_agent.py "Add a new feature"

Environment Variables:
  CLINE_API_KEY         API key for the provider
  CLINE_API_PROVIDER    Provider (anthropic, openai, openrouter, ollama)
  CLINE_MODEL           Model name (default: claude-sonnet-4-5-20250929)
  CLINE_WORKING_DIR     Working directory (default: current directory)
  CLINE_AUTO_APPROVE    Auto-approve commands (true/false, default: false)
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task to accomplish (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=[e.value for e in ApiProvider],
        help="API provider"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="API key"
    )
    parser.add_argument(
        "--working-dir", "-d",
        help="Working directory"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve command execution"
    )
    parser.add_argument(
        "--list-conversations",
        action="store_true",
        help="List saved conversations"
    )

    args = parser.parse_args()

    # Handle list conversations
    if args.list_conversations:
        state = StateManager()
        convs = state.list_conversations()
        if convs:
            print("Saved conversations:")
            for conv_id in convs:
                print(f"  - {conv_id}")
        else:
            print("No saved conversations.")
        return

    print_banner()

    # Build config
    config = AgentConfig(
        api_provider=args.provider or os.getenv("CLINE_API_PROVIDER", ApiProvider.ANTHROPIC.value),
        api_key=args.api_key or os.getenv("CLINE_API_KEY", ""),
        model=args.model or os.getenv("CLINE_MODEL", "claude-sonnet-4-5-20250929"),
        working_dir=args.working_dir or os.getenv("CLINE_WORKING_DIR", "."),
        auto_approve_commands=args.auto_approve or os.getenv("CLINE_AUTO_APPROVE", "false").lower() == "true"
    )

    # Check API key
    if not config.api_key and config.api_provider != ApiProvider.OLLAMA.value:
        print(f"⚠️  API key not set for {config.api_provider}")
        print(f"   Set CLINE_API_KEY environment variable or use --api-key")
        api_key = input(f"\nEnter {config.api_provider} API key: ").strip()
        if not api_key:
            print("❌ API key required. Exiting.")
            return
        config.api_key = api_key

    # Create agent
    agent = CodingAgent(config)

    # Get task
    if args.task:
        task = args.task
    else:
        print("\nEnter your task (or 'quit' to exit):")
        task = input("> ").strip()

        if not task or task.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            return

    # Run agent
    try:
        asyncio.run(agent.run(task))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
