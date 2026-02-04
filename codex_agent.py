#!/usr/bin/env python3
"""
Codex Agent - A single-file coding agent inspired by OpenAI's Codex CLI.

Features:
- LLM integration (OpenAI API)
- File operations (read, write, edit, search)
- Shell command execution with safety checks
- Planning and execution loop
- Conversation history
"""

import os
import re
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

# Optional: Import OpenAI for LLM capabilities
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")

# Optional: Import rich for better terminal UI
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: rich package not installed. Install with: pip install rich")


class ToolResponse(Enum):
    """Response types for tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"


@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    content: str
    error: Optional[str] = None


class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str, requires_approval: bool = False):
        self.name = name
        self.description = description
        self.requires_approval = requires_approval

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool. Override in subclasses."""
        raise NotImplementedError


class ReadFileTool(Tool):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file. Use this to see what's in a file before editing.",
            requires_approval=False
        )

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return ToolResult(success=False, content="", error=f"File not found: {file_path}")
            if not path.is_file():
                return ToolResult(success=False, content="", error=f"Not a file: {file_path}")

            # Check file size
            if path.stat().st_size > 100_000:  # 100KB limit
                return ToolResult(success=False, content="", error=f"File too large (>100KB): {file_path}")

            content = path.read_text(encoding='utf-8', errors='replace')
            return ToolResult(success=True, content=content)
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class WriteFileTool(Tool):
    """Tool for writing file contents."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file, overwriting existing content.",
            requires_approval=True
        )

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            path = Path(file_path).resolve()
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return ToolResult(success=True, content=f"Successfully wrote {len(content)} bytes to {file_path}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class EditFileTool(Tool):
    """Tool for editing files with search and replace."""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="Edit a file by replacing an exact string match with new content.",
            requires_approval=True
        )

    async def execute(self, file_path: str, old_string: str, new_string: str, **kwargs) -> ToolResult:
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return ToolResult(success=False, content="", error=f"File not found: {file_path}")

            content = path.read_text(encoding='utf-8')
            if old_string not in content:
                return ToolResult(success=False, content="", error=f"Old string not found in file")

            new_content = content.replace(old_string, new_string, 1)  # Replace first occurrence only
            path.write_text(new_content, encoding='utf-8')
            return ToolResult(success=True, content=f"Successfully edited {file_path}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class ShellTool(Tool):
    """Tool for executing shell commands."""

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__(
            name="shell",
            description="Execute a shell command. Use for git, npm, build tools, etc.",
            requires_approval=True
        )
        self.allowed_commands = allowed_commands or []

    def _is_command_safe(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if a command is safe to execute."""
        # Dangerous commands that are blocked
        dangerous_patterns = [
            r'\brm\s+-rf\s+[/.]',  # rm -rf from root
            r'\bdeltree\s+',  # deltree
            r'\bformat\s+[a-z]:',  # format drive
            r'\bdel\s+/[sfq]',  # del with dangerous flags
            r'>\s*/[a-z]',  # redirect to device
            r'\bmkfs\.',  # make filesystem
            r'\bdd\s+if=',  # dd command
            r'\bsudo\s+rm',  # sudo rm
            r'\bsu\s+-c\s+"rm',  # su with rm
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches dangerous pattern: {pattern}"

        if self.allowed_commands:
            cmd_start = command.strip().split()[0]
            if cmd_start not in self.allowed_commands:
                return False, f"Command '{cmd_start}' not in allowed list"

        return True, None

    async def execute(self, command: str, cwd: Optional[str] = None, timeout: int = 30, **kwargs) -> ToolResult:
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return ToolResult(success=False, content="", error=f"Command blocked: {reason}")

        try:
            working_dir = cwd if cwd else os.getcwd()
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')

                if process.returncode == 0:
                    result = output
                    if error:
                        result += f"\n[stderr]\n{error}"
                    return ToolResult(success=True, content=result)
                else:
                    return ToolResult(
                        success=False,
                        content=output,
                        error=f"Command failed with exit code {process.returncode}: {error}"
                    )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(success=False, content="", error=f"Command timed out after {timeout}s")

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class SearchFilesTool(Tool):
    """Tool for searching files by pattern."""

    def __init__(self):
        super().__init__(
            name="search_files",
            description="Search for files matching a glob pattern (e.g., '**/*.py').",
            requires_approval=False
        )

    async def execute(self, pattern: str, path: str = ".", **kwargs) -> ToolResult:
        try:
            search_path = Path(path).resolve()
            if not search_path.exists():
                return ToolResult(success=False, content="", error=f"Path not found: {path}")

            matches = list(search_path.glob(pattern))
            # Convert to relative paths for readability
            relative_paths = [str(m.relative_to(search_path)) for m in matches if m.is_file()]

            if not relative_paths:
                return ToolResult(success=True, content=f"No files found matching pattern: {pattern}")

            content = f"Found {len(relative_paths)} file(s):\n" + "\n".join(relative_paths)
            return ToolResult(success=True, content=content)
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class SearchContentTool(Tool):
    """Tool for searching content within files."""

    def __init__(self):
        super().__init__(
            name="search_content",
            description="Search for text/regex patterns in file contents.",
            requires_approval=False
        )

    async def execute(self, pattern: str, path: str = ".", file_pattern: str = "*", **kwargs) -> ToolResult:
        try:
            import fnmatch

            search_path = Path(path).resolve()
            results = []

            for file_path in search_path.rglob(file_pattern):
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            results.append(f"{file_path.relative_to(search_path)}:{i}: {line.strip()}")
                except Exception:
                    continue

            if not results:
                return ToolResult(success=True, content=f"No matches found for pattern: {pattern}")

            return ToolResult(success=True, content=f"Found {len(results)} match(es):\n" + "\n".join(results[:50]))
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class Agent:
    """Main coding agent class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        auto_approve: bool = False,
        allowed_commands: Optional[List[str]] = None,
        working_dir: str = "."
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_iterations = max_iterations
        self.auto_approve = auto_approve
        self.working_dir = Path(working_dir).resolve()

        # Initialize conversation history
        self.messages: List[Message] = []

        # Initialize tools
        self.tools: Dict[str, Tool] = {
            "read_file": ReadFileTool(),
            "write_file": WriteFileTool(),
            "edit_file": EditFileTool(),
            "shell": ShellTool(allowed_commands),
            "search_files": SearchFilesTool(),
            "search_content": SearchContentTool(),
        }

        # Initialize OpenAI client if available
        self.client = None
        if HAS_OPENAI and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)

        # Initialize console
        self.console = Console() if HAS_RICH else None

        # System prompt
        self.system_prompt = """You are Codex, a coding agent that helps users with software engineering tasks.

You have access to the following tools:
- read_file: Read file contents
- write_file: Write content to a file (overwrites existing)
- edit_file: Edit a file by replacing exact string match
- shell: Execute shell commands (git, npm, python, etc.)
- search_files: Find files by glob pattern
- search_content: Search text/regex in file contents

Guidelines:
1. Always read files before editing them
2. Use shell for git operations, running tests, builds, etc.
3. Break complex tasks into smaller steps
4. Explain what you're doing before doing it
5. If something fails, analyze the error and try a different approach
6. Be concise and focused on solving the user's problem

Current working directory: {working_dir}

Respond with clear, actionable steps. When you need to use a tool, format your response as:
TOOL: tool_name
PARAMS: {{"param1": "value1", "param2": "value2"}}
"""

    def _print(self, text: str, style: str = None):
        """Print text with optional styling."""
        if self.console:
            if style:
                self.console.print(text, style=style)
            else:
                self.console.print(text)
        else:
            print(text)

    def _print_markdown(self, text: str):
        """Print markdown formatted text."""
        if self.console:
            self.console.print(Markdown(text))
        else:
            print(text)

    def _print_code(self, code: str, language: str = "python"):
        """Print syntax highlighted code."""
        if self.console:
            self.console.print(Syntax(code, language, line_numbers=True))
        else:
            print(f"```{language}")
            print(code)
            print("```")

    def add_message(self, role: str, content: str, tool_calls: List[Dict] = None):
        """Add a message to the conversation history."""
        self.messages.append(Message(
            role=role,
            content=content,
            tool_calls=tool_calls or []
        ))

    def get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        descriptions = []
        for name, tool in self.tools.items():
            approval_note = " [requires approval]" if tool.requires_approval else ""
            descriptions.append(f"- {name}: {tool.description}{approval_note}")
        return "\n".join(descriptions)

    async def execute_tool(self, tool_name: str, **params) -> ToolResult:
        """Execute a tool with the given parameters."""
        if tool_name not in self.tools:
            return ToolResult(success=False, content="", error=f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]

        # Check if approval is needed
        if tool.requires_approval and not self.auto_approve:
            self._print(f"\n[yellow]⚠ Tool '{tool_name}' requires approval[/yellow]")
            self._print(f"Parameters: {json.dumps(params, indent=2)}")

            response = input("Approve? (y/n/a=always): ").strip().lower()
            if response == 'a':
                self.auto_approve = True
            elif response != 'y':
                return ToolResult(success=False, content="", error="User denied approval")

        # Execute the tool
        try:
            return await tool.execute(**params)
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Tool execution error: {str(e)}")

    def parse_tool_call(self, content: str) -> Optional[tuple[str, Dict]]:
        """Parse a tool call from the LLM response."""
        # Match TOOL: tool_name followed by PARAMS: {json}
        tool_match = re.search(r'TOOL:\s*(\w+)', content, re.IGNORECASE)
        if not tool_match:
            return None

        tool_name = tool_match.group(1)

        params_match = re.search(r'PARAMS:\s*(\{.*?\})', content, re.DOTALL | re.IGNORECASE)
        if not params_match:
            return None

        try:
            params = json.loads(params_match.group(1))
            return tool_name, params
        except json.JSONDecodeError:
            return None

    async def get_llm_response(self, user_message: str) -> str:
        """Get response from LLM."""
        if not self.client:
            return "Error: OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."

        # Build messages for API
        api_messages = [{"role": "system", "content": self.system_prompt.format(working_dir=self.working_dir)}]

        # Add conversation history (last 10 messages to avoid token limits)
        for msg in self.messages[-10:]:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        api_messages.append({"role": "user", "content": user_message})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    async def run_step(self, user_input: str) -> bool:
        """Run a single step of the agent loop."""
        # Add user message
        self.add_message("user", user_input)
        self._print(f"\n[bold blue]You:[/bold blue] {user_input}")

        # Get LLM response
        response = await self.get_llm_response(user_input)
        self.add_message("assistant", response)

        # Check for tool calls in response
        tool_call = self.parse_tool_call(response)

        if tool_call:
            tool_name, params = tool_call
            self._print(f"\n[yellow]🔧 Executing: {tool_name}[/yellow]")

            # Add working directory to params if not present
            if 'path' not in params and 'file_path' not in params and tool_name in ['shell', 'search_files', 'search_content']:
                params['path'] = str(self.working_dir)
            elif 'file_path' in params:
                # Convert relative paths to absolute
                file_path = params['file_path']
                if not Path(file_path).is_absolute():
                    params['file_path'] = str(self.working_dir / file_path)

            result = await self.execute_tool(tool_name, **params)

            # Display result
            if result.success:
                self._print(f"[green]✓ Success[/green]")
                if result.content:
                    # Check if content looks like code
                    if len(result.content) < 1000 and '\n' in result.content:
                        self._print(f"\n{result.content}")
                    else:
                        self._print(f"\n[dim]{result.content[:500]}...[/dim]")
                # Add result as assistant message for context
                self.add_message("assistant", f"Tool {tool_name} completed: {result.content}")
                return True  # Continue loop
            else:
                self._print(f"[red]✗ Error: {result.error}[/red]")
                self.add_message("assistant", f"Tool {tool_name} failed: {result.error}")
                return True  # Continue loop to let agent recover
        else:
            # No tool call, just display response
            self._print(f"\n[bold green]Codex:[/bold green]")
            self._print_markdown(response)
            return False  # End loop

    async def run(self, prompt: str):
        """Run the agent with a given prompt."""
        self._print(Panel.fit("Codex Agent", style="bold cyan"))
        self._print(f"Working directory: {self.working_dir}")
        self._print(f"Model: {self.model}")
        self._print(f"Auto-approve: {self.auto_approve}")
        self._print("")

        if not self.client:
            self._print("[yellow]Warning: OpenAI API key not set. Running in local mode only.[/yellow]")
            self._print("Set OPENAI_API_KEY environment variable for full functionality.\n")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            should_continue = await self.run_step(prompt if iteration == 1 else "Please continue with the next step.")
            if not should_continue:
                break

        self._print("\n[dim]--- Session ended ---[/dim]")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Codex Agent - A single-file coding agent")
    parser.add_argument("prompt", nargs="?", help="The task/prompt for the agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all tool executions")
    parser.add_argument("--working-dir", default=".", help="Working directory for the agent")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--allowed-commands", help="Comma-separated list of allowed shell commands")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Parse allowed commands
    allowed_commands = None
    if args.allowed_commands:
        allowed_commands = [c.strip() for c in args.allowed_commands.split(',')]

    # Create agent
    agent = Agent(
        api_key=args.api_key,
        model=args.model,
        max_iterations=args.max_iterations,
        auto_approve=args.auto_approve,
        allowed_commands=allowed_commands,
        working_dir=args.working_dir
    )

    if args.interactive:
        # Interactive mode
        print("Codex Agent - Interactive Mode")
        print("Type 'exit' or 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue

                await agent.run(user_input)
                # Reset messages for next interaction but keep some context
                agent.messages = agent.messages[-6:]  # Keep last 3 exchanges
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
    elif args.prompt:
        # Single prompt mode
        await agent.run(args.prompt)
    else:
        # No prompt provided, show help and enter interactive mode
        parser.print_help()
        print("\nEntering interactive mode...\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue

                await agent.run(user_input)
                agent.messages = agent.messages[-6:]
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
