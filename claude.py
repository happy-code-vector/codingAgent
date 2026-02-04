#!/usr/bin/env python3
"""
Unified Coding Agent - A single-file autonomous coding agent.

This agent can:
- Analyze problem statements
- Write and modify code
- Run tests
- Fix bugs
- Handle file operations

Usage:
    python unified_coding_agent.py --problem "Your problem statement"
    python unified_coding_agent.py --fix-bug --repo-path /path/to/repo
"""

from __future__ import annotations
import os
import re
import sys
import json
import time
import random
import logging
import argparse
import traceback
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

try:
    import requests
    from pydantic import BaseModel
except ImportError:
    print("Required dependencies: requests, pydantic")
    print("Install with: pip install requests pydantic")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Model:
    name: str
    timeout: int

DEFAULT_MODELS = [
    Model(name="deepseek-ai/DeepSeek-V3", timeout=60),
    Model(name="anthropic/claude-sonnet-4-20250514", timeout=120),
]

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "900"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("CodingAgent")
    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)

    return logger

logger = setup_logging()

# =============================================================================
# UTILITIES
# =============================================================================

class Utils:
    @staticmethod
    def count_tokens(text: str | List[Dict]) -> int:
        """Estimate token count."""
        if isinstance(text, list):
            text = " ".join(str(m.get("content", "")) for m in text)

        words = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for word in words:
            if word.isspace():
                continue
            count += max(1, (len(word) + 2) // 3)
        return count

    @staticmethod
    def truncate_string(text: str, max_lines: int = 1000) -> str:
        """Truncate string to max lines."""
        lines = text.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return text

    @staticmethod
    def parse_json(text: str) -> Dict:
        """Parse JSON with fallback."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return eval(text)
            except Exception:
                raise ValueError(f"Invalid JSON: {text[:200]}")

# =============================================================================
# NETWORK LAYER
# =============================================================================

class NetworkErrorType(Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"

class NetworkLayer:
    """Handles communication with LLM API."""

    def __init__(self, api_url: str = DEFAULT_API_URL):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()

    def make_request(
        self,
        messages: List[Dict],
        model: Model,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> str:
        """Make API request with retry logic."""
        url = f"{self.api_url}/v1/chat/completions"

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url,
                    json={
                        "model": model.name,
                        "messages": messages,
                        "temperature": temperature,
                    },
                    timeout=(30, model.timeout)
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Request timeout after {max_retries} attempts")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Request failed: {e}")

        raise RuntimeError("All retry attempts failed")

# =============================================================================
# CODE PARSING
# =============================================================================

class CodeParser:
    """Parse and analyze code."""

    @staticmethod
    def extract_function_body(file_path: str, function_name: str) -> str:
        """Extract function body from file."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            in_function = False
            indent_level = 0
            function_lines = []

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Find function definition
                if f"def {function_name}" in stripped:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    function_lines.append(line)
                    continue

                # Collect function body
                if in_function:
                    current_indent = len(line) - len(line.lstrip())

                    # End of function
                    if line.strip() and current_indent <= indent_level:
                        break

                    function_lines.append(line)

            return "".join(function_lines)

        except Exception as e:
            logger.error(f"Error extracting function: {e}")
            return ""

    @staticmethod
    def detect_language(file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }
        return lang_map.get(ext, "unknown")

# =============================================================================
# FILE OPERATIONS
# =============================================================================

class FileOperations:
    """Handle file system operations."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.created_files: List[str] = []
        self.modified_files: List[str] = []

    def read_file(self, file_path: str, lines: Tuple[int, int] = None) -> str:
        """Read file content."""
        full_path = os.path.join(self.repo_path, file_path)

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                if lines:
                    all_lines = f.readlines()
                    start, end = lines
                    return "".join(all_lines[start-1:end])
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to file."""
        full_path = os.path.join(self.repo_path, file_path)

        try:
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            if file_path not in self.created_files:
                self.created_files.append(file_path)

            return f"✓ File written: {file_path}"

        except Exception as e:
            return f"✗ Error writing file: {e}"

    def edit_file(self, file_path: str, search: str, replace: str) -> str:
        """Search and replace in file."""
        content = self.read_file(file_path)

        if "Error:" in content:
            return content

        if search not in content:
            return f"✗ Search pattern not found in {file_path}"

        if search == replace:
            return "✗ Search and replace are identical"

        new_content = content.replace(search, replace)
        return self.write_file(file_path, new_content)

    def search_in_file(self, file_path: str, pattern: str, context: int = 3) -> str:
        """Search for pattern in file."""
        content = self.read_file(file_path)

        if "Error:" in content:
            return content

        lines = content.split("\n")
        matches = []

        for i, line in enumerate(lines):
            if pattern in line:
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                matches.append(f"\nLine {i+1}:\n" + "\n".join(lines[start:end]))

        if not matches:
            return f"Pattern '{pattern}' not found in {file_path}"

        return f"Found {len(matches)} matches:\n" + "\n---\n".join(matches)

    def list_dir(self, path: str = ".", max_depth: int = 2) -> str:
        """List directory structure."""
        full_path = os.path.join(self.repo_path, path)

        if not os.path.exists(full_path):
            return f"Error: Path not found: {path}"

        ignore = {".git", "__pycache__", "node_modules", ".venv", "venv"}

        result = []
        for root, dirs, files in os.walk(full_path):
            depth = root[len(full_path):].count(os.sep)

            if depth > max_depth:
                dirs[:] = []
                continue

            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in ignore and not d.startswith(".")]

            level = "  " * depth
            result.append(f"{level}{os.path.basename(root)}/")

            for file in files:
                if not file.startswith("."):
                    result.append(f"{level}  {file}")

        return "\n".join(result)

# =============================================================================
# TEST EXECUTION
# =============================================================================

class TestRunner:
    """Run tests and validate code."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.file_ops = FileOperations(repo_path)

    def run_command(self, command: List[str], timeout: int = 30) -> Dict:
        """Run a shell command."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.repo_path
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1
            }

    def run_tests(self, test_path: str = None) -> str:
        """Run tests using appropriate test runner."""

        # Detect test framework
        if os.path.exists(os.path.join(self.repo_path, "pytest.ini")) or \
           os.path.exists(os.path.join(self.repo_path, "pyproject.toml")):
            result = self.run_command(["python", "-m", "pytest", "-v"])
        elif os.path.exists(os.path.join(self.repo_path, "package.json")):
            result = self.run_command(["npm", "test"])
        else:
            # Try running specific test file
            if test_path:
                result = self.run_command(["python", test_path])
            else:
                result = self.run_command(["python", "-m", "unittest", "discover"])

        output = f"Return code: {result['returncode']}\n"
        output += f"STDOUT:\n{result['stdout']}\n"
        if result['stderr']:
            output += f"STDERR:\n{result['stderr']}\n"

        output += "\n" + ("✓ Tests passed" if result['success'] else "✗ Tests failed")

        return output

    def check_syntax(self, file_path: str) -> str:
        """Check syntax of a file."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".py":
            result = self.run_command(["python", "-m", "py_compile", file_path])
        elif ext in [".js", ".ts"]:
            result = self.run_command(["node", "--check", file_path])
        else:
            return f"Syntax check not available for {ext} files"

        return "✓ Syntax OK" if result['success'] else f"✗ Syntax error:\n{result['stderr']}"

# =============================================================================
# AGENT CORE
# =============================================================================

class ToolResult:
    def __init__(self, success: bool, output: str, data: Any = None):
        self.success = success
        self.output = output
        self.data = data

class CodingAgent:
    """Main autonomous coding agent."""

    def __init__(
        self,
        problem_statement: str,
        repo_path: str = ".",
        models: List[Model] = None,
        api_url: str = DEFAULT_API_URL
    ):
        self.problem_statement = problem_statement
        self.repo_path = repo_path
        self.models = models or DEFAULT_MODELS
        self.network = NetworkLayer(api_url)
        self.file_ops = FileOperations(repo_path)
        self.test_runner = TestRunner(repo_path)
        self.code_parser = CodeParser()

        self.steps_taken = 0
        self.conversation_history: List[Dict] = []
        self.observations: List[str] = []

        logger.info(f"Agent initialized for problem: {problem_statement[:100]}...")

    def _get_next_action(self) -> Tuple[str, str, Dict]:
        """Get next action from LLM."""

        prompt = self._build_prompt()

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Add conversation history
        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        response = self.network.make_request(messages, self.models[0])
        return self._parse_response(response)

    def _get_system_prompt(self) -> str:
        return """You are an autonomous coding agent. Your goal is to solve programming problems by:
1. Understanding the problem statement
2. Reading and analyzing existing code
3. Writing or modifying code
4. Running tests to verify solutions
5. Fixing bugs and issues

Available tools:
- read_file(file_path, lines=None): Read file content
- write_file(file_path, content): Write/create file
- edit_file(file_path, search, replace): Edit file with search/replace
- search_in_file(file_path, pattern, context=3): Search in file
- list_dir(path=".", max_depth=2): List directory structure
- run_tests(test_path=None): Run tests
- check_syntax(file_path): Check syntax
- finish(): Mark task as complete

Always respond with:
thought: <your reasoning>
tool_name: <tool name>
tool_args: <JSON args>

Example:
thought: I need to read the main.py file to understand the structure
tool_name: read_file
tool_args: {"file_path": "main.py"}"""

    def _build_prompt(self) -> str:
        prompt = f"""Problem Statement:
{self.problem_statement}

Repository: {self.repo_path}

"""

        if self.observations:
            prompt += f"\nPrevious observations (last 5):\n"
            for obs in self.observations[-5:]:
                prompt += f"  - {obs}\n"

        return prompt

    def _parse_response(self, response: str) -> Tuple[str, str, Dict]:
        """Parse LLM response to extract thought, tool name, and args."""

        # Extract thought
        thought_match = re.search(r"thought:\s*(.*?)(?=\n\s*tool_name:|$)", response, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract tool name
        tool_match = re.search(r"tool_name:\s*(\w+)", response, re.IGNORECASE)
        tool_name = tool_match.group(1).strip() if tool_match else "finish"

        # Extract tool args
        args_match = re.search(r"tool_args:\s*(\{.*?\})", response, re.DOTALL | re.IGNORECASE)
        if args_match:
            try:
                tool_args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                tool_args = {}
        else:
            tool_args = {}

        return thought, tool_name, tool_args

    def _execute_tool(self, tool_name: str, tool_args: Dict) -> ToolResult:
        """Execute a tool and return result."""

        tools = {
            "read_file": self.file_ops.read_file,
            "write_file": self.file_ops.write_file,
            "edit_file": self.file_ops.edit_file,
            "search_in_file": self.file_ops.search_in_file,
            "list_dir": self.file_ops.list_dir,
            "run_tests": self.test_runner.run_tests,
            "check_syntax": self.test_runner.check_syntax,
        }

        if tool_name == "finish":
            return ToolResult(True, "Task marked as complete")

        if tool_name not in tools:
            return ToolResult(False, f"Unknown tool: {tool_name}")

        try:
            func = tools[tool_name]
            result = func(**tool_args)
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, f"Error executing {tool_name}: {e}")

    def run(self, max_steps: int = MAX_STEPS) -> bool:
        """Run the agent until completion or max steps."""

        logger.info("Starting agent execution...")

        while self.steps_taken < max_steps:
            self.steps_taken += 1

            try:
                # Get next action
                thought, tool_name, tool_args = self._get_next_action()

                logger.info(f"Step {self.steps_taken}: {tool_name} - {thought[:100]}")

                # Execute tool
                result = self._execute_tool(tool_name, tool_args)

                # Record observation
                observation = f"{tool_name}: {result.output[:200]}"
                self.observations.append(observation)

                # Add to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"{thought}\nTool: {tool_name}\nArgs: {tool_args}"
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Result: {result.output[:500]}"
                })

                # Check if finished
                if tool_name == "finish" and result.success:
                    logger.info("Agent completed successfully!")
                    return True

            except Exception as e:
                logger.error(f"Error in step {self.steps_taken}: {e}")
                logger.debug(traceback.format_exc())

                self.observations.append(f"Error: {str(e)}")

        logger.warning(f"Agent stopped after reaching max steps ({max_steps})")
        return False

    def solve(self, strategy: str = "iterative") -> Dict:
        """Solve a problem with specified strategy."""

        logger.info(f"Solving with strategy: {strategy}")

        if strategy == "basic":
            return self._basic_solve()
        else:
            success = self.run()
            return {
                "success": success,
                "steps": self.steps_taken,
                "files_created": self.file_ops.created_files,
                "files_modified": self.file_ops.modified_files
            }

    def _basic_solve(self) -> Dict:
        """Basic one-shot solution generation."""

        prompt = f"""Generate a complete solution for this problem:

Problem:
{self.problem_statement}

Output the solution as:
```
file_name.ext
<code content>
```

Requirements:
1. Use only standard libraries
2. Include error handling
3. Add comments for clarity
4. Handle edge cases
"""

        messages = [
            {"role": "system", "content": "You are an expert programmer."},
            {"role": "user", "content": prompt}
        ]

        response = self.network.make_request(messages, self.models[0])

        # Parse and write files
        files_written = 0
        current_file = None
        content_lines = []

        for line in response.split("\n"):
            # Check for file header
            if re.match(r"^[\w-]+\.\w+$", line.strip()):
                if current_file and content_lines:
                    self.file_ops.write_file(current_file, "\n".join(content_lines))
                    files_written += 1
                current_file = line.strip()
                content_lines = []
            else:
                content_lines.append(line)

        if current_file and content_lines:
            self.file_ops.write_file(current_file, "\n".join(content_lines))
            files_written += 1

        return {
            "success": files_written > 0,
            "files_written": files_written,
            "files_created": self.file_ops.created_files
        }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Coding Agent - Autonomous code generation and bug fixing"
    )

    parser.add_argument(
        "--problem", "-p",
        type=str,
        help="Problem statement to solve"
    )

    parser.add_argument(
        "--repo-path", "-r",
        type=str,
        default=".",
        help="Path to repository (default: current directory)"
    )

    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["iterative", "basic"],
        default="iterative",
        help="Solution strategy (default: iterative)"
    )

    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum steps (default: {MAX_STEPS})"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API URL (default: {DEFAULT_API_URL})"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_level)

    # Read problem from file if specified
    if args.problem and args.problem.endswith(".txt"):
        with open(args.problem, "r") as f:
            problem = f.read()
    else:
        problem = args.problem or input("Enter problem statement: ")

    if not problem:
        print("Error: Problem statement is required")
        sys.exit(1)

    # Create and run agent
    agent = CodingAgent(
        problem_statement=problem,
        repo_path=args.repo_path,
        api_url=args.api_url
    )

    result = agent.solve(strategy=args.strategy)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Success: {result['success']}")
    print(f"Steps: {result.get('steps', 'N/A')}")
    print(f"Files created: {result.get('files_created', [])}")
    print(f"Files modified: {result.get('files_modified', [])}")

    return 0 if result['success'] else 1

if __name__ == "__main__":
    sys.exit(main())
