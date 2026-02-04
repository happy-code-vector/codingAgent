"""
Open SWE - Single File Coding Agent

A simplified Python implementation of the Open SWE coding agent using LangGraph.
This agent can:
1. Plan tasks based on user requests
2. Execute code changes
3. Interact with files and shell commands
4. Track progress through a task plan

Requirements:
    pip install langgraph langchain-anthropic langchain-openai

Usage:
    python open_swe_agent.py
"""

import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "claude-sonnet-4-20250514"  # or "gpt-4o"
DEFAULT_MAX_TOKENS = 10000
DEFAULT_TEMPERATURE = 0
DEFAULT_WORK_DIR = os.getcwd()


@dataclass
class AgentConfig:
    """Configuration for the coding agent"""
    model_name: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    work_dir: str = DEFAULT_WORK_DIR
    max_context_actions: int = 75
    github_token: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL),
            max_tokens=int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            work_dir=os.getenv("WORK_DIR", DEFAULT_WORK_DIR),
            github_token=os.getenv("GITHUB_TOKEN"),
            api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )


# =============================================================================
# State Management
# =============================================================================

class PlanItem(TypedDict):
    """A single item in the execution plan"""
    index: int
    plan: str
    completed: bool
    summary: Optional[str]


class PlanRevision(TypedDict):
    """A revision of the plan"""
    revisionIndex: int
    plans: List[PlanItem]
    createdAt: int
    createdBy: Literal["agent", "user"]


class Task(TypedDict):
    """A task in the system"""
    id: str
    taskIndex: int
    request: str
    title: str
    createdAt: int
    completed: bool
    completedAt: Optional[int]
    summary: Optional[str]
    planRevisions: List[PlanRevision]
    activeRevisionIndex: int
    parentTaskId: Optional[str]
    pullRequestNumber: Optional[int]


class TaskPlan(TypedDict):
    """Overall task plan"""
    tasks: List[Task]
    activeTaskIndex: int


class TargetRepository(TypedDict):
    """Target repository information"""
    owner: str
    repo: str
    branch: Optional[str]
    baseCommit: Optional[str]


class AgentState(TypedDict):
    """Main agent state"""
    # Core message state
    messages: Annotated[List[BaseMessage], add_messages]
    internalMessages: Annotated[List[BaseMessage], add_messages]

    # Task management
    taskPlan: TaskPlan
    currentPlanIndex: int

    # Repository context
    targetRepository: Optional[TargetRepository]
    workDir: str
    codebaseTree: str
    dependenciesInstalled: bool

    # Session management
    branchName: str
    sandboxSessionId: str

    # Context gathering
    contextGatheringNotes: str

    # Metadata
    reviewsCount: int
    lastError: Optional[str]


# =============================================================================
# Prompts
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a terminal-based agentic coding assistant designed to enable natural language interaction with local codebases through wrapped LLM models.

<context>
You are in the PLANNING phase. Your job is to break down the user's request into a clear, executable plan.
</context>

<task>
Generate an execution plan to address the user's request. Your plan will guide the implementation phase, so each action must be specific, actionable and detailed.
It should contain enough information to not require many additional context gathering steps to execute.

<user_request>
{user_request}
</user_request>
</task>

<instructions>
Create your plan following these guidelines:

1. **Structure each action item to include:**
   - The specific task to accomplish
   - Key technical details needed for execution
   - File paths, function names, or other concrete references from the context you've gathered.
   - If you're mentioning a file, include the file path in the plan item.

2. **Write actionable items that:**
   - Focus on implementation steps, not information gathering
   - Can be executed independently without additional context discovery
   - Build upon each other in logical sequence
   - Are not open ended, and require additional context to execute

3. **Optimize for efficiency by:**
   - Completing the request in the minimum number of steps
   - Reusing existing code and patterns wherever possible
   - Writing reusable components when code will be used multiple times

4. **Include only what's requested:**
   - Add testing steps only if the user explicitly requested tests
   - Add documentation steps only if the user explicitly requested documentation
   - Focus solely on fulfilling the stated requirements

5. **Follow coding best practices:**
   - Maintain existing code style
   - Add comments only when necessary for clarity
   - Write clean, readable code

6. **Combine simple, related steps:**
   - If you have multiple simple steps that are related, combine them into a single step
</instructions>

<output_format>
When ready, call the 'session_plan' tool with your plan. Each plan item should be a complete, self-contained action that can be executed without referring back to this conversation.

Structure your plan items as clear directives, for example:
- "Implement function X in file Y that performs Z using the existing pattern from file A"
- "Modify the authentication middleware in /src/auth.py to add rate limiting"

Always format your plan items with proper markdown.
</output_format>

{context_gathering_notes}

Remember: Your goal is to create a focused, executable plan that efficiently accomplishes the user's request."""


PROGRAMMER_SYSTEM_PROMPT = """You are a terminal-based agentic coding assistant built to enable natural language interaction with local codebases. You are precise, safe, and helpful.

<identity>
You are executing tasks from a pre-generated plan. You have access to project files, shell commands, and code editing tools.
</identity>

<core_behavior>
- Persistence: Keep working until the current task is completely resolved
- Accuracy: Never guess or make up information. Always use tools to gather accurate data
- Planning: Leverage the plan context and task summaries heavily
</core_behavior>

<task_execution_guidelines>
- You are executing a task from the plan
- Previous completed tasks contain crucial context - review them first
- Only modify the code outlined in the current task
- After completing a task, mark it as complete using the mark_task_completed tool
</task_execution_guidelines>

<file_and_code_management>
<repository_location>{work_dir}</repository_location>
<current_directory>{work_dir}</current_directory>
- All changes should be intentional and planned
- Work only within the existing repository structure
- Use install_dependencies tool if needed for the task
</file_and_code_management>

<tool_usage_best_practices>
- Search: Use the grep tool for all file searches. It respects .gitignore patterns
- Use view command to examine files
- Use str_replace to edit files
- Use shell for commands
- Multiple tools can be called in parallel if they don't depend on each other
</tool_usage_best_practices>

<coding_standards>
- Read files before modifying them
- Fix root causes, not symptoms
- Maintain existing code style
- Remove unnecessary inline comments after completion
- Write concise and clear code
- Never create backup files (git tracks changes)
- If running tests, use proper flags to avoid color formatting
</coding_standards>

<communication_guidelines>
- For coding tasks: Focus on implementation and provide brief summaries
- Use markdown formatting for readability
- Avoid large headers (# or ##), use smaller headings (### or ####) instead
</communication_guidelines>

{plan_information}

<current_task>
{current_task}
</current_task>

Important: Execute the current task to completion, then mark it as complete."""

# =============================================================================
# Tools
# =============================================================================

def get_work_dir(state: AgentState) -> str:
    """Get the working directory from state"""
    return state.get("workDir", DEFAULT_WORK_DIR)


@tool
def grep(
    query: str,
    glob_pattern: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """
    Search for content in files using ripgrep.

    Args:
        query: The search query (string or regex)
        glob_pattern: Optional glob pattern for file types (e.g., "*.py")
        path: Optional path to search in (defaults to work directory)

    Returns:
        Search results with file paths and line numbers
    """
    work_dir = path or os.getcwd()
    cmd = ["rg", query, work_dir, "-N", "--no-heading", "-n"]

    if glob_pattern:
        cmd.extend(["-g", glob_pattern])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        return f"No results found for query: {query}"
    except FileNotFoundError:
        # Fallback to grep if ripgrep not available
        grep_cmd = ["grep", "-r", query, work_dir]
        if glob_pattern:
            grep_cmd.extend(["--include", glob_pattern])
        try:
            result = subprocess.run(
                grep_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
            return result.stdout if result.returncode == 0 else f"No results found"
        except Exception as e:
            return f"Error running grep: {e}"
    except subprocess.TimeoutExpired:
        return "Search timed out"
    except Exception as e:
        return f"Error: {e}"


@tool
def view(path: str, view_range: Optional[List[int]] = None) -> str:
    """
    View the contents of a file or directory.

    Args:
        path: Path to the file or directory to view
        view_range: Optional [start, end] line numbers for files (1-indexed, -1 for end)

    Returns:
        File contents or directory listing
    """
    work_dir = os.getcwd()
    full_path = Path(work_dir) / path

    if not full_path.exists():
        return f"Error: Path not found: {path}"

    if full_path.is_dir():
        # List directory contents
        try:
            items = sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            result = f"Directory: {path}\n\n"
            for item in items:
                prefix = "DIR " if item.is_dir() else "FILE"
                result += f"{prefix} {item.name}\n"
            return result
        except Exception as e:
            return f"Error listing directory: {e}"

    # Read file
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if view_range:
            start, end = view_range
            if end == -1:
                end = len(content.split("\n"))
            lines = content.split("\n")[start-1:end]
            content = "\n".join(lines)

        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def str_replace(path: str, old_str: str, new_str: str) -> str:
    """
    Replace a string in a file with new content.

    Args:
        path: Path to the file to modify
        old_str: The exact string to replace (must match exactly)
        new_str: The new string to insert

    Returns:
        Success message or error
    """
    work_dir = os.getcwd()
    full_path = Path(work_dir) / path

    if not full_path.exists():
        return f"Error: File not found: {path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str not in content:
            return f"Error: old_str not found in file. The string must match exactly (including whitespace)."

        new_content = content.replace(old_str, new_str, 1)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully replaced text in {path}"
    except Exception as e:
        return f"Error modifying file: {e}"


@tool
def create(path: str, file_text: str) -> str:
    """
    Create a new file with specified content.

    Args:
        path: Path where the new file should be created
        file_text: Content to write to the file

    Returns:
        Success message or error
    """
    work_dir = os.getcwd()
    full_path = Path(work_dir) / path

    try:
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(file_text)

        return f"Successfully created file: {path}"
    except Exception as e:
        return f"Error creating file: {e}"


@tool
def shell(command: str, timeout: int = 60) -> str:
    """
    Execute a shell command.

    Args:
        command: The shell command to execute
        timeout: Timeout in seconds (default 60)

    Returns:
        Command output or error
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        output = result.stdout or result.stderr
        if result.returncode != 0:
            output += f"\nCommand exited with status {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def install_dependencies(command: str, timeout: int = 300) -> str:
    """
    Install dependencies for the project.

    Args:
        command: The install command to run (e.g., "npm install", "pip install -r requirements.txt")
        timeout: Timeout in seconds (default 300)

    Returns:
        Installation output or error
    """
    return shell(command, timeout)


@tool
def mark_task_completed(completed_task_summary: str) -> str:
    """
    Mark the current task as completed.

    Args:
        completed_task_summary: Summary of what was accomplished

    Returns:
        Confirmation message
    """
    return f"Task marked as completed: {completed_task_summary}"


@tool
def update_plan(update_plan_reasoning: str) -> str:
    """
    Request to update the current plan.

    Args:
        update_plan_reasoning: Explanation of why and how to update the plan

    Returns:
        Confirmation message
    """
    return f"Plan update requested: {update_plan_reasoning}"


@tool
def request_human_help(help_request: str) -> str:
    """
    Request help from a human when stuck.

    Args:
        help_request: Description of what help is needed

    Returns:
        Help request message
    """
    return f"Human help requested: {help_request}"


# All available tools
ALL_TOOLS = [
    grep,
    view,
    str_replace,
    create,
    shell,
    install_dependencies,
    mark_task_completed,
    update_plan,
    request_human_help,
]


# =============================================================================
# LLM Factory
# =============================================================================

def create_llm(config: AgentConfig) -> BaseChatModel:
    """Create an LLM instance based on configuration"""
    model_name = config.model_name.lower()
    api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

    if "claude" in model_name or "anthropic" in model_name:
        return ChatAnthropic(
            model=config.model_name,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    elif "gpt" in model_name or "openai" in model_name:
        return ChatOpenAI(
            model=config.model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        # Default to Anthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )


# =============================================================================
# Graph Nodes
# =============================================================================

def initialize_planning(state: AgentState, config: Dict) -> Dict:
    """Initialize the planning phase"""
    user_message = state["messages"][-1] if state["messages"] else None

    # Create initial task plan if doesn't exist
    if not state.get("taskPlan") or not state["taskPlan"].get("tasks"):
        new_task: Task = {
            "id": f"task_{datetime.now().timestamp()}",
            "taskIndex": 0,
            "request": user_message.content if user_message else "",
            "title": "Initial Task",
            "createdAt": int(datetime.now().timestamp()),
            "completed": False,
            "completedAt": None,
            "summary": None,
            "planRevisions": [],
            "activeRevisionIndex": 0,
            "parentTaskId": None,
            "pullRequestNumber": None,
        }
        state["taskPlan"] = {
            "tasks": [new_task],
            "activeTaskIndex": 0,
        }

    return {
        "workDir": state.get("workDir", DEFAULT_WORK_DIR),
        "codebaseTree": generate_codebase_tree(state.get("workDir", DEFAULT_WORK_DIR)),
        "dependenciesInstalled": False,
        "currentPlanIndex": 0,
    }


def generate_codebase_tree(state: AgentState, config: Dict) -> Dict:
    """Generate codebase tree for context"""
    work_dir = state.get("workDir", DEFAULT_WORK_DIR)

    try:
        # Try using tree command if available
        result = subprocess.run(
            ["tree", "-L", "3", "-I", "node_modules|__pycache__|.git|venv|env"],
            capture_output=True,
            text=True,
            cwd=work_dir,
            check=False
        )
        if result.returncode == 0:
            return {"codebaseTree": result.stdout}
    except FileNotFoundError:
        pass

    # Fallback: use ls and find
    try:
        result = subprocess.run(
            ["ls", "-R"],
            capture_output=True,
            text=True,
            cwd=work_dir,
            check=False
        )
        return {"codebaseTree": result.stdout}
    except Exception:
        pass

    return {"codebaseTree": "Unable to generate codebase tree"}


def planner_node(state: AgentState, config: Dict) -> Dict:
    """Generate execution plan for the user request"""
    agent_config = config.get("configurable", {})
    llm = create_llm(AgentConfig.from_env())

    user_request = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            break

    context_notes = state.get("contextGatheringNotes", "")

    # Format system prompt
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        user_request=user_request,
        context_gathering_notes=f"\n<context_notes>\n{context_notes}\n</context_notes>" if context_notes else ""
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Bind session_plan tool
    from langchain_core.pydantic_v1 import BaseModel, Field

    class PlanItem(BaseModel):
        plan: str = Field(description="A specific, actionable task")

    class SessionPlan(BaseModel):
        title: str = Field(description="Title for the plan")
        plan: List[PlanItem] = Field(description="List of tasks to execute")

    structured_llm = llm.with_structured_output(SessionPlan)

    try:
        result: SessionPlan = structured_llm.invoke(messages)

        # Create plan items
        plan_items: List[PlanItem] = [
            {
                "index": i,
                "plan": item.plan,
                "completed": False,
                "summary": None,
            }
            for i, item in enumerate(result.plan)
        ]

        # Update task plan with new revision
        task_plan = state.get("taskPlan", {"tasks": [], "activeTaskIndex": 0})
        if task_plan["tasks"]:
            current_task = task_plan["tasks"][0]
            new_revision: PlanRevision = {
                "revisionIndex": len(current_task.get("planRevisions", [])),
                "plans": plan_items,
                "createdAt": int(datetime.now().timestamp()),
                "createdBy": "agent",
            }
            current_task["planRevisions"] = current_task.get("planRevisions", [])
            current_task["planRevisions"].append(new_revision)
            current_task["title"] = result.title
            current_task["activeRevisionIndex"] = new_revision["revisionIndex"]

        # Generate summary message
        plan_summary = f"# Plan: {result.title}\n\n"
        for item in plan_items:
            plan_summary += f"**[{item['index']}]** {item['plan']}\n\n"

        return {
            "taskPlan": task_plan,
            "messages": [AIMessage(content=plan_summary)],
            "internalMessages": [AIMessage(content=plan_summary)],
        }

    except Exception as e:
        error_msg = f"Error generating plan: {e}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "internalMessages": [AIMessage(content=error_msg)],
        }


def format_plan_prompt(plan_items: List[PlanItem]) -> str:
    """Format plan items for display"""
    result = "<current_plan>\n"
    for item in plan_items:
        status = "✓" if item["completed"] else "○"
        result += f"{status} **[{item['index']}]** {item['plan']}\n"
        if item.get("summary"):
            result += f"   *Summary: {item['summary']}*\n"
    result += "</current_plan>\n"
    return result


def programmer_node(state: AgentState, config: Dict) -> Dict:
    """Execute the current task from the plan"""
    agent_config = config.get("configurable", {})
    llm = create_llm(AgentConfig.from_env())
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    task_plan = state.get("taskPlan", {})
    if not task_plan.get("tasks"):
        return {
            "messages": [AIMessage(content="No task plan available. Please provide a request first.")],
        }

    # Get current task
    current_task = task_plan["tasks"][task_plan["activeTaskIndex"]]
    plan_revisions = current_task.get("planRevisions", [])
    if not plan_revisions:
        return {
            "messages": [AIMessage(content="No plan revisions available.")],
        }

    active_revision = plan_revisions[current_task["activeRevisionIndex"]]
    plan_items = active_revision["plans"]

    # Find current incomplete task
    current_item = None
    current_index = state.get("currentPlanIndex", 0)
    for item in plan_items:
        if not item["completed"] and item["index"] >= current_index:
            current_item = item
            break

    if not current_item:
        return {
            "messages": [AIMessage(content="All tasks completed!🎉")],
            "currentPlanIndex": len(plan_items),
        }

    # Format system prompt
    plan_prompt = format_plan_prompt(plan_items)
    current_task_str = f"**[{current_item['index']}]** {current_item['plan']}"

    system_prompt = PROGRAMMER_SYSTEM_PROMPT.format(
        work_dir=state.get("workDir", DEFAULT_WORK_DIR),
        plan_information=plan_prompt,
        current_task=current_task_str,
    )

    messages = [SystemMessage(content=system_prompt)] + state["internalMessages"]

    # Invoke LLM
    response: AIMessage = llm_with_tools.invoke(messages)

    # Check if task was marked complete
    new_plan_index = current_index
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "mark_task_completed":
                new_plan_index = current_item["index"] + 1
                # Mark item as complete
                current_item["completed"] = True
                summary = tool_call.get("args", {}).get("completed_task_summary", "Completed")
                current_item["summary"] = summary
                break

    return {
        "messages": [response],
        "internalMessages": [response],
        "currentPlanIndex": new_plan_index,
    }


def route_after_programmer(state: AgentState) -> str:
    """Route after programmer node"""
    task_plan = state.get("taskPlan", {})
    if not task_plan.get("tasks"):
        return END

    current_task = task_plan["tasks"][task_plan["activeTaskIndex"]]
    plan_revisions = current_task.get("planRevisions", [])
    if not plan_revisions:
        return END

    active_revision = plan_revisions[current_task["activeRevisionIndex"]]
    plan_items = active_revision["plans"]

    # Check if all tasks complete
    if all(item["completed"] for item in plan_items):
        return END

    return "programmer"


# =============================================================================
# Graph Construction
# =============================================================================

def create_agent_graph(config: Optional[AgentConfig] = None):
    """Create the agent workflow graph"""
    if config is None:
        config = AgentConfig.from_env()

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initialize_planning", initialize_planning)
    workflow.add_node("generate_codebase_tree", generate_codebase_tree)
    workflow.add_node("planner", planner_node)
    workflow.add_node("programmer", programmer_node)

    # Add edges
    workflow.add_edge(START, "initialize_planning")
    workflow.add_edge("initialize_planning", "generate_codebase_tree")
    workflow.add_edge("generate_codebase_tree", "planner")
    workflow.add_edge("planner", "programmer")

    # Conditional edge from programmer
    workflow.add_conditional_edges(
        "programmer",
        route_after_programmer,
        {
            "programmer": "programmer",
            END: END,
        }
    )

    return workflow.compile()


# =============================================================================
# Main Execution
# =============================================================================

def run_agent(user_request: str, config: Optional[AgentConfig] = None):
    """Run the agent with a user request"""
    if config is None:
        config = AgentConfig.from_env()

    # Create initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_request)],
        "internalMessages": [],
        "taskPlan": {"tasks": [], "activeTaskIndex": 0},
        "currentPlanIndex": 0,
        "targetRepository": None,
        "workDir": config.work_dir,
        "codebaseTree": "",
        "dependenciesInstalled": False,
        "branchName": "",
        "sandboxSessionId": "",
        "contextGatheringNotes": "",
        "reviewsCount": 0,
        "lastError": None,
    }

    # Create and run graph
    graph = create_agent_graph(config)

    print(f"\n{'='*60}")
    print(f"Open SWE Agent")
    print(f"{'='*60}\n")
    print(f"User Request: {user_request}\n")
    print(f"{'='*60}\n")

    # Stream the execution
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "planner":
                print("[PLANNER] Generated execution plan:")
                for msg in node_output.get("messages", []):
                    print(f"\n{msg.content}\n")
            elif node_name == "programmer":
                print("[PROGRAMMER] Executing task...")
                for msg in node_output.get("messages", []):
                    if hasattr(msg, "content") and msg.content:
                        print(f"\n{msg.content}\n")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            print(f"  → Tool: {tool_name}")
                            if tool_name == "mark_task_completed":
                                args = tool_call.get("args", {})
                                summary = args.get("completed_task_summary", "Completed")
                                print(f"     ✓ {summary}")

    print(f"\n{'='*60}")
    print("Agent execution complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    # Check for API key
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("Error: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Get user request from command line or prompt
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input("Enter your request: ")

    if not user_request.strip():
        print("No request provided. Exiting.")
        sys.exit(0)

    run_agent(user_request)
