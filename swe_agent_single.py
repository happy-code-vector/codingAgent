#!/usr/bin/env python3
"""
SWE-Agent - Single File Version
A simplified, single-file version of the SWE-agent coding agent.

This is a consolidated version that combines the core functionality of SWE-agent
into a single Python file for easier use and distribution.

Usage:
    python swe_agent_single.py --help
    python swe_agent_single.py --model gpt-4o --problem "Fix the bug in main.py"
    python swe_agent_single.py --github-url "https://github.com/user/repo/issues/1"
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shlex
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

# Third-party imports
import litellm
import litellm.types.utils
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler
from rich.text import Text
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    get_deployment,
)
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    CreateBashSessionRequest,
    ReadFileRequest,
    RexCommand,
    UploadRequest,
    WriteFileRequest,
)
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import Self

# Suppress litellm debug info
litellm.suppress_debug_info = True

# ==============================================================================
# LOGGING
# ==============================================================================

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore


class _RichHandlerWithEmoji(RichHandler):
    """Subclass of RichHandler that adds an emoji to the log message."""

    def __init__(self, emoji: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not emoji.endswith(" "):
            emoji += " "
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname.replace("WARNING", "WARN")
        return Text.styled((self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}")


def get_logger(name: str, *, emoji: str = "") -> logging.Logger:
    """Get logger with emoji support."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    log_level = os.environ.get("SWE_AGENT_LOG_STREAM_LEVEL", "INFO")
    if isinstance(log_level, str) and log_level.isnumeric():
        log_level = int(log_level)
    elif isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(log_level)
    logger.setLevel(logging.TRACE)  # type: ignore
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class FormatError(Exception):
    """Raised when the model response cannot properly be parsed into thought and actions."""


class FunctionCallingFormatError(FormatError):
    """Format error exception used by the function calling parser."""

    def __init__(
        self,
        message: str,
        error_code: Literal[
            "missing", "multiple", "incorrect_args", "invalid_json", "invalid_command", "missing_arg", "unexpected_arg"
        ],
        **extra_info: Any,
    ):
        super().__init__(message + f" [error_code={error_code}]")
        self.message = message
        self.extra_info = {"error_code": error_code, **extra_info}


class ContextWindowExceededError(Exception):
    """Raised when the context window of a LM is exceeded"""


class CostLimitExceededError(Exception):
    """Raised when we exceed a cost limit"""


class InstanceCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the cost limit set for one task instance"""


class TotalCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the total cost limit"""


class InstanceCallLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the per instance call limit"""


class ContentPolicyViolationError(Exception):
    """Raised when the model response violates a content policy"""


class ModelConfigurationError(Exception):
    """Raised when the model configuration is invalid/no further retries should be made."""


# ==============================================================================
# TYPES AND DATA STRUCTURES
# ==============================================================================

@dataclass
class StepOutput:
    """Output from a single agent step."""
    query: list[dict] = field(default_factory=dict)
    thought: str = ""
    action: str = ""
    output: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False
    exit_status: int | str | None = None
    submission: str | None = None
    state: dict[str, str] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_ids: list[str] | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_template_format_dict(self) -> dict[str, str | int | float | bool | None]:
        """Used for formatting (error) prompt templates"""
        out = {}
        for k, v in self.__dict__.items():
            if k in ("tool_calls", "tool_call_ids", "state"):
                continue
            out[k] = v
        out |= self.state
        return out


HistoryItem = dict[str, Any]
History = list[HistoryItem]
TrajectoryStep = dict[str, Any]
Trajectory = list[TrajectoryStep]
AgentInfo = dict[str, Any]


# ==============================================================================
# COMMANDS AND TOOLS
# ==============================================================================

ARGUMENT_NAME_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]*"


def _extract_keys(format_string: str) -> set[str]:
    """Given a format string, returns a set of all the keys in the format string."""
    formatter = str.Formatter()
    keys = set()
    for _, field_name, _, _ in formatter.parse(format_string):
        if field_name is not None:
            keys.add(field_name)
    return keys


class Argument(BaseModel):
    """Defines an argument that can be passed to a command."""
    name: str
    type: str
    items: dict[str, str] | None = None
    description: str
    required: bool
    enum: list[str] | None = None
    argument_format: str = "{{value}}"


class Command(BaseModel):
    """Represents an executable command with arguments and documentation."""
    name: str
    docstring: str | None
    signature: str | None = None
    end_name: str | None = None
    arguments: list[Argument] = []

    @cached_property
    def invoke_format(self) -> str:
        """Gets the format string for invoking this command with arguments."""
        if self.signature:
            for arg in self.arguments:
                if not (
                    f"<{arg.name}>" in self.signature
                    or f"[<{arg.name}>]" in self.signature
                    or f"{{{arg.name}}}" in self.signature
                    or f"--{arg.name}" in self.signature
                ):
                    raise ValueError(
                        f"Missing argument {arg.name} in signature: {self.signature}"
                    )
            return re.sub(rf"\[?<({ARGUMENT_NAME_PATTERN})>\]?", r"{\1}", self.signature)
        else:
            _invoke_format = f"{self.name} "
            for arg in self.arguments:
                _invoke_format += f"{{{arg.name}}} "
            return _invoke_format

    def get_function_calling_tool(self) -> dict:
        """Converts this command into an OpenAI function calling tool definition."""
        tool = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.docstring or "",
            },
        }
        properties = {}
        required = []
        if self.arguments:
            for arg in self.arguments:
                properties[arg.name] = {"type": arg.type, "description": arg.description}
                if arg.items:
                    properties[arg.name]["items"] = arg.items
                if arg.required:
                    required.append(arg.name)
                if arg.enum:
                    properties[arg.name]["enum"] = arg.enum
        tool["function"]["parameters"] = {"type": "object", "properties": properties, "required": required}
        return tool


# Default Bash command
BASH_COMMAND = Command(
    name="bash",
    signature="<command>",
    docstring="runs the given command directly in bash",
    arguments=[
        Argument(
            name="command",
            type="string",
            description="The bash command to execute.",
            required=True,
        )
    ],
)


def _should_quote(value: Any, command: Command) -> bool:
    """Returns True if the value should be quoted, False otherwise."""
    if command.name == "bash":
        return False
    return isinstance(value, str) and command.end_name is None


def generate_command_docs(
    commands: list[Command],
    **kwargs,
) -> str:
    """Generate detailed command documentation."""
    docs = ""
    for cmd in commands:
        docs += f"{cmd.name}:\n"
        if cmd.docstring is not None:
            docs += f"  docstring: {cmd.docstring.format(**kwargs)}\n"
        if cmd.signature is not None:
            docs += f"  signature: {cmd.signature}\n"
        else:
            signature = cmd.name
            if cmd.arguments:
                for argument in cmd.arguments:
                    param = argument.name
                    if argument.required:
                        signature += f" <{param}>"
                    else:
                        signature += f" [<{param}>]"
            docs += f"  signature: {signature}\n"
        if cmd.arguments:
            docs += "  arguments:\n"
            for argument in cmd.arguments:
                param = argument.name
                req_string = "required" if argument.required else "optional"
                docs += f"    - {param} ({argument.type}) [{req_string}]: {argument.description}\n"
        docs += "\n"
    return docs


def _guard_multiline_input(action: str, match_fct: Callable[[str], re.Match | None]) -> str:
    """Split action by multiline commands, then append the first line in each multiline command with "<< '{end_name}'"."""
    parsed_action = []
    rem_action = action
    while rem_action.strip():
        first_match = match_fct(rem_action)
        if first_match:
            pre_action = rem_action[: first_match.start()]
            match_action = rem_action[first_match.start() : first_match.end()]
            rem_action = rem_action[first_match.end() :]
            if pre_action.strip():
                parsed_action.append(pre_action)
            if match_action.strip():
                eof = first_match.group(3).strip()
                if not match_action.split("\n")[0].strip().endswith(f"<< '{eof}'"):
                    guarded_command = match_action[first_match.start() :]
                    first_line = guarded_command.split("\n")[0]
                    guarded_command = guarded_command.replace(first_line, first_line + f" << '{eof}'", 1)
                    parsed_action.append(guarded_command)
                else:
                    parsed_action.append(match_action)
        else:
            parsed_action.append(rem_action)
            rem_action = ""
    return "\n".join(parsed_action)


# ==============================================================================
# PARSING
# ==============================================================================

class AbstractParseFunction(ABC):
    """Abstract class for parsing functions."""

    error_message: str

    @abstractmethod
    def __call__(self, model_response, commands: list[Command], strict=False) -> tuple[str, str]:
        raise NotImplementedError

    @property
    def format_error_template(self):
        import textwrap
        return textwrap.dedent(self.error_message)


class ThoughtActionParser(AbstractParseFunction, BaseModel):
    """Expects the model response to be a discussion followed by a command wrapped in backticks."""

    error_message: str = """\
    Your output was not formatted correctly. You must always include one discussion and one command as part of your response. Make sure you do not have multiple discussion/command tags.
    Please make sure your output precisely matches the following format:
    DISCUSSION
    Discuss here with yourself about what your planning and what you're going to do in this step.

    ```
    command(s) that you're going to run
    ```
    """

    type: Literal["thought_action"] = "thought_action"

    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        """Parses the action from the output of the API call."""
        code_block_pat = re.compile(r"^```(\S*)\s*\n|^```\s*$", re.MULTILINE)
        stack = []
        last_valid_block = None
        for match in code_block_pat.finditer(model_response["message"]):
            if stack and not match.group(1):  # Closing of a code block
                start = stack.pop()
                if not stack:
                    last_valid_block = (start, match)
            elif match.group(1) is not None:  # Opening of a code block
                stack.append(match)
        if last_valid_block:
            start, end = last_valid_block
            thought = model_response["message"][: start.start()] + model_response["message"][end.end() :]
            return thought, model_response["message"][start.end() : end.start()]
        raise FormatError("No action found in model response.")


class FunctionCallingParser(AbstractParseFunction, BaseModel):
    """Expects the model response to be a LiteLLM tool call."""

    error_message: str = """\
    {%- if error_code == "missing" -%}
    Your last output did not use any tool calls!
    Please make sure your output includes exactly _ONE_ function call!
    You must invoke the function directly using the function call format.
    You cannot invoke commands with ```, you have to use the function call format.
    If you think you have already resolved the issue, please submit your changes by running the `submit` command.
    If you think you cannot solve the problem, please run `exit_forfeit` (if available) or `submit`.
    Else, please continue with a new tool call!
    {%- elif error_code == "multiple" -%}
    Your last output included multiple tool calls!
    Please make sure your output includes a thought and exactly _ONE_ function call.
    {%- elif error_code == "unexpected_arg" -%}
    Your action could not be parsed properly: {{exception_message}}.
    Make sure your function call doesn't include any extra arguments that are not in the allowed arguments, and only use the allowed commands.
    {%- else -%}
    Your action could not be parsed properly: {{exception_message}}.
    {% endif %}
    """

    type: Literal["function_calling"] = "function_calling"

    def _parse_tool_call(self, tool_call: dict, commands: list[Command]):
        from shlex import quote
        name = tool_call["function"]["name"]
        command = {c.name: c for c in commands}.get(name)
        if not command:
            raise FunctionCallingFormatError(
                f"Command '{name}' not found in list of available commands.",
                "invalid_command"
            )
        if not isinstance(tool_call["function"]["arguments"], dict):
            try:
                values = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                raise FunctionCallingFormatError("Tool call arguments are not valid JSON.", "invalid_json")
        else:
            values = tool_call["function"]["arguments"]

        required_args = {arg.name for arg in command.arguments if arg.required}
        missing_args = required_args - values.keys()
        if missing_args:
            raise FunctionCallingFormatError(
                f"Required argument(s) missing: {', '.join(missing_args)}",
                "missing_arg"
            )

        valid_args = {arg.name for arg in command.arguments}
        extra_args = set(values.keys()) - valid_args
        if command.end_name:
            extra_args.discard(command.end_name)
        if extra_args:
            raise FunctionCallingFormatError(
                f"Unexpected argument(s): {', '.join(extra_args)}",
                "unexpected_arg"
            )

        def get_quoted_arg(value: Any) -> str:
            if isinstance(value, str):
                return quote(value) if _should_quote(value, command) else value
            if value is None:
                return ""
            return value

        formatted_args = {
            arg.name: Template(arg.argument_format).render(value=get_quoted_arg(values[arg.name]))
            if arg.name in values else ""
            for arg in command.arguments
        }
        return command.invoke_format.format(**formatted_args).strip()

    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        message = model_response["message"]
        tool_calls = model_response.get("tool_calls", None)
        if tool_calls is None or len(tool_calls) != 1:
            num_tools = len(tool_calls) if tool_calls else 0
            error_code = "missing" if num_tools == 0 else "multiple"
            raise FunctionCallingFormatError(
                f"Expected exactly one tool call in model response - received {num_tools} tool calls with message: {message}",
                error_code,
                num_tools=num_tools
            )
        tool_call = tool_calls[0]
        action = self._parse_tool_call(tool_call, commands)
        return message, action


class ActionOnlyParser(AbstractParseFunction, BaseModel):
    """Expects the model response to be a single command."""

    error_message: str = "No message found in model response."
    type: Literal["action_only"] = "action_only"

    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        return "", model_response["message"]


ParseFunction = ThoughtActionParser | FunctionCallingParser | ActionOnlyParser


# ==============================================================================
# TOOL CONFIG
# ==============================================================================

class ToolFilterConfig(BaseModel):
    """Filter out commands that are blocked by the environment."""
    blocklist_error_template: str = "Operation '{{action}}' is not supported by this environment."
    blocklist: list[str] = ["vim", "vi", "emacs", "nano", "nohup", "gdb", "less", "tail -f"]
    blocklist_standalone: list[str] = ["python", "python3", "ipython", "bash", "sh", "/bin/bash", "/bin/sh"]
    block_unless_regex: dict[str, str] = {}


class ToolConfig(BaseModel):
    """Configuration for the tools that are made available to the agent."""
    filter: ToolFilterConfig = Field(default_factory=ToolFilterConfig)
    bundles: list[Any] = Field(default_factory=list)
    propagate_env_variables: list[str] = []
    env_variables: dict[str, Any] = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
    }
    registry_variables: dict[str, Any] = {}
    submit_command: str = "submit"
    parse_function: ParseFunction = Field(default_factory=FunctionCallingParser)
    enable_bash_tool: bool = True
    format_error_template: str | None = None
    command_docs: str | None = None
    multi_line_command_endings: dict[str, str] = {}
    submit_command_end_name: str | None = None
    execution_timeout: int = 30
    install_timeout: int = 300
    total_execution_timeout: int = 1800
    max_consecutive_execution_timeouts: int = 3

    @cached_property
    def use_function_calling(self) -> bool:
        return isinstance(self.parse_function, FunctionCallingParser)

    @cached_property
    def state_commands(self) -> list[str]:
        return []

    @cached_property
    def commands(self) -> list[Command]:
        commands = []
        if self.enable_bash_tool:
            commands.append(BASH_COMMAND)
        return commands

    @cached_property
    def tools(self) -> list[dict]:
        return [command.get_function_calling_tool() for command in self.commands]

    def model_post_init(self, __context):
        commands = self.commands
        multi_line_command_endings = {
            command.name: command.end_name for command in commands if command.end_name is not None
        }
        self.tools
        self.multi_line_command_endings = multi_line_command_endings
        self.command_docs = generate_command_docs(self.commands, **self.env_variables)
        if self.format_error_template is None:
            self.format_error_template = self.parse_function.format_error_template


# ==============================================================================
# MODEL CONFIG
# ==============================================================================

GLOBAL_STATS = {"total_cost": 0, "last_query_timestamp": 0}
GLOBAL_STATS_LOCK = Lock()


class RetryConfig(BaseModel):
    """This configuration object specifies how many times to retry a failed LM API call."""
    retries: int = 20
    min_wait: float = 10
    max_wait: float = 120


class GenericAPIModelConfig(BaseModel):
    """This configuration object specifies a LM like GPT4 or similar."""
    name: str = Field(description="Name of the model.")
    per_instance_cost_limit: float = 3.0
    total_cost_limit: float = 0.0
    per_instance_call_limit: int = 0
    temperature: float = 0.0
    top_p: float | None = 1.0
    api_base: str | None = None
    api_version: str | None = None
    api_key: SecretStr | None = None
    stop: list[str] = []
    completion_kwargs: dict[str, Any] = {}
    convert_system_to_user: bool = False
    retry: RetryConfig = Field(default_factory=RetryConfig)
    delay: float = 0.0
    fallbacks: list[dict[str, Any]] = []
    choose_api_key_by_thread: bool = True
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    litellm_model_registry: str | None = None
    custom_tokenizer: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class InstanceStats(BaseModel):
    """This object tracks usage numbers (costs etc.) for a single instance."""
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) + getattr(other, field) for field in self.model_fields.keys()},
        )


class AbstractModel(ABC):
    """Abstract base class for models."""

    def __init__(self, config: GenericAPIModelConfig, tools: ToolConfig):
        self.config = config
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

    def reset_stats(self):
        self.stats = InstanceStats()

    @abstractmethod
    def query(self, history: History) -> dict:
        raise NotImplementedError

    @property
    def instance_cost_limit(self) -> float:
        return 0


class LiteLLMModel(AbstractModel):
    """Model served by the `litellm` library."""

    def __init__(self, config: GenericAPIModelConfig, tools: ToolConfig):
        super().__init__(config, tools)
        if tools.use_function_calling:
            if not litellm.utils.supports_function_calling(model=config.name):
                self.logger.warning(f"Model {config.name} does not support function calling.")

        if config.max_input_tokens is not None:
            self.model_max_input_tokens = config.max_input_tokens
        else:
            self.model_max_input_tokens = litellm.model_cost.get(config.name, {}).get("max_input_tokens")

        if config.max_output_tokens is not None:
            self.model_max_output_tokens = config.max_output_tokens
        else:
            self.model_max_output_tokens = litellm.model_cost.get(config.name, {}).get("max_output_tokens")

        self.lm_provider = litellm.model_cost.get(config.name, {}).get("litellm_provider", config.name)
        self.custom_tokenizer = None

    @property
    def instance_cost_limit(self) -> float:
        return self.config.per_instance_cost_limit

    def _update_stats(self, *, input_tokens: int, output_tokens: int, cost: float) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS["total_cost"] += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        self.logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )

        if 0 < self.config.total_cost_limit < GLOBAL_STATS["total_cost"]:
            raise TotalCostLimitExceededError("Total cost limit exceeded")

        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            raise InstanceCostLimitExceededError("Instance cost limit exceeded")

        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            raise InstanceCallLimitExceededError("Per instance call limit exceeded")

    def _sleep(self) -> None:
        elapsed_time = time.time() - GLOBAL_STATS["last_query_timestamp"]
        if elapsed_time < self.config.delay:
            time.sleep(self.config.delay - elapsed_time)
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS["last_query_timestamp"] = time.time()

    def _single_query(self, messages: list[dict[str, str]]) -> list[dict]:
        self._sleep()
        messages_no_cache_control = copy.deepcopy(messages)
        for message in messages_no_cache_control:
            if "cache_control" in message:
                del message["cache_control"]
            if "thinking_blocks" in message:
                del message["thinking_blocks"]

        input_tokens = litellm.utils.token_counter(
            messages=messages_no_cache_control,
            model=self.config.name,
        )
        if self.model_max_input_tokens and input_tokens > self.model_max_input_tokens > 0:
            raise ContextWindowExceededError(f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}")

        extra_args = {}
        if self.config.api_base:
            extra_args["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools

        completion_kwargs = self.config.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens

        try:
            response = litellm.completion(
                model=self.config.name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
                fallbacks=self.config.fallbacks,
                **completion_kwargs,
                **extra_args,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise

        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.name)
        except Exception:
            if self.config.per_instance_cost_limit > 0 or self.config.total_cost_limit > 0:
                raise ModelConfigurationError(
                    f"Error calculating cost for model {self.config.name}. "
                    "Please set per_instance_cost_limit and total_cost_limit to 0."
                )
            cost = 0

        choices = response.choices
        outputs = []
        output_tokens = 0
        for i in range(len(choices)):
            output = choices[i].message.content or ""
            output_tokens += litellm.utils.token_counter(text=output, model=self.config.name)
            output_dict = {"message": output}
            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:
                    tool_calls = [call.to_dict() for call in response.choices[i].message.tool_calls]
                else:
                    tool_calls = []
                output_dict["tool_calls"] = tool_calls
            outputs.append(output_dict)

        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return outputs

    def _query(self, messages: list[dict[str, str]]) -> list[dict]:
        return self._single_query(messages)

    def query(self, history: History) -> dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            if retry_state.outcome is not None and retry_state.outcome.exception() is not None:
                exception = retry_state.outcome.exception()
                exception_info = f" due to {exception.__class__.__name__}: {str(exception)}"
            else:
                exception_info = ""
            self.logger.warning(
                f"Retrying LM query: attempt {retry_state.attempt_number} "
                f"(slept for {retry_state.idle_for:.2f}s){exception_info}"
            )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(min=self.config.retry.min_wait, max=self.config.retry.max_wait),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    ContextWindowExceededError,
                    CostLimitExceededError,
                    RuntimeError,
                    litellm.exceptions.UnsupportedParamsError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.PermissionDeniedError,
                    litellm.exceptions.ContextWindowExceededError,
                    litellm.exceptions.APIError,
                    litellm.exceptions.ContentPolicyViolationError,
                    TypeError,
                    litellm.exceptions.AuthenticationError,
                    ContentPolicyViolationError,
                    ModelConfigurationError,
                    KeyboardInterrupt,
                    IndexError,
                )
            ),
            before_sleep=retry_warning,
        ):
            with attempt:
                result = self._query(messages)
        return result[0]

    def _history_to_messages(self, history: History) -> list[dict[str, str]]:
        history = copy.deepcopy(history)

        def get_role(history_item: HistoryItem) -> str:
            if history_item["role"] == "system":
                return "user" if self.config.convert_system_to_user else "system"
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)
            if role == "tool":
                message = {
                    "role": role,
                    "content": history_item["content"],
                    "tool_call_id": history_item["tool_call_ids"][0],
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {"role": role, "content": history_item["content"], "tool_calls": tool_calls}
            else:
                message = {"role": role, "content": history_item["content"]}
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]
            messages.append(message)
        return messages


# ==============================================================================
# PROBLEM STATEMENT
# ==============================================================================

class ProblemStatement(Protocol):
    """A problem statement for a task."""
    id: str

    def get_problem_statement(self) -> str: ...

    def get_problem_statement_for_env(self) -> str:
        return self.get_problem_statement()

    def get_extra_fields(self) -> dict[str, Any]: ...


class TextProblemStatement(BaseModel):
    """Simple text-based problem statement."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    type: Literal["text"] = "text"

    model_config = ConfigDict(extra="forbid")

    def get_problem_statement(self) -> str:
        return self.text

    def get_extra_fields(self) -> dict[str, Any]:
        return self.extra_fields


class FileProblemStatement(BaseModel):
    """Problem statement from a file."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    path: Path
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    type: Literal["text_file"] = "text_file"

    model_config = ConfigDict(extra="forbid")

    def get_problem_statement(self) -> str:
        return self.path.read_text()

    def get_extra_fields(self) -> dict[str, Any]:
        return self.extra_fields


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class EnvironmentConfig(BaseModel):
    """Configure data sources and setup instructions for the environment."""
    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )
    repo: Any | None = None
    post_startup_commands: list[str] = []
    post_startup_command_timeout: int = 500
    name: str = "main"
    model_config = ConfigDict(extra="forbid")


class SWEEnv:
    """This class represents the environment in which we solve the tasks."""

    def __init__(
        self,
        *,
        deployment: AbstractDeployment,
        repo,
        post_startup_commands: list[str],
        post_startup_command_timeout: int = 500,
        name: str = "main",
    ):
        self.deployment = deployment
        self.repo = repo
        self._post_startup_commands = post_startup_commands
        self.post_startup_command_timeout = post_startup_command_timeout
        self.logger = get_logger("swea-env", emoji="🪴")
        self.name = name
        self.clean_multi_line_functions = lambda x: x

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> Self:
        config = config.model_copy(deep=True)
        return cls(
            deployment=get_deployment(config.deployment),
            repo=config.repo,
            post_startup_commands=config.post_startup_commands,
            post_startup_command_timeout=config.post_startup_command_timeout,
            name=config.name,
        )

    def start(self) -> None:
        """Start the environment and reset it to a clean state."""
        self._init_deployment()
        self.reset()
        for command in self._post_startup_commands:
            self.communicate(command, check="raise", timeout=self.post_startup_command_timeout)

    def reset(self):
        """Reset the environment to a clean state."""
        self.communicate(input="cd /", check="raise")
        self._copy_repo()
        self._reset_repository()

    def _copy_repo(self) -> None:
        """Clone/copy repository/codebase in container"""
        if self.repo is None:
            return
        if hasattr(self.repo, 'copy'):
            self.repo.copy(self.deployment)

    def _reset_repository(self) -> None:
        """Clean repository of any modifications + Checkout base commit"""
        if self.repo is not None and hasattr(self.repo, 'repo_name'):
            self.logger.debug(f"Resetting repository {self.repo.repo_name} to commit {self.repo.base_commit}")
            startup_commands = [
                f"cd /{self.repo.repo_name}",
                "export ROOT=$(pwd -P)",
            ]
            if hasattr(self.repo, 'get_reset_commands'):
                startup_commands.extend(self.repo.get_reset_commands())
            self.communicate(
                input=" && ".join(startup_commands),
                check="raise",
                error_msg="Failed to clean repository",
                timeout=120,
            )

    def close(self) -> None:
        """Shutdown SWE-ReX deployment etc."""
        self.logger.info("Beginning environment shutdown...")
        asyncio.run(self.deployment.stop())

    def _init_deployment(self) -> None:
        """Handles container initialization."""
        asyncio.run(self.deployment.start())
        asyncio.run(
            self.deployment.runtime.create_session(
                CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=10)
            )
        )
        self.set_env_variables({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PIP_PROGRESS_BAR": "off", "PAGER": "cat"})
        self.logger.info("Environment Initialized")

    def interrupt_session(self):
        self.logger.info("Interrupting session")
        asyncio.run(self.deployment.runtime.run_in_session(BashInterruptAction()))

    def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Executes a command in the running shell."""
        self.logger.log(logging.TRACE, "Input:\n%s", input)
        rex_check = "silent" if check else "ignore"
        r = asyncio.run(
            self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        )
        output = r.output
        self.logger.log(logging.TRACE, "Output:\n%s", output)
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            self.logger.error(msg)
            if check == "raise":
                self.close()
                raise RuntimeError(msg)
        return output

    def read_file(self, path: str | Path, encoding: str | None = None, errors: str | None = None) -> str:
        """Read file contents from container"""
        r = asyncio.run(
            self.deployment.runtime.read_file(ReadFileRequest(path=str(path), encoding=encoding, errors=errors))
        )
        return r.content

    def write_file(self, path: str | Path, content: str) -> None:
        """Write content to file in container"""
        asyncio.run(self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content)))

    def set_env_variables(self, env_variables: dict[str, str]) -> None:
        """Set environment variables in the environment."""
        if not env_variables:
            return
        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
        command = " && ".join(_env_setters)
        self.communicate(command, check="raise")

    def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Execute a command in the environment independent of the session"""
        asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=shell, check=check, env=env, cwd=cwd))
        )


# ==============================================================================
# AGENT
# ==============================================================================

class TemplateConfig(BaseModel):
    """Configuration for message templates."""
    system_template: str = (
        "You are an autonomous AI agent designed to solve software engineering problems. "
        "You have access to a set of tools to help you complete your tasks."
    )
    instance_template: str = "Problem Statement:\n{{problem_statement}}\n\n{{command_docs}}"
    next_step_template: str = "Observation: {{observation}}"
    next_step_truncated_observation_template: str = (
        "Observation: {{observation[:max_observation_length]}}<response clipped>"
        "<NOTE>Observations should not exceeded {{max_observation_length}} characters. "
        "{{elided_chars}} characters were elided. Please try a different command.</NOTE>"
    )
    max_observation_length: int = 100_000
    next_step_no_output_template: str | None = None
    strategy_template: str | None = None
    demonstration_template: str | None = None
    demonstrations: list[Path] = Field(default_factory=list)
    put_demos_in_history: bool = False
    disable_image_processing: bool = False
    shell_check_error_template: str = (
        "Your bash command contained syntax errors and was NOT executed. "
        "Please fix the syntax errors and try again. Here is the output of `bash -n`:\n"
        "{{bash_stdout}}\n{{bash_stderr}}"
    )
    command_cancelled_timeout_template: str = (
        "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. "
        "Please try a different command that completes more quickly."
    )

    def model_post_init(self, __context):
        if self.next_step_no_output_template is None:
            self.next_step_no_output_template = self.next_step_template


class DefaultAgentConfig(BaseModel):
    """Configuration for the default agent."""
    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    model: GenericAPIModelConfig = Field(description="Model options.")
    max_requeries: int = 3
    type: Literal["default"] = "default"
    model_config = ConfigDict(extra="forbid")


class DefaultAgent:
    """The default SWE-agent implementation."""

    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolConfig,
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
    ):
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | None = None
        self.traj_path: Path | None = None
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()
        self._replay_config: BaseModel | None = None
        self._total_execution_time = 0.0
        self._n_consecutive_timeouts = 0

    @classmethod
    def from_config(cls, config: DefaultAgentConfig) -> Self:
        config = config.model_copy(deep=True)
        model = LiteLLMModel(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            model=model,
            max_requeries=config.max_requeries,
        )

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset."""
        filtered_history = [entry for entry in self.history if entry.get("agent") == self.name]
        return filtered_history

    def _append_history(self, item: dict[str, Any]) -> None:
        self.history.append(item)

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance."""
        output_dir.mkdir(parents=True, exist_ok=True)

        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)

        self.tools.install(env)
        self.info = AgentInfo()
        env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self.add_system_message_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(env))

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates"""
        assert self._problem_statement is not None
        assert self._env is not None
        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:
        """Populate selected template(s) with information and add to history."""
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)

        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template."""
        templates: list[str] = []
        assert self.history[-1]["role"] == "system"
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it."""
        step = step.model_copy(deep=True)
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            try:
                submission = self._env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                self.logger.exception("Failed to read submission file, got %s", e)
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment."""
        if self.tools.should_block_action(step.action):
            raise FormatError("Action is blocked")

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = self.tools.get_state(env=self._env)
            return step

        assert self._env is not None
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="ignore",
            )
        except CommandTimeoutError:
            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception:
                pass
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        step.state = self.tools.get_state(env=self._env)

        return self.handle_submission(step)

    def forward(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors."""
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise RuntimeError("Total execution time exceeded")

        step = StepOutput()
        step.query = copy.deepcopy(history)
        try:
            output = self.model.query(history)
            step.output = output["message"]
            step.thought, step.action = self.tools.parse_actions(output)
            step.thinking_blocks = output.get("thinking_blocks", [])
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n🎬 ACTION\n{step.action.strip()}")
            return self.handle_action(step)
        except Exception as e:
            if step.action == step.thought == "":
                step.thought = step.output
            e.step = step
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can."""
        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)
            except KeyboardInterrupt:
                raise
            except EOFError:
                raise
            except FormatError as e:
                n_format_fails += 1
                self.logger.warning("Requerying model after %s (%dth requery)", type(e).__name__, n_format_fails)
                step: StepOutput = getattr(e, "step", StepOutput())
                self.add_step_to_trajectory(step)
                history = history + [
                    {"role": "assistant", "content": step.output, "agent": self.name, "message_type": "assistant"},
                    {"role": "user", "content": self.tools.config.format_error_template, "agent": self.name, "message_type": "user"},
                ]
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                self.logger.warning("Requerying model after %s (%dth requery)", type(e).__name__, n_format_fails)
                step: StepOutput = getattr(e, "step", StepOutput())
                self.add_step_to_trajectory(step)
                history = history + [
                    {"role": "assistant", "content": step.output, "agent": self.name, "message_type": "assistant"},
                    {"role": "user", "content": str(e), "agent": self.name, "message_type": "user"},
                ]
            except ContextWindowExceededError:
                step = StepOutput(thought="Context window exceeded", exit_status="exit_context", done=True)
                return step
            except CostLimitExceededError:
                step = StepOutput(thought="Cost limit exceeded", exit_status="exit_cost", done=True)
                return step
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}")
                step = StepOutput(thought=f"Runtime error: {e}", exit_status="exit_error", done=True)
                return step
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}")
                step = StepOutput(thought=f"Unknown error: {e}", exit_status="exit_error", done=True)
                return step

        self.logger.exception("Exit due to repeated format errors")
        step = StepOutput(thought="Repeated format errors", exit_status="exit_format", done=True)
        return step

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep({
            "action": step.action,
            "observation": step.observation,
            "response": step.output,
            "thought": step.thought,
            "execution_time": step.execution_time,
            "state": step.state,
            "query": step.query,
            "extra_info": step.extra_info,
        })
        self.trajectory.append(trajectory_step)

    def step(self) -> StepOutput:
        """Run a step of the agent."""
        assert self._env is not None

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status
        self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)
        return step_output

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement,
        output_dir: Path = Path("."),
    ) -> tuple[AgentInfo, Trajectory]:
        """Run the agent on a problem instance."""
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        step_output = StepOutput()
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory()

        self.logger.info("Trajectory saved to %s", self.traj_path)

        data = self.get_trajectory_data()
        return data["info"], data["trajectory"]

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""
        assert self._env is not None
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["environment"] = self._env.name
        return attempt_data

    def save_trajectory(self) -> None:
        """Save the trajectory to disk."""
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))


# ==============================================================================
# TOOL HANDLER
# ==============================================================================

class ToolHandler:
    """This class handles most of the tool usage."""

    def __init__(self, tools: ToolConfig):
        self.config = tools.model_copy(deep=True)
        self._reset_commands = []
        self._command_patterns = self._get_command_patterns()
        self.logger = get_logger("swea-tools", emoji="🧰")
        self.mock_state: dict[str, str] | None = None

    def install(self, env: SWEEnv) -> None:
        self.reset(env)

    def reset(self, env: SWEEnv) -> None:
        self.logger.info("Resetting tools")
        env_variables = self.config.env_variables.copy() | {
            var: os.getenv(var) for var in self.config.propagate_env_variables
        }
        env.set_env_variables(env_variables)

    def get_state(self, env: SWEEnv) -> dict[str, str]:
        """Execute state commands from all bundles and combine their results."""
        if self.mock_state is not None:
            return self.mock_state
        return {}

    def should_block_action(self, action: str) -> bool:
        """Check if the command should be blocked."""
        action = action.strip()
        if not action:
            return False
        if any(f.startswith(action) for f in self.config.filter.blocklist):
            return True
        if action in self.config.filter.blocklist_standalone:
            return True
        return False

    def check_for_submission_cmd(self, output: str) -> bool:
        """Function for checking submission request."""
        if r"<<SWE_AGENT_SUBMISSION>>" in output:
            return True
        return False

    def parse_actions(self, output: dict) -> tuple[str, str]:
        """Parse the model output into a thought and action."""
        return self.config.parse_function(output, self.config.commands)

    def guard_multiline_input(self, action: str) -> str:
        """Split action by multiline commands."""
        return _guard_multiline_input(action, self._get_first_multiline_cmd)

    def _get_first_multiline_cmd(self, action: str) -> re.Match | None:
        """Return the first match of a command pattern in the action string."""
        patterns = {
            k: v
            for k, v in self._command_patterns.items()
            if k in self.config.multi_line_command_endings or k == self.config.submit_command
        }
        matches = list()
        for _, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]

    def _get_command_patterns(self) -> dict[str, re.Pattern]:
        """Creates regular expressions for the commands"""
        _command_patterns = {}
        for command in self.config.commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                _command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                _command_patterns[command.name] = pat
        return _command_patterns


# ==============================================================================
# RUN CONFIG
# ==============================================================================

class RunSingleActionConfig(BaseModel):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""
    apply_patch_locally: bool = False
    model_config = ConfigDict(extra="forbid")


class RunSingleConfig(BaseSettings, cli_implicit_flags=False):
    """Configuration for running SWE-agent on a single instance."""
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment options.")
    agent: DefaultAgentConfig = Field(description="Agent options.")
    problem_statement: TextProblemStatement = Field(
        default_factory=TextProblemStatement,
        description="Problem statement options."
    )
    output_dir: Path = Field(default=Path("trajectories"), description="Output directory.")
    actions: RunSingleActionConfig = Field(default_factory=RunSingleActionConfig)
    env_var_path: Path | None = None
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

class RunSingle:
    """Main runner for SWE-agent."""

    def __init__(
        self,
        env: SWEEnv,
        agent: DefaultAgent,
        problem_statement: ProblemStatement,
        *,
        output_dir: Path = Path("."),
        actions: RunSingleActionConfig | None = None,
    ):
        self.logger = get_logger("swea-run", emoji="🏃")
        instance_id = problem_statement.id
        self.env = env
        self.agent = agent
        self.output_dir = output_dir
        self.actions = actions or RunSingleActionConfig()
        self.problem_statement = problem_statement

    @classmethod
    def from_config(cls, config: RunSingleConfig) -> Self:
        """Create a RunSingle instance from a configuration object."""
        config.output_dir.mkdir(parents=True, exist_ok=True)
        agent = DefaultAgent.from_config(config.agent)

        problem_statement: ProblemStatement = config.problem_statement

        return cls(
            env=SWEEnv.from_config(config.env),
            agent=agent,
            problem_statement=problem_statement,
            output_dir=config.output_dir,
            actions=config.actions,
        )

    def run(self):
        """Run the agent."""
        self.logger.info("Starting environment")
        self.env.start()
        self.logger.info("Running agent")
        output_dir = self.output_dir / self.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)

        info, trajectory = self.agent.run(
            problem_statement=self.problem_statement,
            env=self.env,
            output_dir=output_dir,
        )

        self.logger.info("Done")
        self.env.close()

        return info, trajectory


# ==============================================================================
# CLI
# ==============================================================================

def run_from_cli(args: list[str] | None = None):
    """Run SWE-agent from CLI arguments."""
    if args is None:
        args = sys.argv[1:]

    import argparse

    parser = argparse.ArgumentParser(
        description="SWE-Agent - Software Engineering Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--problem", type=str, help="Problem statement text")
    parser.add_argument("--problem-file", type=str, help="Path to problem statement file")
    parser.add_argument("--repo", type=str, help="Path to repository")
    parser.add_argument("--github-url", type=str, help="GitHub issue URL")
    parser.add_argument("--output-dir", type=str, default="trajectories", help="Output directory")
    parser.add_argument("--cost-limit", type=float, default=3.0, help="Per-instance cost limit")
    parser.add_argument("--api-key", type=str, help="API key for the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps")

    parsed = parser.parse_args(args)

    # Create problem statement
    if parsed.github_url:
        # For GitHub URLs, we'd need to fetch the issue
        problem_statement = TextProblemStatement(
            text=f"Please solve the issue at: {parsed.github_url}",
            id=f"github_{parsed.github_url.split('/')[-1]}"
        )
    elif parsed.problem_file:
        problem_statement = FileProblemStatement(path=Path(parsed.problem_file))
    elif parsed.problem:
        problem_statement = TextProblemStatement(text=parsed.problem)
    else:
        problem_statement = TextProblemStatement(text="Please help me solve a software engineering problem.")

    # Create config
    config = RunSingleConfig(
        agent=DefaultAgentConfig(
            model=GenericAPIModelConfig(
                name=parsed.model,
                per_instance_cost_limit=parsed.cost_limit,
                temperature=parsed.temperature,
                api_key=parsed.api_key,
            )
        ),
        problem_statement=problem_statement,
        output_dir=Path(parsed.output_dir),
    )

    # Run
    runner = RunSingle.from_config(config)
    info, trajectory = runner.run()

    print(f"\n{'='*60}")
    print("Run complete!")
    print(f"Exit status: {info.get('exit_status', 'unknown')}")
    print(f"API calls: {info.get('model_stats', {}).get('api_calls', 0)}")
    print(f"Total cost: ${info.get('model_stats', {}).get('instance_cost', 0.0):.4f}")
    print(f"Trajectory saved to: {runner.output_dir / problem_statement.id}")
    print(f"{'='*60}\n")

    return info, trajectory


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    run_from_cli()
