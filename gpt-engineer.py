"""
Unified Coding Agent - A single-file AI-powered code generation and improvement agent.

This module consolidates the entire GPT Engineer architecture into a single file,
providing a complete coding agent capable of:
- Generating new code from prompts
- Improving existing code
- Managing files and execution environments
- Tracking token usage and logging

Author: GPT Engineer (Consolidated)
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import backoff
import math
import openai

try:
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        messages_from_dict,
        messages_to_dict,
    )
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from PIL import Image
    from regex import regex
except ImportError as e:
    raise ImportError(
        "Required dependencies not found. Please install: "
        "langchain, langchain-openai, langchain-anthropic, pillow, regex, backoff"
    ) from e

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_EDIT_REFINEMENT_STEPS = 2

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY CLASSES
# =============================================================================

class FilesDict(dict):
    """
    A dictionary-based container for managing code files.

    Extends the standard dictionary to enforce string keys and values,
    representing filenames and their corresponding code content.
    """

    def __setitem__(self, key: Union[str, Path], value: str):
        if not isinstance(key, (str, Path)):
            raise TypeError("Keys must be strings or Path's")
        if not isinstance(value, str):
            raise TypeError("Values must be strings")
        super().__setitem__(key, value)

    def to_chat(self) -> str:
        """Formats the items for chat display."""
        chat_str = ""
        for file_name, file_content in self.items():
            lines_dict = file_to_lines_dict(file_content)
            chat_str += f"File: {file_name}\n"
            for line_number, line_content in lines_dict.items():
                chat_str += f"{line_number} {line_content}\n"
            chat_str += "\n"
        return f"```\n{chat_str}```"

    def to_log(self) -> str:
        """Formats the items for log display."""
        log_str = ""
        for file_name, file_content in self.items():
            log_str += f"File: {file_name}\n"
            log_str += file_content
            log_str += "\n"
        return log_str


def file_to_lines_dict(file_content: str) -> dict:
    """Converts file content into a dictionary of line numbers to content."""
    return OrderedDict(
        {
            line_number: line_content
            for line_number, line_content in enumerate(file_content.split("\n"), 1)
        }
    )


class Prompt:
    """Represents a user prompt with optional image URLs."""

    def __init__(
        self,
        text: str,
        image_urls: Optional[Dict[str, str]] = None,
        entrypoint_prompt: str = "",
    ):
        self.text = text
        self.image_urls = image_urls
        self.entrypoint_prompt = entrypoint_prompt

    def __repr__(self):
        return f"Prompt(text={self.text!r}, image_urls={self.image_urls!r})"

    def to_langchain_content(self):
        content = [{"type": "text", "text": f"Request: {self.text}"}]
        if self.image_urls:
            for name, url in self.image_urls.items():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"},
                })
        return content

    def to_dict(self):
        return {
            "text": self.text,
            "image_urls": self.image_urls,
            "entrypoint_prompt": self.entrypoint_prompt,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


# =============================================================================
# MEMORY SYSTEM
# =============================================================================

BaseMemory = MutableMapping[Union[str, Path], str]


class DiskMemory(BaseMemory):
    """A file-based key-value store where keys are filenames and values are file contents."""

    def __init__(self, path: Union[str, Path]):
        self.path: Path = Path(path).absolute()
        self.path.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        return (self.path / key).is_file()

    def __getitem__(self, key: str) -> str:
        full_path = self.path / key
        if not full_path.is_file():
            raise KeyError(f"File '{key}' could not be found in '{self.path}'")

        if full_path.suffix in [".png", ".jpeg", ".jpg"]:
            with full_path.open("rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = "image/png" if full_path.suffix == ".png" else "image/jpeg"
                return f"data:{mime_type};base64,{encoded_string}"
        else:
            with full_path.open("r", encoding="utf-8") as f:
                return f.read()

    def get(self, key: str, default: Any = None) -> Any:
        item_path = self.path / key
        try:
            if item_path.is_file():
                return self[key]
            elif item_path.is_dir():
                return DiskMemory(item_path)
            else:
                return default
        except:
            return default

    def __setitem__(self, key: Union[str, Path], val: str) -> None:
        if str(key).startswith("../"):
            raise ValueError(f"File name {key} attempted to access parent path.")
        if not isinstance(val, str):
            raise TypeError("val must be str")
        full_path = self.path / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(val, encoding="utf-8")

    def __delitem__(self, key: Union[str, Path]) -> None:
        item_path = self.path / key
        if not item_path.exists():
            raise KeyError(f"Item '{key}' could not be found in '{self.path}'")
        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)

    def __iter__(self) -> Iterator[str]:
        return iter(
            sorted(
                str(item.relative_to(self.path))
                for item in sorted(self.path.rglob("*"))
                if item.is_file()
            )
        )

    def __len__(self) -> int:
        return len(list(self.__iter__()))

    def log(self, key: Union[str, Path], val: str) -> None:
        """Append to a log file."""
        if str(key).startswith("../"):
            raise ValueError(f"File name {key} attempted to access parent path.")
        if not isinstance(val, str):
            raise TypeError("val must be str")
        full_path = self.path / "logs" / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            full_path.touch()
        with open(full_path, "a", encoding="utf-8") as file:
            file.write(f"\n{datetime.now().isoformat()}\n")
            file.write(val + "\n")


# =============================================================================
# EXECUTION ENVIRONMENT
# =============================================================================

class BaseExecutionEnv(ABC):
    """Abstract base class for an execution environment."""

    @abstractmethod
    def run(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        raise NotImplementedError

    @abstractmethod
    def popen(self, command: str) -> subprocess.Popen:
        raise NotImplementedError

    @abstractmethod
    def upload(self, files: FilesDict) -> "BaseExecutionEnv":
        raise NotImplementedError

    @abstractmethod
    def download(self) -> FilesDict:
        raise NotImplementedError


class DiskExecutionEnv(BaseExecutionEnv):
    """An execution environment that runs code on the local file system."""

    def __init__(self, path: Union[str, Path, None] = None):
        self.working_dir = Path(path or tempfile.mkdtemp(prefix="gpt-engineer-"))
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, files: FilesDict) -> "DiskExecutionEnv":
        for name, content in files.items():
            path = self.working_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        return self

    def download(self) -> FilesDict:
        files = {}
        for path in self.working_dir.glob("**/*"):
            if path.is_file():
                with open(path, "r") as f:
                    try:
                        content = f.read()
                    except UnicodeDecodeError:
                        content = "binary file"
                    files[str(path.relative_to(self.working_dir))] = content
        return FilesDict(files)

    def popen(self, command: str) -> subprocess.Popen:
        p = subprocess.Popen(
            command,
            shell=True,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p

    def run(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        start = time.time()
        print("\n--- Start of run ---")
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.working_dir,
            text=True,
            shell=True,
        )
        print("$", command)
        stdout_full, stderr_full = "", ""

        try:
            while p.poll() is None:
                assert p.stdout is not None
                assert p.stderr is not None
                stdout = p.stdout.readline()
                stderr = p.stderr.readline()
                if stdout:
                    print(stdout, end="")
                    stdout_full += stdout
                if stderr:
                    print(stderr, end="")
                    stderr_full += stderr
                if timeout and time.time() - start > timeout:
                    print("Timeout!")
                    p.kill()
                    raise TimeoutError()
        except KeyboardInterrupt:
            print()
            print("Stopping execution.")
            p.kill()
            print("--- Finished run ---\n")

        return stdout_full, stderr_full, p.returncode


# =============================================================================
# TOKEN USAGE TRACKING
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage statistics for a conversation step."""
    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class Tokenizer:
    """Tokenizer for counting tokens."""

    def __init__(self, model_name):
        self.model_name = model_name
        try:
            import tiktoken
            self._tiktoken_tokenizer = (
                tiktoken.encoding_for_model(model_name)
                if "gpt-4" in model_name or "gpt-3.5" in model_name
                else tiktoken.get_encoding("cl100k_base")
            )
        except ImportError:
            self._tiktoken_tokenizer = None

    def num_tokens(self, txt: str) -> int:
        if self._tiktoken_tokenizer:
            return len(self._tiktoken_tokenizer.encode(txt))
        return len(txt.split())  # Fallback

    def num_tokens_from_messages(self, messages: List) -> int:
        n_tokens = 0
        for message in messages:
            n_tokens += 4
            if isinstance(message.content, str):
                n_tokens += self.num_tokens(message.content)
            elif isinstance(message.content, list):
                for item in message.content:
                    if item.get("type") == "text":
                        n_tokens += self.num_tokens(item["text"])
            n_tokens += 2
        return n_tokens


class TokenUsageLog:
    """Log of token usage statistics."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._cumulative_prompt_tokens = 0
        self._cumulative_completion_tokens = 0
        self._cumulative_total_tokens = 0
        self._log = []
        self._tokenizer = Tokenizer(model_name)

    def update_log(self, messages: List, answer: str, step_name: str) -> None:
        prompt_tokens = self._tokenizer.num_tokens_from_messages(messages)
        completion_tokens = self._tokenizer.num_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens

        self._cumulative_prompt_tokens += prompt_tokens
        self._cumulative_completion_tokens += completion_tokens
        self._cumulative_total_tokens += total_tokens

        self._log.append(
            TokenUsage(
                step_name=step_name,
                in_step_prompt_tokens=prompt_tokens,
                in_step_completion_tokens=completion_tokens,
                in_step_total_tokens=total_tokens,
                total_prompt_tokens=self._cumulative_prompt_tokens,
                total_completion_tokens=self._cumulative_completion_tokens,
                total_tokens=self._cumulative_total_tokens,
            )
        )


# =============================================================================
# AI INTERFACE
# =============================================================================

Message = Union[AIMessage, HumanMessage, SystemMessage]


class AI:
    """
    A class that interfaces with language models for conversation management.
    """

    def __init__(
        self,
        model_name="gpt-4-turbo",
        temperature=0.1,
        azure_endpoint=None,
        streaming=True,
        vision=False,
    ):
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.model_name = model_name
        self.streaming = streaming
        self.vision = (
            ("vision-preview" in model_name)
            or ("gpt-4-turbo" in model_name and "preview" not in model_name)
            or ("claude" in model_name)
        )
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)
        logger.debug(f"Using model {self.model_name}")

    def start(self, system: str, user: Any, *, step_name: str) -> List[Message]:
        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def _extract_content(self, content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content and "text" in content[0]:
            return content[0]["text"]
        else:
            return ""

    def _collapse_text_messages(self, messages: List[Message]):
        collapsed_messages = []
        if not messages:
            return collapsed_messages

        previous_message = messages[0]
        combined_content = self._extract_content(previous_message.content)

        for current_message in messages[1:]:
            if current_message.type == previous_message.type:
                combined_content += "\n\n" + self._extract_content(
                    current_message.content
                )
            else:
                collapsed_messages.append(
                    previous_message.__class__(content=combined_content)
                )
                previous_message = current_message
                combined_content = self._extract_content(current_message.content)

        collapsed_messages.append(previous_message.__class__(content=combined_content))
        return collapsed_messages

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug(
            "Creating a new chat completion: %s",
            "\n".join([m.pretty_repr() for m in messages]),
        )

        if not self.vision:
            messages = self._collapse_text_messages(messages)

        response = self.backoff_inference(messages)

        self.token_usage_log.update_log(
            messages=messages, answer=response.content, step_name=step_name
        )
        messages.append(response)
        logger.debug(f"Chat completion finished: {messages}")

        return messages

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=7, max_time=45)
    def backoff_inference(self, messages):
        return self.llm.invoke(messages)

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        data = json.loads(jsondictstr)
        prevalidated_data = [
            {**item, "tools": {**item.get("tools", {}), "is_chunk": False}}
            for item in data
        ]
        return list(messages_from_dict(prevalidated_data))

    def _create_chat_model(self) -> BaseChatModel:
        if self.azure_endpoint:
            return AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-05-01-preview"),
                deployment_name=self.model_name,
                openai_api_type="azure",
                streaming=self.streaming,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        elif "claude" in self.model_name:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                callbacks=[StreamingStdOutCallbackHandler()],
                streaming=self.streaming,
                max_tokens_to_sample=4096,
            )
        elif self.vision:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
                callbacks=[StreamingStdOutCallbackHandler()],
                max_tokens=4096,
            )
        else:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
                callbacks=[StreamingStdOutCallbackHandler()],
            )


# =============================================================================
# DIFF PARSING AND APPLICATION
# =============================================================================

RETAIN = "retain"
ADD = "add"
REMOVE = "remove"


def is_similar(str1, str2, similarity_threshold=0.9) -> bool:
    """Compares two strings for similarity."""
    return count_ratio(str1, str2) >= similarity_threshold


def count_ratio(str1, str2) -> float:
    """Computes the ratio of common characters to the length of the longer string."""
    str1, str2 = str1.replace(" ", "").lower(), str2.replace(" ", "").lower()
    counter1, counter2 = Counter(str1), Counter(str2)
    intersection = sum((counter1 & counter2).values())
    longer_length = max(len(str1), len(str2))
    return 1 if longer_length == 0 else intersection / longer_length


class Hunk:
    """Represents a section of a file diff."""

    def __init__(
        self,
        start_line_pre_edit,
        hunk_len_pre_edit,
        start_line_post_edit,
        hunk_len_post_edit,
        lines,
    ) -> None:
        self.start_line_pre_edit = start_line_pre_edit
        self.hunk_len_pre_edit = hunk_len_pre_edit
        self.start_line_post_edit = start_line_post_edit
        self.hunk_len_post_edit = hunk_len_post_edit
        self.category_counts = {RETAIN: 0, ADD: 0, REMOVE: 0}
        self.lines = list()
        self.add_lines(lines)
        self.forward_block_len = 10
        self.is_new_file = (
            self.category_counts[RETAIN] == 0 and self.category_counts[REMOVE] == 0
        )

    def add_lines(self, new_lines) -> None:
        for line in new_lines:
            self.lines.append(line)
            self.category_counts[line[0]] += 1

    def hunk_to_string(self) -> str:
        string = f"@@ -{self.start_line_pre_edit},{self.hunk_len_pre_edit} +{self.start_line_post_edit},{self.hunk_len_post_edit} @@\n"
        for line_type, line_content in self.lines:
            line_prefix = (
                " " if line_type == RETAIN else "+" if line_type == ADD else "-"
            )
            string += f"{line_prefix}{line_content}\n"
        return string

    def validate_and_correct(self, lines_dict: dict, problems: list) -> bool:
        """Validates and corrects the hunk (simplified version)."""
        if self.is_new_file:
            return True
        if self.start_line_pre_edit not in lines_dict:
            problems.append(f"Starting line {self.start_line_pre_edit} not found")
            return False
        return True


class Diff:
    """Represents a file diff, containing multiple hunks."""

    def __init__(self, filename_pre, filename_post) -> None:
        self.filename_pre = filename_pre
        self.filename_post = filename_post
        self.hunks = []

    def is_new_file(self) -> bool:
        if self.filename_pre == "/dev/null":
            return True
        return any(hunk.is_new_file for hunk in self.hunks)

    def validate_and_correct(self, lines_dict: dict) -> List[str]:
        problems = []
        for hunk in self.hunks:
            hunk.validate_and_correct(lines_dict, problems)
        return problems


def chat_to_files_dict(chat: str) -> FilesDict:
    """Converts a chat string containing file paths and code blocks into a FilesDict."""
    regex_pattern = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
    matches = re.finditer(regex_pattern, chat, re.DOTALL)

    files_dict = FilesDict()
    for match in matches:
        path = re.sub(r'[\:<>"|?*]', "", match.group(1))
        path = re.sub(r"^\[(.*)\]$", r"\1", path)
        path = re.sub(r"^`(.*)`$", r"\1", path)
        path = re.sub(r"[\]\:]$", "", path)
        content = match.group(2)
        files_dict[path.strip()] = content.strip()

    return files_dict


def apply_diffs(diffs: Dict[str, Diff], files: FilesDict) -> FilesDict:
    """Applies diffs to the provided files."""
    files = FilesDict(files.copy())
    REMOVE_FLAG = "<REMOVE_LINE>"

    for diff in diffs.values():
        if diff.is_new_file():
            files[diff.filename_post] = "\n".join(
                line[1] for hunk in diff.hunks for line in hunk.lines
            )
        else:
            line_dict = file_to_lines_dict(files[diff.filename_pre])
            for hunk in diff.hunks:
                current_line = hunk.start_line_pre_edit
                for line in hunk.lines:
                    if line[0] == RETAIN:
                        current_line += 1
                    elif line[0] == ADD:
                        current_line -= 1
                        if (
                            current_line in line_dict.keys()
                            and line_dict[current_line] != REMOVE_FLAG
                        ):
                            line_dict[current_line] += "\n" + line[1]
                        else:
                            line_dict[current_line] = line[1]
                        current_line += 1
                    elif line[0] == REMOVE:
                        line_dict[current_line] = REMOVE_FLAG
                        current_line += 1

            line_dict = {
                key: line_content
                for key, line_content in line_dict.items()
                if REMOVE_FLAG not in line_content
            }
            files[diff.filename_post] = "\n".join(line_dict.values())
    return files


def parse_diffs(diff_string: str, diff_timeout=3) -> dict:
    """Parses a diff string in the unified git diff format."""
    diff_block_pattern = regex.compile(
        r"```.*?\n\s*?--- .*?\n\s*?\+\+\+ .*?\n(?:@@ .*? @@\n(?:[-+ ].*?\n)*?)*?```",
        re.DOTALL,
    )

    diffs = {}
    try:
        for block in diff_block_pattern.finditer(diff_string, timeout=diff_timeout):
            diff_block = block.group()
            diff = parse_diff_block(diff_block)
            for filename, diff_obj in diff.items():
                if filename not in diffs:
                    diffs[filename] = diff_obj
    except TimeoutError:
        print("Timed out while parsing git diff")

    return diffs


def parse_diff_block(diff_block: str) -> dict:
    """Parses a block of diff text into a Diff object."""
    lines = diff_block.strip().split("\n")[1:-1]
    diffs = {}
    current_diff = None
    hunk_lines = []
    filename_pre = None
    filename_post = None
    hunk_header = None

    for line in lines:
        if line.startswith("--- "):
            filename_pre = line[4:]
        elif line.startswith("+++ "):
            if (
                filename_post is not None
                and current_diff is not None
                and hunk_header is not None
            ):
                current_diff.hunks.append(Hunk(*hunk_header, hunk_lines))
                hunk_lines = []
            filename_post = line[4:]
            current_diff = Diff(filename_pre, filename_post)
            diffs[filename_post] = current_diff
        elif line.startswith("@@ "):
            if hunk_lines and current_diff is not None and hunk_header is not None:
                current_diff.hunks.append(Hunk(*hunk_header, hunk_lines))
                hunk_lines = []
            hunk_header = parse_hunk_header(line)
        elif line.startswith("+"):
            hunk_lines.append((ADD, line[1:]))
        elif line.startswith("-"):
            hunk_lines.append((REMOVE, line[1:]))
        else:
            hunk_lines.append((RETAIN, line[1:]))

    if current_diff is not None and hunk_lines and hunk_header is not None:
        current_diff.hunks.append(Hunk(*hunk_header, hunk_lines))

    return diffs


def parse_hunk_header(header_line) -> Tuple[int, int, int, int]:
    """Parses the header of a hunk from a diff."""
    pattern = re.compile(r"^@@ -\d{1,},\d{1,} \+\d{1,},\d{1,} @@$")
    if not pattern.match(header_line):
        return 0, 0, 0, 0

    pre, post = header_line.split(" ")[1:3]
    start_line_pre_edit, hunk_len_pre_edit = map(int, pre[1:].split(","))
    start_line_post_edit, hunk_len_post_edit = map(int, post[1:].split(","))
    return (
        start_line_pre_edit,
        hunk_len_pre_edit,
        start_line_post_edit,
        hunk_len_post_edit,
    )


# =============================================================================
# PREPROMPTS HOLDER
# =============================================================================

class PrepromptsHolder:
    """A holder for preprompt texts."""

    def __init__(self, preprompts_path: Path):
        self.preprompts_path = preprompts_path

    def get_preprompts(self) -> Dict[str, str]:
        preprompts_repo = DiskMemory(self.preprompts_path)
        return {file_name: preprompts_repo[file_name] for file_name in preprompts_repo}


# =============================================================================
# CODE GENERATION STEPS
# =============================================================================

def curr_fn() -> str:
    """Returns the name of the current function."""
    return inspect.stack()[1].function


def setup_sys_prompt(preprompts: MutableMapping[Union[str, Path], str]) -> str:
    """Sets up the system prompt for generating code."""
    return (
        preprompts.get("roadmap", "")
        + preprompts.get("generate", "").replace("FILE_FORMAT", preprompts.get("file_format", ""))
        + "\nUseful to know:\n"
        + preprompts.get("philosophy", "")
    )


def setup_sys_prompt_existing_code(
    preprompts: MutableMapping[Union[str, Path], str]
) -> str:
    """Sets up the system prompt for improving existing code."""
    return (
        preprompts.get("roadmap", "")
        + preprompts.get("improve", "").replace("FILE_FORMAT", preprompts.get("file_format_diff", ""))
        + "\nUseful to know:\n"
        + preprompts.get("philosophy", "")
    )


def gen_code(
    ai: AI, prompt: Prompt, memory: BaseMemory, preprompts_holder: PrepromptsHolder
) -> FilesDict:
    """Generates code from a prompt using AI."""
    preprompts = preprompts_holder.get_preprompts()
    messages = ai.start(
        setup_sys_prompt(preprompts),
        prompt.to_langchain_content() if isinstance(prompt.to_langchain_content(), str) else str(prompt.to_langchain_content()),
        step_name=curr_fn()
    )
    chat = messages[-1].content.strip()
    memory.log("all_output.txt", "\n\n".join(x.pretty_repr() for x in messages))
    files_dict = chat_to_files_dict(chat)
    return files_dict


def gen_entrypoint(
    ai: AI,
    prompt: Prompt,
    files_dict: FilesDict,
    memory: BaseMemory,
    preprompts_holder: PrepromptsHolder,
) -> FilesDict:
    """Generates an entrypoint for the codebase."""
    user_prompt = prompt.entrypoint_prompt or """
    Make a unix script that
    a) installs dependencies
    b) runs all necessary parts of the codebase (in parallel if necessary)
    """
    preprompts = preprompts_holder.get_preprompts()
    messages = ai.start(
        system=(preprompts.get("entrypoint", "")),
        user=user_prompt + "\nInformation about the codebase:\n\n" + files_dict.to_chat(),
        step_name=curr_fn(),
    )
    print()
    chat = messages[-1].content.strip()
    regex_pattern = r"```\S*\n(.+?)```"
    matches = re.finditer(regex_pattern, chat, re.DOTALL)
    entrypoint_code = FilesDict(
        {"run.sh": "\n".join(match.group(1) for match in matches)}
    )
    memory.log("gen_entrypoint_chat.txt", "\n\n".join(x.pretty_repr() for x in messages))
    return entrypoint_code


def improve_fn(
    ai: AI,
    prompt: Prompt,
    files_dict: FilesDict,
    memory: BaseMemory,
    preprompts_holder: PrepromptsHolder,
    diff_timeout=3,
) -> FilesDict:
    """Improves the code based on user input."""
    preprompts = preprompts_holder.get_preprompts()
    messages = [
        SystemMessage(content=setup_sys_prompt_existing_code(preprompts)),
    ]

    messages.append(HumanMessage(content=f"{files_dict.to_chat()}"))
    messages.append(HumanMessage(content=prompt.to_langchain_content() if isinstance(prompt.to_langchain_content(), str) else str(prompt.to_langchain_content())))
    memory.log(
        "debug_log_file.txt",
        "UPLOADED FILES:\n" + files_dict.to_log() + "\nPROMPT:\n" + prompt.text,
    )
    return _improve_loop(ai, files_dict, memory, messages, diff_timeout=diff_timeout)


def _improve_loop(
    ai: AI, files_dict: FilesDict, memory: BaseMemory, messages: List, diff_timeout=3
) -> FilesDict:
    messages = ai.next(messages, step_name=curr_fn())
    files_dict, errors = salvage_correct_hunks(
        messages, files_dict, memory, diff_timeout=diff_timeout
    )

    retries = 0
    while errors and retries < MAX_EDIT_REFINEMENT_STEPS:
        messages.append(
            HumanMessage(
                content="Some previously produced diffs were not on the requested format, or the code part was not found in the code. Details:\n"
                + "\n".join(errors)
                + "\n Only rewrite the problematic diffs, making sure that the failing ones are now on the correct format and can be found in the code. Make sure to not repeat past mistakes. \n"
            )
        )
        messages = ai.next(messages, step_name=curr_fn())
        files_dict, errors = salvage_correct_hunks(
            messages, files_dict, memory, diff_timeout
        )
        retries += 1

    return files_dict


def salvage_correct_hunks(
    messages: List, files_dict: FilesDict, memory: BaseMemory, diff_timeout=3
) -> tuple[FilesDict, List[str]]:
    error_messages = []
    ai_response = messages[-1].content.strip()

    diffs = parse_diffs(ai_response, diff_timeout=diff_timeout)

    for _, diff in diffs.items():
        if not diff.is_new_file():
            problems = diff.validate_and_correct(
                file_to_lines_dict(files_dict[diff.filename_pre])
            )
            error_messages.extend(problems)
    files_dict = apply_diffs(diffs, files_dict)
    memory.log("improve.txt", "\n\n".join(x.pretty_repr() for x in messages))
    memory.log("diff_errors.txt", "\n\n".join(error_messages))
    return files_dict, error_messages


# =============================================================================
# UNIFIED CODING AGENT
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for an agent that interacts with code."""

    @abstractmethod
    def init(self, prompt: Prompt) -> FilesDict:
        pass

    @abstractmethod
    def improve(self, files_dict: FilesDict, prompt: Prompt) -> FilesDict:
        pass


class UnifiedCodingAgent(BaseAgent):
    """
    A unified coding agent that combines all functionality into a single class.

    This agent is capable of:
    - Generating new code from prompts
    - Improving existing code
    - Managing memory and execution environments
    - Tracking token usage
    """

    def __init__(
        self,
        memory: BaseMemory,
        execution_env: BaseExecutionEnv,
        ai: AI = None,
        code_gen_fn: Callable = None,
        improve_fn: Callable = None,
        process_code_fn: Callable = None,
        preprompts_holder: PrepromptsHolder = None,
        preprompts_path: Union[str, Path] = None,
    ):
        """
        Initialize the Unified Coding Agent.

        Parameters
        ----------
        memory : BaseMemory
            The memory interface for storing logs and data.
        execution_env : BaseExecutionEnv
            The execution environment for running code.
        ai : AI, optional
            The AI model instance. If not provided, creates a default one.
        code_gen_fn : Callable, optional
            Custom code generation function. Defaults to gen_code.
        improve_fn : Callable, optional
            Custom improvement function. Defaults to improve_fn.
        process_code_fn : Callable, optional
            Custom code processing function. Defaults to None.
        preprompts_holder : PrepromptsHolder, optional
            Holder for preprompt templates.
        preprompts_path : Union[str, Path], optional
            Path to preprompts directory.
        """
        self.memory = memory
        self.execution_env = execution_env
        self.ai = ai or AI()

        if preprompts_holder:
            self.preprompts_holder = preprompts_holder
        elif preprompts_path:
            self.preprompts_holder = PrepromptsHolder(Path(preprompts_path))
        else:
            # Use default preprompts
            default_path = Path(__file__).parent / "preprompts"
            self.preprompts_holder = PrepromptsHolder(default_path)

        self.code_gen_fn = code_gen_fn or gen_code
        self.improve_fn = improve_fn or (lambda ai, prompt, files_dict, memory, holder, **kwargs: improve_fn(ai, prompt, files_dict, memory, holder))
        self.process_code_fn = process_code_fn

    @classmethod
    def with_default_config(
        cls,
        project_path: str = None,
        ai: AI = None,
        preprompts_path: Union[str, Path] = None,
    ):
        """
        Creates a UnifiedCodingAgent with default configuration.

        Parameters
        ----------
        project_path : str, optional
            Path to the project directory. If not provided, uses a temp directory.
        ai : AI, optional
            The AI model instance.
        preprompts_path : Union[str, Path], optional
            Path to preprompts directory.

        Returns
        -------
        UnifiedCodingAgent
            An instance configured with defaults.
        """
        if project_path is None:
            project_path = tempfile.mkdtemp(prefix="gpt-engineer-")

        memory_path = Path(project_path) / ".gpteng" / "memory"
        return cls(
            memory=DiskMemory(memory_path),
            execution_env=DiskExecutionEnv(project_path),
            ai=ai,
            preprompts_path=preprompts_path,
        )

    def init(self, prompt: Prompt) -> FilesDict:
        """
        Generate new code from a prompt.

        Parameters
        ----------
        prompt : Prompt
            The prompt describing what code to generate.

        Returns
        -------
        FilesDict
            The generated code files.
        """
        print("Generating code from prompt...")

        # Generate code
        files_dict = self.code_gen_fn(
            self.ai, prompt, self.memory, self.preprompts_holder
        )

        # Generate entrypoint
        entrypoint = gen_entrypoint(
            self.ai, prompt, files_dict, self.memory, self.preprompts_holder
        )

        # Combine
        combined_dict = {**files_dict, **entrypoint}
        files_dict = FilesDict(combined_dict)

        # Process code if configured
        if self.process_code_fn:
            files_dict = self.process_code_fn(
                self.ai,
                self.execution_env,
                files_dict,
                preprompts_holder=self.preprompts_holder,
                prompt=prompt,
                memory=self.memory,
            )

        print(f"Generated {len(files_dict)} files.")
        return files_dict

    def improve(
        self,
        files_dict: FilesDict,
        prompt: Prompt,
        execution_command: Optional[str] = None,
        diff_timeout=3,
    ) -> FilesDict:
        """
        Improve existing code based on a prompt.

        Parameters
        ----------
        files_dict : FilesDict
            The existing code files.
        prompt : Prompt
            The prompt describing what improvements to make.
        execution_command : str, optional
            Optional command to execute the code.
        diff_timeout : int
            Timeout for parsing diffs.

        Returns
        -------
        FilesDict
            The improved code files.
        """
        print("Improving code...")

        files_dict = self.improve_fn(
            self.ai,
            prompt,
            files_dict,
            self.memory,
            self.preprompts_holder,
            diff_timeout=diff_timeout,
        )

        print(f"Improved {len(files_dict)} files.")
        return files_dict

    def execute_code(self, command: str = "run.sh", timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """
        Execute code in the execution environment.

        Parameters
        ----------
        command : str
            The command to execute.
        timeout : int, optional
            Execution timeout in seconds.

        Returns
        -------
        Tuple[str, str, int]
            stdout, stderr, and return code.
        """
        print(f"Executing: {command}")
        return self.execution_env.run(command, timeout=timeout)

    def get_token_usage(self) -> TokenUsageLog:
        """
        Get the token usage log.

        Returns
        -------
        TokenUsageLog
            The token usage statistics.
        """
        return self.ai.token_usage_log

    def save_to_disk(self, files_dict: FilesDict, path: Union[str, Path] = None) -> None:
        """
        Save files to disk.

        Parameters
        ----------
        files_dict : FilesDict
            The files to save.
        path : Union[str, Path], optional
            Destination path. If not provided, uses the execution environment path.
        """
        if path:
            dest_env = DiskExecutionEnv(path)
            dest_env.upload(files_dict)
        else:
            self.execution_env.upload(files_dict)
        print(f"Saved {len(files_dict)} files.")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_agent(
    project_path: str = None,
    model_name: str = "gpt-4-turbo",
    temperature: float = 0.1,
    azure_endpoint: str = None,
    preprompts_path: str = None,
) -> UnifiedCodingAgent:
    """
    Factory function to create a UnifiedCodingAgent with common configurations.

    Parameters
    ----------
    project_path : str, optional
        Path to the project directory.
    model_name : str
        The name of the AI model to use.
    temperature : float
        The temperature for AI generation.
    azure_endpoint : str, optional
        Azure endpoint for Azure OpenAI.
    preprompts_path : str, optional
        Path to preprompts directory.

    Returns
    -------
    UnifiedCodingAgent
        A configured agent instance.
    """
    ai = AI(
        model_name=model_name,
        temperature=temperature,
        azure_endpoint=azure_endpoint,
    )

    return UnifiedCodingAgent.with_default_config(
        project_path=project_path,
        ai=ai,
        preprompts_path=preprompts_path,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Generate code from a simple prompt
    print("=== Example 1: Generate Code ===")

    agent = create_agent(
        project_path="./my_project",
        model_name="gpt-4-turbo",
    )

    prompt = Prompt(
        text="Create a simple Python web server using Flask that has a hello world endpoint",
    )

    # Generate code
    files = agent.init(prompt)

    # Print generated files
    for filename, content in files.items():
        print(f"Generated: {filename} ({len(content)} chars)")

    # Example 2: Improve existing code
    print("\n=== Example 2: Improve Code ===")

    improve_prompt = Prompt(
        text="Add error handling and logging to the server",
    )

    improved_files = agent.improve(files, improve_prompt)

    # Example 3: Get token usage
    print("\n=== Example 3: Token Usage ===")
    usage_log = agent.get_token_usage()
    print(f"Total tokens used: {usage_log.total_tokens()}")

    # Example 4: Save to disk
    print("\n=== Example 4: Save to Disk ===")
    agent.save_to_disk(improved_files)
