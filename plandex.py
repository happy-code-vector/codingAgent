"""
Plandex Coding Agent - Single File Implementation

A simplified Python version of the Plandex coding agent that handles:
- AI model integration (OpenAI, Anthropic Claude, etc.)
- Streaming response processing
- File operation parsing and execution
- Context management
- Build and apply operations
"""

import os
import re
import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, AsyncGenerator, Tuple
from datetime import datetime
from threading import Thread, Event
from queue import Queue
import hashlib

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ============================================================================
# Configuration and Constants
# ============================================================================

class ModelProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure-openai"
    GOOGLE_VERTEX = "google-vertex"
    AMAZON_BEDROCK = "amazon-bedrock"


class OperationType(str, Enum):
    """Types of file operations."""
    FILE = "file"
    MOVE = "move"
    REMOVE = "remove"
    RESET = "reset"


class PlanStatus(str, Enum):
    """Status of a plan."""
    IDLE = "idle"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    BUILDING = "building"
    MISSING_FILE = "missing_file"
    ERROR = "error"
    FINISHED = "finished"


class TellStage(str, Enum):
    """Stages of tell execution."""
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CodingAgent")

# Constants
MAX_STREAM_RATE = 0.070  # 70ms between stream messages
ACTIVE_PLAN_TIMEOUT = 2 * 60 * 60  # 2 hours
OPENING_TAG_REGEX = re.compile(r'<PlandexBlock\s+lang="(.+?)"\s+path="(.+?)".*?>')


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Operation:
    """Represents a file operation."""
    type: OperationType
    path: str
    destination: str = ""
    content: str = ""
    description: str = ""
    num_tokens: int = 0
    reply_before: str = ""

    def name(self) -> str:
        """Get a descriptive name for the operation."""
        result = f"{self.type} | {self.path}"
        if self.destination:
            result += f" → {self.destination}"
        return result


@dataclass
class Context:
    """Represents a file or context item."""
    id: str
    context_type: str
    name: str
    file_path: str = ""
    url: str = ""
    body: str = ""
    body_size: int = 0
    num_tokens: int = 0
    sha: str = ""
    auto_loaded: bool = False
    force_skip_ignore: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StreamMessage:
    """Message sent to subscribers during streaming."""
    type: str
    reply_chunk: str = ""
    missing_file_path: str = ""
    missing_file_auto_context: bool = False
    operations: List[Operation] = field(default_factory=list)
    error: str = ""
    stream_messages: List["StreamMessage"] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for AI model."""
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: int = 120
    reasoning_budget: int = 0
    include_reasoning: bool = False
    hide_reasoning: bool = False


@dataclass
class PlanSettings:
    """Settings for plan execution."""
    build_mode: str = "auto"  # auto, manual, disabled
    auto_context: bool = True
    confirm_dangerous: bool = True
    skip_ignore: bool = False


@dataclass
class ActiveBuild:
    """Represents an active build operation."""
    reply_id: str
    file_description: str
    file_content: str
    file_content_tokens: int
    current_file_tokens: int
    path: str
    success: bool = False
    error: Optional[Exception] = None
    is_move_op: bool = False
    move_destination: str = ""
    is_remove_op: bool = False
    is_reset_op: bool = False

    def build_finished(self) -> bool:
        return self.success or self.error is not None

    def is_file_operation(self) -> bool:
        return self.is_move_op or self.is_remove_op or self.is_reset_op


# ============================================================================
# Reply Parser
# ============================================================================

class ReplyParser:
    """
    Parses AI responses to extract file operations.

    Handles various formats including:
    - XML-style PlandexBlock tags
    - Markdown code blocks
    - Move/Remove/Reset operation blocks
    """

    def __init__(self):
        self.lines: List[str] = [""]
        self.current_file_lines: List[str] = []
        self.line_index: int = 0
        self.maybe_file_path: str = ""
        self.current_file_path: str = ""
        self.current_description_lines: List[str] = [""]
        self.current_description_line_idx: int = 0
        self.num_tokens: int = 0
        self.operations: List[Operation] = []
        self.current_file_operation: Optional[Operation] = None
        self.pending_operations: List[Operation] = []
        self.pending_paths: Dict[str, bool] = {}
        self.is_in_move_block: bool = False
        self.is_in_remove_block: bool = False
        self.is_in_reset_block: bool = False

    def add_chunk(self, chunk: str, add_to_total: bool = True) -> None:
        """Add a chunk of text to the parser."""
        logger.debug(f"Adding chunk: {repr(chunk)}")

        if add_to_total:
            self.num_tokens += 1

        if self.current_file_path and self.current_file_operation:
            self.current_file_operation.num_tokens += 1

        # Handle newlines
        if chunk == "\n":
            self.lines.append("")
            self.line_index += 1
            if self.current_file_operation is None:
                self.current_description_lines.append("")
                self.current_description_line_idx += 1
            return

        # Split by newlines
        chunk_lines = chunk.split("\n")
        current_line = self.lines[self.line_index]
        current_line += chunk_lines[0]
        self.lines[self.line_index] = current_line

        if self.current_file_operation is None:
            current_desc = self.current_description_lines[self.current_description_line_idx]
            current_desc += chunk_lines[0]
            self.current_description_lines[self.current_description_line_idx] = current_desc

        if len(chunk_lines) > 1:
            self.lines.append(chunk_lines[1])
            self.line_index += 1

            if self.current_file_operation is None:
                self.current_description_lines.append(chunk_lines[1])
                self.current_description_line_idx += 1

            if len(chunk_lines) > 2:
                tail = "\n".join(chunk_lines[2:])
                self.add_chunk(tail, False)

        # Process the previous line
        if self.line_index == 0:
            return

        prev_full_line = self.lines[self.line_index - 1]
        prev_full_line_trimmed = prev_full_line.strip()

        def set_current_file(path: str, no_label: bool) -> None:
            self.current_file_path = path
            self.current_file_operation = Operation(
                type=OperationType.FILE,
                path=path
            )
            self.maybe_file_path = ""
            self.current_file_lines = []

            # Extract description from lines before the file
            skip_num = 4 if not no_label else 2
            if len(self.current_description_lines) > skip_num:
                desc = "\n".join(self.current_description_lines[:-skip_num]).strip()
                if desc:
                    self.current_file_operation.description = desc

            self.current_description_lines = [""]
            self.current_description_line_idx = 0
            logger.debug(f"Confirmed file path: {self.current_file_path}")

        # Handle maybe file path
        if self.maybe_file_path and not (self.is_in_move_block or self.is_in_remove_block or self.is_in_reset_block):
            if prev_full_line_trimmed.startswith("<PlandexBlock"):
                set_current_file(self.maybe_file_path, False)
                return
            elif prev_full_line_trimmed:
                self.maybe_file_path = ""

        # Look for new file paths
        if self.current_file_path == "" and not (self.is_in_move_block or self.is_in_remove_block or self.is_in_reset_block):
            # Check for XML-style tag
            if self._line_has_xml_path(prev_full_line_trimmed):
                path = self._extract_file_path(prev_full_line_trimmed)
                if path:
                    set_current_file(path, True)

            got_path = ""
            if self._line_maybe_has_file_path(prev_full_line_trimmed):
                got_path = self._extract_file_path(prev_full_line_trimmed)
            elif prev_full_line_trimmed == "### Move Files":
                self.is_in_move_block = True
            elif prev_full_line_trimmed == "### Remove Files":
                self.is_in_remove_block = True
            elif prev_full_line_trimmed == "### Reset Changes":
                self.is_in_reset_block = True

            if got_path:
                self.maybe_file_path = got_path
                logger.debug(f"Detected possible file path: {got_path}")

        # Handle current file content
        elif self.current_file_path:
            if prev_full_line_trimmed == "</PlandexBlock>":
                self.operations.append(self.current_file_operation)
                self.current_file_path = ""
                self.current_file_operation = None
                logger.debug(f"Added file operation: {self.current_file_operation}")
            else:
                self.current_file_operation.content += prev_full_line + "\n"
                self.current_file_lines.append(prev_full_line)

        # Handle move/remove/reset blocks
        elif self.is_in_move_block or self.is_in_remove_block or self.is_in_reset_block:
            if prev_full_line_trimmed == "<EndPlandexFileOps/>":
                self.is_in_move_block = False
                self.is_in_remove_block = False
                self.is_in_reset_block = False
                self.operations.extend(self.pending_operations)
                self.pending_operations = []
                self.pending_paths = {}
            elif self.is_in_move_block:
                op = self._extract_move_file(prev_full_line_trimmed)
                if op and op.path not in self.pending_paths:
                    self.pending_operations.append(op)
                    self.pending_paths[op.path] = True
            elif self.is_in_remove_block:
                op = self._extract_remove_or_reset_file(OperationType.REMOVE, prev_full_line_trimmed)
                if op and op.path not in self.pending_paths:
                    self.pending_operations.append(op)
                    self.pending_paths[op.path] = True
            elif self.is_in_reset_block:
                op = self._extract_remove_or_reset_file(OperationType.RESET, prev_full_line_trimmed)
                if op and op.path not in self.pending_paths:
                    self.pending_operations.append(op)
                    self.pending_paths[op.path] = True

    def read(self) -> Dict[str, Any]:
        """Read current parser state."""
        return {
            "maybe_file_path": self.maybe_file_path,
            "current_file_path": self.current_file_path,
            "operations": self.operations,
            "is_in_move_block": self.is_in_move_block,
            "is_in_remove_block": self.is_in_remove_block,
            "is_in_reset_block": self.is_in_reset_block,
            "total_tokens": self.num_tokens
        }

    def finish_and_read(self) -> Dict[str, Any]:
        """Finish parsing and return final state."""
        self.add_chunk("\n", False)
        return self.read()

    def get_reply_before_path(self, path: str) -> str:
        """Get reply content before a specific file path."""
        if not path:
            return "\n".join(self.lines)

        idx = -1
        for i in range(len(self.lines) - 1, -1, -1):
            line = self.lines[i]
            if self._line_maybe_has_file_path(line):
                extracted = self._extract_file_path(line)
                if extracted == path:
                    idx = i
                    break

        if idx == -1:
            return "\n".join(self.lines)
        return "\n".join(self.lines[:idx])

    def get_reply_for_missing_file(self) -> str:
        """Get reply content up to the missing file."""
        path = self.current_file_path
        idx = -1
        for i in range(len(self.lines) - 1, -1, -1):
            line = self.lines[i]
            if self._line_maybe_has_file_path(line):
                extracted = self._extract_file_path(line)
                if extracted == path:
                    idx = i
                    break

        if idx == -1:
            return "\n".join(self.lines)

        idx = idx + 2
        if idx > len(self.lines) - 1:
            return "\n".join(self.lines)
        return "\n".join(self.lines[:idx]) + "\n"

    def _line_has_xml_path(self, line: str) -> bool:
        return line.startswith("<PlandexBlock") and 'path="' in line

    def _line_maybe_has_file_path(self, line: str) -> bool:
        could_be = (line.startswith("-") or
                   line.startswith("-file:") or
                   line.startswith("- file:") or
                   (line.startswith("**") and line.endswith("**")) or
                   (line.startswith("#") and line.endswith(":")))

        if could_be:
            extracted = self._extract_file_path(line)
            ext_split = extracted.split(".")
            has_ext = len(ext_split) > 1 and " " not in ext_split[-1]
            has_sep = os.sep in extracted or "/" in extracted
            has_spaces = " " in extracted
            return not (not has_ext and not has_sep and has_spaces)
        return could_be

    def _extract_file_path(self, line: str) -> str:
        # Handle XML-style PlandexBlock tag
        if line.startswith("<PlandexBlock"):
            match = OPENING_TAG_REGEX.search(line)
            if match:
                return match.group(2)
            return ""

        p = line
        p = p.replace("**", "")
        p = p.replace("`", "")
        p = p.replace("'", "")
        p = p.replace('"', "")
        p = p.lstrip("-")
        p = p.lstrip("#")
        p = p.strip()
        p = p.removeprefix("file:")
        p = p.removeprefix("file path:")
        p = p.removeprefix("filepath:")
        p = p.removeprefix("File path:")
        p = p.removeprefix("File Path:")
        p = p.rstrip(":")
        p = p.strip()

        split_result = p.split(": ")
        if len(split_result) > 1:
            p = split_result[-1]

        split_result = p.split(" (")
        if len(split_result) > 1:
            p = split_result[0]

        return p

    def _extract_move_file(self, line: str) -> Optional[Operation]:
        line = line.strip()
        if not line.startswith("-"):
            return None

        line = line.lstrip("-").strip()

        parts = line.split("→")
        if len(parts) != 2:
            return None

        src = parts[0].strip().strip("`")
        dst = parts[1].strip().strip("`")

        return Operation(
            type=OperationType.MOVE,
            path=src,
            destination=dst
        )

    def _extract_remove_or_reset_file(self, op_type: OperationType, line: str) -> Optional[Operation]:
        line = line.strip()
        if not line.startswith("-"):
            return None

        line = line.lstrip("-").strip()
        path = line.strip("`")

        return Operation(
            type=op_type,
            path=path
        )


# ============================================================================
# AI Model Client
# ============================================================================

class ModelClient(ABC):
    """Abstract base class for AI model clients."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion responses."""
        pass


class OpenAIClient(ModelClient):
    """OpenAI API client."""

    def __init__(self, config: ModelConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI client")

        self.config = config
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                yield {
                    "content": chunk.choices[0].delta.content or "",
                    "finish_reason": chunk.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(ModelClient):
    """Anthropic Claude API client."""

    def __init__(self, config: ModelConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for Anthropic client")

        self.config = config
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = anthropic.AsyncAnthropic(**client_kwargs)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion from Anthropic."""
        # Convert OpenAI format to Anthropic format
        system_message = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            async with self.client.messages.stream(
                model=self.config.model_name,
                system=system_message,
                messages=user_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield {"content": text, "finish_reason": None}

                # Get final message for finish reason
                response = await stream.get_final_message()
                yield {"content": "", "finish_reason": response.stop_reason}

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class ModelClientFactory:
    """Factory for creating model clients."""

    @staticmethod
    def create_client(config: ModelConfig) -> ModelClient:
        """Create a model client based on configuration."""
        if config.provider == ModelProvider.OPENAI:
            return OpenAIClient(config)
        elif config.provider == ModelProvider.ANTHROPIC:
            return AnthropicClient(config)
        else:
            # Default to OpenAI-compatible API
            return OpenAIClient(config)


# ============================================================================
# Chunk Processor
# ============================================================================

class ChunkProcessor:
    """Processes streaming chunks from AI responses."""

    def __init__(self):
        self.chunks_received: int = 0
        self.file_open: bool = False
        self.awaiting_block_opening_tag: bool = False
        self.awaiting_block_closing_tag: bool = False
        self.awaiting_op_closing_tag: bool = False
        self.awaiting_backticks: bool = False
        self.content_buffer: str = ""
        self.reply_operations: List[Operation] = []

    def buffer_or_stream(
        self,
        content: str,
        parser_result: Dict[str, Any],
        current_stage: TellStage
    ) -> Dict[str, Any]:
        """Decide whether to buffer or stream content based on state."""
        # No buffering in planning stage
        if current_stage == TellStage.PLANNING:
            return {
                "should_stream": True,
                "content": content,
                "block_lang": "",
                "should_stop": False
            }

        should_stream = False
        block_lang = ""

        awaiting_any = (self.awaiting_block_opening_tag or
                       self.awaiting_block_closing_tag or
                       self.awaiting_op_closing_tag or
                       self.awaiting_backticks)

        if awaiting_any:
            self.content_buffer += content
            content = self.content_buffer

            if self.awaiting_backticks:
                if "```" in content:
                    self.awaiting_backticks = False
                    content = content.replace("```", "\\`\\`\\`")
                    if not (self.awaiting_block_opening_tag or self.awaiting_block_closing_tag):
                        should_stream = True
                elif not content.endswith("`"):
                    self.awaiting_backticks = False
                    if not (self.awaiting_block_opening_tag or self.awaiting_block_closing_tag):
                        should_stream = True

            if self.awaiting_block_opening_tag:
                if parser_result.get("current_file_path"):
                    matched, replaced = self._replace_code_block_opening_tag(
                        content, lambda lang: ("```" + lang, lang)
                    )
                    if matched:
                        should_stream = True
                        self.awaiting_block_opening_tag = False
                        self.file_open = True
                        content = replaced
                        block_lang = lang

            elif self.awaiting_block_closing_tag:
                if not parser_result.get("current_file_path"):
                    if "</PlandexBlock>" in content:
                        should_stream = True
                        self.awaiting_block_closing_tag = False
                        self.file_open = False
                        content = content.replace("</PlandexBlock>", "```")

            elif self.awaiting_op_closing_tag:
                if "<EndPlandexFileOps/>" in content:
                    self.awaiting_op_closing_tag = False
                    content = content.replace("\n<EndPlandexFileOps/>", "", 1)
                    content = content.replace("<EndPlandexFileOps/>", "", 1)
                    should_stream = True
        else:
            # Not awaiting anything
            if parser_result.get("maybe_file_path") and not parser_result.get("current_file_path"):
                self.awaiting_block_opening_tag = True
            elif parser_result.get("current_file_path"):
                if "</PlandexBlock>" in content:
                    self.awaiting_block_closing_tag = True
                elif parser_result.get("is_in_move_block") or parser_result.get("is_in_remove_block") or parser_result.get("is_in_reset_block"):
                    if "<EndPlandexFileOps/>" in content:
                        self.awaiting_op_closing_tag = True

            if self.file_open and ("```" in content or content.endswith("`")):
                self.awaiting_backticks = True

            if parser_result.get("current_file_path"):
                matched, replaced = self._replace_code_block_opening_tag(
                    content, lambda lang: ("```" + lang, lang)
                )
                if matched:
                    self.awaiting_block_opening_tag = False
                    content = replaced
                    block_lang = lang

            should_stream = not (self.awaiting_block_opening_tag or
                               self.awaiting_block_closing_tag or
                               self.awaiting_op_closing_tag or
                               self.awaiting_backticks)

        if should_stream:
            self.content_buffer = ""
        else:
            self.content_buffer = content

        return {
            "should_stream": should_stream,
            "content": content,
            "block_lang": block_lang,
            "should_stop": False
        }

    def _replace_code_block_opening_tag(
        self,
        content: str,
        replace_fn: Callable[[str], Tuple[str, str]]
    ) -> Tuple[bool, str]:
        """Replace XML opening tag with markdown code block."""
        match = OPENING_TAG_REGEX.search(content)
        if match:
            lang = match.group(1)
            replacement, _ = replace_fn(lang)
            return True, content.replace(match.group(0), replacement, 1)
        elif "<PlandexBlock>" in content:
            replacement, _ = replace_fn("")
            return True, content.replace("<PlandexBlock>", replacement, 1)
        return False, content


# ============================================================================
# Active Plan
# ============================================================================

class ActivePlan:
    """
    Manages an active plan with streaming and state tracking.

    This is the central coordinator for plan execution, handling:
    - Streaming messages to subscribers
    - Managing file operations and builds
    - Context tracking
    - Lifecycle management
    """

    def __init__(
        self,
        plan_id: str,
        branch: str,
        prompt: str,
        build_only: bool = False,
        auto_context: bool = True,
        session_id: str = ""
    ):
        self.id = plan_id
        self.branch = branch
        self.prompt = prompt
        self.build_only = build_only
        self.auto_context = auto_context
        self.session_id = session_id

        # State
        self.contexts: List[Context] = []
        self.contexts_by_path: Dict[str, Context] = {}
        self.operations: List[Operation] = []
        self.built_files: Dict[str, bool] = {}
        self.is_building_by_path: Dict[str, bool] = {}

        # Streaming state
        self.current_reply_content: str = ""
        self.current_streaming_reply_id: str = ""
        self.num_tokens: int = 0
        self.message_num: int = 0
        self.build_queues_by_path: Dict[str, List[ActiveBuild]] = {}

        # Control channels
        self.replies_finished: bool = False
        self.stream_done_ch = Queue()
        self.missing_file_path: str = ""
        self.missing_file_response_ch: Queue()
        self.auto_load_context_ch = Queue()
        self.allow_overwrite_paths: Dict[str, bool] = {}
        self.skipped_paths: Dict[str, bool] = {}
        self.stored_reply_ids: List[str] = []
        self.did_edit_files: bool = False

        # Subscription management
        self.subscriptions: Dict[str, "Subscription"] = {}
        self.stream_ch: Queue = Queue()
        self._last_stream_message_sent: float = 0
        self._stream_message_buffer: List[StreamMessage] = []
        self._running = True

        # Start stream manager thread
        self._stream_thread = Thread(target=self._stream_manager, daemon=True)
        self._stream_thread.start()

    def _stream_manager(self):
        """Background thread that sends messages to all subscribers."""
        try:
            while self._running:
                try:
                    msg = self.stream_ch.get(timeout=0.1)
                    for sub in list(self.subscriptions.values()):
                        sub.enqueue_message(msg)
                except:
                    continue
        except Exception as e:
            logger.error(f"Stream manager error: {e}")

    def stream(self, msg: StreamMessage) -> None:
        """Send a message to all subscribers."""
        # Rate limiting
        now = time.time()
        time_since_last = now - self._last_stream_message_sent

        # Special messages bypass buffering
        skip_buffer = msg.type in (
            "prompt_missing_file",
            "load_context",
            "finished",
            "error"
        )

        if not skip_buffer and time_since_last < MAX_STREAM_RATE:
            self._stream_message_buffer.append(msg)
            return

        # Flush buffer if needed
        if self._stream_message_buffer:
            if len(self._stream_message_buffer) == 1:
                self._send_to_channel(self._stream_message_buffer[0])
            else:
                self._send_to_channel(StreamMessage(
                    type="multi",
                    stream_messages=self._stream_message_buffer
                ))
            self._stream_message_buffer = []

        self._send_to_channel(msg)
        self._last_stream_message_sent = now

        if msg.type == "finished":
            # Signal completion after a short delay
            time.sleep(0.05)
            self.stream_done_ch.put(None)

    def _send_to_channel(self, msg: StreamMessage) -> None:
        """Send message to stream channel."""
        msg_json = json.dumps({
            "type": msg.type,
            "reply_chunk": msg.reply_chunk,
            "missing_file_path": msg.missing_file_path,
            "missing_file_auto_context": msg.missing_file_auto_context,
            "operations": [
                {
                    "type": op.type,
                    "path": op.path,
                    "destination": op.destination,
                    "content": op.content,
                    "description": op.description,
                    "num_tokens": op.num_tokens
                }
                for op in msg.operations
            ],
            "error": msg.error
        })
        self.stream_ch.put(msg_json)

    def flush_stream_buffer(self) -> None:
        """Flush any buffered stream messages."""
        if not self._stream_message_buffer:
            return

        if len(self._stream_message_buffer) == 1:
            self._send_to_channel(self._stream_message_buffer[0])
        else:
            self._send_to_channel(StreamMessage(
                type="multi",
                stream_messages=self._stream_message_buffer
            ))
        self._stream_message_buffer = []

    def subscribe(self) -> Tuple[str, Queue]:
        """Subscribe to stream updates. Returns (subscription_id, message_queue)."""
        import uuid
        sub_id = str(uuid.uuid4())
        sub = Subscription()
        self.subscriptions[sub_id] = sub
        return sub_id, sub.ch

    def unsubscribe(self, sub_id: str) -> None:
        """Unsubscribe from stream updates."""
        if sub_id in self.subscriptions:
            sub = self.subscriptions[sub_id]
            sub.stop()
            del self.subscriptions[sub_id]

    def num_subscribers(self) -> int:
        """Get number of active subscribers."""
        return len(self.subscriptions)

    def build_finished(self) -> bool:
        """Check if all builds are finished."""
        for path in self.build_queues_by_path:
            if self.is_building_by_path.get(path, False):
                return False
            if not self._path_queue_empty(path):
                return False
        return True

    def _path_queue_empty(self, path: str) -> bool:
        """Check if a path's build queue is empty."""
        for build in self.build_queues_by_path.get(path, []):
            if not build.build_finished():
                return False
        return True

    def finish(self) -> None:
        """Mark the plan as finished."""
        self.stream(StreamMessage(type="finished"))

    def stop(self) -> None:
        """Stop the plan and cleanup resources."""
        self._running = False
        for sub in list(self.subscriptions.values()):
            sub.stop()
        self.subscriptions.clear()


class Subscription:
    """A subscription to plan updates."""

    def __init__(self):
        self.ch: Queue = Queue()
        self._running = True
        self._message_queue: List[str] = []
        self._thread = Thread(target=self._process_messages, daemon=True)
        self._thread.start()

    def _process_messages(self):
        """Process messages from queue."""
        while self._running:
            if self._message_queue:
                msg = self._message_queue.pop(0)
                try:
                    self.ch.put(msg, timeout=0.1)
                except:
                    pass
            time.sleep(0.01)

    def enqueue_message(self, msg: str):
        """Add a message to the queue."""
        self._message_queue.append(msg)

    def stop(self):
        """Stop the subscription."""
        self._running = False


# ============================================================================
# Coding Agent - Main Class
# ============================================================================

class CodingAgent:
    """
    Main coding agent that orchestrates AI-assisted code generation and file operations.

    This class brings together all components:
    - Model client for AI interactions
    - Stream processing for real-time responses
    - File operation parsing and execution
    - Build and apply management
    - Context tracking
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        plan_settings: Optional[PlanSettings] = None,
        project_path: str = "."
    ):
        self.model_config = model_config or ModelConfig()
        self.plan_settings = plan_settings or PlanSettings()
        self.project_path = Path(project_path).resolve()

        # Initialize model client
        self.model_client = ModelClientFactory.create_client(self.model_config)

        # Active plans
        self.active_plans: Dict[str, ActivePlan] = {}

        # Context
        self.contexts_by_path: Dict[str, Context] = {}

        logger.info(f"CodingAgent initialized with model: {self.model_config.model_name}")
        logger.info(f"Project path: {self.project_path}")

    async def tell(
        self,
        prompt: str,
        contexts: Optional[List[Context]] = None,
        branch: str = "main",
        plan_id: Optional[str] = None
    ) -> AsyncGenerator[StreamMessage, None]:
        """
        Process a user prompt and generate code changes.

        Yields StreamMessage objects with real-time updates.
        """
        import uuid
        if plan_id is None:
            plan_id = str(uuid.uuid4())

        # Create active plan
        active_plan = ActivePlan(
            plan_id=plan_id,
            branch=branch,
            prompt=prompt,
            auto_context=self.plan_settings.auto_context
        )
        self.active_plans[plan_id] = active_plan

        # Subscribe to updates
        sub_id, msg_queue = active_plan.subscribe()

        try:
            # Build messages for AI
            messages = self._build_messages(prompt, contexts or [])

            # Create reply parser and chunk processor
            reply_parser = ReplyParser()
            chunk_processor = ChunkProcessor()

            # Stream from AI model
            current_stage = TellStage.PLANNING

            async for chunk in self.model_client.stream_chat(messages):
                content = chunk.get("content", "")
                if not content:
                    continue

                # Add chunk to parser
                reply_parser.add_chunk(content, True)
                parser_result = reply_parser.read()

                # Update file open state
                if not chunk_processor.file_open and parser_result.get("current_file_path"):
                    chunk_processor.file_open = True

                # Check for closing tag
                if chunk_processor.file_open and "</PlandexBlock>" in (active_plan.current_reply_content + content):
                    parser_result = reply_parser.finish_and_read()
                    chunk_processor.file_open = False

                # Buffer or stream
                buffer_result = chunk_processor.buffer_or_stream(content, parser_result, current_stage)

                # Stream to subscribers
                if buffer_result["should_stream"]:
                    active_plan.stream(StreamMessage(
                        type="reply",
                        reply_chunk=buffer_result["content"]
                    ))

                # Update state
                active_plan.current_reply_content += content
                active_plan.num_tokens += 1

                # Check for new operations
                operations = parser_result.get("operations", [])
                new_operations = operations[len(chunk_processor.reply_operations):]

                if new_operations:
                    chunk_processor.reply_operations.extend(new_operations)
                    active_plan.operations.extend(new_operations)

                    # Queue builds if auto-build is enabled
                    if self.plan_settings.build_mode == "auto":
                        for op in new_operations:
                            active_plan.build_queues_by_path.setdefault(op.path, []).append(
                                ActiveBuild(
                                    reply_id=plan_id,
                                    file_description=op.description or f"Update {op.path}",
                                    file_content=op.content,
                                    file_content_tokens=op.num_tokens,
                                    current_file_tokens=0,
                                    path=op.path,
                                    is_move_op=op.type == OperationType.MOVE,
                                    move_destination=op.destination,
                                    is_remove_op=op.type == OperationType.REMOVE,
                                    is_reset_op=op.type == OperationType.RESET
                                )
                            )

                    # Stream operations
                    active_plan.stream(StreamMessage(
                        type="operations",
                        operations=new_operations
                    ))

                # Check for finish
                if chunk.get("finish_reason"):
                    active_plan.replies_finished = True
                    break

            # Execute builds
            if self.plan_settings.build_mode == "auto":
                await self._execute_builds(active_plan)

            # Finish
            active_plan.finish()

            # Yield any remaining messages
            while not msg_queue.empty():
                msg_json = msg_queue.get()
                msg_data = json.loads(msg_json)
                yield StreamMessage(**msg_data)

        finally:
            active_plan.unsubscribe(sub_id)

    def _build_messages(
        self,
        prompt: str,
        contexts: List[Context]
    ) -> List[Dict[str, Any]]:
        """Build messages for AI model from prompt and contexts."""
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]

        # Add contexts as user messages
        for ctx in contexts:
            if ctx.body:
                if ctx.file_path:
                    messages.append({
                        "role": "user",
                        "content": f"File: {ctx.file_path}\n\n```\n{ctx.body}\n```"
                    })
                elif ctx.url:
                    messages.append({
                        "role": "user",
                        "content": f"URL: {ctx.url}\n\n{ctx.body}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Context: {ctx.name}\n\n{ctx.body}"
                    })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI model."""
        return """You are an AI coding assistant that helps users write and modify code.

When you need to create or modify files, use this format:

<PlandexBlock lang="python" path="path/to/file.py">
# Your code here
</PlandexBlock>

For moving files:
### Move Files
- path/to/old_file.py → path/to/new_file.py
<EndPlandexFileOps/>

For removing files:
### Remove Files
- path/to/file.py
<EndPlandexFileOps/>

For resetting changes:
### Reset Changes
- path/to/file.py
<EndPlandexFileOps/>

Be concise and focus on the specific changes needed."""

    async def _execute_builds(self, active_plan: ActivePlan) -> None:
        """Execute queued builds for the plan."""
        for path, builds in active_plan.build_queues_by_path.items():
            active_plan.is_building_by_path[path] = True

            for build in builds:
                try:
                    if build.is_file_operation():
                        # Handle move, remove, or reset operations
                        await self._execute_file_operation(active_plan, build)
                    else:
                        # Handle file content update
                        await self._execute_file_build(active_plan, build)

                    build.success = True
                    active_plan.built_files[path] = True

                except Exception as e:
                    build.error = e
                    logger.error(f"Build error for {path}: {e}")

            active_plan.is_building_by_path[path] = False

    async def _execute_file_build(self, active_plan: ActivePlan, build: ActiveBuild) -> None:
        """Execute a file content build."""
        file_path = self.project_path / build.path

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file content
        file_path.write_text(build.file_content)

        active_plan.did_edit_files = True
        logger.info(f"Built file: {build.path}")

    async def _execute_file_operation(self, active_plan: ActivePlan, build: ActiveBuild) -> None:
        """Execute a file operation (move, remove, reset)."""
        file_path = self.project_path / build.path

        if build.is_move_op:
            # Move file
            dest_path = self.project_path / build.move_destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(dest_path)
            active_plan.did_edit_files = True
            logger.info(f"Moved file: {build.path} → {build.move_destination}")

        elif build.is_remove_op:
            # Remove file
            if file_path.exists():
                file_path.unlink()
                active_plan.did_edit_files = True
                logger.info(f"Removed file: {build.path}")

        elif build.is_reset_op:
            # Reset file using git
            import subprocess
            try:
                subprocess.run(
                    ["git", "checkout", "--", str(file_path)],
                    cwd=self.project_path,
                    capture_output=True,
                    check=True
                )
                active_plan.did_edit_files = True
                logger.info(f"Reset file: {build.path}")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Git reset failed: {e.stderr.decode()}")

    def load_context_from_file(self, file_path: str) -> Context:
        """Load a file as context."""
        full_path = self.project_path / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = full_path.read_text()
        sha = hashlib.sha256(content.encode()).hexdigest()

        context = Context(
            id=str(hash((file_path, sha))),
            context_type="file",
            name=file_path,
            file_path=file_path,
            body=content,
            body_size=len(content),
            sha=sha,
            num_tokens=len(content.split())  # Rough estimate
        )

        self.contexts_by_path[file_path] = context
        return context

    def get_context(self, file_path: str) -> Optional[Context]:
        """Get a loaded context by file path."""
        return self.contexts_by_path.get(file_path)

    def get_active_plan(self, plan_id: str) -> Optional[ActivePlan]:
        """Get an active plan by ID."""
        return self.active_plans.get(plan_id)

    def stop_plan(self, plan_id: str) -> None:
        """Stop an active plan."""
        if plan_id in self.active_plans:
            self.active_plans[plan_id].stop()
            del self.active_plans[plan_id]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_agent(
    model_provider: str = "openai",
    model_name: str = "gpt-4",
    api_key: str = "",
    project_path: str = ".",
    **kwargs
) -> CodingAgent:
    """
    Convenience function to create a CodingAgent.

    Args:
        model_provider: Provider name (openai, anthropic, etc.)
        model_name: Model name
        api_key: API key for the provider
        project_path: Path to the project directory
        **kwargs: Additional configuration

    Returns:
        Configured CodingAgent instance
    """
    # Get API key from environment if not provided
    if not api_key:
        if model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif model_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")

    config = ModelConfig(
        provider=ModelProvider(model_provider),
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )

    return CodingAgent(model_config=config, project_path=project_path)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        agent = create_agent(
            model_provider="openai",
            model_name="gpt-4",
            project_path="."
        )

        # Load some context
        try:
            context = agent.load_context_from_file("README.md")
            print(f"Loaded context: {context.name}")
        except FileNotFoundError:
            print("No README.md found, proceeding without context")

        # Process a prompt
        prompt = "Create a simple Python function to calculate fibonacci numbers"

        print(f"Processing: {prompt}\n")

        async for msg in agent.tell(prompt):
            if msg.type == "reply" and msg.reply_chunk:
                print(msg.reply_chunk, end="", flush=True)
            elif msg.type == "operations":
                for op in msg.operations:
                    print(f"\n\n[Operation: {op.name()}]")
            elif msg.type == "finished":
                print("\n\n[Done]")

    asyncio.run(main())
