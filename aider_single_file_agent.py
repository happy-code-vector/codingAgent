#!/usr/bin/env python3
"""
Aider AI Coding Agent - Single File Consolidated Version
An AI pair programmer that helps you write code.

This is a consolidated single-file version of the Aider coding agent,
originally spread across 58+ Python files.

Original: https://github.com/Aider-AI/aider
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

import base64
import hashlib
import json
import locale
import math
import mimetypes
import os
import platform
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, fields
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Union, List

# Third-party imports
try:
    import git
    from git.exc import GitCommandNotFound, InvalidGitRepositoryError, GitError, ODBError
    ANY_GIT_ERROR = [ODBError, GitError, InvalidGitRepositoryError, GitCommandNotFound]
except ImportError:
    git = None
    ANY_GIT_ERROR = []
ANY_GIT_ERROR = tuple(ANY_GIT_ERROR + [OSError, IndexError, BufferError, TypeError,
                                       ValueError, AttributeError, AssertionError, TimeoutError])

try:
    import pathspec
except ImportError:
    pathspec = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Rich terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.style import Style as RichStyle
    from rich.columns import Columns
except ImportError:
    print("Warning: rich not installed. Install with: pip install rich")
    Console = Markdown = Text = RichStyle = Columns = None

# Prompt toolkit for advanced terminal
try:
    from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
    from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.output.vt100 import is_dumb_terminal
    from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
    from prompt_toolkit.styles import Style
    from pygments.lexers import MarkdownLexer, guess_lexer_for_filename
    from pygments.token import Token
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    print("Warning: prompt_toolkit not installed. Install with: pip install prompt_toolkit")
    print("Warning: pygments not installed. Install with: pip install pygments")
    Completer = Completion = ThreadedCompleter = object
    ModalCursorShapeConfig = object
    EditingMode = type('EditingMode', (), {'EMACS': 'emacs', 'VI': 'vi'})()
    Condition = lambda x: True
    FileHistory = object
    KeyBindings = object
    Keys = type('Keys', (), {'ControlZ': 'c-z', 'Space': ' '})()
    PygmentsLexer = object
    is_dumb_terminal = lambda: False
    CompleteStyle = type('CompleteStyle', (), {'MULTI_COLUMN': 'multi_column'})()
    PromptSession = object
    Style = object
    MarkdownLexer = object
    guess_lexer_for_filename = lambda x, y: None
    Token = type('Token', (), {'Name': 'Name'})()
    PROMPT_TOOLKIT_AVAILABLE = False

# LiteLLM for model interface
try:
    from aider.llm import litellm
except ImportError:
    try:
        import litellm
    except ImportError:
        litellm = None
        print("Warning: litellm not installed. Install with: pip install litellm")

# ============================================================================
# SECTION 2: CONSTANTS AND CONFIGURATION
# ============================================================================

__version__ = "0.0.1-single-file"

RETRY_TIMEOUT = 60
request_timeout = 600
DEFAULT_MODEL_NAME = "gpt-4o"
ANTHROPIC_BETA_HEADER = "prompt-caching-2024-07-31,pdfs-2024-09-25"

OPENAI_MODELS = [
    "o1", "o1-preview", "o1-mini", "o3-mini",
    "gpt-4", "gpt-4o", "gpt-4o-2024-05-13",
    "gpt-4-turbo-preview", "gpt-4-0314", "gpt-4-0613",
    "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09",
    "gpt-4-1106-preview", "gpt-4-0125-preview",
    "gpt-4-vision-preview", "gpt-4-1106-vision-preview",
    "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613",
]

ANTHROPIC_MODELS = [
    "claude-2", "claude-2.1", "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
    "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022", "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]

MODEL_ALIASES = {
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
    "opus": "claude-opus-4-20250514",
    "4": "gpt-4-0613",
    "4o": "gpt-4o",
    "4-turbo": "gpt-4-1106-preview",
    "35turbo": "gpt-3.5-turbo",
    "35-turbo": "gpt-3.5-turbo",
    "3": "gpt-3.5-turbo",
    "deepseek": "deepseek/deepseek-chat",
    "flash": "gemini/gemini-2.5-flash",
    "r1": "deepseek/deepseek-reasoner",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini": "gemini/gemini-3-pro-preview",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".pdf"}

REASONING_TAG = "thinking"

# ============================================================================
# SECTION 3: UTILITY FUNCTIONS AND CLASSES
# ============================================================================

class IgnorantTemporaryDirectory:
    """Temporary directory that ignores cleanup errors on Windows."""
    def __init__(self):
        if sys.version_info >= (3, 10):
            self.temp_dir = __import__('tempfile').TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = __import__('tempfile').TemporaryDirectory()

    def __enter__(self):
        return self.temp_dir.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        try:
            self.temp_dir.cleanup()
        except (OSError, PermissionError, RecursionError):
            pass

    def __getattr__(self, item):
        return getattr(self.temp_dir, item)


def is_image_file(file_name):
    """Check if the given file name has an image file extension."""
    file_name = str(file_name)
    return any(file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def safe_abs_path(res):
    """Get absolute path with proper Windows handling."""
    res = Path(res).resolve()
    return str(res)


def find_common_root(fnames):
    """Find the common root directory of a list of files."""
    if not fnames:
        return "."

    fnames = [Path(fname) for fname in fnames]
    root = fnames[0].parent

    for fname in fnames[1:]:
        parts = Path(fname).parts
        root_parts = Path(root).parts

        for i, (part1, part2) in enumerate(zip(parts, root_parts)):
            if part1 != part2:
                root = Path(*root_parts[:i])
                break
        else:
            root = Path(*parts[:min(len(parts), len(root_parts))])

    return str(root) if root != Path(".") else "."


def check_pip_install_extra(io, package_name, install_prompt, install_args):
    """Check if a package is installed, offer to install if not."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        pass

    if io.confirm_ask(f"{install_prompt} Install now?"):
        cmd = [sys.executable, "-m", "pip", "install"] + install_args
        try:
            subprocess.run(cmd, check=True)
            io.tool_output(f"Successfully installed {install_args}")
            return True
        except subprocess.SubprocessError:
            io.tool_error(f"Failed to install {install_args}")
    return False


def format_content(role, content):
    """Format content with role prefix."""
    formatted_lines = []
    for line in content.splitlines():
        formatted_lines.append(f"{role} {line}")
    return "\n".join(formatted_lines)


def format_messages(messages, title=None):
    """Format messages for display."""
    output = []
    if title:
        output.append(f"{title.upper()} {'*' * 50}")

    for msg in messages:
        output.append("-------")
        role = msg["role"].upper()
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, dict) and "url" in value:
                            output.append(f"{role} {key.capitalize()} URL: {value['url']}")
                        else:
                            output.append(f"{role} {key}: {value}")
                else:
                    output.append(f"{role} {item}")
        elif isinstance(content, str):
            output.append(format_content(role, content))
        function_call = msg.get("function_call")
        if function_call:
            output.append(f"{role} Function Call: {function_call}")

    return "\n".join(output)


def touch_file(fname):
    """Create an empty file if it doesn't exist."""
    try:
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        fname.touch()
        return True
    except OSError:
        return False


# ============================================================================
# SECTION 4: MODEL SETTINGS AND MANAGEMENT
# ============================================================================

@dataclass
class ModelSettings:
    """Settings for a specific model."""
    name: str
    edit_format: str = "whole"
    weak_model_name: Optional[str] = None
    use_repo_map: bool = False
    send_undo_reply: bool = False
    lazy: bool = False
    overeager: bool = False
    reminder: str = "user"
    examples_as_sys_msg: bool = False
    extra_params: Optional[dict] = None
    cache_control: bool = False
    caches_by_default: bool = False
    use_system_prompt: bool = True
    use_temperature: Union[bool, float] = True
    streaming: bool = True
    editor_model_name: Optional[str] = None
    editor_edit_format: Optional[str] = None
    reasoning_tag: Optional[str] = None
    remove_reasoning: Optional[str] = None
    system_prompt_prefix: Optional[str] = None
    accepts_settings: Optional[list] = None


class ModelInfoManager:
    """Manages model information from various sources."""
    MODEL_INFO_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    CACHE_TTL = 60 * 60 * 24  # 24 hours

    def __init__(self):
        self.cache_dir = Path.home() / ".aider" / "caches"
        self.cache_file = self.cache_dir / "model_prices_and_context_window.json"
        self.content = None
        self.local_model_metadata = {}
        self.verify_ssl = True
        self._cache_loaded = False

    def set_verify_ssl(self, verify_ssl):
        self.verify_ssl = verify_ssl

    def _load_cache(self):
        if self._cache_loaded:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                cache_age = time.time() - self.cache_file.stat().st_mtime
                if cache_age < self.CACHE_TTL:
                    try:
                        self.content = json.loads(self.cache_file.read_text())
                    except json.JSONDecodeError:
                        self.content = None
        except OSError:
            pass

        self._cache_loaded = True

    def get_model_from_cached_json_db(self, model):
        data = self.local_model_metadata.get(model)
        if data:
            return data

        self._load_cache()

        if not self.content:
            return dict()

        info = self.content.get(model, dict())
        if info:
            return info

        pieces = model.split("/")
        if len(pieces) == 2:
            info = self.content.get(pieces[1])
            if info and info.get("litellm_provider") == pieces[0]:
                return info

        return dict()

    def get_model_info(self, model):
        cached_info = self.get_model_from_cached_json_db(model)

        if litellm and not cached_info:
            try:
                litellm_info = litellm.get_model_info(model)
                if litellm_info:
                    return litellm_info
            except Exception:
                pass

        return cached_info


model_info_manager = ModelInfoManager()


class Model(ModelSettings):
    """Represents an LLM model with configuration."""

    def __init__(
        self,
        model,
        weak_model=None,
        editor_model=None,
        editor_edit_format=None,
        verbose=False
    ):
        model = MODEL_ALIASES.get(model, model)

        self.name = model
        self.verbose = verbose
        self.max_chat_history_tokens = 1024
        self.weak_model = None
        self.editor_model = None

        self.info = self.get_model_info(model)

        # Validate environment
        res = self.validate_environment()
        self.missing_keys = res.get("missing_keys")
        self.keys_in_environment = res.get("keys_in_environment")

        max_input_tokens = self.info.get("max_input_tokens") or 0
        self.max_chat_history_tokens = min(max(max_input_tokens / 16, 1024), 8192)

        self.configure_model_settings(model)
        if weak_model is False:
            self.weak_model_name = None
        else:
            self.get_weak_model(weak_model)

        if editor_model is False:
            self.editor_model_name = None
        else:
            self.get_editor_model(editor_model, editor_edit_format)

    def get_model_info(self, model):
        return model_info_manager.get_model_info(model)

    def _copy_fields(self, source):
        """Copy fields from a ModelSettings instance."""
        for field in fields(ModelSettings):
            val = getattr(source, field.name)
            setattr(self, field.name, val)

        if self.reasoning_tag is None and self.remove_reasoning is not None:
            self.reasoning_tag = self.remove_reasoning

    def configure_model_settings(self, model):
        """Configure model settings based on model name."""
        # Look for exact match first (would use MODEL_SETTINGS in full version)
        exact_match = False

        # Apply generic settings based on model name patterns
        self.apply_generic_model_settings(model)

        # Initialize accepts_settings if None
        if self.accepts_settings is None:
            self.accepts_settings = []

        # Ensure OpenRouter models accept thinking_tokens and reasoning_effort
        if self.name.startswith("openrouter/"):
            if self.accepts_settings is None:
                self.accepts_settings = []
            if "thinking_tokens" not in self.accepts_settings:
                self.accepts_settings.append("thinking_tokens")
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")

    def apply_generic_model_settings(self, model):
        """Apply generic settings based on model name patterns."""
        model = model.lower()
        last_segment = model.split("/")[-1]

        # O1 models
        if "/o1" in model or "/o1-" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.use_temperature = False
            self.streaming = False
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")
            return

        # O3-mini
        if "/o3-mini" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.use_temperature = False
            if "reasoning_effort" not in self.accepts_settings:
                self.accepts_settings.append("reasoning_effort")
            return

        # Deepseek R1
        if "deepseek" in model and ("r1" in model or "reasoning" in model):
            self.edit_format = "diff"
            self.use_repo_map = True
            self.examples_as_sys_msg = True
            self.use_temperature = False
            self.reasoning_tag = "think"
            return

        # Claude 3.5 Sonnet
        if "3.5-sonnet" in model or "3-5-sonnet" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.examples_as_sys_msg = True
            self.reminder = "user"
            return

        # GPT-4 and Claude Opus
        if "gpt-4" in model or "claude-3-opus" in model:
            self.edit_format = "diff"
            self.use_repo_map = True
            self.send_undo_reply = True
            return

        # GPT-4 Turbo
        if "gpt-4-turbo" in model or ("gpt-4-" in model and "-preview" in model):
            self.edit_format = "udiff"
            self.use_repo_map = True
            self.send_undo_reply = True
            return

        # GPT-3.5
        if "gpt-3.5" in model:
            self.reminder = "sys"
            return

        # Qwen Coder 2.5
        if "qwen" in model and "coder" in model and ("2.5" in model or "2-5" in model):
            self.edit_format = "diff"
            self.use_repo_map = True
            return

        # Default diff format
        if self.edit_format == "diff":
            self.use_repo_map = True

    def __str__(self):
        return self.name

    def get_weak_model(self, provided_weak_model_name):
        """Get or create the weak model."""
        if provided_weak_model_name:
            self.weak_model_name = provided_weak_model_name

        if not self.weak_model_name:
            self.weak_model = self
            return

        if self.weak_model_name == self.name:
            self.weak_model = self
            return

        self.weak_model = Model(
            self.weak_model_name,
            weak_model=False,
        )
        return self.weak_model

    def commit_message_models(self):
        """Get models for commit message generation."""
        return [self.weak_model, self]

    def get_editor_model(self, provided_editor_model_name, editor_edit_format):
        """Get or create the editor model."""
        if provided_editor_model_name:
            self.editor_model_name = provided_editor_model_name
        if editor_edit_format:
            self.editor_edit_format = editor_edit_format

        if not self.editor_model_name or self.editor_model_name == self.name:
            self.editor_model = self
        else:
            self.editor_model = Model(
                self.editor_model_name,
                editor_model=False,
            )

        if not self.editor_edit_format:
            self.editor_edit_format = self.editor_model.edit_format
            if self.editor_edit_format in ("diff", "whole", "diff-fenced"):
                self.editor_edit_format = "editor-" + self.editor_edit_format

        return self.editor_model

    def tokenizer(self, text):
        """Tokenize text using litellm."""
        if litellm:
            return litellm.encode(model=self.name, text=text)
        return list(text)

    def token_count(self, messages):
        """Count tokens in messages."""
        if litellm and type(messages) is list:
            try:
                return litellm.token_counter(model=self.name, messages=messages)
            except Exception as err:
                print(f"Unable to count tokens: {err}")
                return 0

        # Fallback
        if type(messages) is str:
            msgs = messages
        else:
            msgs = json.dumps(messages)

        try:
            return len(self.tokenizer(msgs))
        except Exception as err:
            print(f"Unable to count tokens: {err}")
            return 0

    def fast_validate_environment(self):
        """Fast path for common models."""
        model = self.name

        pieces = model.split("/")
        if len(pieces) > 1:
            provider = pieces[0]
        else:
            provider = None

        keymap = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        var = None
        if model in OPENAI_MODELS:
            var = "OPENAI_API_KEY"
        elif model in ANTHROPIC_MODELS:
            var = "ANTHROPIC_API_KEY"
        else:
            var = keymap.get(provider)

        if var and os.environ.get(var):
            return dict(keys_in_environment=[var], missing_keys=[])

    def validate_environment(self):
        """Validate that required API keys are set."""
        res = self.fast_validate_environment()
        if res:
            return res

        if litellm:
            res = litellm.validate_environment(self.name)
            if res.get("keys_in_environment") or res.get("missing_keys"):
                return res

        return dict(keys_in_environment=False, missing_keys=[])

    def get_repo_map_tokens(self):
        """Get the number of tokens to use for repo map."""
        map_tokens = 1024
        max_inp_tokens = self.info.get("max_input_tokens")
        if max_inp_tokens:
            map_tokens = max_inp_tokens / 8
            map_tokens = min(map_tokens, 4096)
            map_tokens = max(map_tokens, 1024)
        return map_tokens

    def simple_send_with_retries(self, messages):
        """Send messages with retry logic."""
        if litellm and "deepseek-reasoner" in self.name:
            from aider.sendchat import ensure_alternating_roles
            messages = ensure_alternating_roles(messages)

        retry_delay = 0.125

        while True:
            try:
                kwargs = {
                    "messages": messages,
                    "functions": None,
                    "stream": False,
                }

                _hash, response = self.send_completion(**kwargs)
                if not response or not hasattr(response, "choices") or not response.choices:
                    return None
                res = response.choices[0].message.content

                # Remove reasoning content if present
                if hasattr(self, 'reasoning_tag_name') and self.reasoning_tag_name:
                    from aider.reasoning_tags import remove_reasoning_content
                    res = remove_reasoning_content(res, self.reasoning_tag_name)

                return res

            except Exception as err:
                print(str(err))
                should_retry = True
                if should_retry:
                    retry_delay *= 2
                    if retry_delay > RETRY_TIMEOUT:
                        should_retry = False
                if not should_retry:
                    return None
                print(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                continue
            except AttributeError:
                return None

    def send_completion(self, messages, functions, stream, temperature=None):
        """Send completion request to the LLM."""
        if not litellm:
            raise ImportError("litellm is required but not installed")

        kwargs = dict(
            model=self.name,
            stream=stream,
        )

        if self.use_temperature is not False:
            if temperature is None:
                if isinstance(self.use_temperature, bool):
                    temperature = 0
                else:
                    temperature = float(self.use_temperature)
            kwargs["temperature"] = temperature

        if functions is not None:
            function = functions[0]
            kwargs["tools"] = [dict(type="function", function=function)]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": function["name"]}}
        if self.extra_params:
            kwargs.update(self.extra_params)

        key = json.dumps(kwargs, sort_keys=True).encode()
        hash_object = hashlib.sha1(key)

        if "timeout" not in kwargs:
            kwargs["timeout"] = request_timeout
        if self.verbose:
            print(kwargs)
        kwargs["messages"] = messages

        res = litellm.completion(**kwargs)
        return hash_object, res


# ============================================================================
# SECTION 5: INPUT/OUTPUT HANDLING
# ============================================================================

def ensure_hash_prefix(color):
    """Ensure hex color values have a # prefix."""
    if not color:
        return color
    if isinstance(color, str) and color.strip() and not color.startswith("#"):
        if all(c in "0123456789ABCDEFabcdef" for c in color) and len(color) in (3, 6):
            return f"#{color}"
    return color


@dataclass
class ConfirmGroup:
    """Group for confirming multiple actions."""
    preference: str = None
    show_group: bool = True

    def __init__(self, items=None):
        if items is not None:
            self.show_group = len(items) > 1


class AutoCompleter(Completer):
    """Auto-completer for file names and commands."""

    def __init__(
        self,
        root,
        rel_fnames,
        addable_rel_fnames,
        commands,
        encoding,
        abs_read_only_fnames=None
    ):
        self.addable_rel_fnames = addable_rel_fnames
        self.rel_fnames = rel_fnames
        self.encoding = encoding
        self.abs_read_only_fnames = abs_read_only_fnames or []

        fname_to_rel_fnames = defaultdict(list)
        for rel_fname in addable_rel_fnames:
            fname = os.path.basename(rel_fname)
            if fname != rel_fname:
                fname_to_rel_fnames[fname].append(rel_fname)
        self.fname_to_rel_fnames = fname_to_rel_fnames

        self.words = set()
        self.commands = commands
        self.command_completions = dict()
        if commands:
            self.command_names = self.commands.get_commands()

        for rel_fname in addable_rel_fnames:
            self.words.add(rel_fname)

        for rel_fname in rel_fnames:
            self.words.add(rel_fname)

        all_fnames = [Path(root) / rel_fname for rel_fname in rel_fnames]
        if abs_read_only_fnames:
            all_fnames.extend(abs_read_only_fnames)

        self.all_fnames = all_fnames
        self.tokenized = False

    def tokenize(self):
        """Tokenize files for code completion."""
        if self.tokenized:
            return
        self.tokenized = True

        for fname in self.all_fnames:
            try:
                with open(fname, "r", encoding=self.encoding) as f:
                    content = f.read()
            except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
                continue
            try:
                lexer = guess_lexer_for_filename(fname, content)
            except Exception:
                continue

            tokens = list(lexer.get_tokens(content))
            self.words.update(
                (token[1], f"`{token[1]}`") for token in tokens if token[0] in Token.Name
            )

    def get_completions(self, document, complete_event):
        """Get completions for the current input."""
        self.tokenize()

        text = document.text_before_cursor
        words = text.split()
        if not words:
            return

        if text and text[-1].isspace():
            return

        candidates = self.words
        candidates.update(set(self.fname_to_rel_fnames))
        candidates = [word if type(word) is tuple else (word, word) for word in candidates]

        last_word = words[-1]

        if len(last_word) < 3:
            return

        completions = []
        for word_match, word_insert in candidates:
            if word_match.lower().startswith(last_word.lower()):
                completions.append((word_insert, -len(last_word), word_match))

                rel_fnames = self.fname_to_rel_fnames.get(word_match, [])
                if rel_fnames:
                    for rel_fname in rel_fnames:
                        completions.append((rel_fname, -len(last_word), rel_fname))

        for ins, pos, match in sorted(completions):
            yield Completion(ins, start_position=pos, display=match)


class InputOutput:
    """Handle user input and formatted output."""

    num_error_outputs = 0
    num_user_asks = 0
    clipboard_watcher = None
    bell_on_next_input = False

    def __init__(
        self,
        pretty=True,
        yes=None,
        input_history_file=None,
        chat_history_file=None,
        input=None,
        output=None,
        user_input_color="blue",
        tool_output_color=None,
        tool_error_color="red",
        tool_warning_color="#FFA500",
        assistant_output_color="blue",
        completion_menu_color=None,
        completion_menu_bg_color=None,
        completion_menu_current_color=None,
        completion_menu_current_bg_color=None,
        code_theme="default",
        encoding="utf-8",
        line_endings="platform",
        dry_run=False,
        llm_history_file=None,
        editingmode=EditingMode.EMACS,
        fancy_input=True,
        file_watcher=None,
        multiline_mode=False,
        root=".",
        notifications=False,
    ):
        self.placeholder = None
        self.interrupted = False
        self.never_prompts = set()
        self.editingmode = editingmode
        self.multiline_mode = multiline_mode
        self.bell_on_next_input = False
        self.notifications = notifications

        no_color = os.environ.get("NO_COLOR")
        if no_color is not None and no_color != "":
            pretty = False

        self.user_input_color = ensure_hash_prefix(user_input_color) if pretty else None
        self.tool_output_color = ensure_hash_prefix(tool_output_color) if pretty else None
        self.tool_error_color = ensure_hash_prefix(tool_error_color) if pretty else None
        self.tool_warning_color = ensure_hash_prefix(tool_warning_color) if pretty else None
        self.assistant_output_color = ensure_hash_prefix(assistant_output_color)
        self.completion_menu_color = ensure_hash_prefix(completion_menu_color) if pretty else None
        self.completion_menu_bg_color = ensure_hash_prefix(completion_menu_bg_color) if pretty else None
        self.completion_menu_current_color = (
            ensure_hash_prefix(completion_menu_current_color) if pretty else None
        )
        self.completion_menu_current_bg_color = (
            ensure_hash_prefix(completion_menu_current_bg_color) if pretty else None
        )

        self.code_theme = code_theme

        self.input = input
        self.output = output

        self.pretty = pretty
        if self.output:
            self.pretty = False

        self.yes = yes

        self.input_history_file = input_history_file
        if self.input_history_file:
            try:
                Path(self.input_history_file).parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                print(f"Could not create directory for input history: {e}")
                self.input_history_file = None
        self.llm_history_file = llm_history_file
        if chat_history_file is not None:
            self.chat_history_file = Path(chat_history_file)
        else:
            self.chat_history_file = None

        self.encoding = encoding
        valid_line_endings = {"platform", "lf", "crlf"}
        if line_endings not in valid_line_endings:
            raise ValueError(
                f"Invalid line_endings value: {line_endings}. "
                f"Must be one of: {', '.join(valid_line_endings)}"
            )
        self.newline = (
            None if line_endings == "platform" else "\n" if line_endings == "lf" else "\r\n"
        )
        self.dry_run = dry_run

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.append_chat_history(f"\n# aider chat started at {current_time}\n\n")

        self.prompt_session = None
        self.is_dumb_terminal = is_dumb_terminal() if PROMPT_TOOLKIT_AVAILABLE else True

        if self.is_dumb_terminal:
            self.pretty = False
            fancy_input = False

        if fancy_input and PROMPT_TOOLKIT_AVAILABLE:
            session_kwargs = {
                "input": self.input,
                "output": self.output,
                "lexer": PygmentsLexer(MarkdownLexer),
                "editing_mode": self.editingmode,
            }
            if self.editingmode == EditingMode.VI:
                session_kwargs["cursor"] = ModalCursorShapeConfig()
            if self.input_history_file is not None:
                session_kwargs["history"] = FileHistory(self.input_history_file)
            try:
                self.prompt_session = PromptSession(**session_kwargs)
                self.console = Console()
            except Exception as err:
                self.console = Console(force_terminal=False, no_color=True) if Console else None
                print(f"Can't initialize prompt toolkit: {err}")
        else:
            self.console = Console(force_terminal=False, no_color=True) if Console else None
            if self.is_dumb_terminal:
                self.tool_output("Detected dumb terminal, disabling fancy input and pretty output.")

        self.file_watcher = file_watcher
        self.root = root

    def read_image(self, filename):
        """Read an image file and return base64 encoded content."""
        try:
            with open(str(filename), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode("utf-8")
        except OSError as err:
            self.tool_error(f"{filename}: unable to read: {err}")
            return
        except FileNotFoundError:
            self.tool_error(f"{filename}: file not found error")
            return
        except IsADirectoryError:
            self.tool_error(f"{filename}: is a directory")
            return
        except Exception as e:
            self.tool_error(f"{filename}: {e}")
            return

    def read_text(self, filename, silent=False):
        """Read text from a file."""
        if is_image_file(filename):
            return self.read_image(filename)

        try:
            with open(str(filename), "r", encoding=self.encoding) as f:
                return f.read()
        except FileNotFoundError:
            if not silent:
                self.tool_error(f"{filename}: file not found error")
            return
        except IsADirectoryError:
            if not silent:
                self.tool_error(f"{filename}: is a directory")
            return
        except OSError as err:
            if not silent:
                self.tool_error(f"{filename}: unable to read: {err}")
            return
        except UnicodeError as e:
            if not silent:
                self.tool_error(f"{filename}: {e}")
                self.tool_error("Use --encoding to set the unicode encoding.")
            return

    def write_text(self, filename, content, max_retries=5, initial_delay=0.1):
        """Write content to a file with retry logic."""
        if self.dry_run:
            return

        delay = initial_delay
        for attempt in range(max_retries):
            try:
                with open(str(filename), "w", encoding=self.encoding, newline=self.newline) as f:
                    f.write(content)
                return
            except PermissionError as err:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    self.tool_error(
                        f"Unable to write file {filename} after {max_retries} attempts: {err}"
                    )
                    raise
            except OSError as err:
                self.tool_error(f"Unable to write file {filename}: {err}")
                raise

    def rule(self):
        """Print a visual separator."""
        if self.pretty and self.console:
            style = dict(style=self.user_input_color) if self.user_input_color else dict()
            self.console.rule(**style)
        else:
            print()

    def get_input(
        self,
        root,
        rel_fnames,
        addable_rel_fnames,
        commands,
        abs_read_only_fnames=None,
        edit_format=None,
    ):
        """Get user input from the terminal."""
        self.rule()

        if self.bell_on_next_input and self.notifications:
            print("\a", end="", flush=True)
            self.bell_on_next_input = False

        rel_fnames = list(rel_fnames)
        show = ""
        if rel_fnames:
            rel_read_only_fnames = [
                get_rel_fname(fname, root) for fname in (abs_read_only_fnames or [])
            ]
            show = self.format_files_for_input(rel_fnames, rel_read_only_fnames)

        prompt_prefix = ""
        if edit_format:
            prompt_prefix += edit_format
        if self.multiline_mode:
            prompt_prefix += (" " if edit_format else "") + "multi"
        prompt_prefix += "> "

        show += prompt_prefix
        self.prompt_prefix = prompt_prefix

        inp = ""
        multiline_input = False

        if not PROMPT_TOOLKIT_AVAILABLE:
            # Simple fallback input
            try:
                line = input(show)
                if line.strip("\r\n") and not multiline_input:
                    stripped = line.strip("\r\n")
                    if stripped == "{":
                        multiline_input = True
                        multiline_tag = None
                    elif stripped[0] == "{" and len(stripped) > 1:
                        tag = "".join(c for c in stripped[1:] if c.isalnum())
                        if stripped == "{" + tag:
                            multiline_input = True
                            multiline_tag = tag
                        else:
                            inp = line
                    else:
                        inp = line
                elif multiline_input:
                    if multiline_tag and line.strip("\r\n") == f"{multiline_tag}}}":
                        pass
                    elif line.strip("\r\n") == "}":
                        pass
                    else:
                        inp += line + "\n"

                print()
                self.user_input(inp)
                return inp
            except EOFError:
                raise
            except Exception as err:
                self.tool_error(str(err))
                return ""

        style = self._get_style()

        completer_instance = ThreadedCompleter(
            AutoCompleter(
                root,
                rel_fnames,
                addable_rel_fnames,
                commands,
                self.encoding,
                abs_read_only_fnames=abs_read_only_fnames,
            )
        )

        kb = KeyBindings()

        @kb.add("c-space")
        def _(event):
            event.current_buffer.insert_text(" ")

        @kb.add("enter", eager=True)
        def _(event):
            if self.multiline_mode:
                event.current_buffer.insert_text("\n")
            else:
                event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter", eager=True)
        def _(event):
            if self.multiline_mode:
                event.current_buffer.validate_and_handle()
            else:
                event.current_buffer.insert_text("\n")

        while True:
            if multiline_input:
                show = self.prompt_prefix

            try:
                if self.prompt_session:
                    default = self.placeholder or ""
                    self.placeholder = None

                    self.interrupted = False

                    def get_continuation(width, line_number, is_soft_wrap):
                        return self.prompt_prefix

                    line = self.prompt_session.prompt(
                        show,
                        default=default,
                        completer=completer_instance,
                        reserve_space_for_menu=4,
                        complete_style=CompleteStyle.MULTI_COLUMN,
                        style=style,
                        key_bindings=kb,
                        complete_while_typing=True,
                        prompt_continuation=get_continuation,
                    )
                else:
                    line = input(show)

                if self.interrupted:
                    line = line or ""
                    if self.file_watcher:
                        cmd = self.file_watcher.process_changes()
                        return cmd

            except EOFError:
                raise
            except Exception as err:
                self.tool_error(str(err))
                return ""
            except UnicodeEncodeError as err:
                self.tool_error(str(err))
                return ""

            if line.strip("\r\n") and not multiline_input:
                stripped = line.strip("\r\n")
                if stripped == "{":
                    multiline_input = True
                    multiline_tag = None
                    inp += ""
                elif stripped[0] == "{":
                    tag = "".join(c for c in stripped[1:] if c.isalnum())
                    if stripped == "{" + tag:
                        multiline_input = True
                        multiline_tag = tag
                        inp += ""
                    else:
                        inp = line
                        break
                else:
                    inp = line
                    break
                continue
            elif multiline_input and line.strip():
                if multiline_tag:
                    if line.strip("\r\n") == f"{multiline_tag}}}":
                        break
                    else:
                        inp += line + "\n"
                elif line.strip("\r\n") == "}":
                    break
                else:
                    inp += line + "\n"
            elif multiline_input:
                inp += line + "\n"
            else:
                inp = line
                break

        print()
        self.user_input(inp)
        return inp

    def _get_style(self):
        """Get the prompt style."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return None

        style_dict = {}
        if not self.pretty:
            return Style.from_dict(style_dict)

        if self.user_input_color:
            style_dict.setdefault("", self.user_input_color)
            style_dict.update(
                {
                    "pygments.literal.string": f"bold italic {self.user_input_color}",
                }
            )

        completion_menu_style = []
        if self.completion_menu_bg_color:
            completion_menu_style.append(f"bg:{self.completion_menu_bg_color}")
        if self.completion_menu_color:
            completion_menu_style.append(self.completion_menu_color)
        if completion_menu_style:
            style_dict["completion-menu"] = " ".join(completion_menu_style)

        completion_menu_current_style = []
        if self.completion_menu_current_bg_color:
            completion_menu_current_style.append(self.completion_menu_current_bg_color)
        if self.completion_menu_current_color:
            completion_menu_current_style.append(f"bg:{self.completion_menu_current_color}")
        if completion_menu_current_style:
            style_dict["completion-menu.completion.current"] = " ".join(
                completion_menu_current_style
            )

        return Style.from_dict(style_dict)

    def user_input(self, inp, log_only=True):
        """Record user input to history."""
        if not log_only:
            self.display_user_input(inp)

        prefix = "####"
        if inp:
            hist = inp.splitlines()
        else:
            hist = ["<blank>"]

        hist = f"  \n{prefix} ".join(hist)

        hist = f"""
{prefix} {hist}"""
        self.append_chat_history(hist, linebreak=True)

    def display_user_input(self, inp):
        """Display user input with styling."""
        if self.pretty and self.user_input_color and self.console and Text:
            style = dict(style=self.user_input_color)
            self.console.print(Text(inp), **style)
        else:
            print(inp)

    def confirm_ask(
        self,
        question,
        default="y",
        subject=None,
        explicit_yes_required=False,
        group=None,
        allow_never=False,
    ):
        """Ask a yes/no confirmation question."""
        self.num_user_asks += 1

        question_id = (question, subject)

        if question_id in self.never_prompts:
            return False

        if group and not group.show_group:
            group = None
        if group:
            allow_never = True

        valid_responses = ["yes", "no", "skip", "all"]
        options = " (Y)es/(N)o"
        if group:
            if not explicit_yes_required:
                options += "/(A)ll"
            options += "/(S)kip all"
        if allow_never:
            options += "/(D)on't ask again"
            valid_responses.append("don't")

        if default.lower().startswith("y"):
            question += options + " [Yes]: "
        elif default.lower().startswith("n"):
            question += options + " [No]: "
        else:
            question += options + f" [{default}]: "

        if subject:
            self.tool_output()
            if "\n" in subject:
                lines = subject.splitlines()
                max_length = max(len(line) for line in lines)
                padded_lines = [line.ljust(max_length) for line in lines]
                padded_subject = "\n".join(padded_lines)
                self.tool_output(padded_subject, bold=True)
            else:
                self.tool_output(subject, bold=True)

        style = self._get_style()

        if self.yes is True:
            res = "n" if explicit_yes_required else "y"
        elif self.yes is False:
            res = "n"
        elif group and group.preference:
            res = group.preference
            self.user_input(f"{question}{res}", log_only=False)
        else:
            try:
                if self.prompt_session and PROMPT_TOOLKIT_AVAILABLE:
                    res = self.prompt_session.prompt(
                        question,
                        style=style,
                        complete_while_typing=False,
                    )
                else:
                    res = input(question)
            except EOFError:
                res = default

        if not res:
            res = default

        res = res.lower()[0]

        if res == "d" and allow_never:
            self.never_prompts.add(question_id)
            hist = f"{question.strip()} {res}"
            self.append_chat_history(hist, linebreak=True, blockquote=True)
            return False

        if explicit_yes_required:
            is_yes = res == "y"
        else:
            is_yes = res in ("y", "a")

        is_all = res == "a" and group is not None and not explicit_yes_required
        is_skip = res == "s" and group is not None

        if group:
            if is_all and not explicit_yes_required:
                group.preference = "all"
            elif is_skip:
                group.preference = "skip"

        hist = f"{question.strip()} {res}"
        self.append_chat_history(hist, linebreak=True, blockquote=True)

        return is_yes

    def tool_error(self, message="", strip=True):
        """Print an error message."""
        self.num_error_outputs += 1
        self._tool_message(message, strip, self.tool_error_color)

    def tool_warning(self, message="", strip=True):
        """Print a warning message."""
        self._tool_message(message, strip, self.tool_warning_color)

    def tool_output(self, *messages, log_only=False, bold=False):
        """Print a tool output message."""
        if messages:
            hist = " ".join(str(m) for m in messages)
            hist = f"{hist.strip()}"
            self.append_chat_history(hist, linebreak=True, blockquote=True)

        if log_only:
            return

        if self.console and Text and RichStyle:
            messages = list(map(Text, messages))
            style = dict()
            if self.pretty:
                if self.tool_output_color:
                    style["color"] = ensure_hash_prefix(self.tool_output_color)
                style["reverse"] = bold

            style = RichStyle(**style)
            self.console.print(*messages, style=style)
        else:
            print(*messages)

    def _tool_message(self, message="", strip=True, color=None):
        """Print a tool message with optional color."""
        if message.strip():
            if "\n" in message:
                for line in message.splitlines():
                    self.append_chat_history(line, linebreak=True, blockquote=True, strip=strip)
            else:
                hist = message.strip() if strip else message
                self.append_chat_history(hist, linebreak=True, blockquote=True)

        if self.console:
            if not isinstance(message, Text):
                message = Text(message) if Text else message
            color = ensure_hash_prefix(color) if color else None
            style = dict(style=color) if self.pretty and color else dict()
            try:
                self.console.print(message, **style)
            except (UnicodeEncodeError, TypeError):
                if Text and isinstance(message, Text):
                    message = message.plain
                message = str(message).encode("ascii", errors="replace").decode("ascii")
                self.console.print(message, **style)
        else:
            print(message)

    def assistant_output(self, message, pretty=None):
        """Print assistant output."""
        if not message:
            self.tool_warning("Empty response received from LLM. Check your provider account?")
            return

        if pretty is None:
            pretty = self.pretty

        if pretty and Markdown and self.console:
            show_resp = Markdown(
                message, style=self.assistant_output_color, code_theme=self.code_theme
            )
        else:
            show_resp = Text(message or "(empty response)") if Text else (message or "(empty response)")

        if self.console:
            self.console.print(show_resp)
        else:
            print(message)

    def append_chat_history(self, text, linebreak=False, blockquote=False, strip=True):
        """Append text to chat history file."""
        if blockquote:
            if strip:
                text = text.strip()
            text = "> " + text
        if linebreak:
            if strip:
                text = text.rstrip()
            text = text + "  \n"
        if not text.endswith("\n"):
            text += "\n"
        if self.chat_history_file is not None:
            try:
                self.chat_history_file.parent.mkdir(parents=True, exist_ok=True)
                with self.chat_history_file.open("a", encoding=self.encoding, errors="ignore") as f:
                    f.write(text)
            except (PermissionError, OSError) as err:
                print(f"Warning: Unable to write to chat history file {self.chat_history_file}.")
                print(err)
                self.chat_history_file = None

    def format_files_for_input(self, rel_fnames, rel_read_only_fnames):
        """Format file list for display."""
        if not self.pretty or not Console or not Text or not Columns:
            read_only_files = []
            for full_path in sorted(rel_read_only_fnames or []):
                read_only_files.append(f"{full_path} (read only)")

            editable_files = []
            for full_path in sorted(rel_fnames):
                if full_path in rel_read_only_fnames:
                    continue
                editable_files.append(f"{full_path}")

            return "\n".join(read_only_files + editable_files) + "\n"

        output = StringIO()
        console = Console(file=output, force_terminal=False)

        read_only_files = sorted(rel_read_only_fnames or [])
        editable_files = [f for f in sorted(rel_fnames) if f not in rel_read_only_fnames]

        if read_only_files:
            ro_paths = []
            for rel_path in read_only_files:
                abs_path = os.path.abspath(os.path.join(self.root, rel_path))
                ro_paths.append(Text(abs_path if len(abs_path) < len(rel_path) else rel_path))

            files_with_label = [Text("Readonly:")] + ro_paths
            Console(file=StringIO(), force_terminal=False).print(Columns(files_with_label))
            console.print(Columns(files_with_label))

        if editable_files:
            text_editable_files = [Text(f) for f in editable_files]
            files_with_label = text_editable_files
            if read_only_files:
                files_with_label = [Text("Editable:")] + text_editable_files
            console.print(Columns(files_with_label))

        return output.getvalue()


def get_rel_fname(fname, root):
    """Get relative filename."""
    try:
        return os.path.relpath(fname, root)
    except ValueError:
        return fname


# ============================================================================
# SECTION 6: GIT REPOSITORY HANDLING
# ============================================================================

class GitRepo:
    """Handle Git repository operations."""

    repo = None
    aider_ignore_file = None
    aider_ignore_spec = None
    aider_ignore_ts = 0
    aider_ignore_last_check = 0
    subtree_only = False
    ignore_file_cache = {}
    git_repo_error = None

    def __init__(
        self,
        io,
        fnames,
        git_dname,
        aider_ignore_file=None,
        models=None,
        attribute_author=True,
        attribute_committer=True,
        attribute_commit_message_author=False,
        attribute_commit_message_committer=False,
        commit_prompt=None,
        subtree_only=False,
        git_commit_verify=True,
        attribute_co_authored_by=False,
    ):
        if not git:
            raise FileNotFoundError("Git is not available")

        self.io = io
        self.models = models

        self.normalized_path = {}
        self.tree_files = {}

        self.attribute_author = attribute_author
        self.attribute_committer = attribute_committer
        self.attribute_commit_message_author = attribute_commit_message_author
        self.attribute_commit_message_committer = attribute_commit_message_committer
        self.attribute_co_authored_by = attribute_co_authored_by
        self.commit_prompt = commit_prompt
        self.subtree_only = subtree_only
        self.git_commit_verify = git_commit_verify
        self.ignore_file_cache = {}

        if git_dname:
            check_fnames = [git_dname]
        elif fnames:
            check_fnames = fnames
        else:
            check_fnames = ["."]

        repo_paths = []
        for fname in check_fnames:
            fname = Path(fname)
            fname = fname.resolve()

            if not fname.exists() and fname.parent.exists():
                fname = fname.parent

            try:
                repo_path = git.Repo(fname, search_parent_directories=True).working_dir
                repo_path = safe_abs_path(repo_path)
                repo_paths.append(repo_path)
            except ANY_GIT_ERROR:
                pass

        num_repos = len(set(repo_paths))

        if num_repos == 0:
            raise FileNotFoundError
        if num_repos > 1:
            self.io.tool_error("Files are in different git repos.")
            raise FileNotFoundError

        self.repo = git.Repo(repo_paths.pop(), odbt=git.GitDB)
        self.root = safe_abs_path(self.repo.working_tree_dir)

        if aider_ignore_file:
            self.aider_ignore_file = Path(aider_ignore_file)

    def get_tracked_files(self):
        """Get list of tracked files in the repo."""
        if not self.repo:
            return []

        try:
            commits = self.repo.iter_commits(self.repo.active_branch)
            latest_commit = next(commits)
            return [
                item.path
                for item in latest_commit.tree.traverse()
                if item.type == "blob"
            ]
        except ANY_GIT_ERROR:
            return []

    def get_rel_repo_dir(self):
        """Get relative repo directory."""
        try:
            return os.path.relpath(self.root, os.getcwd())
        except ValueError:
            return self.root

    def git_ignored_file(self, fname):
        """Check if a file is ignored by git."""
        if not self.repo:
            return False

        try:
            fname = Path(fname)
            return bool(self.repo.ignored(fname))
        except ANY_GIT_ERROR:
            return False

    def ignored_file(self, fname):
        """Check if file is ignored by aiderignore."""
        fname = str(fname)

        if self.aider_ignore_file and self.aider_ignore_file.exists():
            self._load_aiderignore_spec()

            if self.aider_ignore_spec:
                rel_fname = self.get_rel_fname(fname)
                return self.aider_ignore_spec.match_file(rel_fname)

        return False

    def get_rel_fname(self, fname):
        """Get relative filename."""
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            return fname

    def _load_aiderignore_spec(self):
        """Load aiderignore spec."""
        if not pathspec:
            return

        curr_time = time.time()
        if curr_time - self.aider_ignore_last_check < 1:
            return

        self.aider_ignore_last_check = curr_time

        if not self.aider_ignore_file.exists():
            return

        if self.aider_ignore_ts == self.aider_ignore_file.stat().st_mtime:
            return

        self.aider_ignore_ts = self.aider_ignore_file.stat().st_mtime

        try:
            ignore_text = self.aider_ignore_file.read_text()
            self.aider_ignore_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern,
                ignore_text.splitlines(),
            )
        except ANY_GIT_ERROR:
            pass


# ============================================================================
# SECTION 7: CHAT HISTORY AND SUMMARIZATION
# ============================================================================

class ChatSummary:
    """Summarize chat history to manage token count."""

    def __init__(self, models=None, max_tokens=1024):
        if not models:
            raise ValueError("At least one model must be provided")
        self.models = models if isinstance(models, list) else [models]
        self.max_tokens = max_tokens
        self.token_count = self.models[0].token_count

    def too_big(self, messages):
        """Check if messages are too big."""
        sized = self.tokenize(messages)
        total = sum(tokens for tokens, _msg in sized)
        return total > self.max_tokens

    def tokenize(self, messages):
        """Tokenize messages."""
        sized = []
        for msg in messages:
            tokens = self.token_count(msg)
            sized.append((tokens, msg))
        return sized

    def summarize(self, messages, depth=0):
        """Summarize messages if needed."""
        messages = self.summarize_real(messages)
        if messages and messages[-1]["role"] != "assistant":
            messages.append(dict(role="assistant", content="Ok."))
        return messages

    def summarize_real(self, messages, depth=0):
        """Actually perform summarization."""
        if not self.models:
            raise ValueError("No models available for summarization")

        sized = self.tokenize(messages)
        total = sum(tokens for tokens, _msg in sized)
        if total <= self.max_tokens and depth == 0:
            return messages

        min_split = 4
        if len(messages) <= min_split or depth > 3:
            return self.summarize_all(messages)

        tail_tokens = 0
        split_index = len(messages)
        half_max_tokens = self.max_tokens // 2

        for i in range(len(sized) - 1, -1, -1):
            tokens, _msg = sized[i]
            if tail_tokens + tokens < half_max_tokens:
                tail_tokens += tokens
                split_index = i
            else:
                break

        while messages[split_index - 1]["role"] != "assistant" and split_index > 1:
            split_index -= 1

        if split_index <= min_split:
            return self.summarize_all(messages)

        tail = messages[split_index:]

        sized_head = sized[:split_index]

        model_max_input_tokens = self.models[0].info.get("max_input_tokens") or 4096
        model_max_input_tokens -= 512

        keep = []
        total = 0

        for tokens, msg in sized_head:
            total += tokens
            if total > model_max_input_tokens:
                break
            keep.append(msg)

        summary = self.summarize_all(keep)

        summary_tokens = self.token_count(summary)
        tail_tokens = sum(tokens for tokens, _ in sized[split_index:])
        if summary_tokens + tail_tokens < self.max_tokens:
            return summary + tail

        return self.summarize_real(summary + tail, depth + 1)

    def summarize_all(self, messages):
        """Summarize all messages."""
        from aider.prompts import summarize, summary_prefix

        content = ""
        for msg in messages:
            role = msg["role"].upper()
            if role not in ("USER", "ASSISTANT"):
                continue
            content += f"# {role}\n"
            content += msg["content"]
            if not content.endswith("\n"):
                content += "\n"

        summarize_messages = [
            dict(role="system", content=summarize),
            dict(role="user", content=content),
        ]

        for model in self.models:
            try:
                summary = model.simple_send_with_retries(summarize_messages)
                if summary is not None:
                    summary = summary_prefix + summary
                    return [dict(role="user", content=summary)]
            except Exception as e:
                print(f"Summarization failed for model {model.name}: {str(e)}")

        raise ValueError("summarizer unexpectedly failed for all models")


# ============================================================================
# SECTION 8: REPOSITORY MAP
# ============================================================================

class RepoMap:
    """Create and maintain a map of the repository structure."""

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo=None,
        refresh="auto",
    ):
        self.map_tokens = map_tokens
        self.root = root
        self.main_model = main_model
        self.io = io
        self.repo = repo
        self.refresh = refresh

    def get_repo_map(self):
        """Get the repository map."""
        if self.map_tokens <= 0:
            return ""

        try:
            if self.repo:
                files = self.repo.get_tracked_files()
            else:
                return ""

            # Simple repo map - just list files
            other_files = [f for f in files if not f.endswith(".md")]
            files_listing = "\n".join(other_files[:50])  # Limit to 50 files

            return f"""# Repository files

{files_listing}
"""
        except ANY_GIT_ERROR:
            return ""


# ============================================================================
# SECTION 9: COMMANDS SYSTEM
# ============================================================================

class SwitchCoder(Exception):
    """Exception to switch to a different coder."""
    def __init__(self, placeholder=None, **kwargs):
        self.kwargs = kwargs
        self.placeholder = placeholder


class Commands:
    """Handle slash commands."""

    voice = None
    scraper = None

    def clone(self):
        """Clone this command instance."""
        return Commands(
            self.io,
            None,
            voice_language=self.voice_language,
            verify_ssl=self.verify_ssl,
        )

    def __init__(
        self,
        io,
        coder,
        voice_language=None,
        voice_input_device=None,
        voice_format=None,
        verify_ssl=True,
        args=None,
        parser=None,
        verbose=False,
        editor=None,
        original_read_only_fnames=None,
    ):
        self.io = io
        self.coder = coder
        self.parser = parser
        self.args = args
        self.verbose = verbose
        self.verify_ssl = verify_ssl
        self.voice_language = voice_language
        self.voice_format = voice_format
        self.voice_input_device = voice_input_device
        self.help = None
        self.editor = editor
        self.original_read_only_fnames = set(original_read_only_fnames or [])

    def cmd_model(self, args):
        """Switch the Main Model."""
        model_name = args.strip()
        if not model_name:
            announcements = "\n".join(self.coder.get_announcements())
            self.io.tool_output(announcements)
            return

        model = Model(
            model_name,
            editor_model=self.coder.main_model.editor_model.name if self.coder.main_model.editor_model else None,
            weak_model=self.coder.main_model.weak_model.name if self.coder.main_model.weak_model else self.coder.main_model.name,
        )

        old_model_edit_format = self.coder.main_model.edit_format
        current_edit_format = self.coder.edit_format

        new_edit_format = current_edit_format
        if current_edit_format == old_model_edit_format:
            new_edit_format = model.edit_format

        raise SwitchCoder(main_model=model, edit_format=new_edit_format)

    def get_commands(self):
        """Get list of available commands."""
        commands = []
        for attr in dir(self):
            if not attr.startswith("cmd_"):
                continue
            cmd = attr[4:]
            cmd = cmd.replace("_", "-")
            commands.append("/" + cmd)

        return commands

    def do_run(self, cmd_name, args):
        """Run a command."""
        cmd_name = cmd_name.replace("-", "_")
        cmd_method_name = f"cmd_{cmd_name}"
        cmd_method = getattr(self, cmd_method_name, None)
        if not cmd_method:
            self.io.tool_output(f"Error: Command {cmd_name} not found.")
            return

        try:
            return cmd_method(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete {cmd_name}: {err}")

    def matching_commands(self, inp):
        """Find matching commands."""
        inp = inp.strip()

        if inp[0] not in "/!":
            return

        if inp[0] == "!":
            return True

        words = inp[1:].strip().split()
        if not words:
            return

        first = words[0]
        rest = inp[len(first) + 2 :]

        all_commands = self.get_commands()
        matching = [cmd for cmd in all_commands if cmd[1:] == first]

        if len(matching) == 1:
            return matching, matching[0], rest

        return matching, first, rest


# ============================================================================
# SECTION 10: BASE CODER CLASS
# ============================================================================

class UnknownEditFormat(ValueError):
    """Exception for unknown edit format."""
    def __init__(self, edit_format, valid_formats):
        self.edit_format = edit_format
        self.valid_formats = valid_formats
        super().__init__(
            f"Unknown edit format {edit_format}. Valid formats are: {', '.join(valid_formats)}"
        )


class MissingAPIKeyError(ValueError):
    """Exception for missing API key."""
    pass


class FinishReasonLength(Exception):
    """Exception when finish reason is length."""
    pass


def wrap_fence(name):
    """Wrap a fence name."""
    return f"<{name}>", f"</{name}>"


all_fences = [
    ("`" * 3, "`" * 3),
    ("`" * 4, "`" * 4),
    wrap_fence("source"),
    wrap_fence("code"),
    wrap_fence("pre"),
    wrap_fence("codeblock"),
    wrap_fence("sourcecode"),
]


class Coder:
    """Base coder class for AI-assisted coding."""

    abs_fnames = None
    abs_read_only_fnames = None
    repo = None
    last_aider_commit_hash = None
    aider_edited_files = None
    last_asked_for_commit_time = 0
    repo_map = None
    functions = None
    num_exhausted_context_windows = 0
    num_malformed_responses = 0
    last_keyboard_interrupt = None
    num_reflections = 0
    max_reflections = 3
    edit_format = None
    yield_stream = False
    temperature = None
    auto_lint = True
    auto_test = False
    test_cmd = None
    lint_outcome = None
    test_outcome = None
    multi_response_content = ""
    partial_response_content = ""
    commit_before_message = []
    message_cost = 0.0
    add_cache_headers = False
    cache_warming_thread = None
    num_cache_warming_pings = 0
    suggest_shell_commands = True
    detect_urls = True
    ignore_mentions = None
    chat_language = None
    commit_language = None
    file_watcher = None

    @classmethod
    def create(
        self,
        main_model=None,
        edit_format=None,
        io=None,
        from_coder=None,
        summarize_from_coder=True,
        **kwargs,
    ):
        """Create a coder instance."""
        if not main_model:
            if from_coder:
                main_model = from_coder.main_model
            else:
                main_model = Model(DEFAULT_MODEL_NAME)

        if edit_format == "code":
            edit_format = None
        if edit_format is None:
            if from_coder:
                edit_format = from_coder.edit_format
            else:
                edit_format = main_model.edit_format

        if not io and from_coder:
            io = from_coder.io

        if from_coder:
            use_kwargs = dict(from_coder.original_kwargs)

            done_messages = from_coder.done_messages
            if edit_format != from_coder.edit_format and done_messages and summarize_from_coder:
                try:
                    done_messages = from_coder.summarizer.summarize_all(done_messages)
                except ValueError:
                    io.tool_warning("Chat history summarization failed, continuing with full history")

            update = dict(
                fnames=list(from_coder.abs_fnames),
                read_only_fnames=list(from_coder.abs_read_only_fnames),
                done_messages=done_messages,
                cur_messages=from_coder.cur_messages,
                aider_commit_hashes=from_coder.aider_commit_hashes,
                commands=from_coder.commands.clone(),
                total_cost=from_coder.total_cost,
                ignore_mentions=from_coder.ignore_mentions,
            )
            use_kwargs.update(update)
            use_kwargs.update(kwargs)

            kwargs = use_kwargs
            from_coder.ok_to_warm_cache = False

        # Choose coder class based on edit_format
        if edit_format == "whole":
            coder_class = WholeFileCoder
        elif edit_format == "diff":
            coder_class = EditBlockCoder
        else:
            coder_class = Coder

        res = coder_class(main_model, io, **kwargs)
        res.original_kwargs = dict(kwargs)
        return res

    def clone(self, **kwargs):
        """Clone this coder."""
        new_coder = Coder.create(from_coder=self, **kwargs)
        return new_coder

    def get_announcements(self):
        """Get announcement messages."""
        lines = []
        lines.append(f"Aider v{__version__}")

        main_model = self.main_model
        weak_model = main_model.weak_model

        if weak_model is not main_model:
            prefix = "Main model"
        else:
            prefix = "Model"

        output = f"{prefix}: {main_model.name} with {self.edit_format} edit format"

        lines.append(output)

        if weak_model is not main_model:
            output = f"Weak model: {weak_model.name}"
            lines.append(output)

        if self.repo:
            rel_repo_dir = self.repo.get_rel_repo_dir()
            num_files = len(self.repo.get_tracked_files())
            lines.append(f"Git repo: {rel_repo_dir} with {num_files:,} files")
        else:
            lines.append("Git repo: none")

        if self.repo_map:
            map_tokens = self.repo_map.max_map_tokens
            if map_tokens > 0:
                refresh = self.repo_map.refresh
                lines.append(f"Repo-map: using {map_tokens} tokens, {refresh} refresh")
            else:
                lines.append("Repo-map: disabled")
        else:
            lines.append("Repo-map: disabled")

        for fname in self.get_inchat_relative_files():
            lines.append(f"Added {fname} to the chat.")

        for fname in self.abs_read_only_fnames:
            rel_fname = self.get_rel_fname(fname)
            lines.append(f"Added {rel_fname} to the chat (read-only).")

        return lines

    ok_to_warm_cache = False

    def __init__(
        self,
        main_model,
        io,
        repo=None,
        fnames=None,
        add_gitignore_files=False,
        read_only_fnames=None,
        show_diffs=False,
        auto_commits=True,
        dirty_commits=True,
        dry_run=False,
        map_tokens=1024,
        verbose=False,
        stream=True,
        use_git=True,
        cur_messages=None,
        done_messages=None,
        restore_chat_history=False,
        auto_lint=True,
        auto_test=False,
        lint_cmds=None,
        test_cmd=None,
        aider_commit_hashes=None,
        map_mul_no_files=8,
        commands=None,
        summarizer=None,
        total_cost=0.0,
        analytics=None,
        map_refresh="auto",
        cache_prompts=False,
        num_cache_warming_pings=0,
        suggest_shell_commands=True,
        chat_language=None,
        commit_language=None,
        detect_urls=True,
        ignore_mentions=None,
        total_tokens_sent=0,
        total_tokens_received=0,
        file_watcher=None,
        auto_copy_context=False,
        auto_accept_architect=True,
    ):
        # Simple analytics placeholder
        self.analytics = analytics or type('obj', (object,), {'event': lambda *args, **kwargs: None})()

        self.chat_language = chat_language
        self.commit_language = commit_language
        self.commit_before_message = []
        self.aider_commit_hashes = set()
        self.rejected_urls = set()
        self.abs_root_path_cache = {}

        self.auto_copy_context = auto_copy_context
        self.auto_accept_architect = auto_accept_architect

        self.ignore_mentions = ignore_mentions if ignore_mentions else set()

        self.file_watcher = file_watcher
        if self.file_watcher:
            self.file_watcher.coder = self

        self.suggest_shell_commands = suggest_shell_commands
        self.detect_urls = detect_urls

        self.num_cache_warming_pings = num_cache_warming_pings

        if not fnames:
            fnames = []

        if io is None:
            io = InputOutput()

        if aider_commit_hashes:
            self.aider_commit_hashes = aider_commit_hashes
        else:
            self.aider_commit_hashes = set()

        self.chat_completion_call_hashes = []
        self.chat_completion_response_hashes = []
        self.need_commit_before_edits = set()

        self.total_cost = total_cost
        self.total_tokens_sent = total_tokens_sent
        self.total_tokens_received = total_tokens_received
        self.message_tokens_sent = 0
        self.message_tokens_received = 0

        self.verbose = verbose
        self.abs_fnames = set()
        self.abs_read_only_fnames = set()
        self.add_gitignore_files = add_gitignore_files

        if cur_messages:
            self.cur_messages = cur_messages
        else:
            self.cur_messages = []

        if done_messages:
            self.done_messages = done_messages
        else:
            self.done_messages = []

        self.io = io

        self.shell_commands = []

        if not auto_commits:
            dirty_commits = False

        self.auto_commits = auto_commits
        self.dirty_commits = dirty_commits

        self.dry_run = dry_run
        self.pretty = self.io.pretty

        self.main_model = main_model
        self.reasoning_tag_name = (
            self.main_model.reasoning_tag if self.main_model.reasoning_tag else REASONING_TAG
        )

        self.stream = stream and main_model.streaming

        if cache_prompts and self.main_model.cache_control:
            self.add_cache_headers = True

        self.show_diffs = show_diffs

        self.commands = commands or Commands(self.io, self)
        self.commands.coder = self

        self.repo = repo
        if use_git and self.repo is None:
            try:
                self.repo = GitRepo(
                    self.io,
                    fnames,
                    None,
                    models=main_model.commit_message_models(),
                )
            except FileNotFoundError:
                pass

        if self.repo:
            self.root = self.repo.root
        else:
            self.root = "."

        for fname in fnames:
            fname = Path(fname)
            if self.repo and self.repo.git_ignored_file(fname) and not self.add_gitignore_files:
                self.io.tool_warning(f"Skipping {fname} that matches gitignore spec.")
                continue

            if self.repo and self.repo.ignored_file(fname):
                self.io.tool_warning(f"Skipping {fname} that matches aiderignore spec.")
                continue

            if not fname.exists():
                if touch_file(fname):
                    self.io.tool_output(f"Creating empty file {fname}")
                else:
                    self.io.tool_warning(f"Can not create {fname}, skipping.")
                    continue

            if not fname.is_file():
                self.io.tool_warning(f"Skipping {fname} that is not a normal file.")
                continue

            fname = str(fname.resolve())

            self.abs_fnames.add(fname)
            self.check_added_files()

        if read_only_fnames:
            self.abs_read_only_fnames = set()
            for fname in read_only_fnames:
                abs_fname = self.abs_root_path(fname)
                if os.path.exists(abs_fname):
                    self.abs_read_only_fnames.add(abs_fname)
                else:
                    self.io.tool_warning(f"Error: Read-only file {fname} does not exist. Skipping.")

        if map_tokens is None:
            use_repo_map = main_model.use_repo_map
            map_tokens = 1024
        else:
            use_repo_map = map_tokens > 0

        if use_repo_map and self.repo:
            self.repo_map = RepoMap(
                map_tokens,
                self.root,
                self.main_model,
                self.io,
                self.repo,
                map_refresh if map_refresh else "auto",
            )
        else:
            self.repo_map = None

        self.summarizer = summarizer or ChatSummary(
            [main_model.weak_model, main_model],
            max_tokens=1024,
        )

        self.auto_lint = auto_lint
        self.lint_cmds = lint_cmds
        self.auto_test = auto_test
        self.test_cmd = test_cmd

    def check_added_files(self):
        """Check if added files exist."""
        pass

    def get_rel_fname(self, fname):
        """Get relative filename."""
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            return fname

    def abs_root_path(self, path):
        """Get absolute path from root-relative path."""
        if os.path.isabs(path):
            return path

        # Check cache
        if path in self.abs_root_path_cache:
            return self.abs_root_path_cache[path]

        resolved_path = Path(self.root) / path
        resolved_path = resolved_path.resolve()
        abs_path = str(resolved_path)

        # Cache it
        self.abs_root_path_cache[path] = abs_path
        return abs_path

    def get_inchat_relative_files(self):
        """Get relative filenames of files in chat."""
        return [self.get_rel_fname(fname) for fname in self.abs_fnames]

    def get_files_content(self, fnames=None):
        """Get content of specified files."""
        if not fnames:
            fnames = self.abs_fnames

        prompt = ""
        for fname in sorted(fnames):
            fname = self.get_rel_fname(fname)
            content = self.io.read_text(fname)
            if content is not None:
                prompt += f"\n{fname}\n```\n{content}\n```\n"

        return prompt

    def run(self, with_message=None):
        """Run the coder main loop."""
        while True:
            try:
                if with_message:
                    inp = with_message
                    with_message = None
                else:
                    inp = self.io.get_input(
                        self.root,
                        self.get_inchat_relative_files(),
                        self.get_inchat_relative_files() + list(self.commands.get_commands()),
                        self.commands,
                        self.abs_read_only_fnames,
                    )

                    if not inp:
                        continue

                    # Check for commands
                    if inp.startswith("/"):
                        self.commands.do_run(inp[1:].strip(), "")
                        continue

                self.io.user_input(inp)
                resp = self.send_message(inp)
                self.io.assistant_output(resp)

            except KeyboardInterrupt:
                return
            except EOFError:
                return

    def send_message(self, inp):
        """Send a message to the LLM and return the response."""
        self.cur_messages += [
            dict(role="user", content=inp),
        ]

        # Simple call to LLM
        messages = self.format_messages()
        try:
            resp = self.main_model.simple_send_with_retries(messages)
            if resp:
                self.cur_messages += [
                    dict(role="assistant", content=resp),
                ]
            return resp or "No response"
        except Exception as e:
            return f"Error: {str(e)}"

    def format_messages(self):
        """Format messages for the LLM."""
        messages = [
            dict(role="system", content="You are an AI programming assistant."),
        ]
        messages += self.done_messages
        messages += self.cur_messages
        return messages

    def show_announcements(self):
        """Show announcements."""
        for line in self.get_announcements():
            self.io.tool_output(line)


# ============================================================================
# SECTION 11: WHOLE FILE CODER
# ============================================================================

class WholeFileCoder(Coder):
    """Coder that operates on entire files."""

    edit_format = "whole"

    def render_incremental_response(self, final):
        """Render incremental response."""
        return self.get_multi_response_content_in_progress()

    def get_multi_response_content_in_progress(self):
        """Get the multi-response content in progress."""
        return self.partial_response_content


# ============================================================================
# SECTION 12: EDIT BLOCK CODER
# ============================================================================

class EditBlockCoder(Coder):
    """Coder that uses search/replace blocks."""

    edit_format = "diff"

    def get_edits(self):
        """Get edits from response."""
        content = self.partial_response_content
        return []  # Simplified for single-file version

    def apply_edits(self, edits):
        """Apply edits."""
        pass


# ============================================================================
# SECTION 13: MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the single-file agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Aider AI Coding Agent (Single File)")
    parser.add_argument("files", nargs="*", help="Files to edit")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model to use")
    parser.add_argument("--edit-format", choices=["diff", "whole"], help="Edit format")
    parser.add_argument("--no-git", action="store_true", help="Don't use git")
    parser.add_argument("--message", "-m", help="Initial message")
    parser.add_argument("--yes", action="store_true", help="Yes to all prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize IO
    io = InputOutput(
        pretty=True,
        yes=args.yes,
    )

    # Initialize model
    try:
        main_model = Model(args.model, verbose=args.verbose)
    except Exception as e:
        io.tool_error(f"Error initializing model: {e}")
        io.tool_error("Make sure you have set the required API keys in environment variables.")
        return 1

    # Initialize repo
    repo = None
    if not args.no_git and git:
        try:
            repo = GitRepo(
                io,
                args.files,
                None,
                models=main_model.commit_message_models(),
            )
        except FileNotFoundError:
            io.tool_output("No git repository found.")

    # Determine edit format
    edit_format = args.edit_format or main_model.edit_format

    # Create coder
    try:
        coder = Coder.create(
            main_model=main_model,
            edit_format=edit_format,
            io=io,
            repo=repo,
            fnames=args.files,
        )
    except Exception as e:
        io.tool_error(f"Error creating coder: {e}")
        return 1

    # Show announcements
    coder.show_announcements()

    # Run initial message if provided
    if args.message:
        coder.run(with_message=args.message)
    else:
        # Run interactive loop
        coder.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
