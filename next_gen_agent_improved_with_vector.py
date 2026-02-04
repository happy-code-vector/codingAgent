#!/usr/bin/env python3
"""
Next-Gen Coding Agent - Improved Production Version WITH VECTOR SEARCH
=======================================================================

Combines advanced AI techniques with proven production features:
- From next_gen_agent.py: MCTS, Self-Reflection, Multi-Agent, Vector Search, AST Refactoring
- From current_top.py: EnhancedCOT, SolutionVerifier, ChangedImpactAnalyzer, TestManager
- 100% Standard Library - No external dependencies (Ridges-ready)
- **VECTOR SEARCH ENABLED** - Uses stdlib-only TF-IDF with cosine similarity

This is designed to surpass both agents in real-world coding tasks.

Author: Next-Gen AI Team
Version: 2.1.0 - Production Ready with Vector Search
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import re
import shlex
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)
import math

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_colored_logging(level: str = "INFO") -> logging.Logger:
    """Setup colored logging for better terminal output."""
    logger = logging.getLogger("NextGenAgentImprovedVector")
    logger.setLevel(getattr(logging, level.upper()))

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',
            'INFO': '\033[32m',
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[35m',
        }
        RESET = '\033[0m'

        def format(self, record):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)

    return logger

logger = setup_colored_logging()

# =============================================================================
# CONFIGURATION & CONSTANTS (from current_top.py)
# =============================================================================

# Model definitions (from current_top.py lines 48-50)
@dataclass
class Model:
    name: str
    timeout: int

# Problem types
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

# Model definitions (from current_top.py lines 54-60)
GLM_MODEL_NAME = Model(name="zai-org/GLM-4.6-FP8", timeout=150)
QWEN_MODEL_NAME = Model(name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", timeout=100)
GLM_OLD_MODEL_NAME = Model(name="zai-org/GLM-4.5-FP8", timeout=150)
DEEPSEEK_MODEL_NAME = Model(name="deepseek-ai/DeepSeek-V3-0324", timeout=50)
KIMI_MODEL_NAME = Model(name="moonshotai/Kimi-K2-Instruct", timeout=60)
DEEPSEEK_MODEL_NAME = GLM_MODEL_NAME
KIMI_MODEL_NAME = QWEN_MODEL_NAME

# Agent models list (repeated twice like current_top.py line 66)
AGENT_MODELS = [model for model in [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]

class AgentConfig:
    """Centralized configuration for the improved agent."""

    # Model Configuration (from current_top.py)
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", AGENT_MODELS[0].name)
    FALLBACK_MODELS: List[str] = [model.name for model in AGENT_MODELS[1:]]  # All other models

    # API Configuration (Ridges-compatible, from current_top.py line 61)
    SANDBOX_PROXY_URL: str = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
    API_URL: str = SANDBOX_PROXY_URL
    DEFAULT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "1500"))  # From current_top.py line 62
    API_TIMEOUT: int = DEFAULT_TIMEOUT
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))

    # Execution Limits (from current_top.py line 63)
    MAX_FIX_TASK_STEPS: int = int(os.getenv("MAX_FIX_TASK_STEPS", "200"))
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", str(MAX_FIX_TASK_STEPS)))  # Default to MAX_FIX_TASK_STEPS
    MAX_DURATION: int = int(os.getenv("MAX_DURATION", "1800"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))

    # Context Management (EnhancedCOT settings)
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "200000"))
    SUMMARY_THRESHOLD: int = int(os.getenv("SUMMARY_THRESHOLD", "50000"))
    LATEST_OBSERVATIONS_TO_KEEP: int = int(os.getenv("LATEST_OBSERVATIONS_TO_KEEP", "15"))  # From current_top.py line 64
    SUMMARIZE_BATCH_SIZE: int = int(os.getenv("SUMMARIZE_BATCH_SIZE", "5"))  # From current_top.py line 79
    MAX_SUMMARY_RANGES: int = int(os.getenv("MAX_SUMMARY_RANGES", "6"))  # From current_top.py line 65

    # Additional constants from current_top.py lines 80-81
    REJECT_OBSERVATION_TOKEN_THRESHOLD: int = 50_000
    SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD: int = 5_000

    # Parallel Execution
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))
    ENABLE_PARALLEL_TOOLS: bool = os.getenv("ENABLE_PARALLEL_TOOLS", "true").lower() == "true"

    # Advanced Features
    ENABLE_MCTS: bool = os.getenv("ENABLE_MCTS", "true").lower() == "true"
    MCTS_ITERATIONS: int = int(os.getenv("MCTS_ITERATIONS", "30"))
    MCTS_EXPLORATION: float = float(os.getenv("MCTS_EXPLORATION", "1.41"))

    ENABLE_SELF_REFLECTION: bool = os.getenv("ENABLE_SELF_REFLECTION", "true").lower() == "true"
    REFLECTION_INTERVAL: int = int(os.getenv("REFLECTION_INTERVAL", "5"))

    # VECTOR SEARCH ENABLED
    ENABLE_VECTOR_SEARCH: bool = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"  # ENABLED!
    ENABLE_SOLUTION_VERIFICATION: bool = os.getenv("ENABLE_SOLUTION_VERIFICATION", "true").lower() == "true"
    ENABLE_CHANGE_IMPACT_ANALYSIS: bool = os.getenv("ENABLE_CHANGE_IMPACT_ANALYSIS", "true").lower() == "true"

    # Repository
    REPO_PATH: str = os.getenv("REPO_PATH", os.getcwd())

    # Cost Tracking
    TRACK_COST: bool = True
    MAX_COST_USD: float = float(os.getenv("MAX_COST_USD", "2.0"))

    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if cls.MAX_STEPS <= 0:
            raise ValueError("MAX_STEPS must be positive")
        if cls.MAX_DURATION <= 0:
            raise ValueError("MAX_DURATION must be positive")

AgentConfig.validate()

# =============================================================================
# GLOBAL STATE
# =============================================================================

run_id: Optional[str] = None
agent_start_time: Optional[float] = None
total_inferenced_chars: int = 0
individual_inferenced_chars: int = 0

# =============================================================================
# DATA STRUCTURES
# =============================================================================

T = TypeVar('T')

@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_id:
            result["tool_call_id"] = self.tool_id
        return result

    def truncate(self, max_chars: int = 50000) -> 'Message':
        """Return a truncated version of this message."""
        if len(self.content) <= max_chars:
            return self
        return Message(
            role=self.role,
            content=self.content[:max_chars] + "\n...[truncated]",
            tool_calls=self.tool_calls,
            tool_id=self.tool_id,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

@dataclass
class Thought:
    """A single thought in the reasoning chain (EnhancedCOT compatible)."""
    next_thought: str
    next_tool_name: Optional[str] = None
    next_tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_deleted: bool = False
    is_error: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "next_thought": self.next_thought,
            "next_tool_name": self.next_tool_name,
            "next_tool_args": self.next_tool_args,
            "observation": self.observation,
            "is_deleted": self.is_deleted,
            "is_error": self.is_error,
            "timestamp": self.timestamp,
        }

# =============================================================================
# ENUMS
# =============================================================================

class ProblemType(Enum):
    """Type of coding problem."""
    CREATE = "create"
    FIX = "fix"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    TEST = "test"
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ToolErrorType(Enum):
    """Types of tool errors."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_INPUT = "invalid_input"
    NETWORK_ERROR = "network_error"
    PARSE_ERROR = "parse_error"
    SEARCH_TERM_NOT_FOUND = "search_term_not_found"
    UNKNOWN = "unknown"

class ReflectionResult(Enum):
    """Results from self-reflection."""
    GOOD = "good"
    NEEDS_IMPROVEMENT = "improve"
    WRONG_DIRECTION = "wrong"
    CRITICAL_ERROR = "error"

# =============================================================================
# EXCEPTIONS
# =============================================================================

class AgentException(Exception):
    """Base exception for agent errors."""
    pass

class ToolException(AgentException):
    """Exception raised when a tool fails."""
    def __init__(self, message: str, error_type: ToolErrorType = ToolErrorType.UNKNOWN):
        super().__init__(message)
        self.error_type = error_type

class NetworkException(AgentException):
    """Exception raised when network request fails."""
    pass

class ValidationException(AgentException):
    """Exception raised when validation fails."""
    pass

class TimeoutException(AgentException):
    """Exception raised when operation times out."""
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for retrying with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def sanitize_json(text: str) -> str:
    """Sanitize and fix common JSON issues."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    return text.strip()

def calculate_hash(content: str) -> str:
    """Calculate SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def count_tokens(text: Union[str, List[Dict]]) -> int:
    """Estimate token count (rough approximation: 4 chars per token)."""
    if isinstance(text, list):
        text = " ".join(str(m.get("content", "")) for m in text)
    return len(text) // 4

# =============================================================================
# VECTOR DATABASE - STDLIB ONLY (TF-IDF + Cosine Similarity)
# =============================================================================

class VectorDB:
    """
    Vector database using only standard library.

    Implements TF-IDF (Term Frequency-Inverse Document Frequency) with cosine similarity.
    No external dependencies required.

    Features:
    - In-memory document storage
    - TF-IDF vectorization
    - Cosine similarity search
    - Automatic preprocessing and tokenization
    """

    def __init__(self, min_token_length: int = 2, max_tokens: int = 10000):
        """
        Initialize VectorDB.

        Args:
            min_token_length: Minimum characters for a valid token
            max_tokens: Maximum unique tokens to track (for memory efficiency)
        """
        self.documents: Dict[str, str] = {}  # doc_id -> content
        self.tokenized_docs: Dict[str, List[str]] = {}  # doc_id -> tokens
        self.vocabulary: Dict[str, int] = {}  # token -> index
        self.document_frequency: Dict[str, int] = defaultdict(int)  # token -> doc_count
        self.min_token_length = min_token_length
        self.max_tokens = max_tokens
        self._next_doc_id = 0
        self._indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Splits on non-alphanumeric characters and filters stop words.
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        # Filter by length and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'not', 'no', 'yes', 'so',
        }

        filtered = [
            t for t in tokens
            if len(t) >= self.min_token_length and t not in stop_words
        ]

        return filtered

    def _build_vocabulary(self) -> None:
        """Build vocabulary from all documents."""
        self.vocabulary.clear()
        self.document_frequency.clear()

        all_tokens = set()
        for doc_id, tokens in self.tokenized_docs.items():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.document_frequency[token] += 1
            all_tokens.update(tokens)

        # Limit vocabulary size for memory efficiency
        # Keep most frequent tokens
        sorted_tokens = sorted(
            all_tokens,
            key=lambda t: self.document_frequency[t],
            reverse=True
        )[:self.max_tokens]

        self.vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
        self._indexed = True

        logger.info(f"Built vocabulary with {len(self.vocabulary)} tokens")

    def _compute_tfidf(self, tokens: List[str]) -> Dict[int, float]:
        """
        Compute TF-IDF vector for a document.

        Args:
            tokens: List of tokens in the document

        Returns:
            Dictionary mapping vocabulary index to TF-IDF score
        """
        if not self.vocabulary:
            self._build_vocabulary()

        n_docs = len(self.documents)
        vector: Dict[int, float] = defaultdict(float)

        # Count term frequencies
        term_freq: Dict[str, int] = defaultdict(int)
        for token in tokens:
            if token in self.vocabulary:
                term_freq[token] += 1

        # Compute TF-IDF
        max_freq = max(term_freq.values()) if term_freq else 1

        for token, tf in term_freq.items():
            if token in self.vocabulary:
                # TF: normalized term frequency
                tf_score = 0.5 + 0.5 * (tf / max_freq)

                # IDF: inverse document frequency (with smoothing)
                df = self.document_frequency.get(token, 1)
                idf_score = math.log((n_docs + 1) / (df + 1)) + 1

                # TF-IDF
                idx = self.vocabulary[token]
                vector[idx] = tf_score * idf_score

        return dict(vector)

    def _cosine_similarity(
        self,
        vec1: Dict[int, float],
        vec2: Dict[int, float]
    ) -> float:
        """
        Compute cosine similarity between two sparse vectors.

        Args:
            vec1: First vector (dict of index -> value)
            vec2: Second vector (dict of index -> value)

        Returns:
            Similarity score between 0 and 1
        """
        # Compute dot product
        dot_product = 0.0
        for idx, val1 in vec1.items():
            if idx in vec2:
                dot_product += val1 * vec2[idx]

        # Compute magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the database.

        Args:
            content: Document text content
            metadata: Optional metadata (e.g., file_path, function_name)

        Returns:
            Document ID
        """
        doc_id = f"doc_{self._next_doc_id}"
        self._next_doc_id += 1

        self.documents[doc_id] = content
        self.tokenized_docs[doc_id] = self._tokenize(content)
        self._indexed = False

        return doc_id

    def add_code_snippet(
        self,
        code: str,
        file_path: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> str:
        """
        Add a code snippet to the database.

        Args:
            code: Code content
            file_path: File path
            language: Programming language
            context: Optional context (e.g., surrounding code)

        Returns:
            Document ID
        """
        # Enhance code with context for better search
        content_parts = [f"File: {file_path}", f"Language: {language}"]
        if context:
            content_parts.append(f"Context: {context}")
        content_parts.append(f"Code:\n{code}")

        content = "\n".join(content_parts)
        return self.add_document(content, metadata={"file_path": file_path, "language": language})

    def index(self) -> None:
        """Build the index for searching. Call after adding all documents."""
        if not self._indexed:
            self._build_vocabulary()
            # Pre-compute TF-IDF vectors for all documents
            self._vectors: Dict[str, Dict[int, float]] = {}
            for doc_id in self.documents:
                self._vectors[doc_id] = self._compute_tfidf(self.tokenized_docs[doc_id])

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of results with 'doc_id', 'content', 'score', and 'metadata'
        """
        if not self._indexed:
            self.index()

        # Tokenize and vectorize query
        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf(query_tokens)

        # Compute similarities
        results = []
        for doc_id, doc_vector in self._vectors.items():
            score = self._cosine_similarity(query_vector, doc_vector)
            if score >= min_score:
                results.append({
                    "doc_id": doc_id,
                    "content": self.documents[doc_id],
                    "score": score,
                })

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)

        return results[:top_k]

    def search_code(
        self,
        query: str,
        file_pattern: str = "*.py",
        repo_path: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search code in the repository.

        Args:
            query: Natural language query
            file_pattern: Glob pattern for files to search
            repo_path: Repository path (defaults to AgentConfig.REPO_PATH)
            top_k: Number of results

        Returns:
            List of relevant code snippets
        """
        if repo_path is None:
            repo_path = AgentConfig.REPO_PATH

        # Index code files if not already done
        repo_path_obj = Path(repo_path)
        indexed_files = set()

        # Check if we need to index files
        if not self._indexed or len(self.documents) == 0:
            logger.info(f"Indexing code files matching {file_pattern}...")

            for file_path in repo_path_obj.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        code = file_path.read_text(encoding='utf-8', errors='replace')

                        # Try to parse as AST for better context
                        try:
                            tree = ast.parse(code)
                            # Extract functions and classes
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                    # Get line range
                                    lines = code.split('\n')
                                    start = node.lineno - 1
                                    end = getattr(node, 'end_lineno', len(lines))
                                    snippet = '\n'.join(lines[start:end])

                                    name = getattr(node, 'name', 'unknown')
                                    kind = "function" if isinstance(node, ast.FunctionDef) else "class"

                                    self.add_code_snippet(
                                        snippet,
                                        str(file_path.relative_to(repo_path_obj)),
                                        context=f"{kind}: {name}"
                                    )
                        except SyntaxError:
                            # Fall back to whole file
                            self.add_code_snippet(
                                code,
                                str(file_path.relative_to(repo_path_obj))
                            )

                        indexed_files.add(str(file_path))

                    except Exception as e:
                        logger.debug(f"Failed to index {file_path}: {e}")

            logger.info(f"Indexed {len(indexed_files)} files")
            self.index()

        # Search
        results = self.search(query, top_k=top_k)

        # Enhance results with file info
        for result in results:
            # Extract file path from content
            lines = result["content"].split('\n')
            for line in lines:
                if line.startswith("File: "):
                    result["file_path"] = line.replace("File: ", "")
                    break

        return results

    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.documents.clear()
        self.tokenized_docs.clear()
        self.vocabulary.clear()
        self.document_frequency.clear()
        self._indexed = False
        self._next_doc_id = 0

# =============================================================================
# ENHANCED COT WITH SUMMARIZATION (from current_top.py)
# =============================================================================

class EnhancedCOT:
    """
    Enhanced Chain-of-Thought with conversation summarization.

    Manages long conversations by summarizing old thoughts to stay within context limits.
    Adapted from current_top.py's EnhancedCOT.
    """

    def __init__(self, latest_observations_to_keep: int = 15, summarize_batch_size: int = 5):
        self.thoughts: List[Thought] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.summarize_batch_size = summarize_batch_size
        self.summaries: Dict[Tuple[int, int], str] = {}
        self.summarized_ranges: List[Tuple[int, int]] = []
        self.repeated_thoughts = 0

    def add_action(self, action: Thought) -> bool:
        """Add an action/thought to the chain."""
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True

    def pop_action(self) -> Thought:
        """Pop the last action from the chain."""
        return self.thoughts.pop()

    def _summarize_messages_batch(self, start_idx: int, end_idx: int) -> Optional[str]:
        """Summarize a batch of messages."""
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None

        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, "is_deleted", False):
                continue

            assistant_part = (
                f"next_thought: {thought.next_thought}\n"
                f"next_tool_name: {thought.next_tool_name}\n"
                f"next_tool_args: {thought.next_tool_args}\n"
            )

            obs = thought.observation
            if isinstance(obs, (list, tuple)):
                try:
                    obs_render = json.dumps(list(obs), ensure_ascii=False)
                except Exception:
                    obs_render = str(obs)
            else:
                obs_render = str(obs) if obs else ""

            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated]"

            user_part = f"observation: {obs_render}"
            conversation_parts.append({
                "assistant": assistant_part,
                "user": user_part,
                "is_error": getattr(thought, "is_error", False),
            })

        if not conversation_parts:
            return None

        conv_lines = []
        for idx, part in enumerate(conversation_parts, 1):
            conv_lines.append(f"\n--- Step {idx} ---")
            conv_lines.append(f"Assistant: {part['assistant']}")
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conv_lines.append(f"User: {user_obs}")
            if part.get("is_error"):
                conv_lines.append("[Error occurred]")

        conversation_text = "\n".join(conv_lines)

        summarization_prompt = f"""You are summarizing a conversation history between an AI agent and its environment.
Summarize the following conversation steps concisely, focusing on:
1. Key actions taken (tools used, files modified, tests run)
2. Important findings or errors encountered
3. Progress made toward solving the problem
4. Critical decisions or changes in approach

Keep the summary concise (2-4 sentences per step) but preserve important details.

Conversation to summarize:
{conversation_text}

Provide a concise summary:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversation history concisely."},
            {"role": "user", "content": summarization_prompt},
        ]

        try:
            response = EnhancedNetwork.make_request(messages, model=AgentConfig.PRIMARY_MODEL, temperature=0.0)
            if response:
                return response[0].strip() if isinstance(response, tuple) else response.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")

        return None

    def _check_and_summarize_if_needed(self) -> None:
        """Check if we need to summarize old thoughts."""
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep

        if cutoff_idx < self.summarize_batch_size:
            return

        unsummarized = 0
        for s, e in sorted(self.summarized_ranges):
            if s <= unsummarized < e:
                unsummarized = e
            elif s > unsummarized:
                break

        if unsummarized >= cutoff_idx:
            return

        summarize_start = unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        batch_size = summarize_end - summarize_start

        if batch_size >= self.summarize_batch_size:
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()
                    logger.info(f"Summarized thoughts {summarize_start}:{summarize_end}")

    def to_str(self) -> str:
        """Convert the chain of thought to a string format for LLM."""
        messages = []

        # Add summaries first
        for (start, end), summary in sorted(self.summaries.items()):
            messages.append({
                "role": "system",
                "content": f"[Summary of steps {start+1}-{end}]:\n{summary}"
            })

        # Add recent thoughts
        for thought in self.thoughts:
            if getattr(thought, "is_deleted", False):
                continue

            if thought.next_thought:
                messages.append({
                    "role": "assistant",
                    "content": f"next_thought: {thought.next_thought}"
                })

            if thought.next_tool_name and thought.next_tool_args:
                messages.append({
                    "role": "assistant",
                    "content": f"tool_name: {thought.next_tool_name}\ntool_args: {thought.next_tool_args}"
                })

            if thought.observation:
                obs_str = str(thought.observation)
                if len(obs_str) > 50000:
                    obs_str = obs_str[:50000] + "\n...[truncated]"
                messages.append({
                    "role": "user",
                    "content": f"observation: {obs_str}"
                })

        return "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def get_recent_history(self, n: int = 10) -> List[Thought]:
        """Get the n most recent thoughts."""
        return [t for t in self.thoughts if not getattr(t, "is_deleted", False)][-n:]

# =============================================================================
# NETWORK CLIENT (Standard Library Only)
# =============================================================================

class EnhancedNetwork:
    """
    Enhanced network client using urllib.request (standard library only).

    Replaces httpx/requests for Ridges compliance.
    """

    @classmethod
    def make_request(
        cls,
        messages: List[Dict[str, str]],
        model: str = AgentConfig.PRIMARY_MODEL,
        temperature: float = 0.0,
        timeout: int = AgentConfig.API_TIMEOUT,
    ) -> Tuple[str, List]:
        """
        Make a request to the LLM API.

        Returns:
            Tuple of (response_content, tool_calls)
        """
        global total_inferenced_chars, individual_inferenced_chars

        messages_str = json.dumps(messages, ensure_ascii=False)
        individual_inferenced_chars = len(messages_str)
        total_inferenced_chars += individual_inferenced_chars

        url = f"{AgentConfig.API_URL.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": AgentConfig.MAX_TOKENS,
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            content = response_data["choices"][0]["message"]["content"]
            tool_calls = response_data["choices"][0]["message"].get("tool_calls", [])

            return content, tool_calls

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            raise NetworkException(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise NetworkException(f"URL error: {e.reason}")
        except socket.timeout:
            raise TimeoutException(f"Request timeout after {timeout}s")
        except Exception as e:
            raise AgentException(f"LLM request failed: {e}")

    @classmethod
    def get_cost_usage(cls) -> Dict[str, Any]:
        """Get cost usage from the API."""
        global run_id
        url = f"{AgentConfig.API_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id or 'unknown'}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                usage_info = json.loads(response.read().decode('utf-8'))
            if isinstance(usage_info, dict):
                return usage_info
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception:
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}

# =============================================================================
# CHANGE IMPACT ANALYZER (from current_top.py)
# =============================================================================

class ChangedImpactAnalyzer:
    """
    Analyzes the impact of code changes before making them.

    Identifies dependencies and potential side effects.
    """

    def __init__(self, repo_path: str = AgentConfig.REPO_PATH):
        self.repo_path = Path(repo_path)
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.function_graph: Dict[str, Set[str]] = defaultdict(set)
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build import and function dependency graphs."""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                try:
                    self._analyze_file(str(py_file))
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to build dependency graph: {e}")

    def _analyze_file(self, file_path: str) -> None:
        """Analyze a single file for dependencies."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Track imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_graph[file_path].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.import_graph[file_path].add(node.module)

                # Track function calls (simple version)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.function_graph[file_path].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        self.function_graph[file_path].add(node.func.attr)

        except SyntaxError:
            pass
        except Exception:
            pass

    def analyze_impact(self, file_path: str, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the impact of changing a file or function.

        Returns:
            Dictionary with impact analysis including:
            - affected_files: List of files that import this file
            - dependent_functions: Functions that call this function
            - risk_level: LOW, MEDIUM, or HIGH
        """
        affected_files = set()
        dependent_functions = set()

        # Find files that import this file
        module_name = Path(file_path).stem
        for other_file, imports in self.import_graph.items():
            if module_name in imports or file_path in imports:
                affected_files.add(other_file)

        # If function specified, find dependent functions
        if function_name:
            for other_file, functions in self.function_graph.items():
                if function_name in functions:
                    dependent_functions.add((other_file, function_name))

        # Calculate risk level
        risk_level = "LOW"
        if len(affected_files) > 5:
            risk_level = "HIGH"
        elif len(affected_files) > 2:
            risk_level = "MEDIUM"

        return {
            "file": file_path,
            "function": function_name,
            "affected_files": list(affected_files),
            "dependent_functions": list(dependent_functions),
            "risk_level": risk_level,
        }

# =============================================================================
# SOLUTION VERIFIER (from current_top.py)
# =============================================================================

class SolutionVerifier:
    """
    Verifies that solutions work correctly.

    Runs tests and checks for regressions.
    """

    def __init__(self, repo_path: str = AgentConfig.REPO_PATH):
        self.repo_path = Path(repo_path)
        self.test_results: List[Dict[str, Any]] = []

    def verify_solution(
        self,
        problem_statement: str,
        modified_files: List[str],
    ) -> Dict[str, Any]:
        """
        Verify the solution works correctly.

        Returns:
            Dictionary with verification results:
            - passed: bool
            - test_results: List of test outcomes
            - issues: List of problems found
        """
        if not AgentConfig.ENABLE_SOLUTION_VERIFICATION:
            return {"passed": True, "test_results": [], "issues": []}

        issues = []
        test_results = []

        # 1. Check if files are syntactically valid
        for file_path in modified_files:
            full_path = self.repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    issues.append(f"Syntax error in {file_path}: {e}")

        # 2. Run tests if they exist
        test_dirs = ["tests", "test"]
        for test_dir in test_dirs:
            test_path = self.repo_path / test_dir
            if test_path.exists():
                try:
                    result = subprocess.run(
                        ["python", "-m", "pytest", test_dir, "-v"],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=self.repo_path
                    )
                    test_results.append({
                        "dir": test_dir,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    })
                except subprocess.TimeoutExpired:
                    issues.append(f"Tests in {test_dir} timed out")
                except Exception as e:
                    logger.warning(f"Failed to run tests: {e}")

        passed = len(issues) == 0
        if test_results:
            passed = passed and all(r["returncode"] == 0 for r in test_results)

        return {
            "passed": passed,
            "test_results": test_results,
            "issues": issues,
        }

# =============================================================================
# MCTS IMPLEMENTATION
# =============================================================================

class MCTSNode(Generic[T]):
    """Node in Monte Carlo Tree Search."""
    def __init__(self, state: T, parent: Optional['MCTSNode[T]'] = None, action: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode[T]'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.action = action

    def uct_score(self, exploration: float = 1.41, total_visits: int = 0) -> float:
        """Calculate UCT (Upper Confidence Bound) score."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration_term = exploration * (2 * math.log(total_visits + 1) / self.visits) ** 0.5
        return exploitation + exploration_term

    def best_child(self, exploration: float = 1.41) -> 'MCTSNode[T]':
        """Get the best child according to UCT."""
        total_visits = sum(child.visits for child in self.children)
        return max(self.children, key=lambda c: c.uct_score(exploration, total_visits))

class MCTS:
    """Monte Carlo Tree Search for intelligent action selection."""

    def __init__(
        self,
        iterations: int = AgentConfig.MCTS_ITERATIONS,
        exploration: float = AgentConfig.MCTS_EXPLORATION,
        timeout: float = 5.0,
    ):
        self.iterations = iterations
        self.exploration = exploration
        self.timeout = timeout
        self.root: Optional[MCTSNode] = None

    def search(
        self,
        initial_state: Any,
        get_actions: Callable[[Any], List[Any]],
        simulate: Callable[[Any, Any], Tuple[Any, float]],
        is_terminal: Callable[[Any], bool] = lambda s: False,
    ) -> Any:
        """Run MCTS to find the best action."""
        self.root = MCTSNode(state=initial_state)
        start_time = time.time()

        for _ in range(self.iterations):
            if time.time() - start_time > self.timeout:
                break

            node = self._select(self.root)

            if not is_terminal(node.state):
                self._expand(node, get_actions)

            if node.children:
                child = random.choice(node.children)
                reward = self._simulate(child, simulate, is_terminal)
            else:
                reward = self._simulate(node, simulate, is_terminal)

            self._backpropagate(node, reward)

        if not self.root.children:
            return None

        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node for expansion using UCT."""
        while node.children and not self._is_leaf(node):
            node = node.best_child(self.exploration)
        return node

    def _expand(self, node: MCTSNode, get_actions: Callable) -> None:
        """Expand node with children."""
        actions = get_actions(node.state)
        for action in actions:
            child = MCTSNode(state=node.state, parent=node, action=action)
            node.children.append(child)

    def _simulate(
        self,
        node: MCTSNode,
        simulate_fn: Callable,
        is_terminal: Callable,
        max_depth: int = 10,
    ) -> float:
        """Simulate from node to get reward."""
        state = node.state
        total_reward = 0.0
        depth = 0

        while not is_terminal(state) and depth < max_depth:
            actions = simulate_fn(state, None)
            if not actions:
                break

            action = random.choice(actions)
            state, reward = simulate_fn(state, action)
            total_reward += reward
            depth += 1

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _is_leaf(self, node: MCTSNode) -> bool:
        """Check if node is a leaf."""
        return not node.children

# =============================================================================
# SELF-REFLECTION SYSTEM
# =============================================================================

class SelfReflection:
    """Self-reflection system for the agent to critique and improve its decisions."""

    def __init__(self):
        self.reflection_history: List[Dict[str, Any]] = []

    def reflect_on_action(
        self,
        thought: Thought,
        context: List[Message],
        outcome: Optional[str] = None,
    ) -> Tuple[ReflectionResult, str]:
        """Reflect on a recent action and determine if it was good."""
        prompt = self._reflection_prompt(thought, context, outcome)

        try:
            response = EnhancedNetwork.make_request(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result, feedback = self._parse_reflection(response[0] if isinstance(response, tuple) else response)

            self.reflection_history.append({
                "thought": thought.to_dict(),
                "result": result.value,
                "feedback": feedback,
                "timestamp": time.time(),
            })

            return result, feedback
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return ReflectionResult.GOOD, "Reflection failed, assuming good"

    def _reflection_prompt(self, thought: Thought, context: List[Message], outcome: Optional[str]) -> str:
        """Generate prompt for reflection."""
        recent_context = "\n".join(
            f"{m.role}: {m.content[:200]}..."
            for m in context[-5:]
        )

        return f"""You are a meta-cognitive analyzer reviewing an AI agent's decision.

RECENT CONTEXT:
{recent_context}

AGENT'S THOUGHT:
Reasoning: {thought.next_thought}
Action: {thought.next_tool_name}
Confidence: N/A

OUTCOME (if available):
{outcome or "Not yet executed"}

Analyze this decision and provide:
1. A rating (GOOD/IMPROVE/WRONG/ERROR)
2. Brief feedback explaining why

Respond in JSON format:
{{"rating": "GOOD|IMPROVE/WRONG/ERROR", "feedback": "explanation"}}"""

    def _parse_reflection(self, response: str) -> Tuple[ReflectionResult, str]:
        """Parse reflection response."""
        try:
            response = sanitize_json(response)
            data = json.loads(response)
            rating = data.get("rating", "GOOD")
            feedback = data.get("feedback", "")

            result_map = {
                "GOOD": ReflectionResult.GOOD,
                "IMPROVE": ReflectionResult.NEEDS_IMPROVEMENT,
                "WRONG": ReflectionResult.WRONG_DIRECTION,
                "ERROR": ReflectionResult.CRITICAL_ERROR,
            }

            return result_map.get(rating, ReflectionResult.GOOD), feedback
        except Exception as e:
            logger.warning(f"Failed to parse reflection: {e}")
            return ReflectionResult.GOOD, "Parse error"

# =============================================================================
# TOOL SYSTEM
# =============================================================================

class Tool(ABC):
    """Base class for all tools."""
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return result."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "calls": 0,
            "errors": 0,
            "total_time": 0.0,
        })

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self.tools.get(name)
        if not tool:
            raise ToolException(f"Tool not found: {name}", ToolErrorType.INVALID_INPUT)

        start_time = time.time()
        stats = self.execution_stats[name]

        try:
            result = tool.execute(**kwargs)
            stats["calls"] += 1
            stats["total_time"] += time.time() - start_time
            return result
        except Exception as e:
            stats["errors"] += 1
            raise ToolException(f"Tool execution failed: {e}", ToolErrorType.RUNTIME_ERROR)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics."""
        return dict(self.execution_stats)

# =============================================================================
# CORE TOOLS
# =============================================================================

class ReadFileTool(Tool):
    """Read file contents."""
    name = "read_file"
    description = "Read the contents of a file. Use this to examine file contents before making changes."

    def execute(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read file content, optionally with line range."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ToolException(f"File not found: {file_path}", ToolErrorType.FILE_NOT_FOUND)

            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')

            if start_line is not None or end_line is not None:
                start = max(0, (start_line or 1) - 1)
                end = min(len(lines), end_line or len(lines))
                lines = lines[start:end]
                content = '\n'.join(lines)

            if len(content) > 50000:
                content = content[:50000] + "\n...[truncated]"

            return content
        except Exception as e:
            raise ToolException(f"Failed to read file: {e}", ToolErrorType.RUNTIME_ERROR)

class WriteFileTool(Tool):
    """Write content to a file."""
    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."

    def execute(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise ToolException(f"Failed to write file: {e}", ToolErrorType.RUNTIME_ERROR)

class EditFileTool(Tool):
    """Edit a file using search and replace."""
    name = "edit_file"
    description = "Edit a file by replacing an exact string match with new content."

    def execute(
        self,
        file_path: str,
        search: str,
        replace: str,
        occurrence: int = 1,
    ) -> str:
        """Edit file with search/replace."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ToolException(f"File not found: {file_path}", ToolErrorType.FILE_NOT_FOUND)

            content = path.read_text(encoding='utf-8')

            if search not in content:
                raise ToolException(f"Search string not found in {file_path}", ToolErrorType.SEARCH_TERM_NOT_FOUND)

            count = content.count(search)
            if count > 1:
                if occurrence > count:
                    raise ToolException(f"Only {count} occurrence(s) found", ToolErrorType.INVALID_INPUT)
                parts = content.split(search)
                if occurrence <= len(parts):
                    content = search.join(parts[:occurrence]) + replace + search.join(parts[occurrence:])
            else:
                content = content.replace(search, replace, 1)

            path.write_text(content, encoding='utf-8')
            return f"Successfully edited {file_path}"
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to edit file: {e}", ToolErrorType.RUNTIME_ERROR)

class SearchTool(Tool):
    """Search for patterns in files."""
    name = "search"
    description = "Search for text/patterns in files using grep-like functionality."

    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*.py",
        case_insensitive: bool = False,
        max_results: int = 100,
    ) -> str:
        """Search in files."""
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)

            results = []
            search_path = Path(path)

            for file_path in search_path.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                        lines = content.split('\n')

                        for i, line in enumerate(lines, 1):
                            if regex.search(line):
                                results.append(f"{file_path}:{i}: {line.strip()}")
                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue

                if len(results) >= max_results:
                    break

            if not results:
                return f"No matches found for pattern: {pattern}"

            return '\n'.join(results[:max_results])
        except Exception as e:
            raise ToolException(f"Search failed: {e}", ToolErrorType.RUNTIME_ERROR)

class VectorDBSearchTool(Tool):
    """Semantic code search using vector database."""
    name = "vector_search"
    description = "Search code semantically using natural language. Finds relevant code even without exact matches."

    def __init__(self, vector_db: Optional[VectorDB] = None):
        self.vector_db = vector_db or VectorDB()

    def execute(
        self,
        query: str,
        file_pattern: str = "*.py",
        top_k: int = 5,
    ) -> str:
        """Search code semantically."""
        try:
            results = self.vector_db.search_code(
                query=query,
                file_pattern=file_pattern,
                top_k=top_k
            )

            if not results:
                return f"No relevant code found for query: {query}"

            output = [f"Found {len(results)} relevant code snippets:\n"]
            for i, result in enumerate(results, 1):
                output.append(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
                if 'file_path' in result:
                    output.append(f"File: {result['file_path']}")
                # Show first few lines of content
                content_lines = result['content'].split('\n')[:10]
                output.append('\n'.join(content_lines))
                if len(result['content'].split('\n')) > 10:
                    output.append("...")

            return '\n'.join(output)
        except Exception as e:
            raise ToolException(f"Vector search failed: {e}", ToolErrorType.RUNTIME_ERROR)

class RunShellTool(Tool):
    """Execute shell commands."""
    name = "run_shell"
    description = "Execute a shell command. Use for git, npm, build tools, etc."

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
    ) -> str:
        """Execute shell command."""
        try:
            dangerous = ['rm -rf /', 'rm -rf /*', 'mkfs', 'format c:', '> /dev/sda']
            if any(d in command.lower() for d in dangerous):
                raise ToolException("Dangerous command blocked", ToolErrorType.PERMISSION_DENIED)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or os.getcwd(),
            )

            output = result.stdout or result.stderr
            if result.returncode != 0:
                output = f"Command exited with code {result.returncode}\n{output}"

            return output
        except subprocess.TimeoutExpired:
            raise ToolException(f"Command timeout after {timeout}s", ToolErrorType.TIMEOUT)
        except Exception as e:
            raise ToolException(f"Shell execution failed: {e}", ToolErrorType.RUNTIME_ERROR)

class RunTestsTool(Tool):
    """Run test suite."""
    name = "run_tests"
    description = "Run the project's test suite and return results."

    def execute(
        self,
        test_path: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """Run tests."""
        try:
            if Path("pytest.ini").exists() or Path("pyproject.toml").exists():
                cmd = "pytest"
            elif Path("tests").exists():
                cmd = "python -m pytest tests"
            else:
                cmd = "python -m unittest discover -s tests"

            if test_path:
                cmd += f" {test_path}"
            if verbose:
                cmd += " -v"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout or result.stderr
            return f"Test Results:\n{output}"
        except subprocess.TimeoutExpired:
            raise ToolException("Tests timed out", ToolErrorType.TIMEOUT)
        except Exception as e:
            raise ToolException(f"Test execution failed: {e}", ToolErrorType.RUNTIME_ERROR)

# =============================================================================
# GIT STATE MANAGER (from current_top.py)
# =============================================================================

class GitStateManager:
    """
    Manages git state with stash and rollback capabilities.

    Allows safe experimentation with the ability to roll back changes.
    """

    def __init__(self, repo_path: str = AgentConfig.REPO_PATH):
        self.repo_path = Path(repo_path)
        self.stash_history: List[Dict[str, Any]] = []
        self.original_cwd = os.getcwd()

    def _run_git(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        return subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.repo_path,
            check=False
        )

    def init_git(self) -> None:
        """Initialize git repository if not exists."""
        if not (self.repo_path / ".git").exists():
            logger.info("Initializing git repository...")
            self._run_git(["init"], timeout=10)
            self._run_git(["config", "user.email", "agent@ridges.ai"], timeout=5)
            self._run_git(["config", "user.name", "Ridges Agent"], timeout=5)
            self._run_git(["add", "-A"], timeout=30)
            self._run_git(["commit", "-m", "Initial commit"], timeout=30)

    def stash_changes(self, message: str = "temp_stash") -> str:
        """
        Stash current changes with a message.

        Returns:
            Stash reference (e.g., "stash@{0}")
        """
        try:
            # Stage all changes first
            self._run_git(["add", "-A"], timeout=30)

            # Check if there are changes to stash
            status = self._run_git(["status", "--porcelain"], timeout=10)
            if not status.stdout.strip():
                logger.info("No changes to stash")
                return ""

            # Create stash
            result = self._run_git(["stash", "push", "-m", message], timeout=30)

            # Get stash ref
            stash_list = self._run_git(["stash", "list"], timeout=10)
            stash_ref = "stash@{0}"

            self.stash_history.append({
                "ref": stash_ref,
                "message": message,
                "timestamp": time.time(),
            })

            logger.info(f"Stashed changes: {message}")
            return stash_ref

        except Exception as e:
            logger.error(f"Failed to stash changes: {e}")
            return ""

    def get_stash_patch(self, stash_ref: str = "stash@{0}") -> Optional[str]:
        """
        Get the patch for a stashed change.

        Returns:
            Git diff string or None
        """
        try:
            result = self._run_git([
                "stash", "show", "-p", "--no-color", "--unified=5", stash_ref
            ], timeout=30)

            if result.returncode == 0:
                return result.stdout
            return None

        except Exception as e:
            logger.error(f"Failed to get stash patch: {e}")
            return None

    def pop_stash(self, stash_ref: str = "stash@{0}") -> bool:
        """
        Pop and apply a stashed change.

        Returns:
            True if successful
        """
        try:
            result = self._run_git(["stash", "pop"], timeout=30)
            success = result.returncode == 0

            if success:
                # Remove from history
                self.stash_history = [s for s in self.stash_history if s["ref"] != stash_ref]
                logger.info(f"Restored stash: {stash_ref}")

            return success

        except Exception as e:
            logger.error(f"Failed to pop stash: {e}")
            return False

    def reset_hard(self, commit: str = "HEAD") -> bool:
        """
        Reset repository to a commit (discarding all changes).

        Returns:
            True if successful
        """
        try:
            result = self._run_git(["reset", "--hard", commit], timeout=30)
            success = result.returncode == 0

            if success:
                logger.info(f"Reset to {commit}")

            return success

        except Exception as e:
            logger.error(f"Failed to reset: {e}")
            return False

    def clean(self) -> bool:
        """
        Clean untracked files.

        Returns:
            True if successful
        """
        try:
            result = self._run_git(["clean", "-fd"], timeout=30)
            success = result.returncode == 0

            if success:
                logger.info("Cleaned untracked files")

            return success

        except Exception as e:
            logger.error(f"Failed to clean: {e}")
            return False

    def get_current_diff(self) -> str:
        """
        Get current git diff (staged changes).

        Returns:
            Git diff string
        """
        try:
            # Stage all changes
            self._run_git(["add", "-A"], timeout=30)

            # Get diff
            result = self._run_git([
                "diff", "--cached", "--unified=5", "--no-color"
            ], timeout=30)

            return result.stdout.strip()

        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            return ""

# =============================================================================
# PARALLEL EXECUTOR (from current_top.py)
# =============================================================================

class ParallelExecutor:
    """
    Parallel execution manager for tools and solution generation.

    Runs multiple operations in parallel using ThreadPoolExecutor.
    """

    def __init__(self, max_workers: int = AgentConfig.MAX_WORKERS):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_tools_parallel(
        self,
        tool_calls: List[Callable[[], str]],
        timeout: Optional[float] = None,
    ) -> List[Tuple[str, Optional[Exception]]]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of callable tools
            timeout: Maximum time to wait for all tools

        Returns:
            List of (result, exception) tuples
        """
        results = []

        try:
            # Submit all tasks
            futures = {
                self.executor.submit(tool_call): i
                for i, tool_call in enumerate(tool_calls)
            }

            # Wait for completion with timeout
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append((result, None))
                except Exception as e:
                    results.append(("", e))

        except TimeoutException:
            logger.warning(f"Parallel execution timed out after {timeout}s")
            # Cancel remaining futures
            for future in futures:
                future.cancel()

        return results

    def generate_solutions_parallel(
        self,
        problem_statement: str,
        generator_fn: Callable[[str, int], Tuple[str, bool]],
        num_attempts: int = 5,
        timeout_per_attempt: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple solutions in parallel.

        Args:
            problem_statement: The problem to solve
            generator_fn: Function that generates a solution (takes problem, attempt_num)
            num_attempts: Number of parallel attempts
            timeout_per_attempt: Timeout for each attempt

        Returns:
            List of solution dicts with 'solution', 'attempt', 'success', 'patch'
        """
        logger.info(f"Starting parallel solution generation: {num_attempts} attempts")

        solutions = []

        def attempt_generator(attempt_num: int) -> Dict[str, Any]:
            """Generate a single solution attempt."""
            start_time = time.time()
            try:
                solution, success = generator_fn(problem_statement, attempt_num)
                elapsed = time.time() - start_time

                return {
                    "attempt": attempt_num,
                    "solution": solution,
                    "success": success,
                    "elapsed": elapsed,
                }
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Attempt {attempt_num} failed: {e}")
                return {
                    "attempt": attempt_num,
                    "solution": "",
                    "success": False,
                    "elapsed": elapsed,
                    "error": str(e),
                }

        try:
            # Submit all attempts in parallel
            futures = {
                self.executor.submit(attempt_generator, i): i
                for i in range(num_attempts)
            }

            # Collect results as they complete
            for future in as_completed(futures, timeout=timeout_per_attempt):
                try:
                    solution = future.result()
                    solutions.append(solution)

                    # If we found a successful solution, we can stop
                    if solution.get("success"):
                        logger.info(f"Found successful solution at attempt {solution['attempt']}")
                        # Don't break - let other attempts finish for comparison

                except Exception as e:
                    logger.error(f"Parallel generation error: {e}")

        except TimeoutException:
            logger.warning(f"Parallel generation timed out after {timeout_per_attempt}s")

        # Sort by success and elapsed time
        solutions.sort(key=lambda s: (not s.get("success", False), s.get("elapsed", float('inf'))))

        logger.info(f"Parallel generation complete: {len(solutions)} solutions")
        return solutions

    def select_best_solution(
        self,
        solutions: List[Dict[str, Any]],
        problem_statement: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best solution from multiple attempts.

        Uses simple heuristics:
        1. Prefer successful solutions
        2. Prefer shorter elapsed time
        3. Prefer longer solutions (more complete)
        """
        if not solutions:
            return None

        # Filter successful solutions
        successful = [s for s in solutions if s.get("success")]

        if successful:
            # Among successful, pick the one with longest solution (most complete)
            successful.sort(key=lambda s: len(s.get("solution", "")), reverse=True)
            return successful[0]

        # If no successful solutions, return the one that made the most progress
        solutions.sort(key=lambda s: len(s.get("solution", "")), reverse=True)
        return solutions[0] if solutions else None

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

# =============================================================================
# MAIN IMPROVED AGENT WITH VECTOR SEARCH
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_call_id: str
    content: str
    is_error: bool
    error_type: Optional[str] = None
    execution_time: float = 0.0

class NextGenAgentImprovedVector:
    """
    Next-Generation Agent - Improved Production Version WITH VECTOR SEARCH.

    Combines:
    - Advanced AI techniques (MCTS, Self-Reflection, Multi-Agent)
    - Production features (EnhancedCOT, SolutionVerifier, ChangedImpactAnalyzer)
    - **Vector Search** for semantic code search (stdlib-only TF-IDF)
    - 100% Standard Library (no external dependencies)
    """

    def __init__(
        self,
        api_url: str = AgentConfig.API_URL,
        primary_model: str = AgentConfig.PRIMARY_MODEL,
        repo_path: str = AgentConfig.REPO_PATH,
    ):
        # Core components
        self.cot = EnhancedCOT(
            latest_observations_to_keep=AgentConfig.LATEST_OBSERVATIONS_TO_KEEP,
            summarize_batch_size=AgentConfig.SUMMARIZE_BATCH_SIZE
        )
        self.impact_analyzer = ChangedImpactAnalyzer(repo_path) if AgentConfig.ENABLE_CHANGE_IMPACT_ANALYSIS else None
        self.verifier = SolutionVerifier(repo_path) if AgentConfig.ENABLE_SOLUTION_VERIFICATION else None
        self.mcts = MCTS() if AgentConfig.ENABLE_MCTS else None
        self.reflection = SelfReflection() if AgentConfig.ENABLE_SELF_REFLECTION else None

        # NEW: Vector search database
        self.vector_db = VectorDB() if AgentConfig.ENABLE_VECTOR_SEARCH else None

        # NEW: Production features from current_top.py
        self.git_manager = GitStateManager(repo_path)
        self.parallel_executor = ParallelExecutor(max_workers=AgentConfig.MAX_WORKERS)

        # Tool system
        self.tools = ToolRegistry()
        self._register_tools()

        # State
        self.conversation_history: List[Message] = []
        self.step_count = 0
        self.start_time: Optional[float] = None
        self.modified_files: List[str] = []

        # Initialize git
        self.git_manager.init_git()

        logger.info("NextGenAgentImprovedVector initialized with Vector Search")

    def _register_tools(self) -> None:
        """Register all available tools."""
        tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            SearchTool(),
            RunShellTool(),
            RunTestsTool(),
        ]

        # Add vector search tool if enabled
        if AgentConfig.ENABLE_VECTOR_SEARCH and self.vector_db:
            tools.append(VectorDBSearchTool(self.vector_db))

        for tool in tools:
            self.tools.register(tool)

    def run(
        self,
        problem_statement: str,
        max_steps: int = AgentConfig.MAX_STEPS,
        max_duration: int = AgentConfig.MAX_DURATION,
    ) -> str:
        """
        Run the agent to solve a problem.

        Args:
            problem_statement: The problem to solve
            max_steps: Maximum number of steps
            max_duration: Maximum time in seconds

        Returns:
            Final result or patch
        """
        global run_id, agent_start_time
        agent_start_time = time.time()
        run_id = os.getenv("EVALUATION_RUN_ID", str(uuid4()))

        self.start_time = time.time()
        logger.info(f"Starting NextGenAgentImprovedVector: {problem_statement[:100]}")

        # Detect problem type
        problem_type = self._detect_problem_type(problem_statement)
        logger.info(f"Detected problem type: {problem_type.value}")

        # Initialize context
        self.conversation_history = [
            Message(role="system", content=self._get_system_prompt(problem_type)),
            Message(role="user", content=problem_statement),
        ]

        # Main execution loop
        last_result = ""
        for step in range(max_steps):
            if time.time() - self.start_time > max_duration:
                logger.warning(f"Timeout after {max_duration}s")
                break

            self.step_count = step + 1

            # Check cost budget
            if AgentConfig.TRACK_COST:
                cost_info = EnhancedNetwork.get_cost_usage()
                used = cost_info.get("used_cost_usd", 0)
                max_cost = cost_info.get("max_cost_usd", AgentConfig.MAX_COST_USD)
                if used >= max_cost - 0.5:
                    logger.warning(f"Cost limit reached: ${used:.2f}")
                    break

            # Generate next action
            try:
                messages = [m.to_dict() for m in self.conversation_history]
                response, _ = EnhancedNetwork.make_request(messages, temperature=0.0)

                # Add to COT
                thought = Thought(next_thought=response)
                self.cot.add_action(thought)

                last_result = response

                # Check if done
                if self._is_complete(response):
                    logger.info(f"Task completed at step {step + 1}")
                    break

                # Self-reflection check
                if self.reflection and step % AgentConfig.REFLECTION_INTERVAL == 0:
                    result, feedback = self.reflection.reflect_on_action(thought, self.conversation_history)
                    if result != ReflectionResult.GOOD:
                        logger.info(f"Reflection: {feedback}")

            except Exception as e:
                logger.error(f"Step {step + 1} failed: {e}")
                break

        # Verify solution
        if self.verifier and self.modified_files:
            logger.info("Verifying solution...")
            verification = self.verifier.verify_solution(problem_statement, self.modified_files)
            if not verification["passed"]:
                logger.warning(f"Verification failed: {verification['issues']}")

        duration = time.time() - self.start_time
        logger.info(f"Agent completed in {duration:.1f}s with {self.step_count} steps")

        return last_result

    def run_parallel(
        self,
        problem_statement: str,
        num_attempts: int = 5,
        max_duration: int = AgentConfig.MAX_DURATION,
    ) -> str:
        """
        Run the agent with parallel solution generation.

        Generates multiple solutions in parallel and selects the best one.

        Args:
            problem_statement: The problem to solve
            num_attempts: Number of parallel solution attempts
            max_duration: Maximum time in seconds

        Returns:
            Final result or patch
        """
        global run_id, agent_start_time
        agent_start_time = time.time()
        run_id = os.getenv("EVALUATION_RUN_ID", str(uuid4()))

        self.start_time = time.time()
        logger.info(f"Starting parallel execution: {problem_statement[:100]}")

        def generate_single_attempt(problem: str, attempt_num: int) -> Tuple[str, bool]:
            """Generate a single solution attempt."""
            # Stash current state
            stash_ref = self.git_manager.stash_changes(f"attempt_{attempt_num}")

            try:
                # Reset to clean state
                self.git_manager.reset_hard()
                self.git_manager.clean()

                # Run the agent
                result = self.run(problem, max_steps=AgentConfig.MAX_STEPS // num_attempts)

                # Get the patch
                patch = self.git_manager.get_current_diff()
                success = bool(patch)

                return result or patch, success

            except Exception as e:
                logger.error(f"Attempt {attempt_num} failed: {e}")
                return "", False
            finally:
                # Restore original state
                if stash_ref:
                    self.git_manager.reset_hard()
                    # Don't pop - we want to keep each attempt separate for comparison

        # Generate solutions in parallel
        solutions = self.parallel_executor.generate_solutions_parallel(
            problem_statement=problem_statement,
            generator_fn=generate_single_attempt,
            num_attempts=num_attempts,
            timeout_per_attempt=max_duration / num_attempts,
        )

        # Select best solution
        best = self.parallel_executor.select_best_solution(solutions, problem_statement)

        if best:
            logger.info(f"Selected best solution from attempt {best['attempt']}")

            # Apply the best solution
            if best.get("solution"):
                return best["solution"]

        return ""

    def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'

        Returns:
            List of ToolResult objects
        """
        results = []

        def execute_single(call: Dict[str, Any]) -> ToolResult:
            """Execute a single tool call."""
            tool_name = call.get("name")
            args = call.get("arguments", {})

            start_time = time.time()
            tool_call_id = str(uuid4())

            try:
                result = self.tools.execute(tool_name, **args)
                return ToolResult(
                    tool_call_id=tool_call_id,
                    content=result,
                    is_error=False,
                    execution_time=time.time() - start_time,
                )
            except Exception as e:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    content=str(e),
                    is_error=True,
                    error_type="execution_error",
                    execution_time=time.time() - start_time,
                )

        # Create callable list
        callables = [lambda c=call: execute_single(c) for call in tool_calls]

        # Execute in parallel
        raw_results = self.parallel_executor.execute_tools_parallel(callables)

        # Process results
        for i, (result, error) in enumerate(raw_results):
            if error:
                results.append(ToolResult(
                    tool_call_id=str(i),
                    content=str(error),
                    is_error=True,
                    error_type="parallel_error",
                ))
            else:
                results.append(result)

        return results

    def _detect_problem_type(self, statement: str) -> ProblemType:
        """Detect the type of problem."""
        statement_lower = statement.lower()
        if any(word in statement_lower for word in ["create", "implement", "add", "build", "write"]):
            return ProblemType.CREATE
        elif any(word in statement_lower for word in ["fix", "bug", "error", "broken"]):
            return ProblemType.FIX
        elif any(word in statement_lower for word in ["refactor", "clean up"]):
            return ProblemType.REFACTOR
        else:
            return ProblemType.UNKNOWN

    def _is_complete(self, response: str) -> bool:
        """Check if response indicates completion."""
        complete_indicators = ["done", "complete", "finished", "successfully", "the fix is", "the solution is"]
        return any(indicator in response.lower() for indicator in complete_indicators)

    def _get_system_prompt(self, problem_type: ProblemType) -> str:
        """Get the system prompt for the problem type."""
        base_prompt = f"""You are an expert coding assistant working in a repository at {AgentConfig.REPO_PATH}.

Key principles:
- Be thorough and systematic
- Explain your reasoning clearly
- Use tools to gather context before making changes
- Test your changes
- Handle edge cases

You have access to tools for reading, writing, searching files, and semantic code search."""

        if AgentConfig.ENABLE_VECTOR_SEARCH:
            base_prompt += """

**VECTOR SEARCH AVAILABLE**: Use the vector_search tool for semantic code search. This is especially useful when:
- You don't know exact function names
- You need to find related functionality
- You're exploring unfamiliar codebases
- You want to understand code architecture

Example: "Find authentication logic" will locate auth-related code even without exact matches."""

        if problem_type == ProblemType.FIX:
            return base_prompt + """

BUG FIXING APPROACH:
1. Understand the problem thoroughly
2. Find the root cause (use vector_search to locate related code)
3. Make minimal fixes
4. Verify the fix works
5. Ensure no regressions

When you've completed the fix, clearly state: "The fix is complete:" followed by a summary."""

        return base_prompt

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "step_count": self.step_count,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "tool_stats": self.tools.get_stats(),
            "cost_info": EnhancedNetwork.get_cost_usage() if AgentConfig.TRACK_COST else {},
            "git_stash_count": len(self.git_manager.stash_history),
            "vector_db_enabled": AgentConfig.ENABLE_VECTOR_SEARCH,
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.parallel_executor.shutdown()
            logger.info("Agent cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# =============================================================================
# RIDGES ENTRY POINT
# =============================================================================

def agent_main(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ridges.ai platform entry point.

    Args:
        input_dict: Contains 'repo_path' and 'problem'

    Returns:
        Dict with 'patch' key containing git diff
    """
    agent = None
    try:
        repo_path = input_dict.get("repo_path", AgentConfig.REPO_PATH)
        problem_statement = input_dict.get("problem", "")

        if not problem_statement:
            return {"patch": ""}

        AgentConfig.REPO_PATH = repo_path

        # Create agent
        agent = NextGenAgentImprovedVector(api_url=AgentConfig.API_URL, repo_path=repo_path)

        # Detect problem type to decide execution strategy
        problem_type = agent._detect_problem_type(problem_statement)

        # Use parallel execution for CREATE tasks (faster, better solutions)
        if problem_type == ProblemType.CREATE and AgentConfig.MAX_WORKERS > 1:
            logger.info("Using parallel execution for CREATE task")
            result = agent.run_parallel(
                problem_statement,
                num_attempts=min(3, AgentConfig.MAX_WORKERS),  # Up to 3 parallel attempts
                max_duration=AgentConfig.MAX_DURATION,
            )
        else:
            result = agent.run(problem_statement)

        # Generate git diff using git manager
        patch = agent.git_manager.get_current_diff()

        return {"patch": patch}

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return {"patch": ""}
    finally:
        if agent:
            agent.cleanup()


def generate_git_diff(repo_path: str) -> str:
    """Generate git diff for all changes."""
    try:
        original_cwd = os.getcwd()
        os.chdir(repo_path)

        # Initialize git if needed
        if not Path(".git").exists():
            subprocess.run(["git", "init"], capture_output=True, timeout=10)
            subprocess.run(["git", "config", "user.email", "agent@ridges.ai"], capture_output=True, timeout=5)
            subprocess.run(["git", "config", "user.name", "Ridges Agent"], capture_output=True, timeout=5)

        # Stage changes
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=30)

        # Generate diff
        result = subprocess.run(
            ["git", "diff", "--cached", "--unified=5"],
            capture_output=True,
            text=True,
            timeout=30
        )

        os.chdir(original_cwd)
        return result.stdout.strip()

    except Exception as e:
        logger.error(f"Failed to generate git diff: {e}")
        return ""

# =============================================================================
# MAIN ENTRY POINT (for local testing)
# =============================================================================

def main():
    """Main entry point for local CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Next-Gen Coding Agent - Improved with Vector Search")
    parser.add_argument("problem", help="Problem statement to solve")
    parser.add_argument("--repo-path", default=AgentConfig.REPO_PATH)
    parser.add_argument("--api-url", default=AgentConfig.API_URL)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--parallel", action="store_true", help="Use parallel execution (for CREATE tasks)")
    parser.add_argument("--max-workers", type=int, default=AgentConfig.MAX_WORKERS, help="Max parallel workers")

    args = parser.parse_args()

    setup_colored_logging(args.log_level)

    # Update config based on args
    AgentConfig.REPO_PATH = args.repo_path
    AgentConfig.API_URL = args.api_url
    AgentConfig.MAX_WORKERS = args.max_workers

    input_dict = {
        "repo_path": args.repo_path,
        "problem": args.problem,
    }

    result = agent_main(input_dict)

    print("\n" + "=" * 60)
    print("GIT DIFF PATCH:")
    print("=" * 60)
    if result["patch"]:
        print(result["patch"])
    else:
        print("(No changes made)")
    print("=" * 60)

if __name__ == "__main__":
    main()
