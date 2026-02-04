from __future__ import annotations
import os
import re
import sys
import json
import time
import random
import inspect
import logging
import tempfile
import requests
import textwrap
import traceback
import threading
import subprocess
import shlex
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel

try:
    from tree_sitter import Parser
    from tree_sitter_language_pack import get_language
except ImportError:
    Parser = None
    get_language = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

run_id = None
agent_start_time = None
_current_tool_manager = None
total_inferenced_chars = 0
individual_inferenced_chars = 0

class Model(BaseModel):
    name: str
    timeout: int

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
GLM_MODEL_NAME = Model(name="zai-org/GLM-4.6-FP8", timeout=150)
QWEN_MODEL_NAME = Model(name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", timeout=100)
GLM_OLD_MODEL_NAME = Model(name="zai-org/GLM-4.5-FP8", timeout=150)
DEEPSEEK_MODEL_NAME = Model(name="deepseek-ai/DeepSeek-V3-0324", timeout=50)
KIMI_MODEL_NAME = Model(name="moonshotai/Kimi-K2-Instruct", timeout=60)
DEEPSEEK_MODEL_NAME = GLM_MODEL_NAME
KIMI_MODEL_NAME = QWEN_MODEL_NAME
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500"))
MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 15
MAX_SUMMARY_RANGES = 6
AGENT_MODELS = [model for model in [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
You are making same mistakes.
Your previous response:
{previous_response}

**Critical**:
1. Notice what you are going to do.
2. Find the reason the same mistake is repeated.
3. Don't make the same mistakes any more and make a real progress.
"""
)
SUMMARIZE_BATCH_SIZE = 5
REJECT_OBSERVATION_TOKEN_THRESHOLD = 50_000
SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD = 5_000

_codeparse_util_language_cache = {}

class EnhancedCOT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries = {}
        self.summarized_ranges = []

    def _summarize_messages_batch(self, start_idx, end_idx):
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, "is_deleted", False):
                continue
            assistant_part = (
                f"next_thought: {thought.next_thought}\n" f"next_tool_name: {thought.next_tool_name}\n" f"next_tool_args: {thought.next_tool_args}\n"
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
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": getattr(thought, "is_error", False),
                }
            )
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
        summarization_prompt = textwrap.dedent(
            f"""
            You are summarizing a conversation history between an AI agent and its environment.
            Summarize the following conversation steps concisely, focusing on:
            1. Key actions taken (tools used, files modified, tests run)
            2. Important findings or errors encountered
            3. Progress made toward solving the problem
            4. Critical decisions or changes in approach
            Keep the summary concise (2-4 sentences per step) but preserve important details.
            Conversation to summarize:
            {conversation_text}
            Provide a concise summary:
        """
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversation history concisely.",
            },
            {"role": "user", "content": summarization_prompt},
        ]
        for _ in range(10):
            try:
                response, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
                return response.strip()
            except Exception:
                time.sleep(1)
        return None

    def _check_and_summarize_if_needed(self):
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

    def add_action(self, action):
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True

    def pop_action(self):
        return self.thoughts.pop()

    def to_str(self):
        messages = []
        last_summary_range = None
        allowed_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:]) if self.summarized_ranges else set()
        total = len(self.thoughts)
        keep_last = self.latest_observations_to_keep
        for i, thought in enumerate(self.thoughts):
            if getattr(thought, "is_deleted", False):
                continue
            recent = i >= total - keep_last
            if not recent:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            cur_range = (start, end)
                            if cur_range not in allowed_ranges:
                                found_range = True
                                break
                            if cur_range != last_summary_range:
                                messages.append(
                                    {"role": "system", "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"}
                                )
                                last_summary_range = cur_range
                            found_range = True
                            break
                    if found_range:
                        continue
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n" f"next_tool_name:{thought.next_tool_name}\n" f"next_tool_args:{thought.next_tool_args}"
                )
                obs = thought.observation
                if isinstance(obs, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs)
                else:
                    obs_render = str(obs) if obs else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                if thought.is_error is None or i == total - 1:
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\n"
                        f"next_tool_name:{thought.next_tool_name}\n"
                        f"next_tool_args:{thought.next_tool_args}"
                    )
                    obs = thought.observation
                    if isinstance(obs, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(obs), ensure_ascii=False)
                        except Exception:
                            obs_render = str(obs)
                    else:
                        obs_render = str(obs)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if obs is None:
                            obs_len = 0
                        elif isinstance(obs, (list, tuple)):
                            obs_len = len(obs)
                        else:
                            obs_len = len(str(obs).splitlines())
                        user_str = f"observation: error occurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if isinstance(obs, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(obs), ensure_ascii=False)
                            except Exception:
                                obs_render = str(obs)
                        else:
                            obs_render = str(obs)
                        user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
        return messages

    def _get_summary_for_index(self, idx):
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None

    def count_repeated_thoughts(self) -> int:
        if len(self.thoughts) < 2:
            return 0
        last_thought = self.thoughts[-1]
        last_tool_name = last_thought.next_tool_name
        last_tool_args = last_thought.next_tool_args
        count = 0
        for i in range(len(self.thoughts) - 1, -1, -1):
            thought = self.thoughts[i]
            if thought.next_tool_name == last_tool_name and thought.next_tool_args == last_tool_args:
                count += 1
            else:
                break
        return max(0, count - 1)

    def is_thought_repeated(self):
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False

class Utils:
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re

        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages

        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)
        return count

    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)

class ChangedImpactAnalyzer:
    """
    Estimate change impact for a given symbol using lightweight repo scans and an LLM decision.
    """

    def __init__(self, *, code_parser: "CodeParseUtil" | None = None):
        self.code_parser = code_parser

    def _run(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        # Run inside repo_root so callers don't need to prefix commands with "cd ... &&".
        return subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _collect_examples(self, symbol: str, *, limit: int = 10) -> list[dict[str, Any]]:
        q = shlex.quote(symbol)
        # Keep it bounded to avoid huge observations.
        cmd = f"grep -rn --binary-files=without-match {q} . | head -{int(limit)}"
        try:
            p = self._run(cmd)
        except Exception:
            return []
        out = (p.stdout or "").strip()
        if not out:
            return []
        examples: list[dict[str, Any]] = []
        for line in out.splitlines():
            # Expected: path:lineno:content
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            path, ln, ctx = parts
            try:
                ln_i = int(ln)
            except Exception:
                ln_i = None
            examples.append({"path": path, "line": ln_i, "context": ctx.strip()[:300]})
        return examples

    def _estimate_counts(self, symbol: str) -> tuple[int, int]:
        q = shlex.quote(symbol)
        # total references (approx): count of matching lines
        cmd_total = f"grep -rn --binary-files=without-match {q} . 2>/dev/null | wc -l"
        # distinct files (approx)
        cmd_files = f"grep -rl --binary-files=without-match {q} . 2>/dev/null | wc -l"
        try:
            p1 = self._run(cmd_total)
            p2 = self._run(cmd_files)
            total = int((p1.stdout or "0").strip() or "0")
            files = int((p2.stdout or "0").strip() or "0")
            return max(0, total), max(0, files)
        except Exception:
            return 0, 0

    def _semantic_signals(self, *, file_path: str, symbol_name: str) -> dict[str, Any]:
        """
        Best-effort semantic context for blast-radius review.
        This is evidence-only (excerpt), not keyword heuristics or a strategy decision.
        """
        signals: dict[str, Any] = {
            "symbol_name": symbol_name,
            "file_path": file_path,
            "function_body_excerpt": "",
            # Kept for schema stability; do not populate via keyword heuristics.
            "signals": [],
        }
        if not self.code_parser:
            return signals
        try:
            body = self.code_parser.get_function_body(file_path, symbol_name, add_line_numbers=False) or ""
        except Exception:
            body = ""
        if not body:
            return signals
        # Keep excerpt bounded.
        excerpt = body if len(body) <= 2500 else (body[:1200] + "\n... [truncated] ...\n" + body[-900:])
        signals["function_body_excerpt"] = excerpt
        return signals

    def _llm_decide(
        self,
        *,
        symbol_name: str,
        distinct_files: int,
        total_refs: int,
        examples: list[dict[str, Any]],
        semantic_signals: dict[str, Any],
    ) -> dict[str, Any]:
        # Keep prompt abstraction-only: no repo/tool/framework specific instructions.
        prompt = textwrap.dedent(
            f"""
            You are helping decide how risky a change is, and whether a strict boundary-local proof is required before editing.

            Inputs:
            - symbol_name: {symbol_name}
            - distinct_files_estimate: {distinct_files}
            - reference_count_estimate: {total_refs}
            - example_references: {json.dumps(examples[: min(8, len(examples))], ensure_ascii=False)}
            - semantic_signals: {json.dumps(semantic_signals, ensure_ascii=False)}

            Task:
            Choose exactly one decision from:
            - "focused_ok": Low impact; direct change likely safe if invariants are preserved.
            - "prefer_focused_change": Moderate impact or uncertainty; default to localized change.
            - "broad_change_risky": High impact; broad change likely to cause regressions; prefer localized change.

            Also decide if the symbol likely has multiple interpretations/variants (a discriminator or multi-variant builder)
            such that editing it requires an explicit "boundary-localization proof" BEFORE any edits.

            Rules:
            - If uncertain, choose "prefer_focused_change".
            - If the excerpt/examples suggest multi-variant behavior (a discriminator, multiple interpretation branches), set requires_boundary_proof=true.
            - Output MUST be a single JSON object only (no markdown, no extra text).

            Output schema:
            {{
              "decision": "focused_ok|prefer_focused_change|broad_change_risky",
              "requires_boundary_proof": true/false,
              "why": ["short bullet 1", "short bullet 2"]
            }}
            """
        ).strip()

        messages = [{"role": "user", "content": prompt}]

        retry = 0
        selected_model = QWEN_MODEL_NAME
        max_retries = 10

        while retry < max_retries:
            try:
                raw, _ = EnhancedNetwork.make_request(messages, model=selected_model, attempt=1, temperature=0.0)

                cleaned = raw.strip()
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

                try:
                    obj = json.loads(cleaned)
                    if not isinstance(obj, dict):
                        raise ValueError("not a JSON object")

                    obj.setdefault("decision", "prefer_focused_change")
                    obj.setdefault("requires_boundary_proof", True)
                    obj.setdefault("why", [])

                    return obj
                except (json.JSONDecodeError, ValueError):
                    json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', cleaned, re.DOTALL)
                    if json_match:
                        try:
                            obj = json.loads(json_match.group())
                            if isinstance(obj, dict):
                                obj.setdefault("decision", "prefer_focused_change")
                                obj.setdefault("requires_boundary_proof", True)
                                obj.setdefault("why", [])
                                return obj
                        except json.JSONDecodeError:
                            pass

                    # If JSON extraction failed, log and retry
                    logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {cleaned[:500]}")

            except Exception as e:
                logger.warning(f"Error in _llm_decide (attempt {retry + 1}/{max_retries}): {e}")

            retry += 1
            if retry < max_retries:
                # Try different model on retry (after 7 attempts)
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)  # Small delay before retry

        # Conservative fallback if all retries failed.
        logger.warning("Failed to get LLM decision after all retries, using conservative fallback")
        return {
            "decision": "prefer_focused_change",
            "requires_boundary_proof": True,
            "why": ["LLM decision failed after all retries", "Conservative fallback: require boundary proof"],
        }

    def analyze_symbol(self, *, file_path: str, symbol_name: str) -> dict[str, Any]:
        sym = (symbol_name or "").strip()
        if not sym:
            return {"error": "missing_symbol", "message": "symbol_name is required."}
        total_refs, distinct_files = self._estimate_counts(sym)
        examples = self._collect_examples(sym, limit=10)
        semantic_signals = self._semantic_signals(file_path=file_path, symbol_name=sym)
        decision_obj = self._llm_decide(
            symbol_name=sym,
            distinct_files=distinct_files,
            total_refs=total_refs,
            examples=examples,
            semantic_signals=semantic_signals,
        )
        decision = decision_obj.get("decision", "prefer_focused_change")
        requires_boundary_proof = bool(decision_obj.get("requires_boundary_proof", True))
        why = decision_obj.get("why", [])
        if not isinstance(why, list):
            why = [str(why)]
        return {
            "symbol_name": sym,
            "fanout": {
                "reference_count_estimate": total_refs,
                "distinct_files_estimate": distinct_files,
                "example_references": examples,
            },
            "semantic_signals": semantic_signals,
            "decision": decision,
            "requires_boundary_proof": requires_boundary_proof,
            "decision_why": why[:4],
            "file_path": file_path,
        }

class ProblemDecomposer:
    """
    Analyzes problem statements to extract structured information for guided debugging.
    This preprocessing step helps the agent understand the problem before diving into code.
    """

    def __init__(self):
        self.decomposition_cache = {}

    def decompose(self, problem_statement: str) -> dict:
        """
        Analyze a problem statement and return structured decomposition.
        """
        cache_key = hash(problem_statement[:500])
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]

        truncated_problem = problem_statement
        if len(problem_statement) > 8000:
            truncated_problem = problem_statement[:4000] + "\n\n[...truncated...]\n\n" + problem_statement[-4000:]

        messages = [
            {"role": "system", "content": PROBLEM_DECOMPOSITION_PROMPT},
            {"role": "user", "content": f"Analyze this problem:\n\n{truncated_problem}"},
        ]

        result = self._default_decomposition()

        for attempt in range(10):
            try:
                response, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
                parsed = self._parse_response(response)
                if parsed:
                    result = parsed
                    break
            except Exception:
                time.sleep(1)
                continue

        self.decomposition_cache[cache_key] = result
        return result

    def _parse_response(self, response: str) -> dict | None:
        """Extract and parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        return None

    def _default_decomposition(self) -> dict:
        """Return a default decomposition structure when parsing fails."""
        return {
            "problem_summary": "",
            "key_entities": {"files": [], "functions": [], "classes": [], "error_messages": [], "other": []},
            "behavior": {"expected": "Not specified", "actual": "Not specified", "trigger": "Not specified"},
            "success_criteria": [],
            "investigation_starting_points": [],
        }

    def format_for_prompt(self, decomposition: dict) -> str:
        """Format the decomposition as a readable string for injection into prompts."""
        sections = []

        if decomposition.get("problem_summary"):
            sections.append(f"**Problem Summary**: {decomposition['problem_summary']}")

        entities = decomposition.get("key_entities", {})
        entity_parts = []
        if entities.get("files"):
            entity_parts.append(f"  - Files: {', '.join(entities['files'][:5])}")
        if entities.get("functions"):
            entity_parts.append(f"  - Functions: {', '.join(entities['functions'][:5])}")
        if entities.get("classes"):
            entity_parts.append(f"  - Classes: {', '.join(entities['classes'][:5])}")
        if entities.get("error_messages"):
            for msg in entities["error_messages"][:2]:
                entity_parts.append(f"  - Error: `{msg[:100]}`")
        if entity_parts:
            sections.append("**Key Entities**:\n" + "\n".join(entity_parts))

        behavior = decomposition.get("behavior", {})
        if behavior.get("expected") != "Not specified" or behavior.get("actual") != "Not specified":
            sections.append(
                f"**Behavior**:\n"
                f"  - Expected: {behavior.get('expected', 'N/A')}\n"
                f"  - Actual: {behavior.get('actual', 'N/A')}\n"
                f"  - Trigger: {behavior.get('trigger', 'N/A')}"
            )

        if decomposition.get("success_criteria"):
            criteria = "\n".join(f"  - {c}" for c in decomposition["success_criteria"][:3])
            sections.append(f"**Success Criteria**:\n{criteria}")

        if decomposition.get("investigation_starting_points"):
            points = []
            for point in decomposition["investigation_starting_points"][:4]:
                if isinstance(point, dict):
                    points.append(f"  - {point.get('location', 'N/A')}: {point.get('reason', '')}")
                else:
                    points.append(f"  - {point}")
            sections.append(f"**Suggested Starting Points**:\n" + "\n".join(points))

        return "\n\n".join(sections)

_problem_decomposer = ProblemDecomposer()

class SolutionVerifier:
    """
    Verifies that the solution fixes the original bug and doesn't introduce regressions.
    Renamed from RegressionVerifier to SolutionVerifier for clarity.
    """

    def __init__(self, cot: "EnhancedCOT" = None, problem_statement: str = None):
        self.cot = cot
        self.problem_statement = problem_statement

    def verify_solution(self) -> str:
        """
        Uses LLM to analyze the conversation history and verify that the agent has:
        1. Fixed the ORIGINAL BUG (hidden tests that were failing are now passing)
        2. Fixed ALL REGRESSIONS (tests that were passing are still passing)
        3. Run comprehensive tests (not just one or several specific test cases)
        4. Not rationalized away any failures

        Returns feedback to the agent explaining:
        - If BOTH original bug AND regressions are fixed → Returns "REGRESSION_AND_BUG_CHECK_PASSED"
        - If issues found → Returns detailed feedback about what needs to be fixed

        CRITICAL: Agent cannot finish unless BOTH conditions are met:
        - Original bug is fixed (hidden tests passing)
        - No regressions introduced (all previously passing tests still pass)
        """
        # Get conversation history and problem statement
        conversation_history = self.cot.to_str() if self.cot else "No conversation history available"
        problem_statement = self.problem_statement or "No problem statement available"

        # Build the regression verification prompt
        regression_check_prompt = textwrap.dedent(
            """
            You are a rigorous QA reviewer checking if an agent has properly fixed BOTH the original bug AND all regressions before finishing.
            
            **PROBLEM STATEMENT (Original Bug Description)**:
            
            {problem_statement}
            
            **Your job**: Analyze the agent's conversation history and verify TWO critical conditions:
            
            1. **NO REGRESSIONS INTRODUCED** - All tests that were passing before changes are still passing
            2. **ORIGINAL BUG IS FIXED** - The hidden tests that were originally failing are now passing
            
            **CRITICAL**: Agent CANNOT finish unless BOTH conditions are met. A solution that fixes the bug but breaks other tests is NOT acceptable. A solution that fixes regressions but re-introduces the original bug is also NOT acceptable.
            
            **CRITICAL FAILURE PATTERNS TO DETECT**:
            
            1. **Selective Test Running** - Agent ran only 1-2 specific test cases instead of the full test suite
               - Example: Agent saw N test failures, but only ran test_case_1 which passed, then called finish
               - Example: Agent found N tests but only ran specific test names instead of full suite
               - Red flag: Agent cherry-picked individual passing test methods
            
            2. **Ignoring Test Failures** - Agent saw test failures but didn't fix them
               - Example: Agent saw test_case_A failed, test_case_B failed, test_case_C failed but never fixed and re-ran
               - Example: Test output shows failed tests (e.g. FAIL, failed, test failed) but agent never fixed them
               - Red flag: Agent acknowledges seeing failures but doesn't address them
            
            3. **Rationalization** - Agent explained away failures as "unrelated" or "acceptable"
               - Example: Agent saw N test failures but thought "not related to problem statement, so bug is fixed" then finish
               - Example: Agent claimed "failing tests are edge cases" or "seem unrelated to my fix"
               - Example: Agent claimed "test failures existed before my changes" without verification
               - Red flag: Any justification for why failing tests are "OK" or "ignorable"
            
            4. **Problem Statement Excuse** - Agent claims failures are unrelated to the problem statement
               - Example: Agent saw N test failures but thought "they are not related to problem_statement, so bug is fixed" then finish
               - Red flag: Agent dismisses regressions by claiming they're outside the scope of the fix
               - Critical: ALL tests that were passing before changes but failing after are regressions, regardless of problem statement
            
            5. **No Full Suite Run** - Agent never ran the full test suite for the affected module
               - Red flag: Only ran individual test methods, never the full module/class
               - Example: Modified a utility function but only tested one caller, not all callers
            
            6. **Custom Scripts Instead of Real Tests** - Agent relied on `run_code` demos instead of actual test suite
               - Red flag: Multiple `run_code` calls with demo scripts, but no `run_tests` with full suite
               - Example: Created verification scripts instead of running project's test suite
            
            7. **Returned Bug (Critical)** - Agent fixed regressions but re-introduced the original bug
               - Example: Agent fixed regressions but then the hidden tests (originally failing) are failing again
               - Red flag: Agent focuses only on fixing regressions and forgets to verify the original bug is still fixed
               - Critical: BOTH original bug AND regressions must be fixed simultaneously
            
            **WHAT CONSTITUTES COMPLETE SUCCESS (BOTH CONDITIONS REQUIRED)**:
            
            ✅ **CONDITION 1: ORIGINAL BUG IS FIXED**
               - The hidden tests mentioned in the problem statement are now passing
               - Agent verified the fix with actual test runs (not just theory)
               - The bug described in the problem is demonstrably resolved
            
            ✅ **CONDITION 2: NO REGRESSIONS**
               - Agent ran the FULL test suite (or at minimum, the full test class) for affected modules
               - Agent saw test failures and FIXED them (re-ran tests after fixes until they passed)
               - The FINAL test run before calling finish showed ALL tests passing (no failed test output)
               - Agent used `run_tests` with the project's test runner, not just `run_code` demos
               - Agent fixed ALL regressions, regardless of whether they seem "related" to the problem statement
               - A regression is ANY test that was passing before changes but failing after
            
            ✅ **CRITICAL VERIFICATION**:
               - Agent must verify BOTH conditions are true in the SAME final test run
               - Cannot assume "bug is fixed" if only regressions are resolved
               - Cannot assume "regressions are fixed" if only the original bug is resolved
               - BOTH must be verified together in the final test output
            
            **YOUR TASK**:
            
            Analyze the conversation history below and verify BOTH conditions:
            
            **CONDITION 1 CHECK - Original Bug Fixed?**
            1. Are the hidden tests (originally failing, mentioned in problem statement) now passing?
            2. Did the agent verify the bug fix with actual test runs?
            3. Is there evidence the original problem is resolved?
            
            **CONDITION 2 CHECK - No Regressions?**
            4. Did the agent run comprehensive regression tests?
            5. Are there any unresolved test failures?
            6. Did the agent rationalize failures away (including "not related to problem statement")?
            7. Did the agent only test specific cases instead of the full suite?
            8. Did the agent use demo scripts instead of the real test suite?
            
            **CRITICAL CHECK - Returned Bug?**
            9. Did the agent fix regressions but then the original bug came back?
            10. Did the final test run verify BOTH original bug fix AND no regressions?
            
            **YOUR RESPONSE FORMAT**:
            
            - **IF BOTH CONDITIONS MET** (original bug fixed AND no regressions): Return exactly "REGRESSION_AND_BUG_CHECK_PASSED" followed by a brief explanation
            - **IF ANY ISSUES FOUND**: Return detailed feedback explaining:
              * Which condition failed (original bug, regressions, or both)
              * What specific evidence shows the failure
              * What the agent must do to fix it
            
            **CONVERSATION HISTORY TO ANALYZE**:
            
            {conversation_history}
            
            **YOUR RESPONSE**:
        """
        ).strip()

        # Call LLM to analyze regression testing
        messages = [
            {"role": "system", "content": "You are a rigorous QA reviewer checking for proper regression testing. Be strict and thorough."},
            {
                "role": "user",
                "content": regression_check_prompt.format(problem_statement=problem_statement, conversation_history=conversation_history),
            },
        ]

        retry = 0
        selected_model = QWEN_MODEL_NAME
        max_retries = 10

        while retry < max_retries:
            try:
                review_result, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
                return review_result.strip()
            except Exception as e:
                logger.warning(f"Error in verify_solution (attempt {retry + 1}/{max_retries}): {e}")

            retry += 1
            if retry < max_retries:
                # Try different model on retry (after 7 attempts)
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)  # Small delay before retry

        # If LLM call fails after all retries, return a conservative response requiring verification
        logger.warning("Failed to verify solution after all retries, returning conservative response")
        return f"⚠️ Regression verification LLM call failed after {max_retries} attempts.\n\nPlease manually verify that ALL regression tests pass before finishing."

class SearchManager:
    def search_in_file(self, file_path: str, search_term: str) -> str:
        def extract_matches(filepath, term, max_output_lines=1000):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                return f"Error reading '{filepath}': {e}"

            match_lines = [i + 1 for i, line in enumerate(lines) if term in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"

            context = 20
            seen = set()
            chunks = []
            for ln in match_lines:
                start = max(1, ln - context)
                end = min(len(lines), ln + context)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start - 1 : end]
                chunks.append(f"(lines {start}-{end}):\n" + "\n".join(chunk))
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

    def search_in_all_files(self, grep_search_command: str) -> str:
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(["bash", "-c", grep_search_command], capture_output=True, text=True, timeout=45)
        except Exception as e:
            return f"Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout

        if not output.strip():
            return "No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

class CodeParseUtil:
    """
    Code parsing utility using tree-sitter for language-aware code analysis.
    Supports extracting function bodies, skeleton structures, and detecting languages.
    """

    def __init__(self):
        self._parsers = {}

    def check_language(self, source: str, file_path: str | None = None) -> str | None:
        global _codeparse_util_language_cache
        if file_path and not os.path.exists(file_path) or not source or not source.strip():
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[file_path]
        stripped_source = source.strip()
        sample = (
            stripped_source
            if len(stripped_source) <= 1000
            else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
        )
        prompt = f"""Detect the programming language of the following code sample.
        Analyze the code and determine which programming language it is written in.
        Return ONLY the language name in lowercase.
        If you cannot determine the language, return "unknown".
        Code sample:
        ```
        {sample}
        ```
        Return ONLY the language name in lowercase, no other text or explanation."""
        retry = 0
        messages = [{"role": "user", "content": prompt}]
        models_to_try = [KIMI_MODEL_NAME, GLM_MODEL_NAME]
        while retry < 3:
            try:
                result, _ = EnhancedNetwork.make_request(
                    messages=messages, model=models_to_try[retry % len(models_to_try)], attempt=1, temperature=0.0
                )
                cleaned = result.strip().lower()
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
                cleaned = cleaned.strip('"').strip("'").strip()

                if cleaned and " " not in cleaned and cleaned.isalpha():
                    detected_language = cleaned if cleaned != "unknown" else None
                else:
                    retry += 1
                    if retry < 3:
                        messages.append({"role": "assistant", "content": result})
                        messages.append(
                            {"role": "user", "content": "Please return ONLY the language name as a single word in lowercase. No other text."}
                        )
                        time.sleep(1)
                    continue
                if file_path:
                    _codeparse_util_language_cache[file_path] = detected_language
                return detected_language
            except Exception as e:
                logger.warning(f"Error detecting language with LLM (attempt {retry + 1}/3): {e}")
                retry += 1
                if retry < 3:
                    time.sleep(1)
                continue
        return None

    def _is_identifier_node(self, node) -> bool:
        return "identifier" in node.type.lower()

    def _get_parser(self, language: str):
        if Parser is None or get_language is None:
            return None
        if language not in self._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None
                parser = Parser(lang_obj)
                self._parsers[language] = parser
            except Exception as e:
                logger.warning(f"Error creating parser for {language}: {e}")
                return None
        return self._parsers[language]

    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""
        if not source or Parser is None:
            return ""
        try:
            source_bytes, source_lines = bytes(source, "utf8"), source.splitlines()
            language = self.check_language(source, file_path=file_path)
            if not language:
                return ""
            parser = self._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = function_name, function_name.split(".")[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None:
                return ""
            start_idx, end_idx = func_info["start_line"] - 1, func_info["end_line"] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx : end_idx + 1]
                return "\n".join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines)) if add_line_numbers else "\n".join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""

    def _classify_node_type(self, node) -> tuple[str, int | None]:
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        return ("other", None)

    def _find_specific_function(
        self, node, source_lines: list[str], target_qualified: str, target_simple: str, class_name: str = "", parent_node=None
    ) -> dict | None:
        if not node.children:
            return None
        node_type, name_child_index = self._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = line[name_start[1] : name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1] :].strip()
            if not name and parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1] : name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1] :].strip()
                            if name:
                                break
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = self._find_specific_function(child, source_lines, target_qualified, target_simple, new_class_name, node)
                    if result is not None:
                        return result

        elif node_type == "function":
            name = internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = line[name_start[1] : name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1] :].strip()
            if parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1] : name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1] :].strip()
                            if name:
                                break
            if not name:
                name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = "." in target_qualified
                is_match = qualified_name == target_qualified or (not is_qualified_target and name == target_simple)
                if is_match:
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if source_lines[i].strip().startswith("@"):
                            at_start = i
                        elif source_lines[i].strip():
                            break
                    return {"start_line": at_start + 1, "end_line": node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None:
                    return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None:
                return result
        return None

class FileOperationsUtil:
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None

    def save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as file:
            file.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
    ) -> str:
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback,
        )

    def set_managers(self, file_system_manager, search_manager):
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager

class FileSystemManager:
    def __init__(self):
        pass

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        search_in_file_callback=None,
    ) -> str:

        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return "\n".join(numbered_lines)

        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content
        return Utils.limit_strings(result, n=limit) if limit != -1 else result

    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        ignore = {".git", "__pycache__", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", ".eggs"}

        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0) -> list[str]:
            if depth > current_max_depth:
                return []
            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {str(e)}]"]
            dirs = [
                i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".") and i not in ignore and not i.endswith(".egg-info")
            ]
            files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]
            lines: list[str] = []
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))
            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")
            return lines

        def count_tokens(text: str) -> int:
            try:
                if "Utils" in globals() and hasattr(Utils, "count_tokens"):
                    return Utils.count_tokens(text)
            except (NameError, AttributeError):
                pass
            return len(text) // 4

        MAX_TOKENS = 3000
        current_depth = max_depth
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)
            token_count = count_tokens(result)
            if token_count <= MAX_TOKENS:
                if current_depth < max_depth:
                    result += (
                        f"\n\n[Note: Requested depth {max_depth} exceeded token limit. Showing depth {current_depth} instead ({token_count} tokens).]"
                    )
                return result
            if current_depth == 0:
                result += f"\n\n[Warning: Result exceeds token limit ({token_count} tokens > {MAX_TOKENS} tokens). Consider using a more specific directory_path.]"
                return result
            current_depth -= 1
        entries = tree(directory_path, "", 0, 0)
        result = f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)
        return result

class TestManager:
    def run_code(self, content: str, file_path: str, generated_test_files: list, run_command: list[str]) -> str:
        if file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)
        self.file_ops.save(file_path, content)
        if file_path not in generated_test_files and not file_exists:
            generated_test_files.append(file_path)
        try:
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
            if result.returncode != 0:
                return f"Error running code: {result.stderr}"
            return f"{result.stdout}\n"
        except Exception as e:
            return f"Error: {e}"

    def __init__(self, runner_hint: str | None = None, runner_mode_hint: str | None = None, file_ops: "FileOperationsUtil" = None):
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.file_ops = file_ops

    @classmethod
    def llm_select_run_command_for_file(cls, file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        retry = 0
        while retry < 10:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"""
                I'd like you to respond with the command to run this file. Make your command as simple as possible.
                ```
                {file_path}
                {file_content}
                ```
                You must respond in JSON format:
                ```
                {{
                    "command": ["bbb", "aaa.js"]
                }}
                ```
                """,
                    }
                ]
                raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                json_result = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
                return json_result.get("command")
            except Exception as e:
                time.sleep(1)
                retry += 1
        return []

    @classmethod
    def is_all_tests_passed(cls, output: str) -> bool:
        check_all_tests_passed_prompt = """
        Check the test output and tell me if all the tests passed successfully or there is any failure or error.
        This is the output:
        ```
        {output}
        ```
        Return only "true" or "false".
        """
        retry = 1
        while retry < 10:
            try:
                result, _ = EnhancedNetwork.make_request(
                    messages=[
                        {
                            "role": "user",
                            "content": check_all_tests_passed_prompt.format(output=output),
                        }
                    ],
                    model=QWEN_MODEL_NAME,
                )

                if result.lower() == "true":
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"[IS_ALL_TESTS_PASSED] Exception: {e}")
                retry += 1
                time.sleep(1)
        return False

class EnhancedNetwork:
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        tool_name_match = re.search(r"tool_name\s*:\s*([^\s]+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip("\"'")
        args_match = re.search(r"tool_args\s*:\s*\{", block, re.IGNORECASE)
        if not args_match:
            return None
        args_start = args_match.end() - 1
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                try:
                    tool_args = json.loads(json_str.replace("'", '"'))
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except Exception:
                    pass
        return None

    @classmethod
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "HTTP ERROR: Request failed for model" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def inference(
        cls,
        messages: list[dict],
        model: str,
        run_id: str = str(uuid4()),
        temperature: float = 0.0,
    ) -> dict:
        models = [model] if isinstance(model, str) else model
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"} and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        result = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        return result

    @classmethod
    def get_cost_usage(cls) -> dict:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id if run_id else str(uuid4())}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            usage_info = response.json()
            if isinstance(usage_info, dict):
                return usage_info
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception:
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}

    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if start_pos >= len(text):
            return None
        brace_count, in_string, escape_next, start = 0, False, False, -1
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start : i + 1]
        return None

    @classmethod
    def _request_next_action_with_retry(
        cls,
        messages: dict,
        models: list[str],
        max_retries: int = 10,
        temperature: float = 0.0,
    ) -> str:
        raw_text = None
        error_counter = cls.get_error_counter()
        next_thought = next_tool_name = next_tool_args = None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else None
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = models[min(current_model_idx, len(models) - 1)]
                used_model = current_model
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                if is_504_error and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    matched = False
                    for key in ["RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE", "TIMEOUT", "Invalid JSON", "Invalid response"]:
                        if key in error_body:
                            attr_name = key if key in cls.ErrorType.__members__ else "INVALID_RESPONSE_FORMAT"
                            error_counter[attr_name] += 1
                            matched = True
                            break
                    if not matched:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    skip_http = any(
                        x in error_body
                        for x in [
                            "HTTP ERROR",
                            "RATE_LIMIT_EXCEEDED",
                            "RESERVED_TOKEN_PRESENT",
                            "EMPTY_RESPONSE",
                            "TIMEOUT",
                            "NETWORK_ERROR",
                            "HTTP ERROR 429",
                            "INCOMPLETE_RESPONSE",
                        ]
                    )
                    if not skip_http:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(3)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = r",\s*".join(rf'"{k}": (.*)' for k in arguments)
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        return {k: match.group(i + 1).strip().strip('"').replace("\\n", "\n") for i, k in enumerate(arguments)}

    @classmethod
    def make_request(
        cls,
        messages: list,
        model: Model,
        attempt: int = 0,
        temperature: float = 0.0,
        tool_mode: str = "none",
        tool_docs: list = [],
        timeout: int = 150,
    ) -> tuple[str, list]:
        global run_id, agent_start_time, total_inferenced_chars, individual_inferenced_chars
        messages_str = json.dumps(messages, ensure_ascii=False)
        individual_inferenced_chars = len(messages_str)
        total_inferenced_chars += individual_inferenced_chars

        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        attempts = max(1, attempt or 1)
        model_name = model.name if isinstance(model, Model) else model
        model_timeout = timeout

        request_data = {
            "evaluation_run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "model": model_name,
            "tool_mode": tool_mode,
            "tools": tool_docs,
        }
        headers = {"Content-Type": "application/json"}
        for i in range(attempts):
            try:
                start_time = time.time()
                print(f"⏳ Sending request using {model_name} and {model_timeout} seconds timeout")
                resp = requests.post(url, json=request_data, timeout=(30, model_timeout), headers=headers)
                resp.raise_for_status()
                print(f"✔ Request success using {model_name} and {time.time() - start_time:.2f} seconds elapsed!")
                try:
                    resp_json = resp.json()
                except JSONDecodeError as e:
                    if i >= attempts - 1:
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model_name} after {attempts} attempts: {e}")
                    continue
                try:
                    raw_text = resp_json["content"]
                    tool_calls = resp_json["tool_calls"]
                except Exception:
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error timeout for model {model_name} after {attempts} attempts")
                if (tool_mode == "none" and not raw_text) or (tool_mode != "none" and not tool_calls):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND Tool model {model_name} after {attempts} attempts")
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model_name} after {attempts} attempts")
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model_name} after {attempts} attempts: {e}")
                    time.sleep(1)
                    continue
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model_name}"
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model_name} after {attempts} attempts")

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub(r"['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub(r"['\"]*tool_call_['\"]*", "tool_call_", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("'").strip('"').strip()
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{re.escape(next_tool_name)}['\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp,
            )
        return text_resp

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        try:
            return Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            try:
                schema_tool_name = tool_name[0] if isinstance(tool_name, list) and tool_name else tool_name
                return cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(schema_tool_name, required_only=True),
                    next_tool_args,
                )
            except (EnhancedToolManager.Error, Exception):
                raise Exception(f"Invalid JSON: {next_tool_args}")

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else.",
            },
            {"role": "user", "content": json_string},
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        while retry < 5:
            try:
                response, _ = cls.make_request(messages, model=selected_model)
                break
            except Exception:
                retry += 1
                remaining = [model for model in AGENT_MODELS if model != selected_model]
                if remaining:
                    selected_model = random.choice(remaining)
                time.sleep(1)
        try:
            response = response.replace("```json", "").strip("```")
            return json.loads(response)
        except Exception:
            return None

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, any, any, str | None]:
        error_msg = None
        text_resp = text_resp.strip()
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        for pat in [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)",
        ]:
            match = re.search(pat, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 2:
                    next_thought = candidate
                    break
        if not next_thought:
            next_thought = "Processing request"
        tool_call_matches = list(re.finditer(r"tool_call_(\d+)\s*:", text_resp, re.IGNORECASE))
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                return next_thought, None, None, "Multi-tool format detected but no valid tool calls extracted"
            tool_names = [c["tool_name"] for c in tool_calls]
            tool_args_list = [c["tool_args"] for c in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            return next_thought, tool_names, tool_args_list, error_msg

        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            name_idx = text_resp.find("next_tool_name:")
            args_idx = text_resp.find("next_tool_args:")
            if text_resp.find("next_thought:") < name_idx < args_idx:
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip()
                next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip()
                try:
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
                    next_tool_args_list = parsed_args if isinstance(parsed_args, list) else [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    return next_thought, next_tool_names, next_tool_args_list, error_msg
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    return next_thought, None, None, error_msg

        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found (expected next_tool_name: or tool_call_N:)"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        return next_thought, None, None, error_msg

    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str | None]:
        if isinstance(raw_text, dict) and raw_text.get("error"):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        lower = raw_text.lower()
        has_next_thought = "next_thought" in lower or "<next_thought>" in lower
        has_next_tool_name = "next_tool_name" in lower or "<next_tool_name>" in lower
        has_next_tool_args = "next_tool_args" in lower or "<next_tool_args>" in lower
        valid_ending = stripped.endswith("}") or stripped.endswith("}]") or stripped.endswith("</next_tool_args>") or stripped.endswith(">")
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if not raw_text:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        return cls.is_http_response(raw_text)

    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        INCOMPLETE_RESPONSE = 10

class CodeEditManager:
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            lines = original_content.split("\n")
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            best_match_start = -1
            best_similarity = 0
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i : i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                import difflib

                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            return f"{description}\n" + "\n".join(context_lines_list)

        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
            import difflib

            lines = original_content.split("\n")
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:
                    similarities.append((ratio, chunk_desc, chunk_content))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        original = self.file_ops.get_file_content(file_path, limit=-1)
        match original.count(search):
            case 0:
                similar_matches = find_most_similar_content(original, search, 1)
                error_msg = f"Error: search string not found in file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."
                return error_msg
            case 1:
                new_content = original.replace(search, replace)
                try:
                    self.file_ops.save(file_path, new_content)

                    replace_pos = new_content.find(replace)
                    if replace_pos != -1:
                        lines = new_content.split("\n")
                        chars_so_far = 0
                        replace_line_start = 0
                        for i, line in enumerate(lines):
                            if chars_so_far + len(line) >= replace_pos:
                                replace_line_start = i
                                break
                            chars_so_far += len(line) + 1  # +1 for newline
                        replace_lines_count = replace.count("\n") + 1
                        replace_line_end = replace_line_start + replace_lines_count - 1
                        start_line = max(0, replace_line_start - 10)
                        end_line = min(len(lines), replace_line_start + 10)
                        context_lines = []
                        for i in range(start_line, end_line):
                            line_num = i + 1
                            if replace_line_start <= i <= replace_line_end:
                                prefix = ">>> "
                            else:
                                prefix = "    "
                            context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                        context = "\n".join(context_lines)
                        return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                    else:
                        return "ok, code edit applied successfully"
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."

    def __init__(self, file_ops: "FileOperationsUtil" = None):
        self.file_ops = file_ops

class EnhancedToolManager:
    TOOL_LIST = {}

    def get_tool_docs(self) -> str:
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])

    def __init__(self, **kwargs):
        pass

    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description,
                }
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description,
            }
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters,
        }
        return tool_schemas

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j: 0 for j in self.Error.ErrorType.__members__}
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method

    @classmethod
    def get_final_git_patch(cls) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @classmethod
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in cls.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(cls.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return cls.TOOL_LIST[tool_name]["input_schema"]["required"]

    @classmethod
    def get_modified_files_list(cls) -> list[str]:
        """
        Get a list of modified files (not newly created) from the git repository.
        Files that exist in the original repository and have been modified.

        Returns:
            List of file paths relative to repository root, excluding:
            - Newly created files (not in original repo)
            - Agent files (src/agent.py, src/agent_runner.py)
            - Generated test files
        """
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Get modified files (M = modified, not including Added or Deleted)
            # This compares against HEAD (original repository state)
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=M", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,  # Don't fail if no modified files or HEAD doesn't exist
            )

            if result.returncode != 0:
                # If HEAD doesn't exist or other error, return empty list
                logger.warning(f"Git diff failed: {result.stderr}")
                return []

            modified_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]

            # Filter out excluded files
            modified_files = [f for f in modified_files if f not in exclude]

            final_list = []
            for file_path in modified_files:
                # Check if file is tracked in git (exists in HEAD)
                check_result = subprocess.run(["git", "ls-tree", "--name-only", "HEAD", file_path], capture_output=True, text=True, timeout=10)
                if check_result.returncode == 0 and check_result.stdout.strip():
                    final_list.append(file_path)

            return final_list
        except Exception as e:
            logger.warning(f"Error getting modified files list: {e}")
            return []

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message

class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(
        self,
        available_tools: Optional[list[str]] = [],
        runner_hint: str | None = None,
        runner_mode_hint: str | None = None,
        initial_checkpoint=None,
        problem_statement: str = None,
        should_review: bool = True,
        is_fix_task: bool = False,
        initial_structure: str = None,
        function_behaviours: dict = {},
        cot: "EnhancedCOT" = None,
        has_exception_handling_mention: bool = False,
    ):
        self.new_files_created = []
        self.available_tools = available_tools
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        self.observation_dir = ".observation"
        self.problem_statement = problem_statement
        self.initial_structure = initial_structure
        self.repo_dir = "."
        self.saved_observation_counter = 0
        self.is_fix_task = is_fix_task
        self.strategy_counter = 0
        self.strategies = []
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(
            runner_hint=runner_hint,
            runner_mode_hint=runner_mode_hint,
            file_ops=self.file_ops,
        )
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        self.code_parser = CodeParseUtil()
        self.thought_history: list[dict[str, Any]] = []
        self.branches: dict[str, list[dict[str, Any]]] = {}
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
        self.finish_called_count = 0
        self._current_step = 0
        self._cot_snapshot_cache = []
        self.validated_num = 0
        self._test_call_count = 0
        self._pending_run_tests_confirmation: bool = False
        self._last_run_tests_step: int | None = None
        self._last_run_tests_passed: bool | None = None
        self._last_edit_step: int | None = None
        self._edit_count: int = 0
        self._last_blocked_edit_step: int | None = None
        self._blocked_edit_count: int = 0
        self._last_blocked_edit_message: str | None = None
        self.cot = cot
        self.solution_verifier = SolutionVerifier(cot=cot, problem_statement=problem_statement) if cot else None
        self.problem_decomposition: Dict = None
        self.fix_strategy: Dict[str, Dict[str, Any]] = {}
        self.boundary_proofs: Dict[str, Dict[str, Any]] = {}
        self._last_pre_edit_warning_step: int | None = None
        self._pre_edit_warning_count: int = 0
        self._last_pre_edit_warning_message: str | None = None
        self.has_exception_handling_mention = has_exception_handling_mention

    def _has_recent_file_read(self, file_path: str, *, lookback: int = 40) -> bool:
        """
        Best-effort check: has the agent read this file recently via get_file_content/search_in_file/get_function_body?
        Uses the short cot snapshot cache maintained by execute_agent_workflow.
        """
        try:
            snap = getattr(self, "_cot_snapshot_cache", []) or []
            recent = snap[-lookback:]
            for item in reversed(recent):
                tool = str(item.get("tool", ""))
                args = str(item.get("args", ""))
                if tool in {"get_file_content", "search_in_file", "get_function_body"} and file_path in args:
                    return True
        except Exception:
            pass
        return False

    def _enforce_pre_edit_gates(self, *, file_path: str, search: str | None = None) -> str | None:
        """
        Returns an error message if editing must be blocked, else None.
        """
        if not self._has_recent_file_read(file_path):
            return textwrap.dedent(
                f"""
                ⚠️  PRE-EDIT WARNING: Missing Recent Context
                You are editing `{file_path}` without reading it recently.

                Recommended next step (before further edits):
                - Call get_file_content(file_path=..., search_term=... or line range) for the relevant region.
                """
            ).strip()

        strat = self.fix_strategy.get(file_path)
        if not strat:
            return textwrap.dedent(
                f"""
                ⚠️  PRE-EDIT WARNING: Missing Change-Impact Strategy
                You attempted to edit `{file_path}` without first generating a fix strategy based on
                change-impact analysis (fanout + semantic signals) for the target symbol.

                Required next step:
                - Call analyze_change_impact(file_path="{file_path}", symbol_name="...") for the symbol you intend to change.
                """
            ).strip()

        if bool(strat.get("requires_boundary_proof", False)):
            if file_path not in self.boundary_proofs:
                sym = strat.get("symbol_name", "")
                return textwrap.dedent(
                    f"""
                    ⚠️  PRE-EDIT WARNING: Missing Boundary-Localization Proof
                    The current strategy indicates the target symbol likely has multiple interpretations/variants.
                    Editing it requires an explicit boundary-localization proof BEFORE any edits.

                    Required next step:
                    - Call prove_boundary_localization(file_path="{file_path}", symbol_name="{sym}", planned_change_summary="...").
                    """
                ).strip()
        return None

     
    @EnhancedToolManager.tool
    def compare_with_working_version(self) -> str:
        """
        Compares the current broken code with the last known working version from git history.
        This tool helps identify what changed between working and broken states by showing diffs.
        It automatically finds the last commit and compares current state against it.
        This is a language-agnostic tool that works with any programming language.
        
        **IMPORTANT**: Use this tool only ONCE at the beginning of your investigation to understand
        what might have broken. Do not call it multiple times.
        
        Arguments:
            None
        Output:
            Detailed comparison showing differences between current state and last working version,
            or error message if comparison fails
        """
        try:
            check_repo = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            if check_repo.returncode != 0:
                return "Error: Not in a git repository. Cannot compare with working version."
            
            # Get the last commit (assumed to be the last working version)
            last_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            if last_commit.returncode != 0:
                return f"Error getting last commit: {last_commit.stderr}"
            
            commit_hash = last_commit.stdout.strip()
            
            # Get commit info for context
            commit_info = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H%n%an%n%ad%n%s", commit_hash],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            if commit_info.returncode == 0:
                info_lines = commit_info.stdout.strip().split('\n')
                if len(info_lines) >= 4:
                    commit_details = f"\n=== Last Known Working Version ===\n"
                    commit_details += f"Commit: {info_lines[0][:8]}\n"
                    commit_details += f"Author: {info_lines[1]}\n"
                    commit_details += f"Date: {info_lines[2]}\n"
                    commit_details += f"Message: {info_lines[3]}\n"
                    commit_details += "=" * 35 + "\n\n"
                else:
                    commit_details = f"\n=== Last Working Version: {commit_hash[:8]} ===\n\n"
            else:
                commit_details = f"\n=== Last Working Version: {commit_hash[:8]} ===\n\n"
            
            # Check if there are any uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            if not status_result.stdout.strip():
                return commit_details + "No differences found. Current working directory matches the last commit.\nThere are no uncommitted changes to compare."
            
            # Get the diff between HEAD and current working directory
            diff_result = subprocess.run(
                ["git", "diff", "HEAD", "--no-color", "--unified=5"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_dir
            )
            
            if diff_result.returncode != 0:
                return f"Error generating diff: {diff_result.stderr}"
            
            diff_output = diff_result.stdout
            
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            untracked_files = untracked_result.stdout.strip().split('\n') if untracked_result.stdout.strip() else []
            
            # Build comprehensive comparison report
            result = commit_details
            
            if diff_output:
                result += "=== Changes in Tracked Files ===\n\n"
                result += diff_output
                result += "\n"
            
            if untracked_files and untracked_files[0]:
                result += "\n=== New Untracked Files ===\n"
                for file in untracked_files:
                    result += f"  + {file}\n"
                result += "\n"
            
            # Get summary statistics
            stat_result = subprocess.run(
                ["git", "diff", "HEAD", "--stat", "--no-color"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_dir
            )
            
            if stat_result.returncode == 0 and stat_result.stdout.strip():
                result += "\n=== Summary of Changes ===\n"
                result += stat_result.stdout
                result += "\n"
            
            result += "\n✓ Comparison complete. Review the changes above to identify what might have caused the issue."
            return result
            
        except subprocess.TimeoutExpired:
            logger.error("Comparison timed out")
            return "Error: Comparison with working version timed out"
        except Exception as e:
            logger.error(f"Error comparing with working version: {str(e)}\n{traceback.format_exc()}")
            return f"Error comparing with working version: {str(e)}"



    def _is_soft_pre_edit_warning(self, msg: str) -> bool:
        try:
            return isinstance(msg, str) and msg.strip().startswith("⚠️  PRE-EDIT WARNING:")
        except Exception:
            return False

    def _record_pre_edit_warning(self, msg: str) -> None:
        try:
            self._last_pre_edit_warning_step = self._current_step
            self._pre_edit_warning_count = int(getattr(self, "_pre_edit_warning_count", 0)) + 1
            self._last_pre_edit_warning_message = msg
        except Exception:
            pass

    @EnhancedToolManager.tool
    def apply_bug_fix_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within bug file. Use this tool, not use apply_code_edit when you are going to modify the bug file.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        if self.is_fix_task:
            gate = self._enforce_pre_edit_gates(file_path=file_path, search=search)
            if gate:
                if self._is_soft_pre_edit_warning(gate):
                    self._record_pre_edit_warning(gate)
                    result = self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
                    try:
                        if isinstance(result, str) and "ok, code edit applied successfully" in result.lower():
                            self._last_edit_step = self._current_step
                            self._edit_count += 1
                    except Exception:
                        pass
                    return f"{gate}\n\n✅ Proceeded with edit (soft gate). Please run the required tool(s) next.\n\n{result}"
                try:
                    self._last_blocked_edit_step = self._current_step
                    self._blocked_edit_count += 1
                    self._last_blocked_edit_message = gate
                except Exception:
                    pass
                return gate
        result = self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
        try:
            if isinstance(result, str) and "ok, code edit applied successfully" in result.lower():
                self._last_edit_step = self._current_step
                self._edit_count += 1
        except Exception:
            pass
        return result

    
    @EnhancedToolManager.tool
    def profile_performance(self, file_path: str, profiling_command: List[str], baseline_command: Optional[List[str]] = None, num_runs: int = 3) -> str:
        """
        Profiles code execution to identify performance bottlenecks and optimization opportunities.
        Runs profiling tools to collect detailed performance data including execution time, resource usage, and hotspots.
        Supports comparing multiple runs and baseline comparisons to validate optimizations.
        
        Arguments:
            file_path: path to the code file to profile
            profiling_command: command to execute profiling (use language-appropriate profiling tools)
            baseline_command: optional baseline command to compare against (for before/after analysis)
            num_runs: number of times to run the profiling for statistical accuracy (default: 3)
        
        Output:
            structured profiling data with execution metrics and performance characteristics
        """
        import time
        import statistics
        try:
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist."
            
            output_parts = []
            output_parts.append(f"=== Performance Profiling Report ===")
            output_parts.append(f"Target file: {file_path}")
            output_parts.append(f"Number of runs: {num_runs}")
            output_parts.append("")
            
            def run_profiling(command, label):
                execution_times = []
                outputs = []
                
                for run_num in range(num_runs):
                    start_time = time.time()
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=os.getcwd()
                    )
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    outputs.append({
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    })
                
                # Calculate statistics
                avg_time = statistics.mean(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                stddev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                return {
                    'times': execution_times,
                    'avg': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'stddev': stddev_time,
                    'outputs': outputs
                }
            
            # Run main profiling
            output_parts.append(f"--- Profiling Command: {' '.join(profiling_command)} ---")
            main_results = run_profiling(profiling_command, "Main")
            
            output_parts.append(f"\nExecution Time Statistics:")
            output_parts.append(f"  Average: {main_results['avg']:.4f} seconds")
            output_parts.append(f"  Minimum: {main_results['min']:.4f} seconds")
            output_parts.append(f"  Maximum: {main_results['max']:.4f} seconds")
            output_parts.append(f"  Std Dev: {main_results['stddev']:.4f} seconds")
            
            last_output = main_results['outputs'][-1]
            output_parts.append(f"\nReturn Code: {last_output['returncode']}")
            
            if last_output['stdout']:
                output_parts.append(f"\n--- Profiling Output ---")
                output_parts.append(last_output['stdout'])
            
            if last_output['stderr']:
                output_parts.append(f"\n--- Diagnostic Output ---")
                output_parts.append(last_output['stderr'])
            
            # Run baseline comparison if provided
            if baseline_command:
                output_parts.append(f"\n\n--- Baseline Command: {' '.join(baseline_command)} ---")
                baseline_results = run_profiling(baseline_command, "Baseline")
                
                output_parts.append(f"\nBaseline Execution Time Statistics:")
                output_parts.append(f"  Average: {baseline_results['avg']:.4f} seconds")
                output_parts.append(f"  Minimum: {baseline_results['min']:.4f} seconds")
                output_parts.append(f"  Maximum: {baseline_results['max']:.4f} seconds")
                output_parts.append(f"  Std Dev: {baseline_results['stddev']:.4f} seconds")
                
                # Performance comparison
                speedup = baseline_results['avg'] / main_results['avg'] if main_results['avg'] > 0 else 0
                diff_seconds = baseline_results['avg'] - main_results['avg']
                diff_percent = ((baseline_results['avg'] - main_results['avg']) / baseline_results['avg'] * 100) if baseline_results['avg'] > 0 else 0
                
                output_parts.append(f"\n--- Performance Comparison ---")
                output_parts.append(f"  Time difference: {diff_seconds:+.4f} seconds ({diff_percent:+.2f}%)")
                output_parts.append(f"  Speedup factor: {speedup:.2f}x")
            
            # Add analysis prompts for the LLM
            output_parts.append(f"\n\n--- Analysis Guidelines ---")
            output_parts.append("Review the profiling output above and identify:")
            output_parts.append("1. Which functions or code sections appear most frequently or consume the most time")
            output_parts.append("2. Any patterns indicating inefficient algorithms or unnecessary operations")
            output_parts.append("3. Resource usage characteristics (memory allocations, I/O operations, etc.)")
            output_parts.append("4. Opportunities for optimization based on the bottlenecks discovered")
            if baseline_command:
                output_parts.append("5. Whether the optimizations achieved the intended performance improvement")
            
            return "\n".join(output_parts)
            
        except subprocess.TimeoutExpired:
            return f"Error: Profiling execution timed out after 120 seconds. The code may contain performance issues causing excessive runtime."
        except Exception as e:
            return f"Error during profiling: {str(e)}\n\nNote: Ensure the profiling_command uses appropriate profiling tools for the target language/runtime."



    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        result = self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
        try:
            if isinstance(result, str) and "ok, code edit applied successfully" in result.lower():
                self._last_edit_step = self._current_step
                self._edit_count += 1
        except Exception:
            pass
        return result

    @EnhancedToolManager.tool
    def modify_test_case(self, file_path: str, search: str, replace: str) -> str:
        """
        Modifies test files or test cases when they are incorrect or need correction.
        Use this tool when you identify that a test file or specific test case is wrong and needs to be fixed.
        This tool uses the same underlying mechanism as apply_code_edit but is specifically intended for correcting test files.
        Arguments:
            file_path: path to the test file that needs modification
            search: exact text pattern in the test file to locate and replace (e.g., the incorrect test case code)
            replace: corrected test case code to substitute
        Output:
            Operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)

    
    @EnhancedToolManager.tool
    def rank_fix_confidence(
        self,
        proposed_fix_description: str,
        modified_files: List[str] = None,
        test_results: str = None,
        implementation_notes: str = None,
    ) -> str:
        """
        Evaluates and scores the confidence level of a proposed fix solution.
        This tool helps assess whether your fix is likely to resolve the issue successfully
        by analyzing the fix approach, affected files, test outcomes, and implementation quality.

        Arguments:
            proposed_fix_description: Detailed description of what was fixed and how it addresses the root cause
            modified_files: List of file paths that were modified as part of the fix (optional)
            test_results: Output from test execution showing pass/fail status (optional)
            implementation_notes: Any additional context about the implementation, edge cases handled, or concerns (optional)

        Output:
            Structured confidence assessment including:
            - Confidence score (0-100)
            - Confidence level (very_low/low/medium/high/very_high)
            - Strengths of the proposed fix
            - Potential risks or weaknesses
            - Recommendations for improvement
        """
        code_context = ""
        max_files = 5
        max_tokens_per_file = 8000
        max_chars_for_large_file = 15000
        total_context_tokens = 0
        max_total_context_tokens = 20000

        if modified_files:
            for file_path in modified_files[:max_files]:
                if total_context_tokens >= max_total_context_tokens:
                    break

                try:
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        file_tokens = Utils.count_tokens(content)

                        if file_tokens > max_tokens_per_file:
                            content = content[:max_chars_for_large_file]
                            file_tokens = Utils.count_tokens(content)

                        if total_context_tokens + file_tokens > max_total_context_tokens:
                            remaining_tokens = max_total_context_tokens - total_context_tokens
                            if remaining_tokens > 1500:
                                remaining_chars = remaining_tokens * 3
                                content = content[:remaining_chars]
                                file_tokens = Utils.count_tokens(content)
                            else:
                                break

                        code_context += f"\n\n--- {file_path} ---\n{content}\n"
                        total_context_tokens += file_tokens
                except Exception as e:
                    pass

        RANKING_PROMPT = textwrap.dedent(f"""
        You are an expert code reviewer evaluating the quality and confidence level of a proposed bug fix.
        Analyze the fix comprehensively to determine how confident we should be that it will successfully resolve the issue.

        Proposed Fix Description:
        {proposed_fix_description}

        {f"Modified Files:{code_context}" if code_context else ""}

        {f"Test Results:\n{test_results}" if test_results else "No test results provided yet."}

        {f"Implementation Notes:\n{implementation_notes}" if implementation_notes else ""}

        Problem Statement Context:
        {self.problem_statement if self.problem_statement else "Not provided"}

        Evaluate the proposed fix considering:
        1. **Root Cause Alignment**: Does the fix address the actual root cause or just symptoms?
        2. **Completeness**: Are all affected areas covered? Any missing edge cases?
        3. **Test Coverage**: Do tests validate the fix? Are there sufficient test scenarios?
        4. **Code Quality**: Is the implementation clean, maintainable, and follows best practices?
        5. **Risk Assessment**: What could still go wrong? Any potential regressions?
        6. **Backward Compatibility**: Does it maintain compatibility with existing functionality?

        Return a JSON object with this structure:
        {{
            "confidence_score": <integer 0-100>,
            "confidence_level": "very_low|low|medium|high|very_high",
            "overall_assessment": "Brief summary of the fix quality",
            "strengths": ["Strength 1", "Strength 2", ...],
            "weaknesses": ["Weakness 1", "Weakness 2", ...],
            "risks": ["Risk 1", "Risk 2", ...],
            "recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "test_coverage_assessment": "Assessment of whether tests adequately validate the fix",
            "root_cause_alignment": "How well the fix addresses the actual root cause vs symptoms"
        }}

        Confidence Score Guide:
        - 90-100 (very_high): Comprehensive fix, well-tested, addresses root cause, minimal risk
        - 70-89 (high): Solid fix with good test coverage, minor concerns
        - 50-69 (medium): Reasonable fix but has gaps in testing or coverage
        - 30-49 (low): Fix may work but has significant concerns or gaps
        - 0-29 (very_low): Fix is incomplete, risky, or doesn't address root cause

        Be honest and thorough in your assessment.
        """).strip()

        retry = 0
        while retry < 3:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert code reviewer specializing in fix quality assessment. Provide structured, honest evaluation in JSON format.",
                    },
                    {"role": "user", "content": RANKING_PROMPT},
                ]
                response, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)

                cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                ranking = json.loads(cleaned)

                if not hasattr(self, "fix_confidence_rankings"):
                    self.fix_confidence_rankings = []
                ranking["timestamp"] = time.time()
                ranking["step"] = len(getattr(self, "tool_invocations", {}))
                self.fix_confidence_rankings.append(ranking)

                output = ["=== FIX CONFIDENCE RANKING ===\n"]
                output.append(
                    f"Confidence Score: {ranking.get('confidence_score', 'N/A')}/100 ({ranking.get('confidence_level', 'unknown').upper()})"
                )
                output.append(f"\nOverall Assessment:\n{ranking.get('overall_assessment', 'Not provided')}")

                if ranking.get("strengths"):
                    output.append(
                        f"\nStrengths:\n" + "\n".join(f"  ✓ {strength}" for strength in ranking["strengths"])
                    )

                if ranking.get("weaknesses"):
                    output.append(
                        f"\nWeaknesses:\n" + "\n".join(f"  ⚠ {weakness}" for weakness in ranking["weaknesses"])
                    )

                if ranking.get("risks"):
                    output.append(f"\nRisks:\n" + "\n".join(f"  ⚠ {risk}" for risk in ranking["risks"]))

                if ranking.get("recommendations"):
                    output.append(
                        f"\nRecommendations:\n"
                        + "\n".join(f"  → {rec}" for rec in ranking["recommendations"])
                    )

                if ranking.get("test_coverage_assessment"):
                    output.append(f"\nTest Coverage Assessment:\n{ranking['test_coverage_assessment']}")

                if ranking.get("root_cause_alignment"):
                    output.append(f"\nRoot Cause Alignment:\n{ranking['root_cause_alignment']}")

                confidence_score = ranking.get("confidence_score", 0)
                if confidence_score >= 70:
                    output.append(
                        "\n✓ High confidence - this fix appears solid. Consider running final verification tests."
                    )
                elif confidence_score >= 50:
                    output.append(
                        "\n⚠ Medium confidence - the fix is reasonable but consider addressing the weaknesses mentioned above."
                    )
                else:
                    output.append(
                        "\n✗ Low confidence - significant concerns exist. Review the risks and recommendations before proceeding."
                    )

                return "\n".join(output)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                retry += 1
                if retry < 3:
                    time.sleep(2)
                else:
                    return f"Error: Failed to parse fix confidence ranking after 3 attempts. Raw response: {response[:500]}"
            except Exception as e:
                retry += 1
                if retry < 3:
                    time.sleep(2)
                else:
                    return f"Error: Failed to rank fix confidence: {str(e)}"

        return "Error: Failed to complete fix confidence ranking"


    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str, run_command: List[str]) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
            run_command=run_command,
        )

    @EnhancedToolManager.tool
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
    ) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(
            file_path,
            search_start_line,
            search_end_line,
            search_term,
            add_line_numbers=True,
            limit=1000,
        )

    @EnhancedToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(directory_path=directory_path, max_depth=max_depth)

    @EnhancedToolManager.tool
    def analyze_edge_cases(self, file_contents: Dict[str, str], target_identifier: str = None) -> str:
        """
        General-purpose tool to identify edge cases, boundary conditions, and implicit requirements by analyzing provided code and test content.
        Works with any available information sources without hardcoding specific sources.

        Use this tool when:
        - Tests fail but logic seems correct
        - Need to identify implicit requirements not explicitly stated
        - Want to discover edge cases (empty input, boundaries, null handling)
        - Formatting issues cause failures
        - Type or structure requirements are unclear

        Arguments:
            file_contents: Required dictionary mapping file paths/identifiers to their content (use get_file_content tool first to get file contents)
            target_identifier: Optional identifier to focus analysis (function name, class name, or pattern)

        Output:
            Comprehensive analysis identifying:
            - Edge cases and boundary conditions
            - Implicit format/structure requirements
            - Type handling requirements
            - Error handling expectations
            - Missing or ambiguous requirements
        """
        logger.info(f"🔍 [EDGE_CASE] Analyzing edge cases for: {target_identifier or 'general codebase'}")

        if not file_contents or not isinstance(file_contents, dict):
            return "Error: file_contents parameter is required and must be a dictionary mapping file identifiers to content. Use get_file_content tool first to get file contents."

        if not file_contents:
            return "Error: file_contents dictionary is empty. Please provide file content using get_file_content tool first."

        # Prepare file contents for analysis
        MAX_CONTENT_SIZE = 50000  # 50KB per file
        prepared_contents = []

        for file_id, content in list(file_contents.items())[:10]:  # Limit to 10 files
            if not content or not isinstance(content, str):
                continue

            # Limit content size
            if len(content) > MAX_CONTENT_SIZE:
                content = content[:MAX_CONTENT_SIZE] + f"\n... [content truncated, original size: {len(content)} chars]"

            prepared_contents.append({"identifier": file_id, "content": content, "size": len(content)})
            logger.info(f"📄 [EDGE_CASE] Prepared content: {file_id} ({len(content)} chars)")

        if not prepared_contents:
            return "Error: No valid file content provided."

        # Prepare general analysis prompt (no hardcoded sources)
        ANALYSIS_PROMPT = textwrap.dedent("""
            You are an expert code analyzer specializing in identifying edge cases, implicit requirements, and specification mismatches.
            
            Analyze the provided code and test information systematically to identify discrepancies and implicit requirements.
            
            **1. Edge Cases and Boundary Conditions:**
            - Empty/null/undefined input handling and expected output
            - Single item vs multiple items behavior differences
            - Boundary values (min, max, zero, negative, overflow)
            - Invalid input scenarios and error handling
            - Special characters and encoding issues
            
            **2. Implicit Requirements Discovery:**
            - Requirements evident from tests but not in problem statement or function signature
            - Format specifications not documented but required by tests
            - Type conversion requirements (e.g., object to formatted string)
            - Error message format requirements (exact text, not just type)
            
            **3. Type and Conversion Handling:**
            - Type coercion requirements
            - Format conversions
            - Type checking and validation requirements
            
            **4. Error Handling:**
            - When errors should be thrown vs returned
            - Exact error message format requirements
            - Exception types expected
            
            **Analysis Output Format:**
            Provide a structured analysis with:
            1. **Format/Type Mismatches**: Specific discrepancies between signatures, problem statements, and tests
            2. **Edge Cases**: All identified edge cases with expected behavior
            3. **Implicit Requirements**: Requirements in tests but not documented elsewhere
            4. **Actionable Recommendations**: Specific fixes needed with examples
            
            Focus on what will make the code pass all tests.
            """)

        # Build context from available information
        context_parts = []
        for file_info in prepared_contents:
            context_parts.append(f"=== {file_info['identifier']} ({file_info['size']} chars) ===\n{file_info['content']}")

        problem_context = ""
        if hasattr(self, "problem_statement") and self.problem_statement:
            problem_context = f"\n\nAdditional Context:\n{self.problem_statement[:2000]}"

        messages = [
            {"role": "system", "content": ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"""Target: {target_identifier or 'General codebase analysis'}

Content Provided:
{chr(10).join(context_parts)}{problem_context}

**Analysis Instructions:**
1. Compare function signatures/type hints with test expectations to find format/type mismatches
2. Extract exact format requirements from test cases
3. Identify edge cases by examining test inputs and expected outputs
4. Find implicit requirements that are in tests but not in problem statements or signatures
5. Note any discrepancies between different sources of information

Focus on discovering what will make the code pass all tests. Be specific about format and edge case requirements.""",
            },
        ]

        # Get analysis from LLM
        retry = 0
        selected_model = GLM_MODEL_NAME
        while retry < 5:
            try:
                analysis, _ = EnhancedNetwork.make_request(messages, model=selected_model, attempt=1, temperature=0.0)
                logger.info("✅ [EDGE_CASE] Edge case analysis completed")
                return f"=== Edge Case Analysis ===\n\n{analysis}"
            except Exception as e:
                retry += 1
                if retry < 5:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = random.choice(other_models)
                    time.sleep(1)
                else:
                    return f"Error: Failed to analyze edge cases after {retry} attempts: {str(e)}"

        return "Error: Failed to analyze edge cases"

    @EnhancedToolManager.tool
    def analyze_change_impact(self, file_path: str, symbol_name: str) -> str:
        """
        Deterministically analyze change impact for a specific symbol BEFORE editing.

        This tool estimates call-site fanout (how widely referenced a symbol is), collects example references,
        computes semantic signals for multi-mode/discriminator-like code, and then uses an LLM to classify:
        - focused vs broad change risk
        - whether an explicit boundary-localization proof is required before editing

        The result is stored as a stable per-file fix strategy to prevent prompt dilution.

        Arguments:
            file_path: repository-relative path of the target file you intend to modify
            symbol_name: identifier to analyze (use the name as it appears in code, e.g., a function name)
        Output:
            JSON string containing fanout estimates, semantic signals, a decision, and required gates.
        """
        analyzer = ChangedImpactAnalyzer(code_parser=self.code_parser)
        report = analyzer.analyze_symbol(file_path=file_path, symbol_name=symbol_name)
        self.fix_strategy[file_path] = report
        try:
            return json.dumps(report, ensure_ascii=False)
        except Exception:
            return str(report)

    @EnhancedToolManager.tool
    def prove_boundary_localization(self, file_path: str, symbol_name: str, planned_change_summary: str) -> str:
        """
        Produce an explicit boundary-localization proof for edits to multi-variant/discriminator-like code.

        Use this when analyze_change_impact requires boundary proof. The proof must:
        - name the distinct interpretations/variants
        - explain where each variant is required
        - explain why the planned change does not collapse variants
        - specify canaries (reference contexts) that must not break

        Arguments:
            file_path: repository-relative target file for modification
            symbol_name: symbol under consideration (must match analyze_change_impact)
            planned_change_summary: brief description of the intended code change
        Output:
            JSON string proof, also stored to satisfy pre-edit gates.
        """
        strat = self.fix_strategy.get(file_path, {})
        prompt = textwrap.dedent(
            f"""
            You are writing a boundary-localization proof for a code change.

            Inputs:
            - file_path: {file_path}
            - symbol_name: {symbol_name}
            - planned_change_summary: {planned_change_summary}
            - fix_strategy_report: {json.dumps(strat, ensure_ascii=False)}

            Task:
            Produce a proof that the change is localized to the correct boundary and preserves all coexisting interpretations/variants.

            Rules:
            - Output MUST be a single JSON object only (no markdown, no extra text).
            - If uncertain, request narrowing: propose changing a call-site boundary instead of shared behavior.

            Output schema:
            {{
              "symbol_name": "...",
              "variants": [
                {{"name": "variant_a", "when_required": "...", "invariant": "..."}},
                {{"name": "variant_b", "when_required": "...", "invariant": "..."}}
              ],
              "boundary_choice": {{"where": "...", "why_here": "..."}},
              "non_collapse_argument": ["bullet 1", "bullet 2"],
              "canaries": ["short canary 1", "short canary 2"]
            }}
            """
        ).strip()
        messages = [{"role": "user", "content": prompt}]

        retry = 0
        selected_model = QWEN_MODEL_NAME
        max_retries = 10

        while retry < max_retries:
            try:
                raw, _ = EnhancedNetwork.make_request(messages, model=selected_model, attempt=1, temperature=0.0)

                cleaned = raw.strip().removeprefix("```").removesuffix("```").strip()

                try:
                    obj = json.loads(cleaned)

                    if isinstance(obj, dict):
                        self.boundary_proofs[file_path] = obj
                        return json.dumps(obj, ensure_ascii=False)
                    else:
                        logger.warning(f"Response is not a dictionary (attempt {retry + 1}/{max_retries})")
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[^{}]*"symbol_name"[^{}]*\}', cleaned, re.DOTALL)
                    if json_match:
                        try:
                            obj = json.loads(json_match.group())
                            if isinstance(obj, dict):
                                self.boundary_proofs[file_path] = obj
                                return json.dumps(obj, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass

                    # If JSON extraction failed, log and retry
                    logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {cleaned[:500]}")

            except Exception as e:
                logger.warning(f"Error in prove_boundary_localization (attempt {retry + 1}/{max_retries}): {e}")

            retry += 1
            if retry < max_retries:
                # Try different model on retry (after 7 attempts)
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)  # Small delay before retry

        # Return error if all retries failed
        return "Error: boundary proof helper failed after all retries. Re-run with a shorter planned_change_summary."

    def _summarize_test_output(self, test_output: str) -> str:
        """Summarize long test output using LLM to preserve critical debugging info."""
        prompt = textwrap.dedent(
            f"""
            Summarize this test execution output. Focus on:
            1. Total tests run, passed, and failed counts
            2. List ALL failed test cases with their exact names
            3. For each failure: exact important short error message, location (file:line), and root cause
            4. Any setup/teardown errors
            5. Critical error traces (keep full stack traces for failures)
            6. Any warnings or important messages

            Keep all specific error details - the summary must be sufficient for debugging.

            Test Output:
            {test_output}

            Provide a concise but complete summary:"""
        )

        messages = [{"role": "user", "content": prompt}]

        retry = 0
        selected_model = QWEN_MODEL_NAME
        max_retries = 10

        while retry < max_retries:
            try:
                summary, _ = EnhancedNetwork.make_request(
                    messages=messages,
                    model=selected_model,
                )
                return f"[TEST OUTPUT SUMMARIZED - Token count exceeded 5000]\n\n{summary}"
            except Exception as e:
                logger.warning(f"Error in _summarize_test_output (attempt {retry + 1}/{max_retries}): {e}")

            retry += 1
            if retry < max_retries:
                # Try different model on retry (after 7 attempts)
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)  # Small delay before retry

        # If summarization fails after all retries, truncate intelligently
        logger.warning("Failed to summarize test output after all retries, using truncation fallback")
        lines = test_output.split("\n")
        if len(lines) > 200:
            return (
                f"[TEST OUTPUT TRUNCATED]\n\n"
                + "\n".join(lines[:100])
                + f"\n\n... ({len(lines)-200} lines omitted) ...\n\n"
                + "\n".join(lines[-100:])
            )
        return test_output

    @EnhancedToolManager.tool
    def run_tests(self, command: List[str], timeout: int = 5) -> str:
        """
        Runs tests with strict timeout.
        Arguments:
            command: list of command line arguments,
            timeout: timeout in seconds (default: 5)
        Output:
            Standard output or error output of the command.
        """
        if self.is_fix_task and self._test_call_count == 0 and not self._pending_run_tests_confirmation:
            self._test_call_count += 1
            self._pending_run_tests_confirmation = True
            return textwrap.dedent(
                f"""
            ⚠️  VERIFICATION WORKFLOW DISCOVERY CHECK ⚠️
            
            You are about to run tests for the first time with command: {' '.join(command)}
            
            Before proceeding, you MUST confirm you have completed the mandatory discovery steps from section 5.5:
            
            ✓ Step 1: Examined repository root structure for verification entry scripts?
            ✓ Step 2: Inspected project documentation for test execution instructions?
            ✓ Step 3: Analyzed test organization and configuration?
            ✓ Step 4: Determined the canonical execution path with proper priority?
            
            CRITICAL QUESTIONS:
            1. Did you try to find a repository-specific test runner (Priority 1)?
               - Custom entry script in repository root?
               - Specialized test execution script?
            
            2. If no custom runner found, did you verify through documentation that 
               generic framework approach is the intended method?
            
            3. Is the command you're about to run the CORRECT way to execute tests 
               for this specific repository?
            
            ⚠️  COMMON MISTAKE: Jumping to framework-specific commands without discovering 
            canonical test runners wastes steps and causes execution failures.
            
            If you have NOT completed the discovery sequence:
            - STOP and complete section 5.5 discovery steps first
            - Use the repository exploration tools to examine structure and inspect relevant files
            - Look for specialized test runners before using generic commands
            
            If you HAVE completed discovery and verified this is the correct command:
            - Call run_tests again with the same command to proceed
            - The actual test execution will happen on the next call
            
            This confirmation only appears once. Subsequent run_tests calls will execute immediately.
            """
            ).strip()

        # Actual test execution (second call onwards)
        if self._pending_run_tests_confirmation:
            self._pending_run_tests_confirmation = False
        try:
            preface_lines: list[str] = []
            if self.is_fix_task:
                try:
                    if self._last_blocked_edit_step is not None and (
                        self._last_edit_step is None or self._last_blocked_edit_step > self._last_edit_step
                    ):
                        preface_lines.append(
                            "⚠️ NOTE: Your most recent code edit attempt was blocked by pre-edit gates. "
                            "This test run will execute against the last successfully applied code state."
                        )
                    if self._last_run_tests_step is not None and (self._last_edit_step is None or self._last_edit_step <= self._last_run_tests_step):
                        preface_lines.append(
                            "ℹ️ NOTE: No new successful code edits have been applied since the last test run; "
                            "this run mainly reconfirms the same code state."
                        )
                except Exception:
                    pass
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            test_output = result.stdout + result.stderr
            # Deterministic pass/fail capture for finish gating (do not rely on LLM heuristics).
            try:
                self._last_run_tests_step = self._current_step
                self._last_run_tests_passed = result.returncode == 0
            except Exception:
                pass

            token_count = Utils.count_tokens(test_output)
            if token_count > SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD:
                print(
                    f"⚠️  Test output large ({token_count} tokens, exceeds {SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD} tokens limit), summarizing with LLM..."
                )
                test_output = self._summarize_test_output(test_output)
                print(f"✅ Test output summarized successfully with {Utils.count_tokens(test_output)} tokens:\n{test_output}")

            if preface_lines:
                return "\n".join(preface_lines).strip() + "\n\n" + test_output
            return test_output

        except subprocess.TimeoutExpired:
            return "Test run timed out."
        except Exception as e:
            return f"Test execution error: {e}"

    def _save_large_observation(self, observation: str, tool_name: str) -> tuple[str, int]:
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        if not os.path.exists(self.observation_dir):
            os.makedirs(self.observation_dir, exist_ok=True)
        file_path = os.path.join(self.observation_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(observation)
            line_count = observation.count("\n") + 1 if observation else 0
            return file_path, line_count
        except Exception as e:
            return f"Error: Failed to save observation: {e}", -1

    def get_final_git_patch(self) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @EnhancedToolManager.tool
    def generate_test_cases_from_root_cause(self, root_cause_code: str, file_path: str = None, function_name: str = None) -> str:
        """
        Generates comprehensive test cases based on the problem statement and the identified root cause code section.
        Call this tool when you have identified the main root cause code part that needs to be fixed.
        The generated test cases will be saved and automatically referenced when you create test files using generate_test_file.
        Arguments:
            root_cause_code: The code section identified as the root cause of the issue (required)
            file_path: Optional file path where the root cause code is located (helps provide context)
            function_name: Optional function name where the root cause code is located (helps provide context)
        Output:
            A structured markdown document containing test cases with descriptions, inputs/setup, expected results, and reasons for each test case
        """
        if not self.problem_statement:
            return "Error: Problem statement not available. Cannot generate test cases."

        TEST_CASE_GENERATION_PROMPT = textwrap.dedent(
            """
        You are an expert test case generator. Your task is to generate comprehensive test cases based on a problem statement and the root cause code section.

        Analyze the problem statement and the root cause code to generate test cases that:
        1. Verify the bug exists (reproduction test)
        2. Verify the fix works correctly
        3. Cover edge cases related to the root cause
        4. Test boundary conditions

        For each test case, provide:
        - Test case description: What the test case does
        - Input/Setup: What inputs or setup are needed
        - Expected result: What should happen when the code is correct
        - Reason: Why this test case is important for verifying the root cause fix

        **NOTE**: Don't ONLY consider the primary issue in the problem statement.
        You should consider all, every possible edge cases.
        Invalid or wrong test cases should be also generated to test thoroughly.
        For those invalid or wrong cases, you should correctly handle error or edge case.

        Format your response as a structured markdown document with clear sections for each test case.
        Be specific and actionable. Focus on test cases that directly relate to the root cause identified.
        """
        )

        retry = 0
        selected_model = QWEN_MODEL_NAME
        root_cause_context = root_cause_code
        if file_path:
            root_cause_context += f"\n\nFile: {file_path}"
        if function_name:
            root_cause_context += f"\n\nFunction: {function_name}"

        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": TEST_CASE_GENERATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Problem Statement:\n{self.problem_statement}\n\nRoot Cause Code:\n{root_cause_context}\n\nGenerate comprehensive test cases for this root cause.",
                    },
                ]
                test_cases, _ = EnhancedNetwork.make_request(messages, model=selected_model, attempt=1, temperature=0.0)
                self.generated_test_cases = test_cases
                print(f"[GENERATE_TEST_CASES_FROM_ROOT_CAUSE] Test cases generated successfully and saved: {test_cases}")
                return f"Test cases generated successfully and saved.\n\n{test_cases}"
            except Exception as e:
                logger.error(f"Error generating test cases: {e}")
                retry += 1
                if retry < 10:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = random.choice(other_models)
                    time.sleep(1)
                else:
                    return f"Error: Failed to generate test cases after {retry} attempts: {str(e)}"
        return "Error: Failed to generate test cases"

    @EnhancedToolManager.tool
    def grep_search(self, grep_search_command: str) -> str:
        """
        Performs grep search on a single file or across multiple files in the codebase
        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        Output:
            locations where pattern was found with file paths and line numbers
        """
        return self.search_manager.search_in_all_files(grep_search_command)

    @EnhancedToolManager.tool
    def think(self, thought: str) -> str:
        """Use the tool to think about something. It will not make any changes to the repository. Use it when reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be correct and most effective. Alternatively, if you receive some test results, you can call this tool to brainstorm ways to fix the failing tests.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"

    @EnhancedToolManager.tool
    def find_symbol_references(self, symbol_identifier: str) -> str:
        """
        Discovers all code locations where a specific function, class, method, or variable is referenced.
        Provides contextual information around each usage to understand how the symbol is being used.
        Particularly valuable before modifying or refactoring code elements.
        Works across all programming languages and file types.
        Arguments:
            symbol_identifier: exact name of the function, class, method, or variable to locate
        Output:
            comprehensive listing of files and line numbers with surrounding context for each reference
        """
        try:
            cmd = f"grep -rn --binary-files=without-match '{symbol_identifier}' . | head -100"
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
            refs = result.stdout.strip()

            if not refs:
                return f"No references discovered for symbol '{symbol_identifier}' in the codebase."

            lines = refs.split("\n")
            if len(lines) > 50:
                summary = f"Found {len(lines)} references for '{symbol_identifier}' (showing first 50):\n\n"
                return summary + "\n".join(lines[:50]) + f"\n\n... and {len(lines) - 50} more references (refine search if needed)"
            return f"References for '{symbol_identifier}' ({len(lines)} found):\n{refs}"
        except subprocess.TimeoutExpired:
            return f"Search timeout: Symbol '{symbol_identifier}' search took too long. Try a more specific identifier."
        except Exception as e:
            return f"Error locating symbol references: {str(e)}"

    @EnhancedToolManager.tool
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Output:
            The complete function body including decorators, or empty string if function not found.
        """
        if not hasattr(self, "code_parser"):
            self.code_parser = CodeParseUtil()
        return self.code_parser.get_function_body(file_path, function_name, add_line_numbers=True)

    @EnhancedToolManager.tool
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(file_path=file_path, search_term=search_term)

    @EnhancedToolManager.tool
    def log_strategy(self, approach: str, reasoning: str) -> str:
        """Record a high-level strategy before attempting it.

        Use this BEFORE making significant code changes to log your planned approach. This creates
        a history that persists across rollbacks, preventing you from retrying failed strategies.

        Arguments:
            approach: Brief description of the approach
            reasoning: Why you think this will work

        Output:
            Confirmation with strategy ID for later reference.
        """
        self.strategy_counter += 1
        strategy = {
            "id": self.strategy_counter,
            "approach": approach,
            "reasoning": reasoning,
            "success": None,
            "reason": None,
            "timestamp": time.time(),
            "created_step": len(getattr(self, "tool_invocations", {})),
        }
        self.strategies.append(strategy)
        return f"Strategy #{self.strategy_counter} logged: {approach}\nReasoning: {reasoning}\nUse mark_strategy_outcome to record results."

    @EnhancedToolManager.tool
    def create_new_file(
        self,
        file_path: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """
        Creates a new file with the specified content.

        Arguments:
            file_path: Path where the new file should be created.
            content: The content to write into the file.
            overwrite: If True, will overwrite the file if it exists. If False and file exists, returns an error.

        Output:
            Status message indicating success or error.
        """
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Set overwrite=True to overwrite."

        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            if hasattr(self, "file_ops") and hasattr(self.file_ops, "new_files_created"):
                self.file_ops.new_files_created.append(file_path)
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"

    @EnhancedToolManager.tool
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.

        Use this to review what approaches you've already attempted. Critical for:
        - Avoiding retry loops (especially after rollbacks)
        - Understanding what doesn't work
        - Building on partially successful strategies

        Arguments:
            None

        Output:
            Formatted list of all strategies with outcomes.
        """
        if not self.strategies:
            return "No strategies recorded yet. Use log_strategy before attempting significant changes."
        output = ["=== STRATEGY HISTORY ===\n"]
        succeeded = [s for s in self.strategies if s["success"] is True]
        failed = [s for s in self.strategies if s["success"] is False]
        pending = [s for s in self.strategies if s["success"] is None]
        output.append(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n")
        for status, strategies in [
            ("SUCCEEDED", succeeded),
            ("FAILED", failed),
            ("PENDING", pending),
        ]:
            if strategies:
                output.append(f"\n{status}:")
                for s in strategies:
                    output.append(f"\n  [{s['id']}] {s['approach']}")
                    output.append(f"      Reasoning: {s['reasoning']}")
                    if s["reason"]:
                        output.append(f"      Outcome: {s['reason']}")
        return "\n".join(output)

    @EnhancedToolManager.tool
    def mark_strategy_outcome(self, strategy_id: int, success: bool, reason: str) -> str:
        """Record whether a strategy worked.

        After attempting a strategy, record the outcome. This is crucial for institutional memory,
        especially when using rollbacks - you'll know what you already tried even after reverting changes.

        Arguments:
            strategy_id: ID from log_strategy (e.g., 1, 2, 3)
            success: True if approach worked (tests passed, bug fixed), False otherwise
            reason: Why it succeeded/failed (e.g., "Tests passed but introduced new bug in edge case")

        Output:
            Updated strategy status.
        """
        for strat in self.strategies:
            if strat["id"] == strategy_id:
                strat["success"] = success
                strat["reason"] = reason
                strat["completed_step"] = len(getattr(self, "tool_invocations", {}))
                status = "SUCCEEDED" if success else "FAILED"
                return f"Strategy #{strategy_id} marked as {status}\nReason: {reason}"
        return f"Error: Strategy #{strategy_id} not found"

    @EnhancedToolManager.tool
    def finish(self):
        """
        Signals completion of the current workflow execution. Validates patch application and solution verification before finishing.
        Arguments:
            None
        Output:
            Review patch prompt with validation results, or "finish" if all checks pass
        """
        if self.has_exception_handling_mention and self.finish_called_count < 2:
            self.finish_called_count += 1
            return textwrap.dedent(
                """
                ⚠️ EXCEPTION HANDLING CHECK REQUIRED - Cannot Finish Yet

                The problem statement mentions exception handling. Please check if your code handles exceptions correctly and make sure you did not miss any exception cases/scenarios for each exception message. For a given exception message, there can be multiple underlying scenarios. Please verify that your code handles all of them properly.
                If you did miss any exception cases, please fix them and call `finish` again.
                """
            ).strip()
        if self.is_fix_task:
            if self._last_run_tests_step is None:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You have not executed `run_tests`. Run the repository-defined verification workflow, ensure it passes, then call `finish` again.
                    """
                ).strip()
            if self._last_edit_step is not None and self._last_run_tests_step < self._last_edit_step:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You edited code after your last verification run. Run `run_tests` again after the last edit and ensure it passes, then call `finish`.
                    """
                ).strip()
            if self._last_run_tests_passed is False:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION FAILED - Cannot Finish Yet

                    Your latest verification run did not pass. Fix the failures, re-run `run_tests`, then call `finish`.
                    """
                ).strip()

        validation_result = self.validate_patch_application()

        # Generate review patch prompt based on validation result
        if "Patch validation passed" not in validation_result:
            if "Patch validation failed" in validation_result or "Patch validation error" in validation_result:
                review_prompt = textwrap.dedent(
                    """
                    ⚠️ Patch Validation: FAILED
                    
                    The patch validation detected issues that may prevent successful application.
                    Please review and fix the following issues before finalizing:
                    
                    {validation_result}
                    
                    Common fixes:
                    - Replace raw newlines (actual line breaks) in strings with \\n escape sequences
                    - Ensure unified diff format is correct (proper @@ hunk headers with line counts)
                    - Remove control characters (\\r, \\0) from file content
                    - Check for encoding issues (ensure UTF-8 encoding)
                    - Verify file paths in the patch match actual file locations
                    
                    After fixing the issues, you can call validate_patch_application again to verify.
                    """
                ).format(validation_result=validation_result)
            else:
                # Validation was skipped or returned a message
                review_prompt = textwrap.dedent(
                    """
                    ℹ️ Patch Validation: {validation_result}
                    
                    Please review your changes before finalizing.
                    """
                ).format(validation_result=validation_result)
            return review_prompt.strip()

        if self.is_fix_task and self.solution_verifier:
            regression_review = self.solution_verifier.verify_solution()

            if "REGRESSION_AND_BUG_CHECK_PASSED" in regression_review:
                # Both conditions verified - return finish
                print("✅ Regression and bug check PASSED - proceeding to finish")
                return "finish"
            else:
                regression_feedback = (
                    textwrap.dedent(
                        """
                    ⚠️ **VERIFICATION FAILED - Cannot Finish Yet**
                    
                    Your solution is not ready to finish. Please address the following issues:
                    
                    {regression_feedback}
                    
                    **CRITICAL REQUIREMENTS (BOTH must be true)**:
                    
                    ✅ **CONDITION 1**: Original bug must be FIXED
                        - The hidden tests (originally failing) must now PASS
                        - Verify with actual test runs, not just theory
                    
                    ✅ **CONDITION 2**: NO regressions introduced
                        - ALL tests that were passing before must STILL pass
                        - Run FULL test suite, not just specific test cases
                        - Fix ALL failures, do not rationalize or ignore any
                    
                    **BEWARE OF RETURNED BUG**:
                    - Do NOT fix regressions in a way that breaks the original bug fix
                    - Your final solution must fix BOTH the original bug AND all regressions
                    - The final test run must show ALL tests passing (both hidden tests and regression tests)
                    
                    **REQUIRED ACTIONS**:
                    1. Identify which condition failed (original bug, regressions, or both)
                    2. Fix the issue without breaking the other condition
                    3. Run the FULL test suite to verify BOTH conditions are met
                    4. Only call `finish` again when ALL tests pass (no fail in output)
                    
                    After fixing the issues, call `finish` again for re-verification.
                """
                    )
                    .format(regression_feedback=regression_review)
                    .strip()
                )

                print("❌ Regression/bug check FAILED - returning feedback to agent")
                print("Feedback:", regression_feedback)

                return regression_feedback

        return "finish"

    def validate_patch_application(self) -> str:
        """
        Validates that the current patch can be applied successfully without errors.
        This tool tests patch application by: generating patch from clean state, applying it to clean state,
        and verifying it works. This prevents generating patches that fail when applied later due to issues
        like raw newlines in strings, malformed unified diff format, or incorrect hunk line counts.
        Arguments:
            None
        Output:
            Validation result - "Patch validation passed" if successful, or detailed error message
            explaining why the patch failed to apply (e.g., corrupt patch, line count mismatches,
            raw newlines in strings, etc.)
        """
        try:
            subprocess.run(["git", "reset", "HEAD"], capture_output=True, text=True, timeout=10, check=False)

            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=30, check=False)

            if status_result.returncode != 0 or not status_result.stdout.strip():
                return "Patch validation skipped: No modified files to validate"

            modified_files = []
            for line in status_result.stdout.splitlines():
                line = line.rstrip()  # Only strip right side to preserve leading spaces if any
                if not line or len(line) < 3:
                    continue

                # Get status code (first 2 characters)
                status = line[:2]

                if status[0] == "R" or status[1] == "R":
                    # For renamed files, we want the new filename (after ->)
                    if " -> " in line:
                        filepath = line.split(" -> ", 1)[1].strip()
                    else:
                        continue
                else:
                    filename_start = 2
                    while filename_start < len(line) and line[filename_start] == " ":
                        filename_start += 1

                    if filename_start >= len(line):
                        continue

                    remaining = line[filename_start:].strip()

                    if remaining.startswith('"') and remaining.endswith('"'):
                        # Remove quotes and unescape
                        filepath = remaining[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                    else:
                        filepath = remaining

                # Check if file is modified (M) or added (A) - exclude deleted (D) and untracked (??)
                # We only want files we can actually patch
                if any(c in status for c in ["M", "A"]) and "D" not in status and "?" not in status:
                    modified_files.append(filepath)

            if not modified_files:
                return "Patch validation skipped: No modified files to validate"

            # Exclude agent files
            exclude = {"src/agent.py", "src/agent_runner.py", "sitecustomize.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            modified_files = [f for f in modified_files if f not in exclude]

            if not modified_files:
                return "Patch validation skipped: No relevant modified files to validate"

            # Stash current changes to get a clean state
            stash_result = subprocess.run(
                ["git", "stash", "push", "-m", "temp_validation_stash", "--"] + modified_files,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if stash_result.returncode != 0:
                return f"Patch validation error: Failed to stash changes. {stash_result.stderr}"

            stash_applied = True

            try:
                # Generate patch from the stash (which contains our changes)
                # The stash contains the diff we want to validate
                stash_diff_result = subprocess.run(
                    ["git", "stash", "show", "-p", "--no-color", "--unified=3", "stash@{0}"], capture_output=True, text=True, timeout=30, check=False
                )

                if stash_diff_result.returncode != 0:
                    # Fallback: restore from stash, stage, generate patch, then restore stash
                    subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                    stash_applied = False

                    # Stage the files
                    if modified_files:
                        subprocess.run(["git", "add", "--"] + modified_files, capture_output=True, text=True, timeout=30, check=False)

                    # Generate patch
                    diff_result = subprocess.run(
                        ["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=False
                    )

                    # Unstage
                    subprocess.run(["git", "reset", "HEAD"], capture_output=True, text=True, timeout=10, check=False)

                    # Stash again
                    stash_result2 = subprocess.run(
                        ["git", "stash", "push", "-m", "temp_validation_stash", "--"] + modified_files,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if stash_result2.returncode == 0:
                        stash_applied = True

                    patch_text = diff_result.stdout or ""
                else:
                    patch_text = stash_diff_result.stdout or ""

                if not patch_text.strip():
                    # Restore and return
                    if stash_applied:
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                        stash_applied = False
                    return "Patch validation skipped: No changes to validate (empty patch)"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp_file:
                    tmp_file.write(patch_text)
                    patch_file = tmp_file.name

                try:
                    # Test patch application using git apply --check (dry run)
                    check_result = subprocess.run(["git", "apply", "--check", patch_file], capture_output=True, text=True, timeout=30, check=False)

                    if check_result.returncode == 0:
                        # If check passes, try actual application
                        apply_result = subprocess.run(["git", "apply", patch_file], capture_output=True, text=True, timeout=30, check=False)

                        # Reset to clean state after applying
                        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "clean", "-fd"], capture_output=True, text=True, timeout=10, check=False)

                        if apply_result.returncode == 0:
                            # Restore stashed changes
                            subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                            return "Patch validation passed: Patch can be applied successfully"
                        else:
                            error_msg = apply_result.stderr.strip() or apply_result.stdout.strip() or "Unknown error"
                            # Restore stashed changes even on failure
                            subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                            return f"Patch validation failed: Patch cannot be applied. Error: {error_msg}\n\nCommon causes:\n- Raw newlines inside quoted strings (use \\n instead)\n- Malformed unified diff format\n- Incorrect hunk line counts\n- Control characters (\\r, \\0) in file content\n- Encoding issues"
                    else:
                        error_msg = check_result.stderr.strip() or check_result.stdout.strip() or "Unknown error"
                        # Restore stashed changes
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                        return f"Patch validation failed: Patch check failed. Error: {error_msg}\n\nCommon causes:\n- Raw newlines inside quoted strings (use \\n instead)\n- Malformed unified diff format\n- Incorrect hunk line counts\n- Control characters (\\r, \\0) in file content\n- Encoding issues"
                finally:
                    try:
                        os.unlink(patch_file)
                    except Exception:
                        pass
            except Exception as e:
                # Restore stashed changes if something went wrong
                if stash_applied:
                    try:
                        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "clean", "-fd"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                    except Exception:
                        pass
                return f"Patch validation error during application test: {str(e)}"
        except Exception as e:
            return f"Patch validation error: {str(e)}"

    @EnhancedToolManager.tool
    def run_shell_cmd(self, command: str) -> str:
        """
        Runs shell commands for the repository. This tool executes shell commands directly.
        Arguments:
            command: A shell command to be run.
        Output:
            The stdout results of the command. Your working directory is the root of the project.
        """
        if not command:
            return "Error: No command provided."

        try:
            result = subprocess.run(command, shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=150)
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after 150 seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"

    @EnhancedToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
        Signals completion of the file finding workflow execution
        Arguments:
            files: The list of files to fix.
        """
        self.files_to_fix = files
        return files

IO_FORMAT_CONSISTENCY_PROMPT = """You are an expert at analyzing a problem statement and function signature.

Your task is to check if the INPUT/OUTPUT FORMATS in the examples are CONSISTENT with the function signature's parameter types and return type.

**What to check:**
1. **Return Type Consistency**: Does the example output format match the function's declared return type?
2. **Input Type Consistency**: Do the example inputs match the function's parameter types?
3. **Format Compatibility**: Are there format mismatches that don't make sense?

**Return ONLY a JSON object with this exact structure:**
{
    "has_format_mismatch": true or false,
    "mismatch_severity": "none" or "moderate" or "severe",
    "input_format_match": true or false,
    "output_format_match": true or false,
    "example_output_format": "<description of output format shown in examples>",
    "function_return_type": "<return type from function signature>",
    "example_input_format": "<description of input format shown in examples>",
    "function_input_types": "<parameter types from function signature>",
    "mismatch_description": "<description of any format mismatches found, or empty string if none>",
    "requires_complex_conversion": true or false,
    "confidence": <0.0 to 1.0, how confident you are in your assessment>,
    "reasoning": "<brief explanation of your analysis>"
}

Do NOT include any other text or markdown formatting. Only return the JSON object."""
EXPLICIT_EXCEPTION_HANDLING_PROMPT = """You are analyzing a problem statement for educational coding exercises.

Your task is to determine if the problem statement contains EXPLICIT MENTIONS of exception handling.

**What constitutes explicit exception handling mentions:**
The problem statement must contain language that:
1. Directly mentions exceptions, exception handling, or raising exceptions
2. Uses imperative or directive language about exceptions
3. References specific exception types
4. Provides instructions about when or how to raise exceptions
5. Includes code examples showing exception handling
6. Mentions exception messages that should be used

**What does NOT constitute explicit mentions:**
- Mere references to errors without mentioning exceptions
- Implied error conditions without explicit exception instructions
- Test descriptions that mention exceptions but don't instruct the implementer
- Vague references to "error handling" without specifying exceptions
- Edge cases mentioned without explicit exception requirements

**Key principle:** The mention must be EXPLICIT and DIRECT. It must create a clear requirement or instruction for the implementer to handle exceptions. If the text only describes error conditions or test expectations without instructing exception handling, it is not explicit.

**Important**: Only return true if there is a DIRECT, EXPLICIT mention of exception handling in the problem statement. Vague references or implied error conditions are not sufficient.

**Return ONLY a JSON object with this exact structure:**
{
    "has_explicit_exception_mention": true or false,
    "explicit_mentions": [
        {
            "mention_text": "<exact text from problem statement that mentions exception handling>",
            "mention_type": "<instruction/example/reference/requirement>",
            "exception_type_mentioned": "<exception type if specified, or empty string>",
            "context": "<brief context around the mention>",
            "confidence": <0.0 to 1.0>
        },
        ...
    ],
    "total_explicit_mentions": <number>,
    "confidence": <0.0 to 1.0, how confident you are in your assessment>,
    "reasoning": "<brief explanation of why you classified it this way>"
}

Do NOT include any other text or markdown formatting. Only return the JSON object."""
EXAMPLES_CHECK_PROMPT = """You are an expert at analyzing problem statements.

Your task is to determine if a problem statement contains CLEAR EXAMPLES.

A problem statement has CLEAR EXAMPLES if it includes:
1. **Input/Output Examples**: Shows concrete examples of inputs and their corresponding expected outputs
2. **Step-by-step Examples**: Demonstrates how the solution works with specific values
3. **Edge Case Examples**: Shows examples of boundary conditions or special cases
4. **Format Examples**: Shows the exact format/formatting of inputs and outputs

**Criteria for CLEAR EXAMPLES:**
- The examples are concrete (not abstract descriptions)
- The examples show actual values/data, not just types
- The examples are complete enough to understand what the function should return
- Multiple examples are provided (at least 1, ideally 2-3)

**Return ONLY a JSON object with this exact structure:**
{
    "has_clear_examples": true or false,
    "example_count": <number of distinct examples found>,
    "example_types": ["input/output", "step-by-step", "edge_case", "format", etc.],
    "confidence": <0.0 to 1.0, how confident you are in your assessment>,
    "reasoning": "<brief explanation of why you classified it this way>"
}

Do NOT include any other text or markdown formatting. Only return the JSON object."""
PROBLEM_DECOMPOSITION_PROMPT = textwrap.dedent(
    """
You are an expert software debugging analyst. Analyze the bug report and extract structured information.

Extract the following from the problem statement:

1. **Problem Summary**: Brief description of the issue type in your own words

2. **Key Entities**: Extract identifiers mentioned (file paths, function names, class names, error messages, etc.)

3. **Behavior**:
   - Expected: What should happen
   - Actual: What actually happens
   - Trigger: Conditions that cause the issue

4. **Success Criteria**: What would indicate a successful fix

5. **Investigation Starting Points**: 3-5 specific places to start looking (files, search terms, code areas)

Respond in JSON:
```json
{
    "problem_summary": "brief description",
    "key_entities": {
        "files": [],
        "functions": [],
        "classes": [],
        "error_messages": [],
        "other": []
    },
    "behavior": {
        "expected": "",
        "actual": "",
        "trigger": ""
    },
    "success_criteria": [],
    "investigation_starting_points": [
        {"location": "", "reason": ""}
    ]
}
```
"""
)
VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""
FORMAT_PROMPT_CREATE = textwrap.dedent(
    """
**Default: Use single tool call format. Use multiple tool calls ONLY when searching multiple files at once for time efficiency.**

## Response Formats

### Format 1: Single Tool Call (DEFAULT - Use this for most operations)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}

### Format 2: Multiple Tool Calls (ONLY for multi-file searches)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}

## When to Use Multiple Tool Calls

**ONLY use multiple tool calls when:**
- Searching multiple files at once (e.g., codebase_search on multiple files/directories simultaneously)

**Examples:**

✅ **Good - Multiple file searches (time efficient)**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function function_name' ."}
tool_call_2:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function_name(' ."}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "file_name.js"}

✅ **Good - Single tool call (default)**:
next_thought: I'll read this file to understand the code
next_tool_name: get_file_content
next_tool_args: {"file_path": "aaa.py"}

✅ **Good - Single tool to edit file**:
next_thought: I'll edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", "search": "old_code", "replace": "new_code"}

✅ **Good - Single tool call to verify**:
next_thought: I'll run a command to verify the changes
next_tool_name: run_tests
next_tool_args: {"command": "node file.js", "timeout": 5}

## Critical Rules
- Default to single tool call format (next_tool_name, next_tool_args)
- Use multiple tool calls ONLY for parallel multi-file searches
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)
CREATE_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior bug-fix engineer working on an open-source repository.

You will be tasked to fix an issue from this repository.

Your thinking should be thorough and so it's fine if it's very long. You should think step by step before and after each action you decide to take.

You already have everything you need to solve this problem in the repository, even without internet connection.

Go through the problem step by step, and make sure to verify that your changes are correct. NEVER GIVE UP without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy

1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. **MANDATORY**: Generate test cases from root cause using `generate_test_cases_from_root_cause` BEFORE creating test files.
6. Debug as needed. Use debugging techniques to isolate and resolve issues.
7. Test frequently. Run tests after each change to verify correctness.
8. Iterate until the root cause is fixed and all tests pass.
9. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

Refer to the detailed sections below for more information on each step.

## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 2. Codebase Investigation
**CRITICAL: Find working examples first, then identify what's broken.**

**EFFICIENCY: Use parallel searches whenever possible!**
- Use multiple `grep_search` tool calls in parallel (tool_call_1, tool_call_2, tool_call_3, etc.) to run multiple searches simultaneously (3-5x faster than sequential)
- Start with broad queries, then narrow based on results
- Run multiple searches with different wording/phrasing in parallel - first-pass results often miss key details
- Example: Search for "authentication", "auth flow", "login process" all at once using parallel grep_search calls

**Investigation Strategy:**
- Search for key terms from the issue throughout the codebase
- **MANDATORY: Run multiple searches with different wording/phrasing in parallel** - first-pass results often miss key details
- Find similar functionality that WORKS correctly - this is your template
- Study how working code accomplishes what you need
- Locate the broken code using same keywords
- Look beyond surface symptoms - search in domains, helpers, utilities, base classes
- Trace to where mechanisms are actually DEFINED, not just where they're called
- Find the ROOT files where functionality is implemented
- Keep searching new areas until you're CONFIDENT nothing important remains

**Trace from final output backwards to root cause:**
- Start with working feature's final output, trace backwards to find generator
- Start with broken feature's final output, trace backwards to find what's missing or different
- Compare the paths: where do they diverge?
- Don't stop at the first file you find - keep tracing back to where the behavior originates

- Read and understand relevant code snippets
- Compare working vs broken code: what's different? Missing calls? Missing imports?
- Identify the root cause by finding what working code does that broken code doesn't
- Validate and update your understanding continuously as you gather more context
- TRACE every symbol back to its definitions and usages so you fully understand it
- Look past the first seemingly relevant result. EXPLORE alternative implementations, edge cases, and varied search terms until you have COMPREHENSIVE coverage of the topic

## 2.1. Parallel Tool Execution Strategy

**CRITICAL INSTRUCTION: For maximum efficiency, whenever you perform multiple operations, invoke all relevant tools concurrently using tool_call_1, tool_call_2, tool_call_3, etc. rather than sequentially.**

**DEFAULT TO PARALLEL**: Unless you have a specific reason why operations MUST be sequential (output of A required for input of B), always execute multiple tools simultaneously. This is not just an optimization - it's the expected behavior. Remember that parallel tool execution can be 3-5x faster than sequential calls, significantly improving efficiency.

**When gathering information, plan your searches upfront and then execute all tool calls together:**
- Multiple `grep_search` calls with different patterns should run simultaneously
- Multiple `get_file_content` calls for different files should run in parallel
- Combining `grep_search` with `get_file_content` can be done all at once
- Any information gathering where you know upfront what you're looking for should use parallel calls

**Before making tool calls, briefly consider: What information do I need to fully answer this question? Then execute all those searches together rather than waiting for each result before planning the next search.**

**Examples of parallel tool calls:**
- Searching for different patterns (imports, usage, definitions) should happen in parallel
- Multiple grep searches with different regex patterns should run simultaneously
- Reading multiple files or searching different directories can be done all at once
- Combining searches with file reads for comprehensive results

Most of the time, parallel tool calls can be used rather than sequential. Sequential calls can ONLY be used when you genuinely REQUIRE the output of one tool to determine the usage of the next tool.

## 3. Root Cause Verification
**Before implementing any fix, verify you understand the root cause:**

**Trace the COMPLETE data flow for both working and broken:**
1. Find similar WORKING feature
2. Trace working feature through all stages from start to final output
3. Trace broken feature through all stages from start to final output
4. Find EXACT point where paths diverge

**Compare working vs broken at EACH stage:**
- What does working code do that broken code doesn't?
- What functions are called? What imports exist?
- Where does the behavior differ?
- Keep tracing backwards until you find the root cause

**Find root, not symptoms:**
- Don't patch surface symptoms - find the missing or different mechanism
- Trace all the way back to where the behavior originates
- The fix location may be far from where symptoms appear
- Compare: How does working feature accomplish the task? How does broken feature differ?

**Search comprehensively:**
- Is this pattern missing in multiple places? Search the whole repository
- Are there similar files/classes that need the same fix?
- Fix all instances, not just the one example in the issue

## 4. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem
- Break down the fix into small, incremental changes
- Think through all the steps you need to take to fix the problem

## 5. Making Code Changes
**Copy patterns from working code. Make minimal focused changes.**
- **When returning multi-line or verse-by-verse text: (1) The entire returned string must end with a newline (\n), including when returning a single verse or line — e.g. return verse + '\\n' or return verses.join('\\n\\n') + '\\n'. (2) Between verses use a blank line (i.e. \n\n), not a single newline (\n). Single verse: "...verse.\\n"; multiple verses: "verse1\\n\\nverse2\\n\\nverse3\\n". Tests use exact string equality; missing trailing newline or wrong separator causes failures.**
- Before editing, always read the relevant file contents or section to ensure complete context
- If a patch is not applied correctly, attempt to reapply it
- **Use the EXACT same pattern as working code**: same functions, same imports, same structure
- Make small, testable, incremental changes that logically follow from your investigation
- **Search for similar locations**: Is this pattern needed elsewhere? Fix all instances if it's systemic
- Keep changes minimal and focused - don't refactor or change unrelated code

## 6. Debugging
**CRITICAL: Fix root cause, not symptoms. Search broadly across the repository.**

- Make code changes only if you have high confidence they can solve the problem
- When debugging, determine the ROOT CAUSE rather than addressing surface symptoms
- Don't just patch the calling code - trace back to where the mechanism is defined
- Trace from working feature backwards to find where behavior is implemented
- The fix location is often far from where the problem is first noticed

**Search across the entire repository:**
- Broadly search like domain logic files, helper/utility modules, base classes, configuration files, handler classes...
- Look beyond the obvious files mentioned in error messages

- Look for similar patterns that might need the same fix in multiple locations
- Debug for as long as needed to identify the root cause and identify a fix
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening
- Revisit your assumptions if unexpected behavior occurs.

## 6. Testing
- Run tests frequently using the available testing tools (for example, by calling the `run_code` tool).
- After each change, verify correctness by running relevant tests via the testing tool rather than invoking shell commands directly.
- If tests fail, analyze failures and revise your patch.
- Write additional tests if needed to capture important behaviors or edge cases.
- Ensure all tests pass before finalizing.

## 7. Final Verification
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.

## 8. Final Reflection and Additional Testing
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests that would need to pass to fully validate the correctness of your solution.
- Run these new tests and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.

# Tool Documentation
You have access to the following tools:-
{tools_docs}

# Tool Usage Guidelines
- Use appropriate tools to gather context before making changes.
- **CRITICAL: Maximize parallel tool calls** - Use multiple tool_call_N (tool_call_1, tool_call_2, tool_call_3, etc.) to execute searches, file reads, and other independent operations simultaneously. This is 3-5x faster than sequential calls.
- **MANDATORY for searches**: Run multiple `grep_search` calls with different wording/phrasing in parallel - first-pass results often miss key details. For example, search for "authentication", "auth flow", "login process" all at once.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `grep_search` to find all occurrences of an issue before fixing.
- Plan your information gathering upfront, then execute all tool calls together rather than sequentially.

# Meta-Cognitive Checkpoints
Every 15 steps, you will receive a META-COGNITIVE CHECKPOINT that analyzes your recent activity and progress:
- **Progress Analysis**: Shows what tools you've used and whether you're making measurable progress
- **Pattern Detection**: Alerts you if you're stuck in repetitive behavior (e.g., using same tools repeatedly)
- **Mandatory Reflection**: You MUST address these reflection questions in your next_thought:
  1. Am I measurably closer to solving this problem than 15 steps ago?
  2. Is my current approach working, or am I stuck in a loop?
  3. What is the ONE most important thing to do next?

**How to respond to meta-cognitive prompts:**
- Honestly evaluate your progress with concrete evidence (not assumptions)
- If you haven't made progress, identify which assumption was WRONG
- If stuck in a pattern, CHANGE your approach (different files, different strategy, or rollback)
- Be specific about what you'll learn from your next action that you don't already know

**Critical**: These checkpoints exist to prevent wasted effort. Take them seriously and be willing to pivot when not making progress.

# Cognitive Tools for Knowledge Persistence

You have access to powerful cognitive tools designed to preserve knowledge across rollbacks and prevent retry loops:

## Strategy Memory

**Purpose**: Remember what approaches you've tried, even after rolling back changes.

**Tools**:
- **log_strategy(approach, reasoning)**: Record planned approach BEFORE implementing
  - Use when: About to make significant code changes
  - Example: "Update function in <file> at line <N>" because "this fixes the root cause"

- **mark_strategy_outcome(strategy_id, success, reason)**: Record whether it worked
  - Use when: After testing the strategy (tests pass/fail)
  - Example: Mark strategy #1 as failed: "Tests passed but broke edge case in rare input scenario"

- **list_attempted_strategies()**: Review all strategies and outcomes
  - Use when: After rollbacks (to see what doesn't work), during reflection, or when choosing next approach
  - Shows: Which strategies succeeded/failed/pending

**When to Use These Tools**:

1. **Before Making Changes** (Before edits):
   - Use `log_strategy` to record your planned approach

2. **After Testing** (After running tests):
   - Use `mark_strategy_outcome` to record whether strategy worked

3. **During Meta-Cognitive Checkpoints** (Every 15 steps):
   - Use `list_attempted_strategies` to avoid retrying failed approaches

4. **After Rollbacks**:
   - IMMEDIATELY use `list_attempted_strategies` to see what you tried
   - This prevents retry loops since file state resets but cognitive state persists

**Critical**: These tools create institutional memory that survives rollbacks. Use them consistently to avoid wasting effort.

# Step Efficiency
You have a limited step budget (target: 10 steps, maximum: 20 steps). Prioritize simpler, faster solutions and make forward progress with each step. Test frequently to catch issues early. Don't over-investigate - once you understand the issue, implement the fix.

Here is the problem statement:
{problem_statement}

# Response Format Requirements
{format_prompt}
"""
)
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
## Role & Mission
You are a **senior bug-fix engineer** working on a local repository (**no internet access**).

**Mission**: Fix the bug described in the Problem Statement with minimal, correct changes and no regressions.

## Quick Workflow (MUST FOLLOW)
0) **Plan**: call `create_execution_plan` before edits. Use `get_execution_plan` to review and `update_execution_plan` to revise as you learn.
1) **Investigate**: use `grep_search` + `get_file_content` / `get_function_body` to locate working vs broken behavior.
2) **Pre-edit gates**: run `analyze_change_impact` (and `prove_boundary_localization` if required).
3) **Edit**: apply a minimal change using `apply_bug_fix_edit` / `apply_code_edit`.
4) **Test**: run `run_tests` (or `run_code`) soon after edits.
5) **Finish**: only call `finish` when verification is clean.

## Core Principles
1. Think step-by-step and keep your reasoning aligned with the workflow above.
2. Be systematic: do not skip gates or testing.
3. Test rigorously: insufficient testing is the #1 failure mode.
4. Stay focused: fix only what the problem requires.

## Pre-Edit Guidance
Before editing, you SHOULD (strongly recommended):
- Read the relevant region using `get_file_content` (or `get_function_body` for the exact symbol).
- Run `analyze_change_impact(file_path=..., symbol_name=...)` for any symbol with medium/high fanout.
- If the analysis indicates multiple semantics/modes, run `prove_boundary_localization(...)` BEFORE broad/shared edits.

These steps improve quality, but the agent must be able to iterate; avoid “bureaucracy blocks” that waste step budget.

## Critical Strategy Rules (MUST FOLLOW)
You must follow these rules for the entire run:

1) **Preserve existing behavior by default**
- Treat the current behavior as a contract relied upon by other parts of the repository.
- Add the missing/correct behavior **without breaking existing behaviors**, unless you have strong evidence the old behavior is universally wrong.

2) **Do not merge distinct behaviors into one**
- If the code supports multiple variants/interpretations, your fix must preserve them all.
- Prefer tightening the discriminator at a boundary, or route only the affected operation through a specialized path.
- Avoid broad “one-size-fits-all” edits to shared logic used by multiple variants.

3) **Use impact radius (fanout) to decide focused vs broad**
- Before changing a symbol, consult change-impact analysis output (use `analyze_change_impact`).
- If fanout is high: default to a **focused/localized** fix (boundary layer/entry-point/caller-side helper/variant-specific helper).
- Change shared behavior only when you can justify it for **all** known usage shapes.

4) **Canary preservation is mandatory**
- Use the example references from change-impact analysis as canaries.
- Your edit must not break the canaries unless you explicitly justify why the previous behavior was incorrect for them.

5) **No toy validation**
- Do not “validate” by only printing intermediate strings or inspecting partial representations.
- Validation must exercise real repository behavior through the repository-defined verification mechanism.

6) **Fix strategy stabilization**
- Once you choose a fix strategy (focused vs broad + invariants), do not drift.
- If new evidence forces a change, explicitly revise the strategy and explain why.

7) **Required default approach (follow strictly)**
- Prefer a **boundary-localized change**: adjust behavior at the narrowest boundary where semantics must change (e.g., a handler/adapter/entry point), rather than changing shared utilities used by many usage shapes.
- Avoid broad “global semantic” changes to shared formatting/parsing/path-building logic unless change-impact evidence shows it is safe across all usage shapes.
- Avoid modifying verification sources unless explicitly required by the problem statement. Treat verification sources as the contract you must satisfy.
- Before `finish`: ensure a **passing verification run via `run_tests` occurred after the last code edit**. Do not rely on demos/scripts as the final proof.

8) **Required engineering discipline**
- **Explicit invariants before edits**: write down what must remain true (existing behavior, interface/contract shape, error types, performance constraints) and keep your patch consistent with those invariants.
- **Localize the blast radius**:
  - Fix at the *boundary of the bug* (boundary layer/entry point/edge-case handling) instead of changing shared utilities.
  - If you touch shared logic, treat it as a widely depended-on contract: verify all major usage shapes (expressions, annotations, ordering/grouping, subqueries, etc.).
- **Respect provenance in multi-variant code**:
  - If the same token/value can mean different things depending on how it was produced (direct args vs transform chain), your fix must not collapse those meanings into one rule.
- **Small diffs + measurable checkpoints**:
  - Make one minimal conceptual change, then verify. Don't stack multiple conceptual changes before running verification.
  - If a change causes regressions, roll back and re-localize rather than patching regressions with more global logic.
- **Verification discipline**:
  - Ad-hoc reproduction scripts are *supplemental* only; repository-defined verification is authoritative.
  - Run canaries after any change with medium/high fanout or any edit that changes shared semantics.

## Detailed Guides (use as a checklist; keep steps concrete)

### 1) Deeply Understand the Problem
- Restate the bug as: **input → expected behavior → actual behavior**.
- Identify success criteria (what must change, what must not change).

### 2) Codebase Investigation (CRITICAL: find a working analogue)
- Search for key terms from the issue throughout the codebase.
- Find similar functionality that WORKS correctly; use it as the template.
- Trace to where mechanisms are defined, not just called.

**Trace from final output backwards to root cause:**
- Trace working feature from output back to generator.
- Trace broken feature from output back to generator.
- Compare paths and find the **first divergence point**.

### 3) Root Cause Verification (do not patch symptoms)
- Trace the complete data flow for both working and broken.
- Confirm the exact divergence point.
- Keep tracing until you can explain the root cause precisely.

### 4) Develop a Detailed Plan
- Keep the plan simple and verifiable.
- Include: investigation steps, suspected root cause, fix approach, verification strategy.
- Use `get_execution_plan` to review your plan before implementing, and `update_execution_plan` when your understanding changes.

### 5) Making Code Changes (minimal, focused)
- Read the file region before editing.
- Prefer small targeted edits; avoid refactors.
- Use the working code pattern when possible.
- Fix all instances if the root cause is systemic (search the repo to confirm).

### 6) Debugging (root cause, not symptoms)
- If results are unexpected, revisit assumptions and the divergence point.
- Use `run_code` only for targeted repro/inspection; do not rely on toy validation.

### 7) Testing
- Run verification frequently using the repository-defined mechanism.
- After each edit, verify with `run_tests` (or `run_code`) as soon as feasible.
- If tests fail, analyze failures and revise the patch.

### 8) Final Verification
- Confirm the root cause is fixed.
- Confirm no regressions.

### 9) Final Reflection
- Consider edge cases not covered by existing tests.
- Remember there are hidden tests.

## Step Efficiency
You have a limited step budget. Prioritize simpler, faster solutions and make forward progress with each step.
Batch tool calls when possible. Do not over-investigate once you have a verified root cause and a clear fix.

# Tool Documentation (authoritative)
{tools_docs}

# Problem Statement (authoritative)
{problem_statement}

# Response Format Requirements (authoritative)
{format_prompt}
"""
)
STOP_INSTRUCTION = textwrap.dedent(
    """
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- You can make MULTIPLE tool calls in one response using tool_call_1, tool_call_2, tool_call_3, etc.
- For efficiency: Batch related operations together (e.g., edit + test in ONE response)
- Format: next_thought: ... followed by one or more tool_call_N blocks
"""
)
FORMAT_PROMPT_FIX = textwrap.dedent(
    """
**CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
## Response Formats
### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
### Format 2: Single Tool Call (Legacy, less efficient)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
## When to Use Multiple Tool Calls
**ALWAYS batch these operations:**
1. **Edit + Test**: After code edit, MUST test in same response
2. **Multiple Searches**: Batch all search operations together
3. **Multiple File Reads**: Read all needed files at once
4. **Multiple Tests**: Run all test files together
## Examples
✅ **Excellent - Edit and Test Together**:
next_thought: I'll fix the bug and immediately verify with tests
tool_call_1:
    tool_name: apply_code_edit
    tool_args: {"file_path": "abcd.py", "search": "old_code", "replace": "fixed_code"}
tool_call_2:
    tool_name: run_code
    tool_args: {"content": "test_content", "file_path": "file.js", "run_command": ["node", "file.js"]}
✅ **Good - Batch Multiple Searches**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function problematic_func' ."}
tool_call_2:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'problematic_func(' ."}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "abcd.js"}
❌ **Bad - One tool per response (too slow)**:
Response 1:
next_thought: Let me edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", ...}
Response 2 (next turn):
next_thought: Now let me test it
next_tool_name: run_code
...  # ← Should have been in previous response!
## Critical Rules
- Use multiple tool_call_N when possible (tool_call_1, tool_call_2, tool_call_3, ...)
- After any edit: MUST include test in same response
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
Now let's start.

Here is the problem statement:
```
{problem_statement}
```
"""
)
CREATE_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
Now let's start.
```
{problem_statement}
```
"""
)

def generate_function_behaviours(initial_structure: str, problem_statement: str) -> str:
    """Generate function behaviours for all functions in one LLM call."""
    prompt = f"""Problem Statement:
        {problem_statement}

        Initial Structure (Code Skeleton):
        {initial_structure}

        Analyze the code skeleton and provide step-by-step behavior including the final return value for each function/method defined in it.

        Return the response as a JSON dict with the following format:
        {{
            "function_name_1": {{
                "steps": [
                    "Step 1: ...",
                    "Step 2: ...",
                    "Step 3: ..."
                ]
            }},
            "ClassName.method_name": {{
                "steps": [
                    "Step 1: ...",
                    "Step 2: ..."
                ]
            }}
        }}

        Important guidelines:
        - For standalone functions: use just the function name as the key
        - For class methods: use "ClassName.method_name" format as the key
        - Each function should have a "steps" array containing strings
        - Each step should be a clear, detailed description of what the function does
        - Focus on the logical flow and behavior, not implementation details
        - Only include must required steps that is related to final result.
        - Be comprehensive and specific
        - Include the final return value of the function/methods
        """

    messages = [{"role": "user", "content": prompt}]

    retry = 0
    selected_model = QWEN_MODEL_NAME
    max_retries = 10

    logger.info("Generating function behaviours in a single LLM call...")
    print("🚀 Generating function behaviours for all functions...")

    while retry < max_retries:
        try:
            # Call LLM once for all functions
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, timeout=300)

            # Clean up response - remove markdown code blocks if present
            response_cleaned = response.replace("```json", "").replace("```", "").strip()

            # Parse JSON response
            try:
                function_behaviours = json.loads(response_cleaned)

                # Validate that we got a dictionary
                if isinstance(function_behaviours, dict):
                    logger.info(f"✅ Successfully generated behaviours for {len(function_behaviours)} functions")
                    print(f"✅ Generated behaviours for {len(function_behaviours)} functions")
                    return function_behaviours
                else:
                    logger.warning(f"Response is not a dictionary (attempt {retry + 1}/{max_retries})")
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON object from text
                json_match = re.search(r'\{[^{}]*"[^"]*"[^{}]*\{[^{}]*"steps"[^{}]*\}[^{}]*\}', response_cleaned, re.DOTALL)
                if json_match:
                    try:
                        function_behaviours = json.loads(json_match.group())
                        if isinstance(function_behaviours, dict):
                            logger.info(f"✅ Successfully generated behaviours for {len(function_behaviours)} functions (extracted)")
                            print(f"✅ Generated behaviours for {len(function_behaviours)} functions")
                            return function_behaviours
                    except json.JSONDecodeError:
                        pass

                # If JSON extraction failed, log and retry
                logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {response_cleaned[:500]}")

        except Exception as e:
            logger.warning(f"Error in generate_function_behaviours (attempt {retry + 1}/{max_retries}): {e}")

        retry += 1
        if retry < max_retries:
            # Try different model on retry (after 7 attempts)
            if retry > 7:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(0.5)  # Small delay before retry

    # Return default value if all retries failed
    logger.warning("Failed to generate function behaviours after all retries, returning empty dict")
    import traceback

    logger.error(traceback.format_exc())
    return {}

def validate_initial_structure_implementation(
    initial_structure: Dict[str, str],
    modified_files: set,
    model: Model = GLM_MODEL_NAME,
) -> tuple[bool, str]:
    """
    Validates that the modified files correctly implement the code skeleton from initial_structure.
    Only validates files that are in both modified_files and initial_structure.
    Returns (is_valid, validation_message).
    """
    try:
        files_to_validate = {f for f in modified_files if f in initial_structure}

        if not files_to_validate:
            # No modified files to validate, or no overlap with initial_structure
            return True, "No files to validate (no modified files match initial structure)"

        # Read current file contents for modified files only
        current_structure = {}
        initial_structure_subset = {}
        for file_path in files_to_validate:
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        current_structure[file_path] = f.read()
                    initial_structure_subset[file_path] = initial_structure[file_path]
                else:
                    return False, f"File {file_path} from initial structure does not exist."
            except Exception as e:
                return False, f"Error reading file {file_path}: {e}"

        # Prepare comparison content - only for modified files
        comparison_content = "## Initial Structure (Expected Code Skeleton)\n\n"
        comparison_content += "NOTE: Only validating files that were modified by the agent.\n\n"
        for file_path, content in initial_structure_subset.items():
            comparison_content += f"### File: {file_path}\n```\n{content}\n```\n\n"

        comparison_content += "\n## Current Implementation (Actual Code)\n\n"
        for file_path, content in current_structure.items():
            comparison_content += f"### File: {file_path}\n```\n{content}\n```\n\n"

        # Create validation prompt
        validation_prompt = textwrap.dedent(
            """
        You are a code validation expert. Your task is to STRICTLY validate whether the current implementation 
        correctly and EXACTLY implements the code skeleton provided in the initial structure.
        
        The validation must check that the code skeleton structure matches EXACTLY:
        1. All classes from the initial structure are present with the same names and based from the same base class
        2. All functions/methods from the initial structure are present with the same names
        3. All function/method parameters from initial structure match exactly (same parameter names, same order, same types if specified)
        4. The class hierarchy and inheritance structure matches exactly
        
        CRITICAL REQUIREMENTS:
        - The functions in code skeleton (class definitions, function signatures, parameter structures) must match EXACTLY
        - Implementation details (function bodies, logic) can differ, but the STRUCTURE must be identical
        - Any missing classes, functions, or parameters should be flagged as validation failure
        - Any renamed classes, functions, or parameters should be flagged as validation failure
        - Any changes to parameter order, names, or structure should be flagged as validation failure
        
        Things that are allowed:
        - *You can add as many new functions/methods as you wish that is not exist in code skeleton.*
        
        IMPORTANT: This is a STRICT structural validation. The skeleton must match exactly. Only implementation 
        details within function bodies can differ.
        
        Respond with a JSON object in this exact format:
        {{
            "is_valid": true/false,
            "message": "Detailed validation message explaining the result",
            "issues": ["List of specific issues found, if any"]
        }}
        
        {comparison_content}
        """
        ).format(comparison_content=comparison_content)

        messages = [
            {
                "role": "system",
                "content": "You are a code validation expert. Respond only with valid JSON.",
            },
            {"role": "user", "content": validation_prompt},
        ]

        retry = 0
        selected_model = model
        max_retries = 10

        while retry < max_retries:
            try:
                # Call LLM for validation
                response, _ = EnhancedNetwork.make_request(messages, model=selected_model)

                # Parse response
                try:
                    # Extract JSON from response - try to find JSON object with balanced braces
                    response_cleaned = response.replace("```json", "").replace("```", "").strip()
                    # Try to find JSON object starting with {
                    json_start = response_cleaned.find("{")
                    if json_start >= 0:
                        # Use the existing balanced braces extraction method
                        json_str = EnhancedNetwork._extract_balanced_braces(response_cleaned, json_start)
                        if json_str:
                            response_cleaned = json_str
                    validation_result = json.loads(response_cleaned)

                    # Validate that we got the expected structure
                    if isinstance(validation_result, dict):
                        is_valid = validation_result.get("is_valid", False)
                        message = validation_result.get("message", "Validation completed")
                        issues = validation_result.get("issues", [])

                        if issues:
                            message += "\n\nIssues found:\n" + "\n".join(f"- {issue}" for issue in issues)

                        return is_valid, message
                    else:
                        logger.warning(f"Response is not a dictionary (attempt {retry + 1}/{max_retries})")
                except (json.JSONDecodeError, KeyError) as e:
                    # If JSON parsing fails, try fallback parsing
                    if retry < max_retries - 1:
                        logger.warning(f"Failed to parse validation response (attempt {retry + 1}/{max_retries}): {e}")
                    else:
                        # Last attempt - try fallback text parsing
                        logger.warning(f"Failed to parse validation response: {e}, response: {response}")
                        # Fallback: try to determine validity from response text
                        if "is_valid" in response.lower() and "true" in response.lower():
                            return True, "Validation passed (parsed from response)"
                        elif "is_valid" in response.lower() and "false" in response.lower():
                            return False, f"Validation failed (parsed from response): {response[:500]}"
                        else:
                            # If all retries failed, return error
                            break

            except Exception as e:
                logger.warning(f"Error in validate_initial_structure_implementation (attempt {retry + 1}/{max_retries}): {e}")

            retry += 1
            if retry < max_retries:
                # Try different model on retry (after 7 attempts)
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)  # Small delay before retry

        # Return default value if all retries failed
        logger.warning("Failed to validate initial structure implementation after all retries")
        return True, "Could not parse validation response after all retries"

    except Exception as e:
        logger.error(f"Error in validate_initial_structure_implementation: {e}")
        import traceback

        return True, f"Validation error: {str(e)}\n{traceback.format_exc()}"

def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours: Optional[Dict[str, str]] = None,
    files_to_modify=[],
    root_cause_analysis: Optional[str] = None,
):
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repo_dir = repo_path.split("/")[-1]
    if os.path.exists(repo_dir):
        logger.info(f"📂 [WORKFLOW] Changing to repo directory: {repo_dir}")
        os.chdir(repo_dir)
    logger.info("⚙️ [WORKFLOW] Setting up agent environment...")

    set_env_for_agent()

    global run_id, _current_tool_manager
    print("🎯 [WORKFLOW] fix_task_solve_workflow started")
    logger.info("🎯 [WORKFLOW] fix_task_solve_workflow started")
    run_id = run_id_1
    logger.info(f"🆔 [WORKFLOW] Run ID set: {run_id}")

    # ========== PROBLEM DECOMPOSITION PHASE ==========
    # Run structured analysis before the main agent loop
    decomposition = None
    decomposition_text = ""
    try:
        logger.info("🔍 [WORKFLOW] Starting problem decomposition...")
        decomposition = _problem_decomposer.decompose(problem_statement)
        decomposition_text = _problem_decomposer.format_for_prompt(decomposition)
        logger.info("✅ [WORKFLOW] Problem decomposition completed")
    except Exception as e:
        logger.warning(f"⚠️ [WORKFLOW] Problem decomposition failed: {e}")
        pass  # Decomposition is optional enhancement, don't fail on errors

    logger.info("🧠 [WORKFLOW] Initializing EnhancedCOT...")
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    logger.info("🛠️ [WORKFLOW] Creating FixTaskEnhancedToolManager with available tools...")
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "analyze_edge_cases",
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "analyze_change_impact",
            "prove_boundary_localization",
            "find_symbol_references",
            "profile_performance",
            "rank_fix_confidence",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "apply_bug_fix_edit",
            "modify_test_case",
            "compare_with_working_version",
            "create_new_file",
            "run_code",
            "run_tests",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "finish",
        ],
        initial_structure=initial_structure,
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
        is_fix_task=True,
        cot=cot,
    )
    _current_tool_manager = tool_manager

    tool_manager.problem_decomposition = decomposition

    logger.info("📝 [WORKFLOW] Formatting system prompt...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )

    # Build enhanced problem with decomposition
    enhanced_problem = problem_statement
    if decomposition_text:
        logger.info("📊 [WORKFLOW] Adding decomposition analysis to problem statement...")
        enhanced_problem = problem_statement + "\n\n---\n\n# Structured Problem Analysis\n\n" + decomposition_text
    if enhancement:
        logger.info("✨ [WORKFLOW] Applying enhancement to problem statement...")
        enhanced_problem = enhanced_problem + "\n\n---\n\n# Additional Context\n\n" + enhancement
    logger.info("📋 [WORKFLOW] Creating instance prompt...")
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
    if root_cause_analysis:
        instance_prompt += "\n\n---\n\n# Preliminary analysis on the git repository for the given problem statement\n\n" + root_cause_analysis
    print("🚀 [WORKFLOW] Executing agent workflow...")
    logger.info("🚀 [WORKFLOW] Executing agent workflow...")
    patch, is_success = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [QWEN_MODEL_NAME, KIMI_MODEL_NAME],
        log_prefix="FIX_MAIN_AGENT",
        initial_structure=initial_structure,
        function_behaviours=function_behaviours,
        files_to_modify=files_to_modify,
    )
    print("✅ [WORKFLOW] fix_task_solve_workflow completed")
    logger.info("✅ [WORKFLOW] fix_task_solve_workflow completed")
    return patch, is_success

def set_env_for_agent():
    logger.debug("Setting up environment for agent")

    work_dir = os.getcwd()
    original_cwd = os.getcwd()

    pythonpath = os.environ.get("PYTHONPATH", "")
    if work_dir not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{work_dir}:{pythonpath}"

    # Optional lib dir
    lib_dir = os.path.join(work_dir, "lib")
    if os.path.exists(lib_dir) and lib_dir not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] += f":{lib_dir}"

    # Write sitecustomize.py
    with open(os.path.join(work_dir, "sitecustomize.py"), "w") as f:
        f.write(VERSION_COMPATIBILITY_FIX)

    try:
        os.chdir(work_dir)

        if not os.path.exists(".git"):
            logger.info("Initializing git repository")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.warning(f"Error setting up environment: {e}")
    finally:
        os.chdir(original_cwd)

def validate_before_finish(
    initial_structure: Optional[Dict[str, str]], files_to_modify: List[str], modified_files: set, function_behaviours: dict
) -> tuple[bool, str]:
    if not initial_structure:
        return False, ""

    # Step 1: Validate initial structure
    print("🔍 [WORKFLOW] Validating initial structure implementation...")
    logger.info("🔍 [WORKFLOW] Validating initial structure implementation...")
    logger.info(f"🔍 [WORKFLOW] Modified files: {modified_files}")
    logger.info(f"🔍 [WORKFLOW] Files in initial_structure: {set(initial_structure.keys())}")
    is_valid, validation_message = validate_initial_structure_implementation(initial_structure, modified_files)

    if not is_valid:
        print(f"❌ [WORKFLOW] Initial structure validation failed: {validation_message}")
        logger.warning(f"❌ [WORKFLOW] Initial structure validation failed: {validation_message}")
        validation_observation = (
            f"VALIDATION FAILED: The implementation does not correctly match the initial structure.\n\n"
            f"Validation Result:\n{validation_message}\n\n"
            f"Please review the initial structure requirements and ensure all code skeleton elements "
            f"are implemented correctly and exactly as specified. Do not call the finish tool until "
            f"the validation passes."
        )
        return True, validation_observation

    # Step 1 passed
    print(f"✅ [WORKFLOW] Initial structure validation passed: {validation_message}")
    logger.info(f"✅ [WORKFLOW] Initial structure validation passed: {validation_message}")

    # Step 2: Combined validation (completeness + undefined functions + logic)
    logger.info("🔍 [WORKFLOW] Validating implementation completeness, dependencies, and logic...")
    all_incomplete_functions = []
    all_undefined_functions = []
    all_logic_issues = []

    for modified_file in modified_files:
        try:
            if modified_file not in files_to_modify:
                continue

            with open(modified_file, "r") as f:
                code = f.read()

            # Run combined validation
            validation_results = validate_implementation_and_dependencies(code)
            print(f"Validate Implementation and Dependencies results: \n\n{json.dumps(validation_results, indent=4)}")

            # Collect incomplete functions
            incomplete_funcs = validation_results.get("incomplete_functions", [])
            all_incomplete_functions.extend([x.get("name", "") for x in incomplete_funcs])

            # Collect undefined functions
            undefined_funcs = validation_results.get("undefined_functions", [])
            all_undefined_functions.extend(undefined_funcs)

            # Collect logic issues
            logic_issues = validation_results.get("logic_issues", [])
            all_logic_issues.extend(logic_issues)

        except Exception as e:
            logger.error(f"Error validating {modified_file}: {e}")
            pass

    # Check if there are any issues
    has_incomplete = len(all_incomplete_functions) > 0
    has_undefined = len(all_undefined_functions) > 0
    has_logic_issues = len(all_logic_issues) > 0

    validation_failed = has_incomplete or has_undefined or has_logic_issues

    logger.info(f"🔍 [WORKFLOW] Validation Results:")
    logger.info(f"  - Incomplete functions: {len(all_incomplete_functions)}")
    logger.info(f"  - Undefined functions: {len(all_undefined_functions)}")
    logger.info(f"  - Logic issues: {len(all_logic_issues)}")

    if not validation_failed:
        return False, ""

    # Build combined validation observation
    validation_observation = "⚠️ **IMPLEMENTATION ISSUES DETECTED** ⚠️\n\n"

    # Section 1: Incomplete Functions
    if has_incomplete:
        validation_observation += f"## 1. INCOMPLETE IMPLEMENTATIONS ({len(all_incomplete_functions)} function(s))\n\n"
        validation_observation += "The following functions need proper implementation:\n\n"

        for idx, function_name in enumerate(all_incomplete_functions, 1):
            validation_observation += f"{idx}. **{function_name}**\n"
            expected_behaviour = function_behaviours.get(function_name)

            if expected_behaviour:
                if isinstance(expected_behaviour, dict) and "steps" in expected_behaviour:
                    validation_observation += "   Expected behaviour:\n"
                    for step_idx, step in enumerate(expected_behaviour["steps"], 1):
                        if isinstance(step, str):
                            validation_observation += f"   - Step {step_idx}: {step}\n"
                        elif isinstance(step, dict) and "description" in step:
                            validation_observation += f"   - Step {step_idx}: {step['description']}\n"
                        else:
                            validation_observation += f"   - Step {step_idx}: {step}\n"
                else:
                    validation_observation += f"   Expected behaviour: {expected_behaviour}\n"
            else:
                validation_observation += "   Expected behaviour: Not specified\n"
            validation_observation += "\n"

    # Section 2: Undefined Functions
    if has_undefined:
        validation_observation += f"\n## 2. UNDEFINED FUNCTION CALLS ({len(all_undefined_functions)} function(s))\n\n"
        validation_observation += "The following functions are being called but are NOT defined:\n\n"

        for idx, undefined_func in enumerate(all_undefined_functions, 1):
            func_name = undefined_func.get("name", "unknown")
            code_snippet = undefined_func.get("code_snippet", "N/A")
            validation_observation += f"{idx}. **{func_name}**\n"
            validation_observation += f"   Used in: `{code_snippet}`\n\n"

    # Section 3: Logic Issues
    if has_logic_issues:
        validation_observation += f"\n## 3. LOGIC ERRORS ({len(all_logic_issues)} issue(s))\n\n"
        validation_observation += "The following logic errors were detected:\n\n"

        for idx, logic_issue in enumerate(all_logic_issues, 1):
            func_name = logic_issue.get("function_name", "unknown")
            issue_type = logic_issue.get("issue_type", "unknown")
            description = logic_issue.get("description", "N/A")
            problematic_code = logic_issue.get("problematic_code", "N/A")

            validation_observation += f"{idx}. **{func_name}** - {issue_type}\n"
            validation_observation += f"   Issue: {description}\n"
            validation_observation += f"   Code: `{problematic_code}`\n\n"

    # Action required
    validation_observation += "\n**ACTION REQUIRED:**\n"
    if has_incomplete:
        validation_observation += "- Implement all incomplete functions according to their expected behaviours\n"
    if has_undefined:
        validation_observation += "- Define all missing functions or fix the function calls\n"
    if has_logic_issues:
        validation_observation += "- Fix all logic errors and return type mismatches\n"
    validation_observation += "\nDo not call the finish tool until all issues are resolved."

    print(f"🔍 [WORKFLOW] Validation Observation: \n\n{validation_observation}")

    return validation_failed, validation_observation

def validate_implementation_completeness(code: str) -> str:
    prompt = textwrap.dedent(
        """
        You are a code validation expert. Your task is to analyze ALL functions/methods in the provided code and determine their implementation status.
        
        Follow these steps:
        1. Find ALL functions/methods defined in the code.
        2. For EACH function/method, analyze its body to determine:
           - is_empty: True if the function body is empty
           - is_only_null_return: True if the function body only returns None/null without doing anything else (no method calls, no assignments, no other operations)
           - reason: Brief explanation of what you found in the function body
        
        You must respond in JSON format with information for ALL functions.
        
        Return a JSON object in this exact format:
        {{
            "functions": [
                {{
                    "name": "function_name or ClassName.method_name",
                    "is_empty": true or false,
                    "is_only_null_return": true or false,
                    "reason": "reason why you think it's incomplete"
                }}
            ]
        }}
        
        Code to Analyze:
        ```
        {code}
        ```
        """
    ).format(code=code)

    messages = [
        {"role": "user", "content": prompt},
    ]

    retry = 0
    selected_model = QWEN_MODEL_NAME
    max_retries = 10

    logger.info("Validating implementation completeness...")
    print("🔍 Checking for incomplete function implementations...")

    while retry < max_retries:
        try:
            # Call LLM for validation
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, timeout=120)

            # Clean up response - remove markdown code blocks if present
            response_cleaned = response.replace("```json", "").replace("```", "").strip()

            # Parse and validate JSON response
            try:
                validation_result = json.loads(response_cleaned)

                # Validate that we got the expected structure
                if isinstance(validation_result, dict):
                    validation_result.setdefault("functions", [])

                    print(f"[DEBUG] Validation Results:\n\n {json.dumps(validation_result, indent=4)} ")

                    # Filter to keep only incomplete functions (is_empty=True OR is_only_empty_return=True)
                    if "functions" in validation_result:
                        all_functions = validation_result["functions"]
                        incomplete_functions = [
                            func for func in all_functions if func.get("is_empty", False) or func.get("is_only_null_return", False)
                        ]
                        validation_result["functions"] = incomplete_functions
                        logger.info(f"Found {len(all_functions)} total functions, {len(incomplete_functions)} incomplete")
                        print(f"📊 Analyzed {len(all_functions)} functions, found {len(incomplete_functions)} incomplete")

                    return validation_result
                else:
                    logger.warning(f"Response is not a dictionary (attempt {retry + 1}/{max_retries})")
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON object from text
                json_match = re.search(r'\{[^{}]*"functions"[^{}]*\}', response_cleaned, re.DOTALL)
                if json_match:
                    try:
                        validation_result = json.loads(json_match.group())
                        if isinstance(validation_result, dict):
                            validation_result.setdefault("functions", [])
                            # Filter incomplete functions
                            if "functions" in validation_result:
                                all_functions = validation_result["functions"]
                                incomplete_functions = [
                                    func for func in all_functions if func.get("is_empty", False) or func.get("is_only_null_return", False)
                                ]
                                validation_result["functions"] = incomplete_functions
                            return validation_result
                    except json.JSONDecodeError:
                        pass

                # If JSON extraction failed, log and retry
                logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {response_cleaned[:500]}")

        except Exception as e:
            logger.warning(f"Error in validate_implementation_completeness (attempt {retry + 1}/{max_retries}): {e}")

        retry += 1
        if retry < max_retries:
            # Try different model on retry (after 7 attempts)
            if retry > 7:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(0.5)  # Small delay before retry

    # Return default value if all retries failed
    logger.warning("Failed to validate implementation completeness after all retries, returning default values")
    import traceback

    logger.error(traceback.format_exc())
    return {
        "functions": [],
        "error": "Failed to parse LLM response after all retries",
        "summary": "Validation error: Failed to get response from LLM after all retries",
    }

def get_files_to_modify(problem_statement: str) -> tuple[str, list[str]]:
    tool_manager = FixTaskEnhancedToolManager(available_tools=["get_file_content", "list_directory_structure", "finish_find_files_to_fix"])
    FIND_FILES_TO_MODIFY = textwrap.dedent(
        """
        You are a helpful assistant that finds the files to modify related to the problem statement.
        You must check the directory structure using `list_directory_structure` tool and then determine which files are needed for the problem statement.
        You must then use the `finish_find_files_to_fix` tool to signal the completion of the file finding workflow execution.
        You have access to the following tools:-
        {tools_docs}
        {format_prompt}
        """
    ).format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_CREATE)
    try:
        cot = EnhancedCOT(latest_observations_to_keep=10, summarize_batch_size=10)
        instance_prompt = f"Problem Statement:\n{problem_statement}"
        result, __cached__ = execute_agent_workflow(
            cot,
            tool_manager,
            FIND_FILES_TO_MODIFY,
            instance_prompt,
            30,
            300,
            [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME],
            finish_tool_name="finish_find_files_to_fix",
            log_prefix="FINISH_FIND_FILES_TO_MODIFY",
        )
        if not result:
            return "", []
        if not isinstance(result, list):
            result = [result]
        contents = []
        for file_path in result:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contents.append(f"{file_path}\n{f.read()}")
            except Exception as e:
                logger.error(f"Failed to open file {file_path}: {e}")
        return "\n\n".join(contents), result
    except Exception as e:
        logger.error(f"Error in get files to modify: {e}")
        return "", []

def check_not_defined_functions(code: str) -> dict:
    prompt = f"""Analyze the following code and identify HELPER/UTILITY functions/methods that are CALLED but NOT implemented in the code.

        Code to analyze:
        ```
        {code}
        ```
        
        Steps to follow:
        1. Working through the codebase, find all function/method calls and list them in the reasoning steps.
        2. For each function/method call
            - Check if it is a standard library or third-party library function. If yes, continue with the next call.
            - Check if it is a direct call to the parent class method. If yes, continue with the next call.
            - Check if it is a call to a method of the class that is not defined in the code. If yes, continue with the next call.
            - Do not consider a parent class method as defined unless it is explicitly called using super or the parent class name. If the code calls that method is not implemented in the current class, flag it as undefined even if the parent class has it.
        
        CRITICAL: Return ONLY valid JSON in the exact format shown below. No explanations, reasoning, or markdown.
        
        Required JSON format (return this EXACTLY):
        {{
            "reasonings": [
                "reasoning step 1",
                "reasoning step 2"
            ],
            "undefined_functions": [
                {{
                    "name": "function_name",
                    "code_snippet": "result = function_name(x, y)"
                }}
            ]
        }}
        """

    messages = [{"role": "user", "content": prompt}]

    retry = 0
    selected_model = QWEN_MODEL_NAME
    max_retries = 10

    logger.info("🔍 Calling LLM to check for undefined functions...")
    print("🔍 Analyzing code for undefined functions using LLM...")

    while retry < max_retries:
        try:
            # Call LLM
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, timeout=120)

            # Clean up response - remove markdown code blocks if present
            response_cleaned = response.replace("```json", "").replace("```", "").strip()

            try:
                analysis = json.loads(response_cleaned)
                # Validate that we got the expected structure
                if isinstance(analysis, dict):
                    analysis.setdefault("reasonings", [])
                    analysis.setdefault("undefined_functions", [])
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON object from text
                json_match = re.search(r'\{[^{}]*"undefined_functions"[^{}]*\}', response_cleaned, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        analysis.setdefault("reasonings", [])
                        analysis.setdefault("undefined_functions", [])
                        return analysis
                    except json.JSONDecodeError:
                        pass

                # If JSON extraction failed, log and retry
                logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {response_cleaned[:500]}")

        except Exception as e:
            logger.warning(f"Error in check_not_defined_functions (attempt {retry + 1}/{max_retries}): {e}")

        retry += 1
        if retry < max_retries:
            # Try different model on retry (after 7 attempts)
            if retry > 7:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(0.5)  # Small delay before retry

    # Return default value if all retries failed
    logger.warning("Failed to get undefined functions analysis after all retries, returning default values")
    import traceback

    logger.error(traceback.format_exc())
    return {"reasonings": [], "undefined_functions": []}

def check_explicit_exception_handling_mention(problem_statement: str, model: str = QWEN_MODEL_NAME, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Check if a problem statement contains explicit mentions of exception handling.
    This is a focused check that specifically looks for explicit instructions about exceptions.

    Args:
        problem_statement: The problem statement text to analyze
        model: Model name to use for analysis
        temperature: Temperature for LLM inference

    Returns:
        Dictionary with explicit exception handling mention detection results
    """
    retry = 0
    selected_model = model
    max_retries = 10

    while retry < max_retries:
        try:
            messages = [
                {"role": "system", "content": EXPLICIT_EXCEPTION_HANDLING_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n\n{problem_statement}\n\nDetermine if this problem statement contains explicit mentions of exception handling. Return the JSON result.",
                },
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_clean[json_start : json_end + 1]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        result.setdefault("has_explicit_exception_mention", False)
                        result.setdefault("explicit_mentions", [])
                        result.setdefault("total_explicit_mentions", 0)
                        result.setdefault("confidence", 0.5)
                        result.setdefault("reasoning", "Analysis completed")
                        if "explicit_mentions" in result:
                            result["total_explicit_mentions"] = len(result["explicit_mentions"])
                        try:
                            confidence = float(result.get("confidence", 0.5))
                            result["confidence"] = max(0.0, min(1.0, confidence))
                        except (ValueError, TypeError):
                            result["confidence"] = 0.5
                        return result
                    else:
                        raise ValueError("Parsed JSON is not a dictionary")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {e}")
                    logger.warning(f"Response snippet: {response_clean[:200]}")
                except Exception as e:
                    logger.warning(f"Error processing JSON response (attempt {retry + 1}/{max_retries}): {e}")
            else:
                logger.warning(f"No JSON object found in response (attempt {retry + 1}/{max_retries})")
                logger.warning(f"Response snippet: {response_clean[:200]}")
        except Exception as e:
            logger.error(f"Error in check_explicit_exception_handling_mention (attempt {retry + 1}/{max_retries}): {e}")
        retry += 1
        if retry < max_retries:
            other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
            if other_models:
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
    logger.warning("Failed to get explicit exception handling mention analysis after all retries, returning default values")
    return {
        "has_explicit_exception_mention": False,
        "explicit_mentions": [],
        "total_explicit_mentions": 0,
        "confidence": 0.0,
        "reasoning": "Analysis failed - unable to parse LLM response",
        "error": "All retries failed",
    }

def execute_agent_workflow(
    cot: EnhancedCOT,
    tool_manager: EnhancedToolManager,
    system_prompt: str,
    instance_prompt: str,
    n_max_steps: int,
    timeout: int,
    models: List[str],
    log_prefix: str = "AGENT",
    finish_tool_name="finish",
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours=None,
    files_to_modify=[],
    cost_limit: float = 1.0,
    cost_usage_threshold: float = 0.15,
    reject_observation_token_threshold: int = 50000,
    save_observation_to_file_token_threshold: int = 4000,
) -> tuple[str, bool]:
    global run_id
    print(f"🚀 [WORKFLOW] execute_agent_workflow started (max_steps={n_max_steps}, timeout={timeout}s)")
    logger.info(f"🚀 [WORKFLOW] execute_agent_workflow started (max_steps={n_max_steps}, timeout={timeout}s)")
    logger.info(f"{log_prefix} Starting agent execution... ")
    start_time = time.time()
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = None
    next_tool_name = None
    next_tool_args = None
    modified_files = set()
    files_with_syntax_errors = set()
    current_model_index = 0

    def _safe_call_tool(tool_manager: EnhancedToolManager, tool_name: str, tool_args):
        tool_fn = tool_manager.get_tool(tool_name)
        if isinstance(tool_fn, str):
            return tool_fn

        if tool_args is None or tool_args == {}:
            return tool_fn()
        if not isinstance(tool_args, dict):
            return tool_fn()
        try:
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
        except Exception:
            allowed = set(tool_args.keys())
        cleaned = {k: v for k, v in tool_args.items() if k in allowed}
        try:
            for k in list(cleaned.keys()):
                v = cleaned[k]
                p = sig.parameters.get(k)
                ann = str(getattr(p, "annotation", ""))
                if v is not None and isinstance(v, str) and ("List" in ann or "list" in ann):
                    cleaned[k] = v.split() if v.strip() else []
        except Exception:
            pass
        return tool_fn(**cleaned) if cleaned else tool_fn()

    for step in range(n_max_steps):
        selected_model = models[current_model_index]
        elapsed_time = time.time() - start_time
        logger.info(f"📊 [WORKFLOW] Step {step}/{n_max_steps} - Elapsed: {elapsed_time:.2f}s/{timeout}s")
        logger.info("=" * 40 + f"[{log_prefix}] Step {step}" + "=" * 40)
        cost_usage = EnhancedNetwork.get_cost_usage()
        logger.info(
            f"[{log_prefix}] Elapsed time: {elapsed_time}/{timeout} seconds, Usage: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
        )
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0) * cost_limit - cost_usage_threshold:
            print(f"⚠️ [WORKFLOW] Cost limit exceeded: {cost_usage.get('used_cost_usd', 0)}/{cost_usage.get('max_cost_usd', 0)} USD")
            logger.warning(f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD")
            break
        if time.time() - start_time > timeout:
            print(f"⏱️ [WORKFLOW] Global timeout reached ({elapsed_time:.2f}s)")
            logger.warning(f"[{log_prefix}] Global timeout reached")
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break
        logger.info(f"💬 [WORKFLOW] Preparing messages for inference (step {step})...")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        if cot.is_thought_repeated():
            logger.warning(f"🔄 [WORKFLOW] Thought repeated {cot.repeated_thoughts} times - adjusting temperature")
            logger.info(f"[TEMPERATURE] Thought repeated {cot.repeated_thoughts} times")
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            temperature = 0.5
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]
                logger.info(f"🔄 [WORKFLOW] Switching to model index {model_idx}")
        else:
            temperature = 0.0
        try:
            logger.info(f"🤖 [WORKFLOW] Calling inference API (model: {selected_model}, temp: {temperature})...")
            inference_start_time = time.time()
            models_to_try = [selected_model] + [m for m in models if m != selected_model]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = EnhancedNetwork.inference(messages, model=models_to_try, run_id=run_id, temperature=temperature)
            logger.info("next_thought: %s", next_thought)
            logger.info(f"next_tool_name: {next_tool_name}")
            logger.info(f"next_tool_args {json.dumps(next_tool_args, indent=4)}")

            selected_model = used_model
            inference_duration = time.time() - inference_start_time
            logger.info(f"✅ [WORKFLOW] Inference completed in {inference_duration:.2f}s")
        except Exception as e:
            inference_duration = 0
            print(f"❌ [WORKFLOW] Inference error: {e}")
            logger.error(f"[{log_prefix}] Inference error: {e}")
        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]

        try:
            logger.info(f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s")
            logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
            logger.info(f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n")
            logger.info(f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n")
        except Exception as e:
            logger.error(f"[{log_prefix}] Error in logging tool information: {e}")

        tool_manager._current_step = step
        tool_manager._cot_snapshot_cache = [
            {
                "thought": t.next_thought,
                "tool": t.next_tool_name,
                "args": str(t.next_tool_args)[:200],
                "success": not t.is_error,
            }
            for t in cot.thoughts[-10:]
        ]
        if hasattr(tool_manager, "is_fix_task") and tool_manager.is_fix_task:
            if not tool_manager.cot:
                tool_manager.cot = cot
                tool_manager.solution_verifier = SolutionVerifier(cot=cot, problem_statement=tool_manager.problem_statement)
            elif tool_manager.cot != cot:
                tool_manager.cot = cot
                if tool_manager.solution_verifier:
                    tool_manager.solution_verifier.cot = cot
        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            logger.info(f"🔧 [WORKFLOW] Executing tool {idx+1}/{len(tool_names_list)}: {tool_name}")
            try:
                if '"' in tool_name or "'" in tool_name:
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = _safe_call_tool(tool_manager, tool_name, tool_args)
                if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                    file_path = tool_args["file_path"]
                    if "ok, code edit applied successfully" in str(observation).lower():
                        modified_files.add(file_path)
                        logger.info(f"✅ [WORKFLOW] Code edit applied successfully to: {file_path}")
                    elif "syntax error" in str(observation).lower():
                        files_with_syntax_errors.add(file_path)
                        logger.error(f"❌ [WORKFLOW] Syntax error detected in: {file_path}")

                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = f"Error: Tool output from '{tool_name}' exceeded token limit ({estimated_tokens} tokens > 50000 tokens limit). The response is too large to process. Please use more specific queries, target smaller file ranges, or break the request into smaller operations."
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    observation_path, line_count = tool_manager._save_large_observation(str(observation), tool_name)
                    observation = f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens > 4000 tokens limit). The full output has been saved to: {observation_path}. You can use search tool to find specific lines in the file and you can read this file using the get_file_content tool, but specify the start and end line numbers to read the file. The file has {line_count} lines."
                all_observations.append(observation)

            except EnhancedToolManager.Error as e:
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
                all_successful = False
            except Exception as e:
                import traceback

                error_traceback = traceback.format_exc()
                error_msg = f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                all_observations.append(error_msg)
                all_successful = False
        validation_failed = False
        if finish_tool_name in tool_names_list:
            if finish_tool_name == "finish_find_files_to_fix":
                logger.info("🎯 [WORKFLOW] finish_find_files_to_fix called")
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs, False
            elif finish_tool_name == "finish":
                print("🎯 [WORKFLOW] finish tool called")
                logger.info("🎯 [WORKFLOW] finish tool called")
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        if obs != "finish":
                            break
                        if initial_structure:
                            if tool_manager.validated_num >= 5:  # only validate up to 5 times
                                return tool_manager.get_final_git_patch(), True
                            tool_manager.validated_num += 1
                            validation_failed, validation_observation = validate_before_finish(
                                initial_structure=initial_structure,
                                modified_files=modified_files,
                                files_to_modify=files_to_modify,
                                function_behaviours=function_behaviours,
                            )
                            if validation_failed:
                                for i, (n, o) in enumerate(zip(tool_names_list, all_observations)):
                                    if n == finish_tool_name:
                                        all_observations[i] = validation_observation
                                        break
                                if "does not correctly match the initial structure" in validation_observation:
                                    break

                            if not validation_failed:
                                print("✅ [WORKFLOW] Workflow completed successfully, generating final patch...")
                                logger.info("✅ [WORKFLOW] Workflow completed successfully, generating final patch...")
                                return tool_manager.get_final_git_patch(), True

                        else:
                            return tool_manager.get_final_git_patch(), True
        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [f"Tool {i+1} ({tool_names_list[i]}):\n{obs}" for i, obs in enumerate(all_observations)]
            )
        logger.info(f"[{log_prefix}] Combined observation: {combined_observation}\n\n")
        cot.add_action(
            EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=combined_observation,
                is_error=not all_successful or validation_failed,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
            )
        )
        if validation_failed:
            continue
    logger.info("📝 [WORKFLOW] Workflow ended, generating final patch...")
    return tool_manager.get_final_git_patch(), False

def validate_implementation_and_dependencies(code: str) -> dict:
    try:
        print(f"Code to validate: \n\n{code}")
        # Step 1: Check implementation completeness
        completeness_result = validate_implementation_completeness(code)
        incomplete_functions = completeness_result.get("functions", [])

        # Step 2: Check for undefined functions
        undefined_result = check_not_defined_functions(code)
        undefined_functions = undefined_result.get("undefined_functions", [])
        print(f"Undefined Result: \n\n {json.dumps(undefined_result, indent=4)}")
        # undefined_functions = []

        # Step 3: Check for logic errors
        logic_issues = []
        # logic_issues = []

        # Combine results
        return {
            "incomplete_functions": incomplete_functions,
            "undefined_functions": undefined_functions,
            "logic_issues": logic_issues,
            "has_issues": len(incomplete_functions) > 0 or len(undefined_functions) > 0 or len(logic_issues) > 0,
        }
    except Exception as e:
        logger.error(f"Error in validate_implementation_and_dependencies: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"incomplete_functions": [], "undefined_functions": [], "logic_issues": [], "has_issues": False, "error": str(e)}

def create_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours: Optional[Dict[str, str]] = None,
    files_to_modify=[],
    cost_usage_threshold: float = 0.15,
    has_exception_handling_mention: bool = False,
):
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repo_dir = repo_path.split("/")[-1]
    if os.path.exists(repo_dir):
        logger.info(f"📂 [WORKFLOW] Changing to repo directory: {repo_dir}")
        os.chdir(repo_dir)
    logger.info("⚙️ [WORKFLOW] Setting up agent environment...")

    global run_id, _current_tool_manager
    print("🎯 [WORKFLOW] create_task_solve_workflow started")
    logger.info("🎯 [WORKFLOW] create_task_solve_workflow started")
    run_id = run_id_1
    logger.info(f"🆔 [WORKFLOW] Run ID set: {run_id}")
    logger.info("🧠 [WORKFLOW] Initializing EnhancedCOT...")
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    logger.info("🛠️ [WORKFLOW] Creating FixTaskEnhancedToolManager with available tools...")
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "analyze_edge_cases",
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "modify_test_case",
            "create_new_file",
            "run_code",
            "run_tests",
            "think",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "finish",
        ],
        initial_structure=initial_structure,
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
        has_exception_handling_mention=has_exception_handling_mention,
    )
    _current_tool_manager = tool_manager

    logger.info("📝 [WORKFLOW] Formatting system prompt...")
    system_prompt = CREATE_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )
    enhanced_problem = problem_statement
    if enhancement:
        logger.info("✨ [WORKFLOW] Applying enhancement to problem statement...")
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    logger.info("📋 [WORKFLOW] Creating instance prompt...")
    instance_prompt = CREATE_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
    if not has_exception_handling_mention:
        instance_prompt = (
            instance_prompt
            + "\n\n"
            + (
                """
            ### Exception Handling (VERY IMPORTANT) - safe behavior for invalid input
            - Do not raise or throw any errors.
            - Return the default value for the declared return type, such as a false value for booleans, a null equivalent for reference types, an empty string for text types, or an empty collection for collection types.
        """
            )
        )
    print("🚀 [WORKFLOW] Executing agent workflow...")
    logger.info("🚀 [WORKFLOW] Executing agent workflow...")
    patch, is_success = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [QWEN_MODEL_NAME, KIMI_MODEL_NAME],
        log_prefix="CREATE_MAIN_AGENT",
        initial_structure=initial_structure,
        function_behaviours=function_behaviours,
        files_to_modify=files_to_modify,
        cost_usage_threshold=cost_usage_threshold,
    )
    print("✅ [WORKFLOW] create_task_solve_workflow completed")
    logger.info("✅ [WORKFLOW] create_task_solve_workflow completed")
    return patch, is_success

def generate_problem_details_for_io(problem_statement: str, initial_structure: str) -> str:
    PROBLEM_DETAILS_PROMPT = textwrap.dedent(
        """
        Analyze the provided problem statement and project file structure.
        Identify all relevant questions that could be asked about the only outputs(not input) of the functions defined in the project, along with their accurate answers.
        - The output MUST BE accurate including exceptions, edge cases.

        Then, transform each question–answer pair into a concise descriptive statement that explains the function behavior.

        Output only the final list of descriptions.
        Do not include questions, answers, explanations, or any additional text.
        Highly emphasis important points in the description.
        """
    )
    USER_CONTENT = f"Problem Statement:\n\n{problem_statement}\n\nFile Structure:{initial_structure}"
    return ask_ai(PROBLEM_DETAILS_PROMPT, USER_CONTENT)

def check_problem_statement_has_examples(problem_statement: str, model: str = QWEN_MODEL_NAME, temperature: float = 0.0) -> Dict:
    messages = [
        {"role": "system", "content": EXAMPLES_CHECK_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n\n{problem_statement}\n\nAnalyze if this problem statement has clear examples and return the JSON result.",
        },
    ]
    retry = 0
    selected_model = model
    max_retries = 10

    while retry < max_retries:
        try:
            result, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            try:
                analysis = json.loads(result)
                return analysis
            except json.JSONDecodeError:
                json_match = re.search(r'\{[^{}]*"has_clear_examples"[^{}]*\}', result, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
                logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {result[:200]}")
        except Exception as e:
            logger.warning(f"Error in check_problem_statement_has_examples (attempt {retry + 1}/{max_retries}): {e}")
        retry += 1
        if retry < max_retries:
            if retry > 7:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(0.5)
    logger.warning("Failed to get examples analysis after all retries, returning default values")
    return {
        "has_clear_examples": None,
        "error": "Failed to parse LLM response after all retries",
        "raw_response": result if "result" in locals() else "",
    }

def check_io_format_consistency(problem_statement: str, function_signature: str, model: str = QWEN_MODEL_NAME, temperature: float = 0.0) -> Dict:
    """
    Check if the input/output formats in examples are consistent with the function signature.

    Args:
        problem_statement: The problem statement text containing examples
        function_signature: The function signature code
        model: Model name to use for analysis
        temperature: Temperature for LLM inference

    Returns:
        Dictionary with I/O format consistency analysis results
    """
    messages = [
        {"role": "system", "content": IO_FORMAT_CONSISTENCY_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement (with examples):\n\n{problem_statement}\n\nFunction Signature:\n\n{function_signature}\n\nAnalyze if the example input/output formats are consistent with the function signature. Return the JSON result.",
        },
    ]
    retry = 0
    selected_model = model
    max_retries = 10

    while retry < max_retries:
        try:
            result, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            try:
                analysis = json.loads(result)
                return analysis
            except json.JSONDecodeError:
                json_match = re.search(r'\{[^{}]*"has_format_mismatch"[^{}]*\}', result, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
                logger.warning(f"Failed to parse JSON from response (attempt {retry + 1}/{max_retries}): {result[:200]}")
        except Exception as e:
            logger.warning(f"Error in check_io_format_consistency (attempt {retry + 1}/{max_retries}): {e}")
        retry += 1
        if retry < max_retries:
            if retry > 7:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(0.5)
    logger.warning("Failed to get I/O format consistency analysis after all retries, returning default values")
    return {
        "has_format_mismatch": None,
        "error": "Failed to parse LLM response after all retries",
        "raw_response": result if "result" in locals() else "",
    }

def get_complexity_category(score: float) -> str:
    """Categorize problem based on complexity score."""
    if score >= 8.0:
        return "Very High (8.0-10.0)"
    elif score >= 6.0:
        return "High (6.0-7.9)"
    elif score >= 4.0:
        return "Medium (4.0-5.9)"
    elif score >= 2.0:
        return "Low (2.0-3.9)"
    else:
        return "Very Low (0.0-1.9)"

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    def extract_file_names_using_llm(initial_solution: str) -> list:
        retry = 0
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                file_names_prompt = f"""
                Extract the file names from the initial solution. Return only the file names in a list only.
                This is the initial solution:
                ```
                {initial_solution}
                ```
                Return only the file names in a list.
                Example:
                ```
                ["a.py", "b.js"]
                ```
                """
                result, _ = EnhancedNetwork.make_request(
                    messages=[{"role": "user", "content": file_names_prompt}],
                    model=selected_model,
                )
                return json.loads(result.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                retry += 1
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(1)
        return []

    if not initial_solution.strip():
        return []
    file_names = extract_file_names_using_llm(initial_solution)
    created_files = []
    current_file, content = None, []

    def write_file():
        if current_file and content:
            path = os.path.join(base_dir, current_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                file_content = "\n".join(content)
                file_content = file_content.rstrip() + "\n" if file_content.strip() else file_content
                f.write(file_content)
            created_files.append(path)
    filename_set = set(file_names)
    for fname in file_names:
        filename_set.add(fname.split("/")[-1])
    for line in initial_solution.split("\n"):
        stripped = line.strip()
        if stripped in filename_set:
            write_file()
            current_file = next(
                (f for f in file_names if f == stripped or f.endswith("/" + stripped) or f.split("/")[-1] == stripped),
                stripped,
            )
            current_file, content = current_file, []
        elif current_file:
            content.append(line)
    write_file()
    return created_files
def check_problem_type(problem_statement):
    def get_problem_type(problem_statement: str, enhancement: str) -> str:
        retry = 0
        PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
            """
            You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
            Classify development tasks as either:
            - FIX: If the PROJECT DESCRIPTION is about fixing a bug, creating a new functionality or improving the existing codebase.
            - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
            Output ONLY: "CREATE" or "FIX"
            """
        )
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                    {"role": "user", "content": f"{problem_statement}\n# Enhanced Problem: \n{enhancement}"},
                ]
                response, _ = EnhancedNetwork.make_request(messages, model=selected_model)
                if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                    retry += 1
                    logger.warning(f"Invalid response from get_problem_type (attempt {retry}/10): {response}")
                else:
                    return response
            except Exception as e:
                retry += 1
                logger.warning(f"Error in get_problem_type (attempt {retry}/10): {e}")
            if retry < 10:
                if retry > 7:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)
        return PROBLEM_TYPE_FIX
    type_count = {PROBLEM_TYPE_CREATE: 0, PROBLEM_TYPE_FIX: 0}
    enhancement = ""
    for _ in range(3):
        problem_type = get_problem_type(problem_statement, enhancement)
        type_count[problem_type] += 1
    if type_count[PROBLEM_TYPE_CREATE] > type_count[PROBLEM_TYPE_FIX]:
        return PROBLEM_TYPE_CREATE, enhancement
    elif type_count[PROBLEM_TYPE_FIX] > type_count[PROBLEM_TYPE_CREATE]:
        return PROBLEM_TYPE_FIX, enhancement
    return PROBLEM_TYPE_FIX, enhancement

def process_fix_task(problem_text: str, enhancement: str):
    cwd = os.getcwd()
    global run_id, agent_start_time
    print("🔧 [WORKFLOW] process_fix_task started")
    logger.info("🔧 [WORKFLOW] process_fix_task started")
    patch_text = ""
    try:
        results = []
        for attempt in range(3):
            cost_usage = EnhancedNetwork.get_cost_usage()
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > 850 or cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0) - 0.6:
                break
            os.system("git reset --hard")
            os.system("git clean -fd")
            remaining_time = max(10, 1250 - elapsed_time)
            logger.info(
                f"⏱️ [WORKFLOW] Attempt {attempt + 1}/3 - Elapsed time: {elapsed_time:.2f}s, Starting fix_task_solve_workflow with timeout: {remaining_time:.2f}s..."
            )
            patch_text, is_success = fix_task_solve_workflow(
                problem_text, timeout=remaining_time, run_id_1=run_id, enhancement=enhancement, should_review=True
            )
            modified_files = EnhancedToolManager.get_modified_files_list()
            modified_files_content = {}
            result = ""
            if modified_files:
                temp_file_ops = FileOperationsUtil(new_files_created=[])
                temp_file_ops.file_system_manager = FileSystemManager()
                temp_file_ops.search_manager = SearchManager()
                for file_path in modified_files:
                    file_content = temp_file_ops.get_file_content(file_path, limit=-1)
                    modified_files_content[file_path] = file_content
                result = "\n\n".join([f"{file}\n{content}" for file, content in modified_files_content.items()])
            observation = "Success" if is_success else "Failed"
            if len(results) == 0 or is_success:
                results.append(
                    {
                        "solution_code": result,
                        "patch": patch_text,
                        "modified_files": modified_files,
                        "modified_files_content": modified_files_content,
                        "summary": observation,
                    }
                )
        best_solution = select_best_solution(results, problem_text)
        os.system("git reset --hard")
        os.system("git clean -fd")
        if best_solution and best_solution.get("modified_files_content"):
            file_ops = FileOperationsUtil(new_files_created=[])
            for file_path, file_content in best_solution["modified_files_content"].items():
                try:
                    file_ops.save(file_path, file_content)
                    logger.info(f"[PROCESS_TASK] Restored file: {file_path}")
                except Exception as e:
                    logger.error(f"[PROCESS_TASK] Error restoring file {file_path}: {e}")
            if best_solution.get("patch"):
                patch_text = best_solution["patch"]
        logger.info("🧹 [WORKFLOW] Resetting git state...")
        print("✅ [WORKFLOW] process_fix_task completed successfully")
        logger.info("✅ [WORKFLOW] process_fix_task completed successfully")
    except Exception as e:
        print(f"❌ [WORKFLOW] Error in process_fix_task: {e}")
        logger.error(f"Error in process_fix_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
        logger.info("📁 [WORKFLOW] Restored original working directory")
    return patch_text

def get_category_from_score(score: float) -> str:
    """Categorize problem based on granularity score."""
    if score >= 9.0:
        return "Excellent (9.0-10.0)"
    elif score >= 7.0:
        return "Good (7.0-8.9)"
    elif score >= 4.0:
        return "Moderate (4.0-6.9)"
    elif score >= 1.0:
        return "Poor (1.0-3.9)"
    else:
        return "Insufficient (0.0-0.9)"

def check_granularity_of_statement(problem_statement: str) -> Dict[str, Any]:
    """
    Evaluates whether the problem_statement has sufficient granularity to be solvable.
    Uses an LLM to assess the level of detail and completeness of the problem statement.

    Args:
        problem_statement: The problem description to evaluate

    Returns:
        A dictionary with keys: 'score' (float 0-10), 'strongness' (str), 'weakness' (str)
    """
    GRANULARITY_ASSESSMENT_PROMPT = textwrap.dedent(
        """
        You are an expert problem analyst. Your task is to evaluate the granularity and detail level
        of a problem statement to determine if it has sufficient information to be solvable.
        
        Assess the problem statement based on:
        
        1. **Clarity and Specificity**:
           - Are the requirements clearly stated?
           - Are there specific examples or test cases?
           - Is the expected behavior well-defined?
        
        2. **Completeness**:
           - Are all necessary details provided?
           - Are edge cases mentioned or implied?
           - Are constraints and limitations clearly specified?
        
        3. **Actionability**:
           - Can a developer understand what needs to be implemented?
           - Are input/output formats clearly defined?
           - Are there enough details to write code without guessing?
        
        4. **Precision**:
           - Is the problem statement precise enough to avoid ambiguity?
           - Are technical terms and concepts clearly explained?
           - Is the scope of the problem well-defined?
        
        Provide your assessment as a JSON object with the following exact format:
        {
            "score": <number between 0 and 10 with exactly 2 decimal places>,
            "strongness": "<description of strengths>",
            "weakness": "<description of weaknesses>"
        }
        
        Where:
        - score: A numeric value between 0 and 10 with exactly 2 decimal places (e.g., 7.51, 8.23, 9.00)
          - 10 = Perfect detail, completely solvable without any ambiguity
          - 7-9 = Good detail, mostly clear with minor ambiguities
          - 4-6 = Moderate detail, some important information missing
          - 1-3 = Poor detail, significant information missing
          - 0 = Insufficient detail, problem is not solvable
        - strongness: A brief description of the strengths and positive aspects of the problem statement
        - weakness: A brief description of the weaknesses and areas that need improvement
        
        Respond with ONLY valid JSON in the exact format specified above. Do not include any additional text or explanation outside the JSON.
        """
    )
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": GRANULARITY_ASSESSMENT_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nEvaluate the granularity of this problem statement and provide a JSON response with score, strongness, and weakness.",
                },
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            response_clean = response.strip()
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_clean[json_start : json_end + 1]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        score = result.get("score", 5.0)
                        strongness = result.get("strongness", "No strengths identified")
                        weakness = result.get("weakness", "No weaknesses identified")
                        try:
                            score = float(score)
                            score = max(0.0, min(10.0, score))
                            score = round(score, 2)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid score value in JSON: {score}, using default 5.00")
                            score = 5.00
                        strongness = str(strongness) if strongness else "No strengths identified"
                        weakness = str(weakness) if weakness else "No weaknesses identified"
                        return {"score": score, "strongness": strongness, "weakness": weakness}
                    else:
                        raise ValueError("Parsed JSON is not a dictionary")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from response: {e}. Response: {response_clean[:200]}")
                except Exception as e:
                    logger.warning(f"Error processing JSON response: {e}. Response: {response_clean[:200]}")
            else:
                logger.warning(f"No JSON object found in response: {response_clean[:200]}")
            logger.warning(f"Could not extract valid JSON from LLM response: {response[:200]}")
            return {"score": 5.00, "strongness": "Unable to assess strengths", "weakness": "Unable to assess weaknesses"}

        except Exception as e:
            logger.error(f"Error in check_granularity_of_statement: {e}")
            retry += 1
            if retry < 10:
                other_models = [model for model in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
    logger.warning("Failed to get granularity assessment after all retries, returning default values")
    return {
        "score": 5.00,
        "strongness": "Unable to assess strengths - all retries failed",
        "weakness": "Unable to assess weaknesses - all retries failed",
    }

def generate_problem_details_for_io_combined(problem_statement: str, initial_structure: str) -> str:
    logger.info("🔄 Generating problem details (attempt 1/3)...")
    details_1 = generate_problem_details_for_io(problem_statement, initial_structure)
    logger.info("🔄 Generating problem details (attempt 2/3)...")
    details_2 = generate_problem_details_for_io(problem_statement, initial_structure)
    logger.info("🔄 Generating problem details (attempt 3/3)...")
    details_3 = generate_problem_details_for_io(problem_statement, initial_structure)
    if details_1 == details_2 == details_3:
        logger.info("✅ All 3 versions are identical, returning first version")
        return details_1
    COMBINE_PROMPT = textwrap.dedent(
        """
        You are an expert at analyzing and combining problem analysis results.
        
        You have been given 3 different analyses of the same problem statement and file structure.
        Your task is to combine them into ONE comprehensive, accurate, and conflict-free analysis.
        
        **Guidelines:**
        1. Identify all unique insights from each version
        2. Resolve any conflicts or contradictions between versions
        3. If versions contradict each other, choose the most accurate one
        4. Combine complementary information from all versions
        5. Ensure the final output is comprehensive and accurate
        6. Maintain the same format as the input (list of descriptive statements)
        
        **Conflict Resolution Rules:**
        - If two versions say different things about the same function, prefer the version that:
          * Is more specific and detailed
          * Better aligns with the problem statement
          * Includes more edge cases and exception handling
          * Is more technically accurate
        - If versions have complementary information, merge them intelligently
        - Remove any duplicate or redundant information
        - Ensure consistency across all descriptions
        
        **Output Format:**
        Output only the final combined list of descriptions.
        Do not include questions, answers, explanations, or any additional text.
        Highly emphasize important points in the description.
        """
    )

    USER_CONTENT = textwrap.dedent(
        f"""
        Problem Statement:
        {problem_statement}
        
        File Structure:
        {initial_structure}
        
        ---
        
        Version 1:
        {details_1}
        
        ---
        
        Version 2:
        {details_2}
        
        ---
        
        Version 3:
        {details_3}
        
        ---
        
        Please combine these 3 versions into one comprehensive, conflict-free analysis.
        Resolve any conflicts and merge complementary information.
    """
    )

    retry = 0
    selected_model = QWEN_MODEL_NAME
    max_retries = 10

    while retry < max_retries:
        try:
            messages = [
                {"role": "system", "content": COMBINE_PROMPT},
                {"role": "user", "content": USER_CONTENT},
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            combined = response.strip()
            if combined.startswith("```"):
                end_idx = combined.find("```", 3)
                if end_idx != -1:
                    combined = combined[3:end_idx].strip()
                else:
                    combined = combined[3:].strip()
            logger.info("✅ Successfully combined 3 versions of problem details")
            return combined
        except Exception as e:
            logger.warning(f"Error combining problem details (attempt {retry + 1}/{max_retries}): {e}")
            retry += 1
            if retry < max_retries:
                if retry > 7:
                    other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                    if other_models:
                        selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(0.5)
    logger.warning("⚠️ Failed to combine problem details after all retries, returning the longest version")
    versions = [details_1, details_2, details_3]
    return max(versions, key=len)

def ask_ai(system_prompt: str, user_content: str, temperature: float = 0.1, delay: int = 1) -> Optional[str]:
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            return response
        except Exception as e:
            retry += 1
            if retry < 10:
                if retry > 7:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = random.choice(other_models)
                time.sleep(delay)
    return ""

def check_problem_complexity(problem_statement: str, model: str = QWEN_MODEL_NAME, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Evaluates the complexity of a problem statement across multiple dimensions.
    Uses an LLM to assess various aspects of problem complexity.

    Args:
        problem_statement: The problem description to evaluate
        model: Model name to use for analysis
        temperature: Temperature for LLM inference

    Returns:
        A dictionary with complexity scores and analysis
    """
    COMPLEXITY_ASSESSMENT_PROMPT = textwrap.dedent(
        """
        You are an expert problem analyst. Your task is to evaluate the complexity of a problem statement
        across multiple dimensions to help understand how difficult it would be to solve.
        
        Assess the problem statement based on the following complexity dimensions:
        
        1. **Algorithmic Complexity**:
           - Does it require complex algorithms?
           - Are there multiple algorithmic approaches possible?
           - What is the time/space complexity likely to be?
        
        2. **Implementation Difficulty**:
           - How much code is likely needed?
           - Are there many edge cases to handle?
           - Does it require advanced language features?
           - How many functions/classes/components are needed?
        
        3. **Conceptual Complexity**:
           - How many concepts are involved (data structures, design patterns, mathematical concepts)?
           - Are the concepts advanced or fundamental?
           - Is domain-specific knowledge required?
           - How abstract is the problem?
        
        4. **Edge Cases and Special Handling**:
           - How many edge cases need to be considered?
           - Are there special conditions or constraints?
           - Does it require extensive input validation?
           - Are there multiple failure modes?
        
        5. **Cognitive Load**:
           - How hard is it to understand the problem?
           - Is the problem statement clear or ambiguous?
           - How much mental modeling is required?
           - Is it easy to break down into smaller parts?
        
        6. **Testing Complexity**:
           - How many test cases are likely needed?
           - Are the test cases straightforward or complex?
           - Does it require mocking or special test setup?
        
        Provide your assessment as a JSON object with the following exact format:
        {
            "overall_complexity": <number between 0 and 10 with exactly 2 decimal places>,
            "dimensions": {
                "algorithmic_complexity": <0.0 to 10.0 with 2 decimal places>,
                "implementation_difficulty": <0.0 to 10.0 with 2 decimal places>,
                "conceptual_complexity": <0.0 to 10.0 with 2 decimal places>,
                "edge_cases_complexity": <0.0 to 10.0 with 2 decimal places>,
                "cognitive_load": <0.0 to 10.0 with 2 decimal places>,
                "testing_complexity": <0.0 to 10.0 with 2 decimal places>
            },
            "estimated_code_length": "<short/medium/long/very_long>",
            "estimated_time_to_solve": "<minutes/hours estimate>",
            "key_complexity_factors": [
                "<factor 1>",
                "<factor 2>",
                ...
            ],
            "complexity_breakdown": {
                "algorithms_required": ["<algorithm 1>", "<algorithm 2>", ...],
                "data_structures_needed": ["<structure 1>", "<structure 2>", ...],
                "concepts_involved": ["<concept 1>", "<concept 2>", ...],
                "edge_cases_count": <estimated number>,
                "functions_classes_needed": <estimated number>
            },
            "difficulty_level": "<beginner/intermediate/advanced/expert>",
            "reasoning": "<brief explanation of the complexity assessment>"
        }
        
        Where:
        - overall_complexity: A numeric value between 0 and 10 with exactly 2 decimal places
          - 0-2 = Very simple, trivial problems
          - 2-4 = Simple, straightforward problems
          - 4-6 = Moderate complexity, requires some thought
          - 6-8 = Complex, challenging problems
          - 8-10 = Very complex, expert-level problems
        - Each dimension score: 0.0 to 10.0 with 2 decimal places
        - estimated_code_length: rough estimate of code size
        - estimated_time_to_solve: rough time estimate for an experienced developer
        - key_complexity_factors: list of main factors that make this problem complex
        - complexity_breakdown: detailed breakdown of what contributes to complexity
        - difficulty_level: overall difficulty categorization
        
        Respond with ONLY valid JSON in the exact format specified above. Do not include any additional text or explanation outside the JSON.
        """
    )

    retry = 0
    selected_model = model
    max_retries = 10

    while retry < max_retries:
        try:
            messages = [
                {"role": "system", "content": COMPLEXITY_ASSESSMENT_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nEvaluate the complexity of this problem statement and provide a JSON response with all complexity dimensions.",
                },
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            response_clean = response.strip()
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_clean[json_start : json_end + 1]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        overall = result.get("overall_complexity", 5.0)
                        try:
                            overall = float(overall)
                            overall = max(0.0, min(10.0, overall))
                            overall = round(overall, 2)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid overall_complexity value: {overall}, using default 5.00")
                            overall = 5.00
                        result["overall_complexity"] = overall
                        dimensions = result.get("dimensions", {})
                        for dim_name in [
                            "algorithmic_complexity",
                            "implementation_difficulty",
                            "conceptual_complexity",
                            "edge_cases_complexity",
                            "cognitive_load",
                            "testing_complexity",
                        ]:
                            dim_value = dimensions.get(dim_name, 5.0)
                            try:
                                dim_value = float(dim_value)
                                dim_value = max(0.0, min(10.0, dim_value))
                                dim_value = round(dim_value, 2)
                            except (ValueError, TypeError):
                                dim_value = 5.00
                            dimensions[dim_name] = dim_value
                        result["dimensions"] = dimensions
                        result.setdefault("estimated_code_length", "medium")
                        result.setdefault("estimated_time_to_solve", "unknown")
                        result.setdefault("key_complexity_factors", [])
                        result.setdefault("complexity_breakdown", {})
                        result.setdefault("difficulty_level", "intermediate")
                        result.setdefault("reasoning", "Complexity analysis completed")
                        breakdown = result.get("complexity_breakdown", {})
                        breakdown.setdefault("algorithms_required", [])
                        breakdown.setdefault("data_structures_needed", [])
                        breakdown.setdefault("concepts_involved", [])
                        breakdown.setdefault("edge_cases_count", 0)
                        breakdown.setdefault("functions_classes_needed", 1)
                        result["complexity_breakdown"] = breakdown
                        return result
                    else:
                        raise ValueError("Parsed JSON is not a dictionary")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from response: {e}. Response: {response_clean[:200]}")
                except Exception as e:
                    logger.warning(f"Error processing JSON response: {e}. Response: {response_clean[:200]}")
            else:
                logger.warning(f"No JSON object found in response: {response_clean[:200]}")
            logger.warning(f"Could not extract valid JSON from LLM response: {response[:200]}")
            retry += 1
            if retry < max_retries:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]

        except Exception as e:
            logger.error(f"Error in check_problem_complexity: {e}")
            retry += 1
            if retry < max_retries:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]

    logger.warning("Failed to get complexity assessment after all retries, returning default values")
    return {
        "overall_complexity": 5.00,
        "dimensions": {
            "algorithmic_complexity": 5.0,
            "implementation_difficulty": 5.0,
            "conceptual_complexity": 5.0,
            "edge_cases_complexity": 5.0,
            "cognitive_load": 5.0,
            "testing_complexity": 5.0,
        },
        "estimated_code_length": "medium",
        "estimated_time_to_solve": "unknown",
        "key_complexity_factors": ["Unable to assess"],
        "complexity_breakdown": {
            "algorithms_required": [],
            "data_structures_needed": [],
            "concepts_involved": [],
            "edge_cases_count": 0,
            "functions_classes_needed": 1,
        },
        "difficulty_level": "intermediate",
        "reasoning": "Analysis failed - unable to parse LLM response",
    }

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, run_id, agent_start_time
    print("🚀 [WORKFLOW] Starting agent_main - Entry point")
    logger.info("🚀 [WORKFLOW] Starting agent_main - Entry point")
    agent_start_time = time.time()
    run_id = os.getenv("EVALUATION_RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
        logger.info(f"📁 [WORKFLOW] Changed directory to: {repo_dir}")
    logger.info("⚙️ [WORKFLOW] Setting up environment...")
    set_env_for_agent()

    timeout = 1400
    result = None
    exception_occurred = None
    task_completed = threading.Event()

    def run_task():
        nonlocal result, exception_occurred
        logger.info("🔄 [WORKFLOW] Starting task execution thread...")
        enhancement = ""
        try:
            global _current_tool_manager

            _current_tool_manager = EnhancedToolManager()
            problem_statement = input_dict.get("problem_statement")
            problem_type, _ = check_problem_type(input_dict.get("problem_statement"))
            if problem_type == PROBLEM_TYPE_FIX:
                result = process_fix_task(problem_statement, "")
            else:
                result = process_create_task(problem_statement, "")
        finally:
            task_completed.set()

    logger.info("🧵 [WORKFLOW] Creating and starting task thread...")
    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    task_thread.join(timeout=timeout)

    timed_out = task_thread.is_alive()
    if timed_out:
        print(f"⏱️ [WORKFLOW] Task execution timed out after {timeout} seconds")
        logger.warning(f"Task execution timed out after {timeout} seconds, killing thread")

    print("Result: \n\n", result)

    global _current_tool_manager
    if _current_tool_manager is not None:
        try:
            final_patch = _current_tool_manager.get_final_git_patch()
            if final_patch:
                result = final_patch
                logger.info("✅ [WORKFLOW] Final patch generated successfully")
        except Exception as e:
            logger.error(f"❌ [WORKFLOW] Failed to get final patch: {e}")
            logger.warning(f"Failed to get final patch from tool manager: {e}")
        finally:
            _current_tool_manager = None

    try:
        logger.info("🧹 [WORKFLOW] Cleaning up git state...")
        subprocess.Popen(["git", "reset", "--hard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    print("🎯 [WORKFLOW] agent_main completed")
    logger.info("🎯 [WORKFLOW] agent_main completed")

    logger.info("📝 [WORKFLOW] Generating final git patch...")

    return result if result else ""

def clean_code_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response

def process_create_task(problem_statement: str, enhancement: str):
    global run_id, agent_start_time
    patch_text = ""
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repo_dir = repo_path.split("/")[-1]
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    os.system("git reset --hard")
    os.system("git clean -fd")
    initial_structure_str, files_to_modify = get_files_to_modify(problem_statement)
    has_exception_handling_mention = check_explicit_exception_handling_mention(problem_statement, model=QWEN_MODEL_NAME, temperature=0.0).get(
        "has_explicit_exception_mention"
    )
    def run_basic_attempt(attempt: int):
        os.system("git reset --hard")
        try:
            attempt_solution, _ = basic_approach(initial_structure_str, problem_statement)
            return attempt, attempt_solution
        except Exception as e:
            return attempt, None
    assessment = check_granularity_of_statement(problem_statement)
    category = get_category_from_score(assessment["score"])
    if "Excellent" in category:
        problem_statement = (
            f"{problem_statement}\n\nThis is a very clear and well-defined problem. Please try to follow the problem statement concisely."
        )
    elif "Moderate" in category:
        assessment = check_problem_complexity(problem_statement, model=QWEN_MODEL_NAME, temperature=0.0)
        complexity_category = get_complexity_category(assessment['overall_complexity'])
        if "Low" in complexity_category:
            retry = 30
            initial_solution = None
            max_workers = max(1, min(2, int(retry / 10)))
            logger.info(f"[PARALLEL_GEN] Starting parallel generation with {max_workers} workers for {retry} attempts")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_basic_attempt, attempt): attempt for attempt in range(retry)}
                for fut in as_completed(futures):
                    elapsed_time = time.time() - agent_start_time
                    if elapsed_time > 1250:
                        logger.info("[PARALLEL_GEN] Time limit reached, stopping parallel generation")
                        break
                    
                    attempt_num, solution = fut.result()
                    if solution is not None:
                        logger.info(f"[PARALLEL_GEN] Solution found in attempt {attempt_num}")
                        initial_solution = solution
                        break
            if initial_solution is not None:
                os.system("git reset --hard")
                extract_and_write_files(initial_solution)
                patch = EnhancedToolManager.get_final_git_patch()
                return patch
    initial_structure = {}
    for file_path in files_to_modify:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                initial_structure[file_path] = f.read()
        except Exception as e:
            logger.warning(f"Could not read initial content for {file_path}: {e}")
            continue
    function_behaviours = generate_function_behaviours(initial_structure_str, problem_statement)
    analysis = check_problem_statement_has_examples(problem_statement, model=QWEN_MODEL_NAME, temperature=0.0)
    has_examples = analysis.get("has_clear_examples")
    if has_examples:
        initial_structure_content = "\n\n".join([f"===== {file_path} =====\n{content}" for file_path, content in initial_structure.items()])
        io_format_analysis = check_io_format_consistency(problem_statement, initial_structure_content, model=QWEN_MODEL_NAME, temperature=0.0)
        has_mismatch = io_format_analysis.get("has_format_mismatch")
        if has_mismatch:
            severity = io_format_analysis.get("mismatch_severity", "unknown")
            if severity == "severe":
                problem_details = generate_problem_details_for_io_combined(problem_statement, initial_structure_str)
                problem_statement = f"{problem_statement}\n\nProblem Breakdown:\n{problem_details}"
    try:
        results = []
        for attempt in range(3):
            cost_usage = EnhancedNetwork.get_cost_usage()
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > 950 or cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0) - 0.6:
                break
            os.system("git reset --hard")
            os.system("git clean -fd")
            remaining_time = max(10, 1300 - elapsed_time)
            logger.info(
                f"⏱️ [WORKFLOW] Attempt {attempt + 1}/3 - Elapsed time: {elapsed_time:.2f}s, Starting create_task_solve_workflow with timeout: {remaining_time:.2f}s..."
            )
            patch_text, is_success = create_task_solve_workflow(
                problem_statement,
                timeout=remaining_time,
                run_id_1=run_id,
                enhancement=enhancement,
                should_review=True,
                n_max_steps=200,
                initial_structure=initial_structure,
                function_behaviours=function_behaviours,
                files_to_modify=files_to_modify,
                has_exception_handling_mention=has_exception_handling_mention,
            )
            modified_files = EnhancedToolManager.get_modified_files_list()
            modified_files_content = {}
            result = ""
            if modified_files:
                temp_file_ops = FileOperationsUtil(new_files_created=[])
                temp_file_ops.file_system_manager = FileSystemManager()
                temp_file_ops.search_manager = SearchManager()
                for file_path in modified_files:
                    file_content = temp_file_ops.get_file_content(file_path, limit=-1)
                    modified_files_content[file_path] = file_content
                result = "\n\n".join([f"{file}\n{content}" for file, content in modified_files_content.items()])

            observation = "Success" if is_success else "Failed"
            if len(results) == 0 or is_success:
                results.append(
                    {
                        "solution_code": result,
                        "patch": patch_text,
                        "modified_files": modified_files,
                        "modified_files_content": modified_files_content,
                        "summary": observation,
                    }
                )
        best_solution = select_best_solution(results, problem_statement) if results else None
        os.system("git reset --hard")
        os.system("git clean -fd")
        if best_solution and best_solution.get("modified_files_content"):
            file_ops = FileOperationsUtil(new_files_created=[])
            for file_path, file_content in best_solution["modified_files_content"].items():
                try:
                    file_ops.save(file_path, file_content)
                    logger.info(f"[PROCESS_TASK] Restored file: {file_path}")
                except Exception as e:
                    logger.error(f"[PROCESS_TASK] Error restoring file {file_path}: {e}")
            if best_solution.get("patch"):
                patch_text = best_solution["patch"]
        logger.info("🧹 [WORKFLOW] Resetting git state...")
        logger.info("✅ [WORKFLOW] process_create_task completed successfully")
    except Exception as e:
        logger.error(f"Error in process_create_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
        logger.info("📁 [WORKFLOW] Restored original working directory")
    return patch_text

def select_best_solution(
    solutions: List[dict],
    problem_statement: str,
) -> dict:
    """
    Use LLM to select the best solution among multiple candidates.
    Each solution dict has: {'solution_code': str, 'test_cases': str, 'patch': str}
    Returns the best solution dict.
    """
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]

    SELECT_BEST_SOLUTION_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer.

        You are given:
        1. A problem statement
        2. Multiple candidate solutions

        Your task is to carefully compare all candidate solutions against the problem statement.
        You must explicitly analyze the differences between the solutions, not just evaluate them in isolation.

        Evaluate each solution on:
        - Correctness with respect to the stated requirements
        - Coverage of edge cases and invalid or unexpected inputs
        - Completeness in solving the full problem, not just part of it
        - Logical soundness and absence of bugs
        - Clarity and readability of the approach
        - Safety and robustness under realistic usage

        If a solution fails any required condition, it must not be selected, even if it is partially correct.

        Select the SINGLE best solution overall.
        If multiple solutions satisfy the requirements, prefer the one that is:
        - Easier to understand and reason about
        - Less error prone
        - More maintainable

        Strict rules:
        - Do NOT modify any solution
        - Do NOT combine solutions
        - Do NOT add new code
        - Do NOT assume missing behavior unless explicitly stated in the problem

        Return ONLY the final selection result as instructed elsewhere.

        Return ONLY a valid JSON object with exactly the following structure:
        {
            "selected_index": <0-based index of the best solution>,
            "reasoning": "<concise explanation of why this solution is best compared to the others>"
        }
        """
    )
    solutions_context = ""
    for i, sol in enumerate(solutions):
        code_truncated = sol.get("solution_code", "")
        if len(code_truncated) > 60000:
            patch = sol.get("patch", "")
            if patch:
                code_truncated = patch
        summary = sol.get("summary", "")
        solutions_context += f"\n\n=========== SOLUTION {i} ===========\n```\n{code_truncated}\n```\n\nSolution Summary:\n{summary}"
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": SELECT_BEST_SOLUTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\n{solutions_context}\n\nSelect the best solution and explain why.",
                },
            ]
            result = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            if isinstance(result, tuple):
                response_text, _ = result
            else:
                response_text = result
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                selected_index = selection.get("selected_index", 0)
                reasoning = selection.get("reasoning", "No reasoning provided")
                if 0 <= selected_index < len(solutions):
                    logger.info(f"[SELECT_BEST_SOLUTION] Selected solution {selected_index}: {reasoning}")
                    return solutions[selected_index]
                else:
                    logger.warning(f"[SELECT_BEST_SOLUTION] Invalid index {selected_index}, using first solution")
                    return solutions[0]
            else:
                selection = json.loads(response_text)
                selected_index = selection.get("selected_index", 0)
                if 0 <= selected_index < len(solutions):
                    return solutions[selected_index]
                return solutions[0]
        except Exception as e:
            retry += 1
            logger.warning(f"[SELECT_BEST_SOLUTION] Retry {retry}/5: {e}")
            if retry >= 5:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)
    logger.warning("[SELECT_BEST_SOLUTION] All retries failed, returning second solution")
    return solutions[0]

def basic_approach(initial_structure: str, problem_statement: str, temperature: float = 0.0) -> tuple[str, str] | tuple[None, None]:
    def extract_core_concepts_for_search(problem_statement: str) -> dict:
        EXTRACT_CONCEPTS_PROMPT = textwrap.dedent(
            """
            You are an expert at analyzing programming problems and extracting their core concepts.
            Your task is to identify the fundamental concepts and domain of the problem using the exact terminology from the problem statement.
            
            Rules:
            1. Identify the core domain related to the problem statement.
            2. Extract key algorithmic concepts related to the problem statement.
            3. Identify common edge cases that similar problems typically have
            4. Can use synonyms, related terms, or broader categories instead of exact words from the problem
            5. Focus on what types of inputs/outputs and validation might be needed
            
            Return a JSON object with:
            - "search_terms": array of 2-4 search terms
            - "domain": the problem domain in general terms
            - "common_edge_cases": array of typical edge cases for this type of problem
        """
        )
        retry = 0
        selected_model = GLM_MODEL_NAME
        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": EXTRACT_CONCEPTS_PROMPT},
                    {"role": "user", "content": f"Problem Statement:\n{problem_statement}"},
                ]
                response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
                json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
                if json_match:
                    try:
                        concepts = json.loads(json_match.group(0))
                        if isinstance(concepts, dict) and "search_terms" in concepts:
                            return concepts
                    except json.JSONDecodeError:
                        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                        if code_block_match:
                            try:
                                concepts = json.loads(code_block_match.group(1))
                                if isinstance(concepts, dict) and "search_terms" in concepts:
                                    return concepts
                            except json.JSONDecodeError:
                                pass
                return {"search_terms": [], "domain": "", "common_edge_cases": []}
            except Exception as e:

                retry += 1
                if retry > 2:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(1)
        return {"search_terms": [], "domain": "", "common_edge_cases": []}

    def generate_single_testset(problem_statement: str, files_to_test: str, initial_structure: str, temperature: float = 0.0) -> str:
        GENERATE_TESTCASES_PROMPT = textwrap.dedent(
            """
            You are an expert testcase developer.
            Important points:-
            - Follow the best practices and conventions of the language of the code skeleton.
            - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
            - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
            - Use the only built-in testing framework for the language of the code skeleton. **MUST** use the built-in testing framework.
                - For python, use `unittest` to write a test.
                - For javascript, **MUST** use **`node:test` and `node:assert`** to write a test.
                - For other languages, use built-in test frameworks as well.
            You must respond directly with the test cases in the following format.
            =========TEST_CASES
            <<test cases>>
            Do not include anything else. For Example (JavaScript):
            =========TEST_CASES
            import { test } from 'node:test';
            import assert from 'node:assert/strict';
            import { main_func } from './main_module.js';
            test('main_func should return expected output', () => {
                assert.strictEqual(main_func(), 'expected_output');
            });
        """
        )
        retry = 0
        test_generation_messages = [
            {"role": "system", "content": GENERATE_TESTCASES_PROMPT},
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nInitial structure:\n{initial_structure}\n\nGenerate the complete and correct testcases.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```\n```javascript\ntest_a.js\ncontents of test_a.js\n\ntest_b.js\ncontents of test_b.js\n```",
            },
        ]
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                result = EnhancedNetwork.make_request(test_generation_messages, model=selected_model, temperature=temperature)
                if isinstance(result, tuple):
                    testcode_response, _ = result
                else:
                    testcode_response = result
                testcases = clean_code_response(testcode_response)
                if not testcases or not testcases.strip():
                    retry += 1
                    continue
                lines = testcases.split("\n")
                if not lines or len(lines) == 0:
                    retry += 1
                    test_generation_messages.append({"role": "assistant", "content": testcode_response})
                    test_generation_messages.append(
                        {
                            "role": "user",
                            "content": f"Include file name in the response. example:\n```python\ntest_a.py\n{{content}}\n\ntest_b.py\n{{content}}\n```\n```javascript\ntest_a.js\n{{content}}\n\ntest_b.js\n{{content}}\n```",
                        }
                    )
                    continue
                return testcases
            except Exception as e:
                retry += 1
                time.sleep(1)
        return ""

    def generate_initial_solution(problem_statement: str, initial_structure: str, temperature: float = 0.7) -> str:

        concepts = extract_core_concepts_for_search(problem_statement)

        edge_case_guidance = ""
        if concepts.get("common_edge_cases"):
            edge_case_guidance = f"\n\n**Common Edge Cases to Consider (based on similar problems):**\n"
            for i, case in enumerate(concepts.get("common_edge_cases", []), 1):
                edge_case_guidance += f"{i}. {case}\n"
        GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = (
            textwrap.dedent(
            """
                You are an expert software engineer. Your task is to generate a complete, working solution for the given problem statement.
                Strict Requirements:
                1. Output the full content of files along with their file names. You **MUST** output the **file name** along with file content.
                2. Do not include explanations, comments, or markdown formatting in the main code.
                3. Use only standard libraries and frameworks (no external libraries).
                4. Implement all required classes and functions exactly with the same names as in the initial code stub.
                5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
                6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
                7. The solution must be executable as-is with no placeholders or TODOs.
                8. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: Comment Prefix + Edge Case: [description of the edge case]`
                9. **IMPORTANT**: Design your code to robustly handle input in all possible formats-implement logic to detect and preprocess various valid input types so the program works regardless of input format.
                10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `Comment prefix + Handled Edge Cases: [list of edge cases]`
                Return only the final code.
                Response Examples:
                ```python
                a.py
                {{content}}
                b.js
                {{content}}
                ```
                """
            ) + edge_case_guidance
        )
        INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
            """
            You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated code for potential infinite loops and provide a corrected version if issues are found.
            CRITICAL INFINITE LOOP DETECTION:
            1. Check for while True: loops without guaranteed exit conditions
            2. Verify all while loops have clear termination conditions
            3. Ensure recursive functions have proper base cases
            4. Look for loops that depend on external state that might never change
            5. Check for patterns that could lead to infinite iteration
            If you find potential infinite loops:
            - Provide a corrected version of the code
            - Ensure all loops have finite termination conditions
            - Add reasonable iteration limits or timeout mechanisms where appropriate
            If no infinite loops are detected:
            - Return the original code unchanged
            STRICT REQUIREMENT: Return the final code along with file names. Do not include any explanations, comments, or additional text.
            example:
            ```python
            a.py
            {{content}}
            b.py
            {{content}}
            ```
        """
        )
        retry = 0
        code_generation_messages = [
            {
                "role": "system",
                "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT,
            },
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nInitial structure:\n{initial_structure}\nGenerate the complete and correct implementation in files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
            },
        ]
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                result = EnhancedNetwork.make_request(code_generation_messages, model=selected_model, temperature=temperature)
                if isinstance(result, tuple):
                    code_response, _ = result
                else:
                    code_response = result
                loop_check_messages = [
                    {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                    {
                        "role": "user",
                        "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final code.",
                    },
                ]
                result2 = EnhancedNetwork.make_request(loop_check_messages, model=selected_model)
                if isinstance(result2, tuple):
                    loop_check_response, _ = result2
                else:
                    loop_check_response = result2
                solution = clean_code_response(loop_check_response)
                return solution
            except Exception as e:
                retry += 1
                time.sleep(1)
        if retry >= 10:
            return ""
        return ""
    initial_solution = generate_initial_solution(problem_statement, initial_structure, temperature)
    if not initial_solution:
        return (None, None)
    created_files = extract_and_write_files(initial_solution)
    test_cases = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature)
    if not test_cases:
        return (None, None)
    test_files = extract_and_write_files(test_cases)
    for file in test_files:
        try:
            run_command = TestManager.llm_select_run_command_for_file(file)
            result = subprocess.run(
                run_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            return (None, None)
        except ValueError as e:
            return (None, None)
        except Exception as e:
            return (None, None)
        if not TestManager.is_all_tests_passed(result.stdout):
            return (None, None)
    return (initial_solution, test_cases)