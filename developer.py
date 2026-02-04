"""
Smol AI Coding Agent - Single File Version

A lightweight AI coding agent that generates complete applications from natural language prompts.
Uses OpenAI's API to plan, structure, and generate code files.

Usage:
    python smol_agent.py --prompt "Create a simple todo app in React"
    python smol_agent.py --prompt "Make a Python web scraper" --model gpt-4-0613 --debug True
"""

import argparse
import asyncio
import os
import re
import shutil
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional

import openai
from openai_function_call import openai_function
from tenacity import retry, stop_after_attempt, wait_random_exponential

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "gpt-4-0613"
DEFAULT_FOLDER = "generated"

SMOL_DEV_SYSTEM_PROMPT = """
You are a top tier AI developer who is trying to write a program that will generate code for the user based on their intent.
Do not leave any todos, fully implement every feature requested.

When writing code, add comments to explain what you intend to do and why it aligns with the program plan and specific instructions from the original prompt.
"""


# =============================================================================
# Step Types for Agent Protocol
# =============================================================================

class StepTypes(str, Enum):
    PLAN = "plan"
    SPECIFY_FILE_PATHS = "specify_file_paths"
    GENERATE_CODE = "generate_code"


# =============================================================================
# File Utilities
# =============================================================================

def generate_folder(folder_path: str) -> None:
    """Create a fresh folder for generated code, removing existing if present."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def write_file(file_path: str, content: str) -> None:
    """Write content to a file, creating parent directories if needed."""
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# OpenAI Function Call Decorators
# =============================================================================

@openai_function
def file_paths(files_to_edit: List[str]) -> List[str]:
    """
    Construct a list of file paths to be generated.
    """
    return files_to_edit


# =============================================================================
# Core AI Functions
# =============================================================================

def specify_file_paths(prompt: str, plan: str, model: str = DEFAULT_MODEL) -> List[str]:
    """
    Determine which files need to be created based on the prompt and plan.

    Args:
        prompt: User's original prompt
        plan: Generated plan for the project
        model: OpenAI model to use

    Returns:
        List of file paths to generate
    """
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.7,
        functions=[file_paths.openai_schema],
        function_call={"name": "file_paths"},
        messages=[
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}
      Given the prompt and the plan, return a list of strings corresponding to the new files that will be generated.
                  """,
            },
            {
                "role": "user",
                "content": f""" I want a: {prompt} """,
            },
            {
                "role": "user",
                "content": f""" The plan we have agreed on is: {plan} """,
            },
        ],
    )
    result = file_paths.from_response(completion)
    return result


def plan(
    prompt: str,
    stream_handler: Optional[Callable[[bytes], None]] = None,
    model: str = DEFAULT_MODEL,
    extra_messages: List[Any] = None
) -> str:
    """
    Generate a detailed plan for the project including file structure.

    Args:
        prompt: User's original prompt
        stream_handler: Optional callback for streaming response
        model: OpenAI model to use
        extra_messages: Additional messages to include in the conversation

    Returns:
        Generated plan in Markdown format with YAML header
    """
    if extra_messages is None:
        extra_messages = []

    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.7,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

    In response to the user's prompt, write a plan using GitHub Markdown syntax. Begin with a YAML description of the new files that will be created.
  In this plan, please name and briefly describe the structure of code that will be generated, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.
                Respond only with plans following the above schema.
                  """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            *extra_messages,
        ],
    )

    collected_messages = []
    for chunk in completion:
        chunk_message_dict = chunk["choices"][0]
        chunk_message = chunk_message_dict["delta"]
        if chunk_message_dict["finish_reason"] is None:
            collected_messages.append(chunk_message)
            if stream_handler:
                try:
                    stream_handler(chunk_message["content"].encode("utf-8"))
                except Exception as err:
                    print(f"\nstream_handler error: {err}", file=sys.stderr)
                    print(chunk_message, file=sys.stderr)

    full_reply_content = "".join([m.get("content", "") for m in collected_messages])
    return full_reply_content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def generate_code_async(
    prompt: str,
    plan: str,
    current_file: str,
    stream_handler: Optional[Callable[[bytes], None]] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Generate code for a specific file.

    Args:
        prompt: User's original prompt
        plan: Generated plan for the project
        current_file: Path to the current file being generated
        stream_handler: Optional callback for streaming response
        model: OpenAI model to use

    Returns:
        Generated code content
    """
    completion = openai.ChatCompletion.acreate(
        model=model,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

  In response to the user's prompt,
  Please name and briefly describe the structure of the app we will generate, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.

  We have broken up the program into per-file generation.
  Now your job is to generate only the code for the file: {current_file}

  only write valid code for the given filepath and file type, and return only the code.
  do not add any other explanation, only return valid code for that file type.
                  """,
            },
            {
                "role": "user",
                "content": f""" the plan we have agreed on is: {plan} """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            {
                "role": "user",
                "content": f"""
    Make sure to have consistent filenames if you reference other files we are also generating.

    Remember that you must obey 3 things:
       - you are generating code for the file {current_file}
       - do not stray from the names of the files and the plan we have decided on
       - MOST IMPORTANT OF ALL - every line of code you generate must be valid code. Do not include code fences in your response, for example

    Bad response (because it contains the code fence):
    ```javascript
    console.log("hello world")
    ```

    Good response (because it only contains the code):
    console.log("hello world")

    Begin generating the code now.

    """,
            },
        ],
        stream=True,
    )

    collected_messages = []
    async for chunk in await completion:
        chunk_message_dict = chunk["choices"][0]
        chunk_message = chunk_message_dict["delta"]
        if chunk_message_dict["finish_reason"] is None:
            collected_messages.append(chunk_message)
            if stream_handler:
                try:
                    stream_handler(chunk_message["content"].encode("utf-8"))
                except Exception as err:
                    print(f"\nstream_handler error: {err}", file=sys.stderr)
                    print(chunk_message, file=sys.stderr)

    code_file = "".join([m.get("content", "") for m in collected_messages])

    # Extract code from markdown code blocks if present
    pattern = r"```[\w\s]*\n([\s\S]*?)```"
    code_blocks = re.findall(pattern, code_file, re.MULTILINE)
    return code_blocks[0] if code_blocks else code_file


def generate_code_sync(
    prompt: str,
    plan: str,
    current_file: str,
    stream_handler: Optional[Callable[[bytes], None]] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Synchronous wrapper for generate_code_async.

    Args:
        prompt: User's original prompt
        plan: Generated plan for the project
        current_file: Path to the current file being generated
        stream_handler: Optional callback for streaming response
        model: OpenAI model to use

    Returns:
        Generated code content
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        generate_code_async(prompt, plan, current_file, stream_handler, model)
    )


# =============================================================================
# Main Agent Function
# =============================================================================

def smol_dev(
    prompt: str,
    generate_folder_path: str = DEFAULT_FOLDER,
    debug: bool = False,
    model: str = DEFAULT_MODEL
) -> None:
    """
    Main entry point for the Smol Dev coding agent.

    Args:
        prompt: User's prompt describing what to build
        generate_folder_path: Where to save the generated code
        debug: Enable debug output
        model: OpenAI model to use
    """
    # Create output folder
    generate_folder(generate_folder_path)

    # Step 1: Generate plan
    if debug:
        print("--------Generating Plan---------")

    start_time = time.time()

    def stream_handler(chunk):
        if debug:
            end_time = time.time()
            sys.stdout.write(
                f"\r \033[93mChars streamed\033[0m: {stream_handler.count}. "
                f"\033[93mChars per second\033[0m: {stream_handler.count / (end_time - start_time):.2f}"
            )
            sys.stdout.flush()
        stream_handler.count += len(chunk)

    stream_handler.count = 0

    shared_deps = plan(prompt, stream_handler, model=model)
    write_file(f"{generate_folder_path}/shared_deps.md", shared_deps)

    if debug:
        print(f"\n{shared_deps}")
        print("--------Plan Complete---------")

    # Step 2: Specify file paths
    if debug:
        print("--------Specifying File Paths---------")

    file_paths_list = specify_file_paths(prompt, shared_deps, model=model)

    if debug:
        print(f"Files to create: {file_paths_list}")
        print("--------File Paths Specified---------")

    # Step 3: Generate code for each file
    for file_path in file_paths_list:
        full_file_path = f"{generate_folder_path}/{file_path}"

        if debug:
            print(f"--------Generating Code: {file_path}---------")

        start_time = time.time()

        def stream_handler(chunk):
            if debug:
                end_time = time.time()
                sys.stdout.write(
                    f"\r \033[93mChars streamed\033[0m: {stream_handler.count}. "
                    f"\033[93mChars per second\033[0m: {stream_handler.count / (end_time - start_time):.2f}"
                )
                sys.stdout.flush()
            stream_handler.count += len(chunk)

        stream_handler.count = 0

        code = generate_code_sync(prompt, shared_deps, file_path, stream_handler, model=model)

        if debug:
            print(f"\n{code}")
            print(f"--------Code Complete: {file_path}---------")

        write_file(full_file_path, code)

    print("\n--------Smol Dev Complete!---------")
    print(f"Generated code saved to: {os.path.abspath(generate_folder_path)}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Command line interface entry point."""
    DEFAULT_PROMPT = """
  a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG.
  The left paddle is controlled by the player, following where the mouse goes.
  The right paddle is controlled by a simple AI algorithm, which slowly moves the paddle toward the ball at every frame, with some probability of error.
  Make the canvas a 400 x 400 black square and center it in the app.
  Make the paddles 100px long, yellow and the ball small and red.
  Make sure to render the paddles and name them so they can controlled in javascript.
  Implement the collision detection and scoring as well.
  Every time the ball bounces off a paddle, the ball should move faster.
  It is meant to run in Chrome browser, so dont use anything that is not supported by Chrome, and don't use the import and export keywords.
    """

    parser = argparse.ArgumentParser(
        description="Smol AI Coding Agent - Generate code from natural language prompts"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the app to be created. Can also be a path to a .md file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}). Can also use gpt-3.5-turbo-0613"
    )
    parser.add_argument(
        "--generate_folder_path",
        type=str,
        default=DEFAULT_FOLDER,
        help=f"Path of the folder for generated code (default: {DEFAULT_FOLDER})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )

    # Handle simple invocation: python smol_agent.py "your prompt here"
    if len(sys.argv) == 2:
        prompt = sys.argv[1]
    else:
        args = parser.parse_args()
        prompt = args.prompt if args.prompt else DEFAULT_PROMPT

    # Read prompt from file if it's a .md file
    if len(prompt) < 100 and prompt.endswith(".md"):
        with open(prompt, "r", encoding="utf-8") as promptfile:
            prompt = promptfile.read()

    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    if len(sys.argv) == 2:
        # Simple invocation
        smol_dev(prompt=prompt)
    else:
        args = parser.parse_args()
        smol_dev(
            prompt=prompt,
            generate_folder_path=args.generate_folder_path,
            debug=args.debug,
            model=args.model
        )


if __name__ == "__main__":
    main()
