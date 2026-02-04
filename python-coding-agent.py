"""
Unified Python Coding Agent
===========================
A single-file implementation combining CrewAI and LangGraph approaches.
Supports OpenAI and Groq (Llama) API providers.

Usage:
    python unified_agent.py

Requirements:
    pip install langchain langchain-community langchain-groq langgraph
    pip install crewai crewai-tools ipython duckduckgo-search
"""

import os
import sys
import time
import uuid
import logging
from typing import Annotated, Dict, TypedDict, List, Optional, Any

# LangChain imports
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# LangGraph imports
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# CrewAI imports
from crewai import Agent, Task, Crew, Process

# Optional imports (handled gracefully)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: langchain-groq not installed. Groq provider unavailable.")

try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class AgentConfig:
    """Configuration for the coding agent."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: str = "",
        model: str = "",
        temperature: float = 0,
        max_iterations: int = 5,
        max_rpm: int = 10
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.max_rpm = max_rpm

        # Set default models based on provider
        if not model:
            if self.provider == "groq":
                self.model = "llama3-70b-8192"
            elif self.provider == "openai":
                self.model = "gpt-4"
            else:
                self.model = model
        else:
            self.model = model

        # Set environment variables
        if self.provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseCodingAgent:
    """Base class for all coding agent implementations."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._setup_tools()

    def _setup_tools(self):
        """Set up common tools for all agents."""
        self.python_repl = PythonREPL()
        self.python_repl_tool = Tool(
            name="python_repl",
            description="Execute python code and shell commands (pip commands for module installation). Use with caution.",
            func=self.python_repl.run,
        )

        self.duckduckgo_search_tool = Tool(
            name="duckduckgo_search",
            description="A wrapper around DuckDuckGo Search.",
            func=DuckDuckGoSearchRun().run,
        )

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt."""
        raise NotImplementedError("Subclasses must implement run()")


# =============================================================================
# CREWAI IMPLEMENTATION
# =============================================================================

class CrewAICodingAgent(BaseCodingAgent):
    """CrewAI-based coding agent implementation."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._setup_llm()
        self._setup_agents()
        self._setup_crew()

    def _setup_llm(self):
        """Set up the LLM based on provider."""
        if self.config.provider == "groq" and GROQ_AVAILABLE:
            self.llm = ChatGroq(
                temperature=self.config.temperature,
                groq_api_key=self.config.api_key,
                model_name=self.config.model
            )
        else:
            # For OpenAI or other providers, use environment variable
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                temperature=self.config.temperature,
                model=self.config.model
            )

    def _setup_agents(self):
        """Set up CrewAI agents."""
        self.coder_agent = Agent(
            role='Senior Software Engineer and Developer',
            goal=f'Write production grade bug free code on this user prompt: {{topic}}',
            verbose=True,
            memory=True,
            backstory="You are an experienced developer in big tech companies. "
                     "You write clean, efficient, and well-documented code.",
            max_iter=self.config.max_iterations,
            llm=self.llm,
            max_rpm=self.config.max_rpm,
            tools=[self.duckduckgo_search_tool],
            allow_delegation=False
        )

        self.debugger_agent = Agent(
            role='Code Debugger and Bug Solving Agent',
            goal='Debug the code line by line and solve bugs and errors using the python_repl tool. '
                 'You also have access to the search tool which can assist you for searching bugs.',
            verbose=True,
            memory=True,
            backstory="You are a debugger agent with access to a python interpreter. "
                     "You methodically test code and identify issues.",
            tools=[self.duckduckgo_search_tool, self.python_repl_tool],
            max_iter=self.config.max_iterations,
            llm=self.llm,
            max_rpm=self.config.max_rpm,
            allow_delegation=True
        )

    def _setup_crew(self):
        """Set up the CrewAI crew."""
        self.coding_task = Task(
            description="Write code for this {topic}.",
            expected_output='A bug-free and production-grade code on {topic}',
            tools=[self.duckduckgo_search_tool],
            agent=self.coder_agent,
        )

        self.debug_task = Task(
            description="Run the python code given by the CoderAgent and check for bugs and errors.",
            expected_output='Communicate to CoderAgent and give feedback on the code if errors occur during execution.',
            tools=[self.duckduckgo_search_tool, self.python_repl_tool],
            agent=self.debugger_agent,
            output_file='temp.py'
        )

        self.final_check = Task(
            description="Finalize the code which is verified by the debugger agent. The code should be error-free with no bugs.",
            expected_output="Communicate to DebuggerAgent. If the code is bug free and executed without errors, return the code to user.",
            agent=self.coder_agent,
            tools=[self.duckduckgo_search_tool]
        )

        self.crew = Crew(
            agents=[self.coder_agent, self.debugger_agent],
            tasks=[self.coding_task, self.debug_task, self.final_check],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=self.config.max_rpm * 2,
            share_crew=True
        )

    def run(self, prompt: str) -> str:
        """Run the CrewAI agent."""
        logger.info(f"Running CrewAI agent with prompt: {prompt[:50]}...")
        result = self.crew.kickoff(inputs={'topic': prompt})
        return str(result)


# =============================================================================
# LANGGRAPH IMPLEMENTATION
# =============================================================================

class CodeSolution(BaseModel):
    """Schema for code solutions."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

    class Config:
        description = "Schema for code solutions to questions about coding problems."


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: CodeSolution
    iterations: int


class LangGraphCodingAgent(BaseCodingAgent):
    """LangGraph-based coding agent implementation."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._setup_llm()
        self._setup_graph()

    def _setup_llm(self):
        """Set up the LLM."""
        if not GROQ_AVAILABLE:
            raise ImportError("langchain-groq is required for LangGraph agent")

        self.llm = ChatGroq(
            temperature=self.config.temperature,
            groq_api_key=self.config.api_key,
            model_name=self.config.model
        )

        # Set up structured output
        self.code_gen_chain = self.llm.with_structured_output(CodeSolution, include_raw=False)

        # Set up prompt template
        self.code_gen_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables defined.
             Structure your answer:
             1) A prefix describing the code solution
             2) The imports
             3) The functioning code block

             Here is the user question:"""
            ),
            ("placeholder", "{messages}"),
        ])

    def _generate_solution(self, state: GraphState) -> Dict:
        """Generate a code solution."""
        print("---GENERATING CODE SOLUTION---")

        messages = state["messages"]
        iterations = state["iterations"]

        # Generate solution
        code_solution = self.code_gen_chain.invoke(messages)
        messages += [
            ("assistant",
             f"Here is my attempt to solve the problem: {code_solution.prefix}\n"
             f"Imports: {code_solution.imports}\n"
             f"Code: {code_solution.code}")
        ]

        iterations += 1
        time.sleep(1)  # Rate limiting

        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations
        }

    def _check_code(self, state: GraphState) -> Dict:
        """Check the generated code."""
        print("---CHECKING CODE---")

        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        # Extract components
        imports = code_solution.imports
        code = code_solution.code

        # Check imports
        try:
            exec(imports)
        except Exception as e:
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [
                ("user",
                 f"Your solution failed the import test. Error: {e}\n"
                 "Reflect on this error and your prior attempt.\n"
                 "(1) State what went wrong with the prior solution\n"
                 "(2) Try to solve this problem again with the FULL SOLUTION.")
            ]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes"
            }

        # Check execution
        try:
            combined_code = f"{imports}\n{code}"
            global_scope = {}
            exec(combined_code, global_scope)
        except Exception as e:
            print("---CODE EXECUTION CHECK: FAILED---")
            error_message = [
                ("user",
                 f"Your solution failed the code execution test. Error: {e}\n"
                 "Reflect on this error and your prior attempt.\n"
                 "(1) State what went wrong with the prior solution\n"
                 "(2) Try to solve this problem again with the FULL SOLUTION.")
            ]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes"
            }

        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no"
        }

    def _decide_to_finish(self, state: GraphState) -> str:
        """Decide whether to finish or retry."""
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations >= self.config.max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            return "generate"

    def _setup_graph(self):
        """Set up the LangGraph workflow."""
        builder = StateGraph(GraphState)

        # Add nodes
        builder.add_node("generate", self._generate_solution)
        builder.add_node("check_code", self._check_code)

        # Build graph
        builder.set_entry_point("generate")
        builder.add_edge("generate", "check_code")
        builder.add_conditional_edges(
            "check_code",
            self._decide_to_finish,
            {"end": END, "generate": "generate"}
        )

        # Compile with memory
        memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = builder.compile(checkpointer=memory)

    def run(self, prompt: str) -> str:
        """Run the LangGraph agent."""
        logger.info(f"Running LangGraph agent with prompt: {prompt[:50]}...")

        # Display graph visualization if available
        if IPYTHON_AVAILABLE:
            try:
                display(Image(self.graph.get_graph(xray=True).draw_mermaid_png()))
            except:
                pass

        # Set up config
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        events = self.graph.stream(
            {"messages": [("user", prompt)], "iterations": 0},
            config,
            stream_mode="values"
        )

        # Collect results
        final_event = None
        for event in events:
            final_event = event

        return f"Final Solution:\n{final_event['generation'].prefix}\n\nImports:\n{final_event['generation'].imports}\n\nCode:\n{final_event['generation'].code}"


# =============================================================================
# AGENT FACTORY
# =============================================================================

class AgentFactory:
    """Factory for creating different types of coding agents."""

    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig) -> BaseCodingAgent:
        """Create an agent of the specified type."""
        agent_type = agent_type.lower()

        if agent_type == "crewai":
            return CrewAICodingAgent(config)
        elif agent_type == "langgraph":
            return LangGraphCodingAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_banner():
    """Print the application banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║         UNIFIED PYTHON CODING AGENT                        ║
║     Combining CrewAI and LangGraph Approaches              ║
╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_provider_choice() -> str:
    """Get the user's choice of LLM provider."""
    print("\n--- Choose LLM Provider ---")
    print("1. Groq (Llama models) - Fast & Free API available")
    print("2. OpenAI (GPT models)")

    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == "1":
            return "groq"
        elif choice == "2":
            return "openai"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_agent_type_choice() -> str:
    """Get the user's choice of agent type."""
    print("\n--- Choose Agent Type ---")
    print("1. CrewAI - Multi-agent collaborative approach")
    print("2. LangGraph - State-based workflow approach (Groq only)")

    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == "1":
            return "crewai"
        elif choice == "2":
            return "langgraph"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_api_key(provider: str) -> str:
    """Get the API key from user or environment."""
    env_var = f"{provider.upper()}_API_KEY"
    key = os.getenv(env_var, "")

    if key:
        print(f"\nFound API key in environment variable {env_var}")
        use_env = input("Use this key? (y/n): ").strip().lower()
        if use_env == 'y':
            return key

    print(f"\nEnter your {provider.upper()} API key:")
    return input().strip()


def interactive_mode(agent: BaseCodingAgent):
    """Run the agent in interactive mode."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter your coding prompts and the agent will generate solutions.")
    print("Type 'quit' or 'exit' to stop the agent.\n")

    while True:
        try:
            prompt = input("\n>>> Enter your coding task: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            print("\n" + "-"*60)
            result = agent.run(prompt)
            print("\n" + "-"*60)
            print(result)
            print("-"*60)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with a new prompt.")
        except Exception as e:
            logger.error(f"Error processing request: {e}")


def single_prompt_mode(agent: BaseCodingAgent, prompt: str):
    """Run the agent with a single prompt."""
    print("\n" + "="*60)
    print("PROCESSING YOUR REQUEST...")
    print("="*60 + "\n")

    result = agent.run(prompt)

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(result)
    print("="*60)


def main():
    """Main entry point."""
    print_banner()

    # Check for command line arguments
    if len(sys.argv) > 1:
        # Non-interactive mode: python unified_agent.py "your prompt here"
        prompt = " ".join(sys.argv[1:])
        provider = os.getenv("LLM_PROVIDER", "groq")
        api_key = os.getenv(f"{provider.upper()}_API_KEY", "")

        if not api_key:
            print(f"Error: Please set {provider.upper()}_API_KEY environment variable")
            sys.exit(1)

        config = AgentConfig(provider=provider, api_key=api_key)
        agent = AgentFactory.create_agent("crewai", config)
        single_prompt_mode(agent, prompt)
    else:
        # Interactive mode
        # Get provider
        provider = get_provider_choice()

        # Get API key
        api_key = get_api_key(provider)

        # Get agent type
        if provider == "groq":
            agent_type = get_agent_type_choice()
        else:
            print("\nNote: CrewAI agent recommended for OpenAI")
            agent_type = "crewai"

        # Set up model
        if provider == "groq":
            models = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
            print("\nAvailable Groq models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
            model_choice = input("Select model (1-3, default 1): ").strip() or "1"
            model = models[int(model_choice) - 1]
        else:
            model = "gpt-4"  # Default for OpenAI

        # Create configuration
        config = AgentConfig(
            provider=provider,
            api_key=api_key,
            model=model
        )

        # Create and run agent
        try:
            agent = AgentFactory.create_agent(agent_type, config)
            interactive_mode(agent)
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            print("\nFailed to initialize agent. Please check your configuration.")
            sys.exit(1)


if __name__ == "__main__":
    main()
