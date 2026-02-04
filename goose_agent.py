#!/usr/bin/env python3
"""
Goose Agent - A standalone Python agent for ACP (Agent Communication Protocol)
Connects to goose acp running on stdio and provides a simple interface for interaction.
"""

import subprocess
import json
import os
import sys
import time
import argparse
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SessionState(Enum):
    """Agent session states."""
    DISCONNECTED = "disconnected"
    INITIALIZED = "initialized"
    SESSION_CREATED = "session_created"
    SESSION_LOADED = "session_loaded"


@dataclass
class Notification:
    """Represents an ACP notification."""
    method: str
    params: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class SessionInfo:
    """Information about a session."""
    session_id: str
    cwd: str
    created_at: float = field(default_factory=time.time)
    message_count: int = 0


class GooseAgent:
    """
    A standalone Python agent for communicating with goose via ACP protocol.

    Features:
    - Initialize connection with goose ACP
    - Create and manage sessions
    - Send prompts and receive streaming responses
    - Load existing sessions
    - Collect and manage notifications
    - Interactive and programmatic modes
    """

    def __init__(self,
                 cargo_path: str = "cargo",
                 package: str = "goose-cli",
                 command: str = "acp"):
        """Initialize the Goose Agent.

        Args:
            cargo_path: Path to cargo executable
            package: Cargo package to run
            command: Command to pass to the package
        """
        self.cargo_path = cargo_path
        self.package = package
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.state = SessionState.DISCONNECTED
        self.current_session: Optional[SessionInfo] = None
        self.notifications_buffer: List[Notification] = []
        self.capabilities: Dict[str, Any] = {}

    def connect(self) -> bool:
        """Establish connection to goose ACP."""
        try:
            self.process = subprocess.Popen(
                [self.cargo_path, 'run', '-p', self.package, '--', self.command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )

            # Test connection
            time.sleep(0.5)
            if self.process.poll() is not None:
                # Process exited
                stderr = self.process.stderr.read() if self.process.stderr else ""
                print(f"Failed to start goose agent: {stderr}", file=sys.stderr)
                return False

            return True
        except Exception as e:
            print(f"Error connecting to goose: {e}", file=sys.stderr)
            return False

    def disconnect(self):
        """Close the connection to goose ACP."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.state = SessionState.DISCONNECTED
        self.current_session = None

    def _send_request(self, method: str, params: Optional[Dict] = None,
                     collect_notifications: bool = False) -> Tuple[Optional[Dict], List[Notification]]:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: JSON-RPC method name
            params: Optional parameters
            collect_notifications: If True, collect notifications until response

        Returns:
            Tuple of (response, notifications)
        """
        if not self.process:
            return None, []

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id,
        }
        if params:
            request["params"] = params

        request_str = json.dumps(request)
        self.process.stdin.write(request_str + '\n')
        self.process.stdin.flush()

        notifications = []

        # Read responses until we get one with our request ID
        max_retries = 100
        retry_count = 0

        while retry_count < max_retries:
            response_line = self.process.stdout.readline()
            if not response_line:
                if self.process.poll() is not None:
                    # Process ended
                    return None, notifications
                retry_count += 1
                time.sleep(0.1)
                continue

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                retry_count += 1
                continue

            # Check if this is a notification (has 'method' but no 'id')
            if 'method' in response and 'id' not in response:
                notification = Notification(
                    method=response['method'],
                    params=response.get('params', {}),
                    raw=response
                )
                if collect_notifications:
                    notifications.append(notification)
                self.notifications_buffer.append(notification)
                continue

            if response.get('id') == self.request_id:
                return response, notifications

            retry_count += 1

        return None, notifications

    def initialize(self, client_name: str = "goose-agent",
                  client_version: str = "1.0.0") -> bool:
        """Initialize the ACP connection.

        Args:
            client_name: Name of the client
            client_version: Version of the client

        Returns:
            True if successful
        """
        response, _ = self._send_request("initialize", {
            "protocolVersion": "v1",
            "clientCapabilities": {},
            "clientInfo": {
                "name": client_name,
                "version": client_version
            }
        })

        if response and 'result' in response:
            self.state = SessionState.INITIALIZED
            self.capabilities = response['result'].get('agentCapabilities', {})
            return True

        return False

    def create_session(self, cwd: Optional[str] = None,
                      mcp_servers: Optional[List] = None) -> Optional[str]:
        """Create a new session.

        Args:
            cwd: Working directory for the session
            mcp_servers: List of MCP servers to connect

        Returns:
            Session ID if successful, None otherwise
        """
        params = {
            "mcpServers": mcp_servers or [],
            "cwd": cwd or os.getcwd()
        }

        response, _ = self._send_request("session/new", params)

        if response and 'result' in response:
            session_id = response['result']['sessionId']
            self.current_session = SessionInfo(
                session_id=session_id,
                cwd=params['cwd']
            )
            self.state = SessionState.SESSION_CREATED
            return session_id

        return None

    def load_session(self, session_id: str, cwd: Optional[str] = None,
                    mcp_servers: Optional[List] = None) -> Tuple[bool, List[Notification]]:
        """Load an existing session.

        Args:
            session_id: ID of the session to load
            cwd: Working directory
            mcp_servers: List of MCP servers

        Returns:
            Tuple of (success, notifications with session history)
        """
        params = {
            "sessionId": session_id,
            "mcpServers": mcp_servers or [],
            "cwd": cwd or os.getcwd()
        }

        response, notifications = self._send_request("session/load", params,
                                                    collect_notifications=True)

        if response and 'result' in response:
            self.current_session = SessionInfo(
                session_id=session_id,
                cwd=params['cwd']
            )
            self.state = SessionState.SESSION_LOADED
            return True, notifications

        return False, notifications

    def send_prompt(self, text: str, session_id: Optional[str] = None) -> Tuple[Optional[Dict], List[Notification]]:
        """Send a prompt to the session.

        Args:
            text: Prompt text
            session_id: Session ID (uses current session if None)

        Returns:
            Tuple of (response, streaming notifications)
        """
        sid = session_id or (self.current_session.session_id if self.current_session else None)
        if not sid:
            return None, []

        if self.current_session:
            self.current_session.message_count += 1

        return self._send_request("session/prompt", {
            "sessionId": sid,
            "prompt": [{
                "type": "text",
                "text": text
            }]
        }, collect_notifications=True)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return self.capabilities

    def get_session_info(self) -> Optional[SessionInfo]:
        """Get current session information."""
        return self.current_session

    def get_notifications(self, clear: bool = False) -> List[Notification]:
        """Get buffered notifications.

        Args:
            clear: If True, clear the buffer after returning

        Returns:
            List of notifications
        """
        notifications = self.notifications_buffer.copy()
        if clear:
            self.notifications_buffer.clear()
        return notifications

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class InteractiveGooseAgent:
    """Interactive CLI interface for Goose Agent."""

    def __init__(self):
        self.agent = GooseAgent()
        self.running = False

    def start(self):
        """Start interactive session."""
        print("="*60)
        print("Goose Agent - Interactive Mode")
        print("="*60)

        if not self.agent.connect():
            print("Failed to connect to goose agent", file=sys.stderr)
            return

        print("\nInitializing...")
        if not self.agent.initialize():
            print("Failed to initialize", file=sys.stderr)
            self.agent.disconnect()
            return

        caps = self.agent.get_capabilities()
        print(f"✓ Connected")
        print(f"  - Load session: {caps.get('loadSession', False)}")
        print(f"  - Capabilities: {list(caps.get('promptCapabilities', {}).keys())}")

        self.running = True
        self._run_interactive_loop()

    def _run_interactive_loop(self):
        """Main interactive loop."""
        while self.running:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input in ('quit', 'exit', 'q'):
                    self.running = False
                    break

                elif user_input == 'new':
                    session_id = self.agent.create_session()
                    if session_id:
                        print(f"✓ Created session: {session_id}")
                    else:
                        print("✗ Failed to create session")

                elif user_input.startswith('load '):
                    session_id = user_input[5:].strip()
                    success, notifs = self.agent.load_session(session_id)
                    if success:
                        print(f"✓ Loaded session: {session_id}")
                        print(f"  Received {len(notifs)} history notifications")
                    else:
                        print(f"✗ Failed to load session: {session_id}")

                elif user_input == 'status':
                    self._show_status()

                elif user_input == 'help':
                    self._show_help()

                else:
                    # Treat as prompt
                    if not self.agent.get_session_info():
                        # Auto-create session
                        print("(Creating new session...)")
                        session_id = self.agent.create_session()
                        if not session_id:
                            print("✗ Failed to create session")
                            continue

                    print("\nSending prompt...")
                    response, notifications = self.agent.send_prompt(user_input)

                    if notifications:
                        print(f"\n📝 {len(notifications)} streaming updates:")
                        for n in notifications:
                            update = n.params.get('update', {})
                            update_type = update.get('sessionUpdate', 'unknown')
                            content = update.get('content', {})
                            if isinstance(content, dict):
                                text = content.get('text', '')
                            else:
                                text = str(content)
                            if text:
                                print(f"  [{update_type}]: {text[:200]}")

                    if response and 'result' in response:
                        print(f"\n✓ Response received")
                    elif response and 'error' in response:
                        print(f"\n✗ Error: {response['error']}")

            except KeyboardInterrupt:
                print("\n(Interrupt)")
                self.running = False
                break
            except EOFError:
                self.running = False
                break

        self.agent.disconnect()
        print("\nGoodbye!")

    def _show_status(self):
        """Show current status."""
        print("\nStatus:")
        print(f"  State: {self.agent.state.value}")
        session = self.agent.get_session_info()
        if session:
            print(f"  Session ID: {session.session_id}")
            print(f"  Working Dir: {session.cwd}")
            print(f"  Messages: {session.message_count}")
        else:
            print(f"  Session: None")
        notifs = self.agent.get_notifications()
        print(f"  Buffered Notifications: {len(notifs)}")

    def _show_help(self):
        """Show help."""
        print("\nCommands:")
        print("  <prompt>     Send a prompt to the agent")
        print("  new          Create a new session")
        print("  load <id>    Load an existing session")
        print("  status       Show current status")
        print("  help         Show this help")
        print("  quit/exit    Exit")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Goose Agent - ACP client for goose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python goose_agent.py                    # Interactive mode
  python goose_agent.py --prompt "Hello"   # Single prompt
  python goose_agent.py --session abc123   # Load and interact with session
        """
    )

    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Run in interactive mode (default)')
    parser.add_argument('-p', '--prompt', type=str,
                       help='Send a single prompt and exit')
    parser.add_argument('-s', '--session', type=str,
                       help='Session ID to load')
    parser.add_argument('-c', '--cwd', type=str,
                       help='Working directory for session')
    parser.add_argument('--cargo-path', type=str, default='cargo',
                       help='Path to cargo executable')
    parser.add_argument('--package', type=str, default='goose-cli',
                       help='Cargo package to run')

    args = parser.parse_args()

    # Single prompt mode
    if args.prompt:
        with GooseAgent(cargo_path=args.cargo_path, package=args.package) as agent:
            if not agent.initialize():
                print("Failed to initialize", file=sys.stderr)
                return 1

            session_id = None
            if args.session:
                success, _ = agent.load_session(args.session, cwd=args.cwd)
                if not success:
                    print(f"Failed to load session: {args.session}", file=sys.stderr)
                    return 1
                session_id = args.session
            else:
                session_id = agent.create_session(cwd=args.cwd)
                if not session_id:
                    print("Failed to create session", file=sys.stderr)
                    return 1

            response, notifications = agent.send_prompt(args.prompt)

            if notifications:
                for n in notifications:
                    update = n.params.get('update', {})
                    content = update.get('content', {})
                    if isinstance(content, dict):
                        text = content.get('text', '')
                    else:
                        text = str(content)
                    if text:
                        print(text)

            if response and 'error' in response:
                print(f"Error: {response['error']}", file=sys.stderr)
                return 1

        return 0

    # Interactive mode (default)
    interactive = InteractiveGooseAgent()
    interactive.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
