#!/usr/bin/env python3
"""
Conversational Assistant Example - End-to-End Use Case.

This example demonstrates a complete neuro-memory assistant that:
1. Maintains conversation memory across interactions
2. Uses moral/risk governance for responses
3. Retrieves relevant context from memory
4. Makes decisions with cognitive rhythm awareness

This is the reference implementation for the "Neuro Memory for LLM Assistant"
use case described in the product specification.

Usage:
    python examples/example_conversational_assistant.py

    # Or with HTTP server:
    mlsdm serve  # In one terminal
    python examples/example_conversational_assistant.py --remote  # In another

Prerequisites:
    pip install -e .
"""

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mlsdm.sdk import NeuroMemoryClient


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuroMemoryAssistant:
    """A conversational assistant powered by MLSDM Neuro Memory Service.
    
    This assistant demonstrates:
    - Session-based memory (facts persist within a session)
    - Moral filtering (inappropriate content is filtered)
    - Cognitive rhythm (awareness of wake/sleep phases)
    - Context retrieval (relevant memories inform responses)
    - Risk-aware decisions (adjusts behavior based on risk level)
    """
    
    def __init__(
        self,
        client: NeuroMemoryClient,
        user_id: str = "user",
        session_id: str | None = None,
    ):
        """Initialize the assistant.
        
        Args:
            client: NeuroMemoryClient instance.
            user_id: User identifier for memory scoping.
            session_id: Session identifier (auto-generated if not provided).
        """
        self.client = client
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.conversation_history: list[ConversationTurn] = []
        self.facts_stored: int = 0
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate a response.
        
        This is the main interaction method. It:
        1. Stores the user input in memory
        2. Retrieves relevant context
        3. Generates a response with governance
        4. Returns the response
        
        Args:
            user_input: The user's message.
        
        Returns:
            The assistant's response.
        """
        # Record user turn
        user_turn = ConversationTurn(role="user", content=user_input)
        self.conversation_history.append(user_turn)
        
        # 1. Store user input in memory (as a fact/observation)
        self._store_interaction(user_input)
        
        # 2. Query relevant context from memory
        context_items = self._retrieve_context(user_input)
        
        # 3. Build enhanced prompt with context
        enhanced_prompt = self._build_prompt(user_input, context_items)
        
        # 4. Generate response with governance
        response, metadata = self._generate_response(enhanced_prompt)
        
        # 5. Store the response as well (for future context)
        self._store_response(response)
        
        # Record assistant turn
        assistant_turn = ConversationTurn(
            role="assistant",
            content=response,
            metadata=metadata,
        )
        self.conversation_history.append(assistant_turn)
        
        return response
    
    def _store_interaction(self, content: str) -> None:
        """Store user interaction in memory."""
        result = self.client.append_memory(
            content=f"[{datetime.now().isoformat()}] User: {content}",
            user_id=self.user_id,
            session_id=self.session_id,
            moral_value=0.9,  # User inputs are typically neutral
        )
        if result.success:
            self.facts_stored += 1
    
    def _store_response(self, response: str) -> None:
        """Store assistant response in memory."""
        self.client.append_memory(
            content=f"[{datetime.now().isoformat()}] Assistant: {response[:200]}",
            user_id=self.user_id,
            session_id=self.session_id,
            moral_value=0.95,  # Assistant responses should be high quality
        )
    
    def _retrieve_context(self, query: str) -> list[str]:
        """Retrieve relevant context from memory."""
        result = self.client.query_memory(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            top_k=3,
        )
        return [item.content for item in result.results]
    
    def _build_prompt(self, user_input: str, context_items: list[str]) -> str:
        """Build an enhanced prompt with context."""
        prompt_parts = []
        
        if context_items:
            prompt_parts.append("Relevant context from memory:")
            for item in context_items:
                prompt_parts.append(f"- {item[:100]}")
            prompt_parts.append("")
        
        # Include recent conversation history
        if len(self.conversation_history) > 2:
            prompt_parts.append("Recent conversation:")
            for turn in self.conversation_history[-4:-1]:  # Last 2 exchanges
                prefix = "User" if turn.role == "user" else "Assistant"
                prompt_parts.append(f"{prefix}: {turn.content[:100]}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _generate_response(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate a response with governance."""
        result = self.client.decide(
            prompt=prompt,
            risk_level="low",
            mode="standard",
            use_memory=True,
            context_top_k=5,
        )
        
        metadata = {
            "accepted": result.accepted,
            "phase": result.phase,
            "decision_id": result.decision_id,
            "memory_context_used": result.memory_context_used,
        }
        
        if not result.accepted:
            return "I'm sorry, I cannot respond to that request.", metadata
        
        return result.response, metadata
    
    def get_session_info(self) -> dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "turns": len(self.conversation_history),
            "facts_stored": self.facts_stored,
        }


def run_interactive_demo(client: NeuroMemoryClient):
    """Run an interactive chat demo."""
    print("\n" + "=" * 60)
    print("MLSDM Neuro Memory Assistant - Interactive Demo")
    print("=" * 60)
    print("\nType your messages below. Commands:")
    print("  /info   - Show session information")
    print("  /clear  - Clear conversation history")
    print("  /quit   - Exit the demo")
    print("-" * 60)

    assistant = NeuroMemoryAssistant(
        client=client,
        user_id="interactive-user",
    )

    print(f"\nSession ID: {assistant.session_id}")
    print("Ready to chat!\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "/info":
            info = assistant.get_session_info()
            print(f"\n--- Session Info ---")
            print(f"Session ID: {info['session_id']}")
            print(f"User ID: {info['user_id']}")
            print(f"Turns: {info['turns']}")
            print(f"Facts Stored: {info['facts_stored']}")
            print("-------------------\n")
            continue
        
        if user_input.lower() == "/clear":
            assistant = NeuroMemoryAssistant(
                client=client,
                user_id="interactive-user",
            )
            print(f"\nConversation cleared. New session: {assistant.session_id}\n")
            continue

        response = assistant.process_input(user_input)
        print(f"Assistant: {response}\n")

    # Final summary
    print("\n" + "=" * 60)
    info = assistant.get_session_info()
    print(f"Session Summary:")
    print(f"  Total Turns: {info['turns']}")
    print(f"  Facts Stored: {info['facts_stored']}")
    print("=" * 60 + "\n")


def run_scripted_demo(client: NeuroMemoryClient):
    """Run a scripted demonstration."""
    print("\n" + "=" * 60)
    print("MLSDM Neuro Memory Assistant - Scripted Demo")
    print("=" * 60)

    assistant = NeuroMemoryAssistant(
        client=client,
        user_id="demo-user",
    )

    print(f"\nSession ID: {assistant.session_id}")
    print("-" * 60)

    # Scripted conversation
    exchanges = [
        "Hello! I'm planning a birthday party for my friend.",
        "She loves chocolate cake and Italian food.",
        "We're expecting about 15 guests.",
        "What should I consider for the menu?",
        "Oh, I forgot to mention - she's vegetarian.",
        "Can you update your recommendation based on that?",
    ]

    print("\n--- Conversation Start ---\n")

    for user_input in exchanges:
        print(f"User: {user_input}")
        response = assistant.process_input(user_input)
        print(f"Assistant: {response[:150]}...")
        print()

    print("--- Conversation End ---\n")

    # Show session summary
    info = assistant.get_session_info()
    print("Session Summary:")
    print(f"  Total Turns: {info['turns']}")
    print(f"  Facts Stored: {info['facts_stored']}")
    print(f"  Session ID: {info['session_id']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MLSDM Neuro Memory Assistant Demo"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote HTTP API mode (requires running server)",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run scripted demo instead of interactive",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for remote mode (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    # Initialize client
    if args.remote:
        # Check if server is running
        try:
            import requests
            response = requests.get(f"{args.url}/health", timeout=5)
            if response.status_code != 200:
                print(f"Error: Server not responding at {args.url}")
                sys.exit(1)
        except Exception as e:
            print(f"Error: Cannot connect to server at {args.url}")
            print(f"  {e}")
            print("\nStart the server with: mlsdm serve")
            sys.exit(1)
        
        print(f"Connecting to {args.url}...")
        client = NeuroMemoryClient(mode="remote", base_url=args.url)
    else:
        print("Using local mode (no HTTP server needed)...")
        client = NeuroMemoryClient(mode="local")

    # Run appropriate demo
    if args.scripted:
        run_scripted_demo(client)
    else:
        run_interactive_demo(client)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Session-based memory (facts persist within session)")
    print("  ✓ Moral filtering (governed responses)")
    print("  ✓ Context retrieval (relevant memories inform responses)")
    print("  ✓ Cognitive rhythm awareness (phase tracking)")
    print("\nSee USAGE_GUIDE.md for more examples and documentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
