#!/usr/bin/env python3
"""
SDK Local Mode Example for MLSDM Neuro Memory Service.

This example demonstrates using the NeuroMemoryClient in local mode,
which runs the engine directly without HTTP networking.

Usage:
    python examples/example_sdk_local.py

Prerequisites:
    pip install -e .
"""

from mlsdm.sdk import NeuroMemoryClient


def demo_memory_operations():
    """Demonstrate memory append and query operations."""
    print("=" * 60)
    print("Memory Operations (Local Mode)")
    print("=" * 60)

    # Initialize client in local mode (no HTTP server needed)
    client = NeuroMemoryClient(mode="local", user_id="demo-user")

    # 1. Append memories
    print("\n1. Appending memories...")
    
    facts = [
        "The user prefers dark roast coffee in the morning.",
        "User's favorite programming language is Python.",
        "The user has a meeting at 3pm on Tuesdays.",
        "User likes hiking on weekends.",
    ]
    
    for fact in facts:
        result = client.append_memory(fact, moral_value=0.9)
        status = "✓" if result.success else "✗"
        print(f"  {status} {fact[:50]}... (phase: {result.phase})")
    
    # 2. Query memories
    print("\n2. Querying memories...")
    
    queries = [
        "What does the user like to drink?",
        "What are the user's hobbies?",
        "Programming preferences?",
    ]
    
    for query in queries:
        result = client.query_memory(query, top_k=3)
        print(f"\n  Query: '{query}'")
        print(f"  Found: {result.total_results} items (phase: {result.query_phase})")
        for item in result.results[:2]:
            print(f"    - Similarity: {item.similarity:.3f}")


def demo_decision_making():
    """Demonstrate decision-making with governance."""
    print("\n" + "=" * 60)
    print("Decision Making (Local Mode)")
    print("=" * 60)

    client = NeuroMemoryClient(mode="local")

    # 1. Standard decision
    print("\n1. Standard mode decision...")
    result = client.decide(
        prompt="Should I recommend coffee to the user?",
        context="User has mentioned they enjoy coffee in the morning.",
        risk_level="low",
        mode="standard",
    )
    
    print(f"  Response: {result.response[:100]}...")
    print(f"  Accepted: {result.accepted}")
    print(f"  Decision ID: {result.decision_id}")
    print("  Contours:")
    for contour in result.contour_decisions:
        status = "✓" if contour.passed else "✗"
        print(f"    {status} {contour.contour}: score={contour.score:.2f}")

    # 2. Cautious decision
    print("\n2. Cautious mode decision...")
    result = client.decide(
        prompt="Should I access the user's private data?",
        risk_level="high",
        mode="cautious",
    )
    
    print(f"  Response: {result.response[:100]}...")
    print(f"  Accepted: {result.accepted}")
    print(f"  Risk Assessment: {result.risk_assessment}")


def demo_agent_protocol():
    """Demonstrate agent step protocol."""
    print("\n" + "=" * 60)
    print("Agent Step Protocol (Local Mode)")
    print("=" * 60)

    client = NeuroMemoryClient(mode="local")

    # Simulate a multi-step agent interaction
    print("\n1. Agent initialization...")
    
    internal_state = {"goal": "Help user with coffee recommendations"}
    
    # Step 1: Initial observation
    result = client.agent_step(
        agent_id="coffee-assistant",
        observation="User says: 'I need help choosing a coffee.'",
        internal_state=internal_state,
    )
    
    print(f"  Action: {result.action.action_type}")
    print(f"  Response: {result.response[:80]}...")
    print(f"  Memory Updated: {result.memory_updated}")
    print(f"  Step ID: {result.step_id}")
    
    # Step 2: Follow-up
    print("\n2. Agent follow-up step...")
    result = client.agent_step(
        agent_id="coffee-assistant",
        observation="User says: 'I prefer something strong.'",
        internal_state=result.updated_state,
    )
    
    print(f"  Action: {result.action.action_type}")
    print(f"  Response: {result.response[:80]}...")
    print(f"  Step Count: {result.updated_state.get('step_count', 0)}")


def demo_generate():
    """Demonstrate basic generation (compatible with NeuroCognitiveClient)."""
    print("\n" + "=" * 60)
    print("Basic Generation (Local Mode)")
    print("=" * 60)

    client = NeuroMemoryClient(mode="local")

    result = client.generate(
        prompt="Explain the benefits of cognitive memory systems.",
        max_tokens=256,
        moral_value=0.8,
    )

    print(f"\n  Response: {result.get('response', '')[:150]}...")
    print(f"  Phase: {result.get('mlsdm', {}).get('phase', 'unknown')}")
    print(f"  Timing: {result.get('timing', {}).get('total', 0):.2f}ms")


def main():
    """Run all local mode demonstrations."""
    print("\n" + "=" * 60)
    print("MLSDM Neuro Memory Service - SDK Local Mode Examples")
    print("=" * 60)

    demo_memory_operations()
    demo_decision_making()
    demo_agent_protocol()
    demo_generate()

    print("\n" + "=" * 60)
    print("All local mode examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - See example_sdk_remote.py for HTTP API examples")
    print("  - See example_agent_integration.py for full agent demo")
    print("  - See SDK_USAGE.md for detailed documentation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
