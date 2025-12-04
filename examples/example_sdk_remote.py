#!/usr/bin/env python3
"""
SDK Remote Mode Example for MLSDM Neuro Memory Service.

This example demonstrates using the NeuroMemoryClient in remote mode,
which connects to a running HTTP API server.

Usage:
    1. Start the server: mlsdm serve
    2. Run this script: python examples/example_sdk_remote.py

Prerequisites:
    pip install -e .
    pip install requests
"""

import sys


def check_server(base_url: str = "http://localhost:8000") -> bool:
    """Check if the server is running."""
    try:
        import requests
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def demo_health_check(client):
    """Demonstrate health check."""
    print("=" * 60)
    print("Health Check (Remote Mode)")
    print("=" * 60)

    try:
        result = client.health_check()
        print(f"\n  Status: {result.get('status', 'unknown')}")
        print(f"  Mode: {result.get('mode', 'unknown')}")
    except Exception as e:
        print(f"\n  Error: {e}")


def demo_memory_operations(client):
    """Demonstrate memory append and query operations."""
    print("\n" + "=" * 60)
    print("Memory Operations (Remote Mode)")
    print("=" * 60)

    # 1. Append memories
    print("\n1. Appending memories to remote service...")
    
    facts = [
        "User's name is Alice.",
        "Alice works as a software engineer.",
        "Alice prefers earl grey tea.",
    ]
    
    for fact in facts:
        try:
            result = client.append_memory(fact, moral_value=0.9)
            status = "✓" if result.success else "✗"
            print(f"  {status} {fact} (ID: {result.memory_id})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # 2. Query memories
    print("\n2. Querying remote memory...")
    
    try:
        result = client.query_memory("What does Alice like?", top_k=3)
        print(f"  Found: {result.total_results} items")
        print(f"  Phase: {result.query_phase}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_decision_making(client):
    """Demonstrate decision-making with governance."""
    print("\n" + "=" * 60)
    print("Decision Making (Remote Mode)")
    print("=" * 60)

    print("\n1. Making decision via remote API...")
    
    try:
        result = client.decide(
            prompt="Should I schedule a meeting with Alice?",
            context="Alice mentioned she's available on Thursday.",
            risk_level="low",
            mode="standard",
        )
        
        print(f"  Response: {result.response[:100]}...")
        print(f"  Accepted: {result.accepted}")
        print(f"  Decision ID: {result.decision_id}")
        print("  Contours:")
        for contour in result.contour_decisions:
            status = "✓" if contour.passed else "✗"
            print(f"    {status} {contour.contour}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_agent_protocol(client):
    """Demonstrate agent step protocol."""
    print("\n" + "=" * 60)
    print("Agent Step Protocol (Remote Mode)")
    print("=" * 60)

    print("\n1. Sending agent step to remote service...")
    
    try:
        result = client.agent_step(
            agent_id="remote-assistant",
            observation="User says: 'Hello, how are you?'",
            internal_state={"initialized": True},
        )
        
        print(f"  Action Type: {result.action.action_type}")
        print(f"  Response: {result.response[:80]}...")
        print(f"  Step ID: {result.step_id}")
        print(f"  Memory Updated: {result.memory_updated}")
    except Exception as e:
        print(f"  Error: {e}")


def demo_generate(client):
    """Demonstrate basic generation."""
    print("\n" + "=" * 60)
    print("Basic Generation (Remote Mode)")
    print("=" * 60)

    try:
        result = client.generate(
            prompt="What is the MLSDM cognitive memory system?",
            max_tokens=256,
            moral_value=0.8,
        )
        
        response = result.get("response", "")
        print(f"\n  Response: {response[:150]}...")
        print(f"  Accepted: {result.get('accepted', False)}")
        print(f"  Phase: {result.get('phase', 'unknown')}")
    except Exception as e:
        print(f"\n  Error: {e}")


def show_curl_examples():
    """Print curl command examples for the Product Layer API."""
    print("\n" + "=" * 60)
    print("Curl Examples for Product Layer API")
    print("=" * 60)
    print("""
# Memory Append
curl -X POST http://localhost:8000/v1/memory/append \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "User likes coffee",
    "user_id": "user-123",
    "moral_value": 0.8
  }'

# Memory Query
curl -X POST http://localhost:8000/v1/memory/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What does the user like?",
    "top_k": 5
  }'

# Decision
curl -X POST http://localhost:8000/v1/decide \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Should I recommend coffee?",
    "risk_level": "low",
    "mode": "standard"
  }'

# Agent Step
curl -X POST http://localhost:8000/v1/agent/step \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": "assistant-1",
    "observation": "User asked for help",
    "max_tokens": 256
  }'
""")


def main():
    """Run all remote mode demonstrations."""
    from mlsdm.sdk import NeuroMemoryClient

    print("\n" + "=" * 60)
    print("MLSDM Neuro Memory Service - SDK Remote Mode Examples")
    print("=" * 60)

    base_url = "http://localhost:8000"
    
    # Check if server is running
    if not check_server(base_url):
        print(f"\n⚠ Server not running at {base_url}")
        print("Start the server with: mlsdm serve")
        print("\nShowing curl examples instead...\n")
        show_curl_examples()
        sys.exit(0)

    print(f"\n✓ Connected to server at {base_url}")

    # Initialize client in remote mode
    client = NeuroMemoryClient(
        mode="remote",
        base_url=base_url,
        user_id="demo-user",
        session_id="demo-session",
    )

    # Run demonstrations
    demo_health_check(client)
    demo_memory_operations(client)
    demo_decision_making(client)
    demo_agent_protocol(client)
    demo_generate(client)

    print("\n" + "=" * 60)
    print("All remote mode examples completed!")
    print("=" * 60)
    print("\nAPI Endpoints Used:")
    print("  - GET  /health")
    print("  - POST /v1/memory/append")
    print("  - POST /v1/memory/query")
    print("  - POST /v1/decide")
    print("  - POST /v1/agent/step")
    print("  - POST /generate")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
