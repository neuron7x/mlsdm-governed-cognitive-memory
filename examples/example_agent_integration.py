#!/usr/bin/env python3
"""
Agent Integration Example for MLSDM Neuro Memory Service.

This example demonstrates how an external LLM/agent can integrate with
MLSDM as its memory and decision backend using the Agent Step Protocol.

The agent protocol:
1. Agent sends observation and internal state
2. MLSDM updates memory with the observation
3. MLSDM retrieves relevant context
4. MLSDM generates a response/action
5. Agent receives action, updated state, and memory updates

Usage:
    python examples/example_agent_integration.py

Prerequisites:
    pip install -e .
"""

from mlsdm.sdk import NeuroMemoryClient


class SimpleAgent:
    """A simple agent that uses MLSDM as its memory backend.
    
    This demonstrates the Agent Step Protocol for integrating
    external agents with MLSDM Neuro Memory Service.
    """
    
    def __init__(self, agent_id: str, client: NeuroMemoryClient):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            client: NeuroMemoryClient instance (local or remote mode).
        """
        self.agent_id = agent_id
        self.client = client
        self.internal_state = {
            "goal": "Assist user with tasks",
            "step_count": 0,
            "conversation_history": [],
        }
        self.running = True
    
    def process_observation(self, observation: str) -> str:
        """Process an observation and return an action.
        
        This is the main loop step for the agent. It:
        1. Sends the observation to MLSDM
        2. Gets back an action and updated state
        3. Executes the action or returns a response
        
        Args:
            observation: The current observation/input.
        
        Returns:
            The agent's response or action result.
        """
        # Add observation to conversation history
        self.internal_state["conversation_history"].append({
            "role": "user",
            "content": observation,
        })
        
        # Send to MLSDM via Agent Step Protocol
        result = self.client.agent_step(
            agent_id=self.agent_id,
            observation=observation,
            internal_state=self.internal_state,
        )
        
        # Update internal state from MLSDM
        if result.updated_state:
            self.internal_state.update(result.updated_state)
        
        # Handle action type
        if result.action.action_type == "terminate":
            self.running = False
            return "Goodbye! Agent terminating."
        
        elif result.action.action_type == "wait":
            return "[Agent is waiting - request was not accepted]"
        
        elif result.action.action_type == "tool_call":
            # In a real agent, this would execute tool calls
            tools = result.action.tool_calls or []
            return f"[Executing {len(tools)} tool(s)...]"
        
        else:  # respond
            response = result.action.content or result.response
            
            # Add response to conversation history
            self.internal_state["conversation_history"].append({
                "role": "assistant",
                "content": response,
            })
            
            return response
    
    def store_fact(self, fact: str) -> bool:
        """Store a fact in memory.
        
        Args:
            fact: The fact to store.
        
        Returns:
            True if storage succeeded.
        """
        result = self.client.append_memory(
            content=fact,
            metadata={"source": "agent", "agent_id": self.agent_id},
            moral_value=0.9,
        )
        return result.success
    
    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """Recall relevant memories.
        
        Args:
            query: Query to search for.
            top_k: Number of results.
        
        Returns:
            List of memory contents.
        """
        result = self.client.query_memory(query, top_k=top_k)
        return [item.content for item in result.results]
    
    def get_state(self) -> dict:
        """Get current agent state."""
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "step_count": self.internal_state.get("step_count", 0),
            "history_length": len(self.internal_state.get("conversation_history", [])),
        }


def demo_agent_conversation():
    """Demonstrate a multi-turn agent conversation."""
    print("=" * 60)
    print("Agent Conversation Demo")
    print("=" * 60)

    # Initialize client and agent
    client = NeuroMemoryClient(mode="local")
    agent = SimpleAgent(agent_id="demo-agent-001", client=client)

    # Simulate a conversation
    observations = [
        "Hello, I need help planning a coffee tasting event.",
        "We're expecting about 20 guests.",
        "The budget is around $200.",
        "What types of coffee should we include?",
        "Thank you, that's very helpful!",
    ]

    print("\n--- Conversation Start ---\n")

    for i, observation in enumerate(observations):
        print(f"User [{i+1}]: {observation}")
        
        response = agent.process_observation(observation)
        print(f"Agent: {response[:120]}...")
        print()
        
        if not agent.running:
            break

    print("--- Conversation End ---\n")

    # Show agent state
    state = agent.get_state()
    print("Agent Final State:")
    print(f"  Steps: {state['step_count']}")
    print(f"  History Length: {state['history_length']}")


def demo_agent_memory():
    """Demonstrate agent memory operations."""
    print("\n" + "=" * 60)
    print("Agent Memory Operations Demo")
    print("=" * 60)

    client = NeuroMemoryClient(mode="local")
    agent = SimpleAgent(agent_id="memory-agent-001", client=client)

    # Store facts
    print("\n1. Storing facts in memory...")
    facts = [
        "User prefers dark roast coffee.",
        "User is lactose intolerant.",
        "User has a coffee tasting event on Saturday.",
    ]
    
    for fact in facts:
        success = agent.store_fact(fact)
        status = "✓" if success else "✗"
        print(f"  {status} {fact}")

    # Recall memories
    print("\n2. Recalling memories...")
    queries = [
        "What are the user's coffee preferences?",
        "Any dietary restrictions?",
    ]
    
    for query in queries:
        memories = agent.recall(query, top_k=2)
        print(f"  Query: '{query}'")
        print(f"  Results: {len(memories)} items found")


def demo_agent_decision():
    """Demonstrate agent decision-making with governance."""
    print("\n" + "=" * 60)
    print("Agent Decision Making Demo")
    print("=" * 60)

    client = NeuroMemoryClient(mode="local")

    scenarios = [
        {
            "prompt": "Should I send a reminder about the coffee event?",
            "context": "The event is tomorrow and user hasn't responded.",
            "risk_level": "low",
            "mode": "standard",
        },
        {
            "prompt": "Should I share user's dietary info with caterers?",
            "context": "Caterers need to prepare suitable options.",
            "risk_level": "high",
            "mode": "cautious",
        },
        {
            "prompt": "The venue has a fire alarm - should I evacuate?",
            "context": "Fire alarm just went off at the venue.",
            "risk_level": "critical",
            "mode": "emergency",
        },
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\n{i+1}. Scenario: {scenario['prompt']}")
        print(f"   Risk Level: {scenario['risk_level']}, Mode: {scenario['mode']}")
        
        result = client.decide(
            prompt=scenario["prompt"],
            context=scenario.get("context"),
            risk_level=scenario["risk_level"],
            mode=scenario["mode"],
        )
        
        print(f"   Decision: {'ACCEPTED' if result.accepted else 'REJECTED'}")
        print(f"   Response: {result.response[:80]}...")
        
        # Show contour decisions
        for contour in result.contour_decisions:
            status = "✓" if contour.passed else "✗"
            print(f"   {status} {contour.contour}: {contour.notes}")


def main():
    """Run all agent integration demonstrations."""
    print("\n" + "=" * 60)
    print("MLSDM Agent Integration Examples")
    print("=" * 60)
    print("\nThis demonstrates the Agent Step Protocol for integrating")
    print("external LLM agents with MLSDM as their memory backend.")

    demo_agent_conversation()
    demo_agent_memory()
    demo_agent_decision()

    print("\n" + "=" * 60)
    print("All agent integration examples completed!")
    print("=" * 60)
    print("\nKey Integration Points:")
    print("  - NeuroMemoryClient.agent_step() for step-by-step agent execution")
    print("  - NeuroMemoryClient.append_memory() for storing facts")
    print("  - NeuroMemoryClient.query_memory() for recalling context")
    print("  - NeuroMemoryClient.decide() for governed decision-making")
    print("\nSee INTEGRATION_GUIDE.md for complete protocol documentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
