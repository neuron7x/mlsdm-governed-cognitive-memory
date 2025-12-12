import examples.production_chatbot_example as chatbot_example


class FakeWrapper:
    def __init__(self, *args, **kwargs):
        self.generate_calls: list[str] = []

    def generate(self, prompt: str, moral_value: float, context_top_k: int):
        self.generate_calls.append(prompt)
        return {
            "accepted": True,
            "response": f"response:{prompt}",
            "phase": "wake",
            "step": 1,
            "context_items": [],
            "moral_threshold": 0.5,
            "moral_ema": 0.5,
            "qilm_stats": {"used": 0, "capacity": 1, "memory_mb": 0.0},
            "synaptic_norms": {"L1": 0.1, "L2": 0.2, "L3": 0.3},
        }

    def get_state(self):
        return {
            "phase": "wake",
            "step": 1,
            "moral_threshold": 0.5,
            "moral_ema": 0.5,
            "qilm_stats": {"used": 0, "capacity": 1, "memory_mb": 0.0},
            "synaptic_norms": {"L1": 0.1, "L2": 0.2, "L3": 0.3},
        }


def test_production_chatbot_process_message(monkeypatch):
    monkeypatch.setattr(chatbot_example, "LLMWrapper", FakeWrapper)

    bot = chatbot_example.ProductionChatbot(dim=8)

    message = bot.process_message("hello world", max_retries=1)

    assert message.accepted is True
    assert bot.stats["total_messages"] == 1
    assert bot.stats["accepted"] == 1
    assert len(bot.history) == 2
    assert bot.history[-1].content.startswith("response:")


def test_production_chatbot_system_state(monkeypatch):
    monkeypatch.setattr(chatbot_example, "LLMWrapper", FakeWrapper)

    bot = chatbot_example.ProductionChatbot(dim=8)
    bot.process_message("hello world", max_retries=1)

    state = bot.get_system_state()

    assert state["phase"] == "wake"
    assert state["chatbot_stats"]["total_messages"] == 1
    assert state["qilm_stats"]["capacity"] == 1
