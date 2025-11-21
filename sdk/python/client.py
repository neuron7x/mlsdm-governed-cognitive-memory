from typing import Any

import requests


class MLSDMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def process_event(self, event_vector: list[float], moral_value: float) -> dict[str, Any]:
        payload = {"event_vector": event_vector, "moral_value": moral_value}
        response = requests.post(f"{self.base_url}/process_event/", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state/", headers=self.headers)
        response.raise_for_status()
        return response.json()
