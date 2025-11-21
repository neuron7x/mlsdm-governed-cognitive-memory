import argparse
import json
import logging

from mlsdm.core.memory_manager import MemoryManager
from mlsdm.utils.config_loader import ConfigLoader


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(log_record)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="mlsdm-governed-cognitive-memory CLI")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()

    if args.api:
        import uvicorn

        from mlsdm.api.app import app

        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    config = ConfigLoader.load_config(args.config)
    manager = MemoryManager(config)
    logger.info("Running simulation...")
    manager.run_simulation(args.steps)
    logger.info("Done.")


if __name__ == "__main__":
    main()
