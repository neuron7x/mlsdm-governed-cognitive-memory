"""
MLSDM CLI - Command-line interface for the MLSDM system.

Commands:
- mlsdm demo: Interactive demo of the LLM wrapper
- mlsdm serve: Start the HTTP API server
- mlsdm check: Check environment and configuration
"""

import argparse
import json
import os
import sys
from typing import Any


def cmd_demo(args: argparse.Namespace) -> int:
    """Run interactive demo of MLSDM."""
    try:
        from mlsdm import create_llm_wrapper
    except ImportError as e:
        print(f"Error: Failed to import mlsdm: {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("MLSDM Demo - Governed Cognitive Memory")
    print("=" * 60)
    print()

    # Create wrapper with demo settings
    wrapper = create_llm_wrapper(
        wake_duration=args.wake_duration,
        sleep_duration=args.sleep_duration,
        initial_moral_threshold=args.moral_threshold,
    )

    print(f"Wrapper created with:")
    print(f"  - Wake duration: {args.wake_duration} steps")
    print(f"  - Sleep duration: {args.sleep_duration} steps")
    print(f"  - Moral threshold: {args.moral_threshold}")
    print()

    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' or 'exit' to exit.")
        print("Type 'state' to show current system state.")
        print("-" * 60)

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if prompt.lower() == "state":
                state = wrapper.get_state()
                print("\nSystem State:")
                print(json.dumps(state, indent=2, default=str))
                continue

            # Generate response
            result = wrapper.generate(prompt=prompt, moral_value=args.moral_value)

            if result["accepted"]:
                print(f"\nMLSDM [{result['phase']}]: {result['response']}")
            else:
                print(f"\n[Rejected - {result['note']}]")

    else:
        # Single prompt mode
        if args.prompt:
            result = wrapper.generate(prompt=args.prompt, moral_value=args.moral_value)

            print(f"Prompt: {args.prompt}")
            print(f"Moral Value: {args.moral_value}")
            print("-" * 60)

            if result["accepted"]:
                print(f"Response: {result['response']}")
                print(f"Phase: {result['phase']}")
            else:
                print(f"Rejected: {result['note']}")

            if args.verbose:
                print("\nFull result:")
                print(json.dumps(result, indent=2, default=str))
        else:
            # Demo prompts
            demo_prompts = [
                ("Hello, how are you?", 0.9),
                ("Tell me about Python", 0.85),
                ("What is machine learning?", 0.8),
            ]

            print("Running demo prompts...")
            print("-" * 60)

            for prompt, moral in demo_prompts:
                result = wrapper.generate(prompt=prompt, moral_value=moral)
                status = "✓" if result["accepted"] else "✗"
                print(f"\n{status} Prompt: {prompt}")
                print(f"  Phase: {result['phase']}, Accepted: {result['accepted']}")
                if result["accepted"]:
                    print(f"  Response: {result['response'][:80]}...")

    print()
    print("=" * 60)
    print("Demo complete.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the HTTP API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 1

    # Set environment variables from args
    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    if args.disable_rate_limit:
        os.environ["DISABLE_RATE_LIMIT"] = "1"

    print("=" * 60)
    print("MLSDM HTTP API Server")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Backend: {os.environ.get('LLM_BACKEND', 'local_stub')}")
    print(f"Config: {os.environ.get('CONFIG_PATH', 'config/default_config.yaml')}")
    print()

    # Import app here to pick up environment variables
    from mlsdm.api.app import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )

    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check environment and configuration."""
    print("=" * 60)
    print("MLSDM Environment Check")
    print("=" * 60)
    print()

    status: dict[str, Any] = {
        "python_version": sys.version,
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # Check Python version
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.version_info < (3, 10):
        status["errors"].append("Python 3.10+ required")
        print("  ✗ Python 3.10+ required")
    else:
        print("  ✓ Python version OK")
    status["checks"]["python"] = sys.version_info >= (3, 10)

    # Check core imports
    print("\nCore dependencies:")
    core_deps = [
        ("numpy", "numpy"),
        ("FastAPI", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
    ]

    for name, module in core_deps:
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
            status["checks"][name] = True
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            status["checks"][name] = False
            status["errors"].append(f"{name} not installed")

    # Check mlsdm import
    print("\nMLSDM package:")
    try:
        from mlsdm import __version__
        print(f"  ✓ mlsdm v{__version__} installed")
        status["checks"]["mlsdm"] = True
        status["mlsdm_version"] = __version__
    except ImportError as e:
        print(f"  ✗ mlsdm NOT installed: {e}")
        status["checks"]["mlsdm"] = False
        status["errors"].append(f"mlsdm not installed: {e}")

    # Check optional dependencies
    print("\nOptional dependencies:")
    optional_deps = [
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("prometheus-client", "prometheus_client"),
    ]

    for name, module in optional_deps:
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
            status["checks"][name] = True
        except ImportError:
            print(f"  ○ {name} not installed (optional)")
            status["checks"][name] = False
            status["warnings"].append(f"{name} not installed (optional)")

    # Check configuration
    print("\nConfiguration:")
    config_path = os.environ.get("CONFIG_PATH", "config/default_config.yaml")
    if os.path.exists(config_path):
        print(f"  ✓ Config file exists: {config_path}")
        status["checks"]["config"] = True
    else:
        print(f"  ○ Config file not found: {config_path} (will use defaults)")
        status["checks"]["config"] = False
        status["warnings"].append(f"Config file not found: {config_path}")

    # Check environment variables
    print("\nEnvironment variables:")
    env_vars = [
        ("LLM_BACKEND", "local_stub"),
        ("OPENAI_API_KEY", None),
        ("CONFIG_PATH", None),
    ]

    for var, default in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "SECRET" in var:
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                print(f"  ✓ {var} = {masked}")
            else:
                print(f"  ✓ {var} = {value}")
        elif default:
            print(f"  ○ {var} not set (default: {default})")
        else:
            print(f"  ○ {var} not set")

    # Summary
    print("\n" + "=" * 60)
    error_count = len(status["errors"])
    warning_count = len(status["warnings"])

    if error_count == 0:
        print("✓ All checks passed!")
        if warning_count > 0:
            print(f"  ({warning_count} warnings)")
    else:
        print(f"✗ {error_count} error(s), {warning_count} warning(s)")
        for error in status["errors"]:
            print(f"  Error: {error}")

    if args.verbose:
        print("\nFull status:")
        print(json.dumps(status, indent=2, default=str))

    return 0 if error_count == 0 else 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mlsdm",
        description="MLSDM - Governed Cognitive Memory CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.2.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    demo_parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Single prompt to process",
    )
    demo_parser.add_argument(
        "-m", "--moral-value",
        type=float,
        default=0.8,
        help="Moral value for prompts (default: 0.8)",
    )
    demo_parser.add_argument(
        "--moral-threshold",
        type=float,
        default=0.5,
        help="Initial moral threshold (default: 0.5)",
    )
    demo_parser.add_argument(
        "--wake-duration",
        type=int,
        default=8,
        help="Wake cycle duration in steps (default: 8)",
    )
    demo_parser.add_argument(
        "--sleep-duration",
        type=int,
        default=3,
        help="Sleep cycle duration in steps (default: 3)",
    )
    demo_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start HTTP API server")
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    serve_parser.add_argument(
        "--backend",
        type=str,
        choices=["local_stub", "openai"],
        help="LLM backend to use",
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--disable-rate-limit",
        action="store_true",
        help="Disable rate limiting (for testing)",
    )

    # Check command
    check_parser = subparsers.add_parser("check", help="Check environment")
    check_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    if args.command == "demo":
        return cmd_demo(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "check":
        return cmd_check(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
