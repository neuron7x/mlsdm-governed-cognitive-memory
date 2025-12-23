import inspect

import mlsdm.entrypoints.serve as serve_mod


def test_canonical_serve_smoke():
    if hasattr(serve_mod, "create_app"):
        app = serve_mod.create_app()
        assert app is not None
        return

    sig = inspect.signature(serve_mod.serve)
    assert "dry_run" in sig.parameters
    assert serve_mod.serve(dry_run=True) in (None, 0)
