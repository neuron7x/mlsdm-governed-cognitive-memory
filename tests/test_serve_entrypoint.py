from fastapi import FastAPI

from mlsdm.serve import get_app


def test_get_app_api_returns_fastapi():
    app = get_app("api")

    assert isinstance(app, FastAPI)
    paths = [route.path for route in app.router.routes]
    assert any("/health" in path for path in paths)


def test_get_app_neuro_returns_fastapi():
    app = get_app("neuro")

    assert isinstance(app, FastAPI)
    paths = [route.path for route in app.router.routes]
    assert "/v1/neuro/generate" in paths
