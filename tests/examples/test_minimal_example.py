from examples import minimal_example


def test_minimal_example_without_otel(monkeypatch, capsys):
    monkeypatch.setattr(
        minimal_example.importlib.util, "find_spec", lambda name: None
    )

    minimal_example.main()

    output = capsys.readouterr().out
    assert "NOT INSTALLED" in output


def test_minimal_example_with_otel(monkeypatch, capsys):
    class DummySpec:
        ...

    monkeypatch.setattr(
        minimal_example.importlib.util, "find_spec", lambda name: DummySpec()
    )

    minimal_example.main()

    output = capsys.readouterr().out
    assert "INSTALLED" in output
