from pathlib import Path

from tests.eval.aphasia_eval_suite import AphasiaEvalSuite


def test_aphasia_eval_suite_basic_metrics() -> None:
    corpus_path = Path("tests/eval/aphasia_corpus.json")
    assert corpus_path.exists(), "aphasia_corpus.json must exist"

    suite = AphasiaEvalSuite(corpus_path=corpus_path)
    result = suite.run()

    # Should confidently detect telegraphic speech
    assert result.true_positive_rate >= 0.8
    # Should not incorrectly flag normal speech
    assert result.true_negative_rate >= 0.8
    # Severity for telegraphic cases should be noticeable
    assert result.mean_severity_telegraphic >= 0.3
