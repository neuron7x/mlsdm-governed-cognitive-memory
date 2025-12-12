from pathlib import Path

import yaml

from scripts.validate_policy_config import PolicyValidator


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_validate_policy_config_success(tmp_path: Path):
    repo_root = tmp_path
    policy_dir = repo_root / "policy"
    policy_dir.mkdir()

    workflows_dir = repo_root / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "security.yml").write_text("", encoding="utf-8")

    security_policy = {
        "required_checks": [
            {"name": "security-check", "workflow_file": ".github/workflows/security.yml"}
        ]
    }
    slo_policy = {"slos": {}, "documentation": {}}

    write_yaml(policy_dir / "security-baseline.yaml", security_policy)
    write_yaml(policy_dir / "observability-slo.yaml", slo_policy)

    validator = PolicyValidator(repo_root, policy_dir)

    assert validator.validate_all()


def test_validate_policy_config_missing_workflow(tmp_path: Path):
    repo_root = tmp_path
    policy_dir = repo_root / "policy"
    policy_dir.mkdir()

    security_policy = {
        "required_checks": [
            {"name": "security-check", "workflow_file": ".github/workflows/missing.yml"}
        ]
    }
    slo_policy = {"slos": {}, "documentation": {}}

    write_yaml(policy_dir / "security-baseline.yaml", security_policy)
    write_yaml(policy_dir / "observability-slo.yaml", slo_policy)

    validator = PolicyValidator(repo_root, policy_dir)

    assert not validator.validate_all()
