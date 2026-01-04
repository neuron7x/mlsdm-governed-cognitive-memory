import math
from types import SimpleNamespace

import numpy as np

from mlsdm.core.governance_kernel import GovernanceKernel


def _build_kernel(
    dim: int = 4, capacity: int = 3, wake_duration: int = 2, sleep_duration: int = 1
) -> GovernanceKernel:
    return GovernanceKernel(
        dim=dim,
        capacity=capacity,
        wake_duration=wake_duration,
        sleep_duration=sleep_duration,
    )


def test_governance_kernel_initializes_proxies() -> None:
    kernel = _build_kernel()

    moral_state = kernel.moral_ro.get_state()
    assert "threshold" in moral_state

    l1, l2, l3 = kernel.synaptic_ro.get_state()
    assert l1.shape == (4,)
    assert l2.shape == (4,)
    assert l3.shape == (4,)
    assert 0 < kernel.synaptic_ro.lambda_l1 <= 1.0
    assert 0 < kernel.synaptic_ro.lambda_l3 <= 1.0
    assert kernel.synaptic_ro.theta_l2 > 0
    assert 0 <= kernel.synaptic_ro.gating12 <= 1.0
    assert 0 <= kernel.synaptic_ro.gating23 <= 1.0
    assert kernel.synaptic_ro.to_dict()["dimension"] == 4

    pelm_stats = kernel.pelm_ro.get_state_stats()
    assert pelm_stats["capacity"] == 3
    assert kernel.pelm_ro.detect_corruption() is False
    assert kernel.rhythm_ro.get_state_label() in {"wake", "sleep"}


def test_moral_adapt_updates_state() -> None:
    kernel = _build_kernel()
    start_ema = kernel.moral_ro.ema_accept_rate

    kernel.moral_adapt(True)
    kernel.moral_adapt(False)

    updated_state = kernel.moral_ro.get_state()
    assert isinstance(kernel.moral_ro.ema_accept_rate, float)
    assert not math.isclose(updated_state["ema"], start_ema)


def test_memory_commit_updates_components() -> None:
    kernel = _build_kernel()
    vector = np.ones(4, dtype=np.float32)

    kernel.memory_commit(vector, phase=0.25)

    l1, l2, l3 = kernel.synaptic_ro.get_state()
    assert l1.shape == (4,)
    assert l2.shape == (4,)
    assert l3.shape == (4,)

    pelm_stats = kernel.pelm_ro.get_state_stats()
    assert pelm_stats["used"] >= 1
    assert isinstance(kernel.pelm_ro.size, int)


def test_rhythm_step_advances_counter() -> None:
    kernel = _build_kernel(wake_duration=3, sleep_duration=2)
    initial_counter = kernel.rhythm_ro.counter
    initial_label = kernel.rhythm_ro.get_state_label()

    kernel.rhythm_step()

    assert kernel.rhythm_ro.counter == initial_counter - 1
    assert kernel.rhythm_ro.get_state_label() == initial_label
    assert kernel.rhythm_ro.counter >= 0


def test_reset_reinitializes_components() -> None:
    kernel = _build_kernel()
    kernel.moral_adapt(True)
    kernel.memory_commit(np.ones(4, dtype=np.float32), phase=0.5)

    synaptic_config = SimpleNamespace(
        lambda_l1=0.4,
        lambda_l2=0.2,
        lambda_l3=0.1,
        theta_l1=1.0,
        theta_l2=2.0,
        gating12=0.3,
        gating23=0.2,
    )

    kernel.reset(
        dim=2,
        capacity=2,
        wake_duration=4,
        sleep_duration=2,
        initial_moral_threshold=0.75,
        synaptic_config=synaptic_config,
    )

    l1, l2, l3 = kernel.synaptic_ro.get_state()
    assert l1.shape == (2,)
    assert l2.shape == (2,)
    assert l3.shape == (2,)
    assert kernel.pelm_ro.size == 0
    assert kernel.rhythm_ro.counter == kernel.rhythm_ro.wake_duration
    assert math.isclose(kernel.moral_ro.threshold, 0.75, rel_tol=1e-6)
    assert math.isclose(kernel.synaptic_ro.lambda_l1, synaptic_config.lambda_l1)
