from __future__ import annotations

import asyncio
import time

import pytest


def test_fast_time_advances_for_sleep(fast_time) -> None:
    start = time.time()
    time.sleep(1.5)
    assert time.time() >= start + 1.5


@pytest.mark.asyncio
async def test_fast_time_advances_for_async_sleep(fast_time) -> None:
    start = time.time()
    await asyncio.sleep(2.0)
    assert time.time() >= start + 2.0
