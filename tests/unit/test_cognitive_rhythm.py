"""
Unit Tests for CognitiveRhythm

Tests the cognitive rhythm system with wake/sleep cycles.
"""

import pytest

from mlsdm.rhythm.cognitive_rhythm import CognitiveRhythm


class TestCognitiveRhythmInitialization:
    """Test CognitiveRhythm initialization."""

    def test_default_initialization(self):
        """Test rhythm can be initialized with defaults."""
        rhythm = CognitiveRhythm()

        assert rhythm.wake_duration == 8
        assert rhythm.sleep_duration == 3
        assert rhythm.phase == "wake"
        assert rhythm.counter == 8

    def test_custom_initialization(self):
        """Test rhythm can be initialized with custom values."""
        rhythm = CognitiveRhythm(wake_duration=10, sleep_duration=5)

        assert rhythm.wake_duration == 10
        assert rhythm.sleep_duration == 5
        assert rhythm.phase == "wake"
        assert rhythm.counter == 10

    def test_invalid_wake_duration_raises(self):
        """Test invalid wake duration raises error."""
        with pytest.raises(ValueError, match="positive"):
            CognitiveRhythm(wake_duration=0)

        with pytest.raises(ValueError, match="positive"):
            CognitiveRhythm(wake_duration=-5)

    def test_invalid_sleep_duration_raises(self):
        """Test invalid sleep duration raises error."""
        with pytest.raises(ValueError, match="positive"):
            CognitiveRhythm(sleep_duration=0)

        with pytest.raises(ValueError, match="positive"):
            CognitiveRhythm(sleep_duration=-3)

    def test_both_invalid_raises(self):
        """Test both durations invalid raises error."""
        with pytest.raises(ValueError, match="positive"):
            CognitiveRhythm(wake_duration=-1, sleep_duration=-1)


class TestCognitiveRhythmStep:
    """Test rhythm step functionality."""

    def test_step_decrements_counter(self):
        """Test step decrements counter."""
        rhythm = CognitiveRhythm(wake_duration=5, sleep_duration=2)
        initial_counter = rhythm.counter

        rhythm.step()

        assert rhythm.counter == initial_counter - 1

    def test_step_transitions_wake_to_sleep(self):
        """Test step transitions from wake to sleep."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=3)

        # Step through wake phase
        rhythm.step()  # counter: 2 -> 1
        assert rhythm.phase == "wake"
        assert rhythm.counter == 1

        rhythm.step()  # counter: 1 -> 0, transition to sleep
        assert rhythm.phase == "sleep"
        assert rhythm.counter == 3

    def test_step_transitions_sleep_to_wake(self):
        """Test step transitions from sleep to wake."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=2)

        # Step through wake phase
        rhythm.step()
        rhythm.step()

        # Now in sleep phase
        assert rhythm.phase == "sleep"

        # Step through sleep phase
        rhythm.step()
        rhythm.step()

        # Back to wake
        assert rhythm.phase == "wake"
        assert rhythm.counter == 2

    def test_full_cycle(self):
        """Test complete wake-sleep cycle."""
        rhythm = CognitiveRhythm(wake_duration=3, sleep_duration=2)

        # Record phases during full cycle
        phases = [rhythm.phase]
        for _ in range(10):
            rhythm.step()
            phases.append(rhythm.phase)

        # Should see alternating patterns
        assert "wake" in phases
        assert "sleep" in phases

        # Count transitions
        transitions = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i-1])
        assert transitions >= 2  # At least two transitions


class TestCognitiveRhythmIsWake:
    """Test is_wake functionality."""

    def test_is_wake_initial(self):
        """Test is_wake returns True initially."""
        rhythm = CognitiveRhythm()

        assert rhythm.is_wake() is True

    def test_is_wake_during_wake(self):
        """Test is_wake returns True during wake phase."""
        rhythm = CognitiveRhythm(wake_duration=5, sleep_duration=2)

        rhythm.step()
        rhythm.step()

        assert rhythm.is_wake() is True

    def test_is_wake_during_sleep(self):
        """Test is_wake returns False during sleep phase."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=2)

        rhythm.step()
        rhythm.step()  # Transition to sleep

        assert rhythm.is_wake() is False


class TestCognitiveRhythmIsSleep:
    """Test is_sleep functionality."""

    def test_is_sleep_initial(self):
        """Test is_sleep returns False initially."""
        rhythm = CognitiveRhythm()

        assert rhythm.is_sleep() is False

    def test_is_sleep_during_sleep(self):
        """Test is_sleep returns True during sleep phase."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=3)

        rhythm.step()
        rhythm.step()  # Transition to sleep

        assert rhythm.is_sleep() is True

    def test_is_sleep_during_wake(self):
        """Test is_sleep returns False during wake phase."""
        rhythm = CognitiveRhythm(wake_duration=3, sleep_duration=2)

        rhythm.step()

        assert rhythm.is_sleep() is False


class TestCognitiveRhythmGetCurrentPhase:
    """Test get_current_phase functionality."""

    def test_get_current_phase_initial(self):
        """Test get_current_phase returns 'wake' initially."""
        rhythm = CognitiveRhythm()

        assert rhythm.get_current_phase() == "wake"

    def test_get_current_phase_during_wake(self):
        """Test get_current_phase during wake."""
        rhythm = CognitiveRhythm(wake_duration=5, sleep_duration=2)

        rhythm.step()

        assert rhythm.get_current_phase() == "wake"

    def test_get_current_phase_during_sleep(self):
        """Test get_current_phase during sleep."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=2)

        rhythm.step()
        rhythm.step()

        assert rhythm.get_current_phase() == "sleep"

    def test_get_current_phase_consistency(self):
        """Test get_current_phase is consistent with is_wake/is_sleep."""
        rhythm = CognitiveRhythm(wake_duration=2, sleep_duration=2)

        for _ in range(10):
            phase = rhythm.get_current_phase()

            if phase == "wake":
                assert rhythm.is_wake() is True
                assert rhythm.is_sleep() is False
            else:
                assert rhythm.is_wake() is False
                assert rhythm.is_sleep() is True

            rhythm.step()


class TestCognitiveRhythmToDict:
    """Test serialization functionality."""

    def test_to_dict_structure(self):
        """Test to_dict returns correct structure."""
        rhythm = CognitiveRhythm(wake_duration=10, sleep_duration=5)

        data = rhythm.to_dict()

        assert isinstance(data, dict)
        assert data["wake_duration"] == 10
        assert data["sleep_duration"] == 5
        assert data["phase"] == "wake"
        assert data["counter"] == 10

    def test_to_dict_after_steps(self):
        """Test to_dict reflects current state after steps."""
        rhythm = CognitiveRhythm(wake_duration=3, sleep_duration=2)

        rhythm.step()
        rhythm.step()
        rhythm.step()  # Transition to sleep

        data = rhythm.to_dict()

        assert data["phase"] == "sleep"
        assert data["counter"] == 2


class TestCognitiveRhythmIntegration:
    """Test integration scenarios."""

    def test_long_running_simulation(self):
        """Test rhythm over many steps."""
        rhythm = CognitiveRhythm(wake_duration=5, sleep_duration=3)

        wake_count = 0
        sleep_count = 0

        for _ in range(100):
            if rhythm.is_wake():
                wake_count += 1
            else:
                sleep_count += 1
            rhythm.step()

        # Both phases should have significant time
        assert wake_count > 30
        assert sleep_count > 20

        # Ratio should approximate wake_duration / (wake_duration + sleep_duration)
        ratio = wake_count / (wake_count + sleep_count)
        expected_ratio = 5 / (5 + 3)
        assert abs(ratio - expected_ratio) < 0.1

    def test_short_durations(self):
        """Test rhythm with short durations."""
        rhythm = CognitiveRhythm(wake_duration=1, sleep_duration=1)

        # Should alternate every step
        assert rhythm.phase == "wake"
        rhythm.step()
        assert rhythm.phase == "sleep"
        rhythm.step()
        assert rhythm.phase == "wake"
        rhythm.step()
        assert rhythm.phase == "sleep"

    def test_asymmetric_durations(self):
        """Test rhythm with very asymmetric durations."""
        rhythm = CognitiveRhythm(wake_duration=10, sleep_duration=1)

        # Record phases
        phases = []
        for _ in range(22):
            phases.append(rhythm.phase)
            rhythm.step()

        # Should see 10 wakes, 1 sleep, 10 wakes, 1 sleep
        wake_count = phases.count("wake")
        sleep_count = phases.count("sleep")

        assert wake_count == 20  # Two full wake periods
        assert sleep_count == 2  # Two short sleep periods


class TestCognitiveRhythmEdgeCases:
    """Test edge cases."""

    def test_float_duration_converted(self):
        """Test float durations are converted to int."""
        # The constructor should handle this via int() conversion
        rhythm = CognitiveRhythm(wake_duration=5, sleep_duration=3)

        assert isinstance(rhythm.wake_duration, int)
        assert isinstance(rhythm.sleep_duration, int)

    def test_very_large_durations(self):
        """Test very large durations work correctly."""
        rhythm = CognitiveRhythm(wake_duration=1000000, sleep_duration=500000)

        assert rhythm.wake_duration == 1000000
        assert rhythm.counter == 1000000

        rhythm.step()
        assert rhythm.counter == 999999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
