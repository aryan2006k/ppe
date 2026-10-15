"""Tests for the combined state machine logic in unified_entry_monitor.py — RED phase.

We test ONLY the pure state-machine logic (no OpenCV, no YOLO, no camera required).
The state machine is extracted as a class so it can be unit-tested in isolation.

Run with: .\\venv\\Scripts\\python.exe -m pytest tests/test_state_machine.py -v
"""
import os
import sys
import time

import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from unified_state_machine import UnifiedStateMachine, State


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sm():
    """Return a state machine with a small mandated set for fast tests."""
    return UnifiedStateMachine(
        mandated_set={'mask'},
        history_len=10,
        min_detect_count=3,
        eval_timeout=2.0,       # short timeout for tests
        granted_hold_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_in_idle(self, sm):
        assert sm.state == State.IDLE

    def test_no_recognized_person_on_init(self, sm):
        assert sm.recognized_person is None

    def test_validated_set_empty_on_init(self, sm):
        assert len(sm.validated_set) == 0


# ---------------------------------------------------------------------------
# IDLE → EVALUATING_PPE transition
# ---------------------------------------------------------------------------

class TestIdleToEvaluatingPPE:
    def test_unknown_face_does_not_trigger_ppe_eval(self, sm):
        """An UNKNOWN match result should keep the state machine in IDLE."""
        sm.update(face_decision="UNKNOWN", face_name=None, ppe_detections=set())
        assert sm.state == State.IDLE

    def test_authorized_face_triggers_ppe_eval(self, sm):
        """An AUTHORIZED face match should start PPE evaluation."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert sm.state == State.EVALUATING_PPE

    def test_denied_face_does_not_trigger_ppe_eval(self, sm):
        """A DENIED face (registered but deauthorized) should stay in IDLE."""
        sm.update(face_decision="DENIED", face_name="Bob", ppe_detections=set())
        assert sm.state == State.IDLE

    def test_recognized_person_stored_on_trigger(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert sm.recognized_person == "Alice"


# ---------------------------------------------------------------------------
# Temporal PPE validation
# ---------------------------------------------------------------------------

class TestTemporalPPEValidation:
    def test_ppe_not_validated_below_min_count(self, sm):
        """PPE appearing in fewer frames than min_detect_count must not be validated."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        # Add mask only 2 times (min_detect_count = 3)
        for _ in range(2):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert 'mask' not in sm.validated_set

    def test_ppe_validated_at_min_count(self, sm):
        """PPE appearing exactly min_detect_count times must be validated."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert 'mask' in sm.validated_set

    def test_validated_set_accumulates(self, sm):
        """Validated set should accumulate items across frames."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        # Validated set must at least contain mask
        assert sm.validated_set >= {'mask'}


# ---------------------------------------------------------------------------
# EVALUATING_PPE → GRANTED transition
# ---------------------------------------------------------------------------

class TestEvaluatingToGranted:
    def test_full_compliance_triggers_granted_after_eval_window(self, sm):
        """Once all mandated PPE is validated and evaluation window finishes, state becomes GRANTED."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        # Fast-forward past eval_timeout (2.0s)
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.GRANTED

    def test_partial_ppe_stays_in_evaluating(self, sm):
        """Only mask mandated; mask is validated → should grant. Use sm with extra items."""
        sm2 = UnifiedStateMachine(
            mandated_set={'mask', 'gloves'},
            history_len=10, min_detect_count=3,
            eval_timeout=2.0, granted_hold_seconds=1.0,
        )
        sm2.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm2.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        # Only mask validated, gloves missing → still evaluating during window
        assert sm2.state == State.EVALUATING_PPE


# ---------------------------------------------------------------------------
# EVALUATING_PPE → DISPENSING (timeout without PPE)
# ---------------------------------------------------------------------------

class TestEvaluatingTimeout:
    def test_timeout_without_ppe_transitions_to_dispensing(self, sm):
        """Exceeding eval_timeout without full compliance should trigger DISPENSING phase."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        # Simulate time passing past eval_timeout (2.0s)
        sm._eval_start_time -= 3.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert sm.state == State.DISPENSING

    def test_dispensing_returns_to_idle_after_hold(self, sm):
        """After dispense_hold_seconds, DISPENSING phase resets to IDLE."""
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        sm._eval_start_time -= 3.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert sm.state == State.DISPENSING
        # Fast-forward dispensing hold timer
        sm._dispense_start_time -= (sm.dispense_hold_seconds + 1.0)
        sm.update(face_decision="UNKNOWN", face_name=None, ppe_detections=set())
        assert sm.state == State.IDLE
        assert len(sm._history) == 0

    def test_violation_records_missing_ppe(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        sm._eval_start_time -= 3.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert 'mask' in sm.last_missing_ppe


# ---------------------------------------------------------------------------
# GRANTED → IDLE (hold timer)
# ---------------------------------------------------------------------------

class TestGrantedToIdle:
    def test_granted_stays_granted_during_hold_period(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.GRANTED
        # Immediately update — should still be GRANTED (hold = 1.0s)
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.GRANTED

    def test_granted_returns_to_idle_after_hold(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.GRANTED
        # Fast-forward hold timer
        sm._granted_start_time -= 2.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.IDLE

    def test_granted_resets_recognized_person(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.state == State.GRANTED
        sm._granted_start_time -= 2.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.recognized_person is None


# ---------------------------------------------------------------------------
# State machine produces log entries
# ---------------------------------------------------------------------------

class TestLogEntries:
    def test_granted_produces_log_entry(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        assert sm.pending_log is not None
        assert sm.pending_log['ppe_status'] == 'COMPLIANT'
        assert sm.pending_log['name'] == 'Alice'
        assert sm.pending_log['decision'] == 'AUTHORIZED'

    def test_violation_produces_log_entry(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        sm._eval_start_time -= 3.0
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        assert sm.pending_log is not None
        assert sm.pending_log['ppe_status'] == 'NON_COMPLIANT'
        assert 'mask' in sm.pending_log.get('ppe_missing', '')

    def test_log_entry_consumed_after_read(self, sm):
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections=set())
        for _ in range(3):
            sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        sm._eval_start_time -= 2.5
        sm.update(face_decision="AUTHORIZED", face_name="Alice", ppe_detections={'mask'})
        _ = sm.consume_log()
        assert sm.pending_log is None

    def test_consume_log_returns_none_when_nothing_pending(self, sm):
        assert sm.consume_log() is None

