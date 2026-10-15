"""Pure state-machine logic for the Unified Face Recognition + PPE Detection system.

Two-Phase Architecture
----------------------
Phase 1 — EVALUATING_PPE  (eval_timeout seconds, default 8 s)
    Camera + ML runs continuously. Outcome decided at end of window:
    • All mandated PPE validated  → Phase 2a GRANTED
    • Any PPE still missing       → Phase 2b DISPENSING

Phase 2a — GRANTED  (granted_hold_seconds, default 5 s)
    Green LED ON via Arduino GRANT command. Returns to IDLE after hold.

Phase 2b — DISPENSING  (dispense_hold_seconds, default 10 s)
    Red LED ON + PPE servo doors opened via Arduino DENY / GLOVES / MASK.
    Arduino closes servos automatically via its own internal timer.
    Returns to IDLE after hold.

States
------
IDLE            Waiting for a recognised (AUTHORIZED) face.
EVALUATING_PPE  Person identified; evaluating PPE compliance (Phase 1).
GRANTED         All mandated PPE validated; green LED active (Phase 2a).
DISPENSING      PPE missing; red LED + servo dispensers active (Phase 2b).

Usage
-----
    sm = UnifiedStateMachine(mandated_set={'mask', 'gloves'}, ...)

    # called once per frame from the camera loop:
    sm.update(face_decision, face_name, ppe_detections)

    # poll for a pending log row (set on entry to GRANTED or DISPENSING):
    log = sm.consume_log()
    if log:
        db.insert_entry_log(**log)
"""
import enum
import time
from collections import deque


class State(enum.Enum):
    IDLE           = "IDLE"
    EVALUATING_PPE = "EVALUATING_PPE"
    GRANTED        = "GRANTED"
    DISPENSING     = "DISPENSING"


class UnifiedStateMachine:
    """Two-phase (evaluate → grant/dispense) entry-control state machine.

    Parameters
    ----------
    mandated_set:
        Set of PPE class names that must all be validated.
    history_len:
        Rolling window size in frames for temporal PPE validation.
    min_detect_count:
        Minimum frames a PPE item must appear in to be "validated".
    eval_timeout:
        Phase 1 duration — evaluation window in seconds (default 8 s).
    granted_hold_seconds:
        Phase 2a duration — green LED hold time in seconds (default 5 s).
    dispense_hold_seconds:
        Phase 2b duration — red LED + open-servo hold time in seconds (default 10 s).
    """

    def __init__(
        self,
        mandated_set: set,
        history_len: int             = 100,
        min_detect_count: int        = 5,
        eval_timeout: float          = 8.0,
        granted_hold_seconds: float  = 5.0,
        dispense_hold_seconds: float = 10.0,
        early_grant_allowed: bool    = False,
    ):
        self.mandated_set          = frozenset(mandated_set)
        self.history_len           = history_len
        self.min_detect_count      = min_detect_count
        self.eval_timeout          = eval_timeout
        self.granted_hold_seconds  = granted_hold_seconds
        self.dispense_hold_seconds = dispense_hold_seconds
        self.early_grant_allowed   = early_grant_allowed

        # --- runtime state ---
        self.state: State                = State.IDLE
        self.recognized_person: str | None   = None
        self.recognized_staff_id: int | None = None
        self.recognized_similarity: float    = 0.0

        self._history: deque[set]    = deque(maxlen=history_len)
        self._eval_start_time: float     = 0.0
        self._granted_start_time: float  = 0.0
        self._dispense_start_time: float = 0.0

        self.validated_set:    set = set()
        self.last_missing_ppe: set = set()

        self.pending_log: dict | None = None   # written once on GRANTED/DISPENSING entry

        # ---- public HUD fields (read by the renderer each frame) ----
        self.hud_eval_remaining:     float = 0.0
        self.hud_granted_remaining:  float = 0.0
        self.hud_dispense_remaining: float = 0.0
        self.hud_missing: set              = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        face_decision: str,          # "AUTHORIZED" | "DENIED" | "UNKNOWN"
        face_name: str | None,
        ppe_detections: set,         # PPE class names detected in this frame
        face_staff_id: int | None = None,
        face_similarity: float    = 0.0,
        now: float | None         = None,
    ) -> None:
        """Advance the state machine by one frame."""
        t = now if now is not None else time.time()

        if self.state == State.IDLE:
            self._handle_idle(face_decision, face_name, face_staff_id, face_similarity, t)

        elif self.state == State.EVALUATING_PPE:
            self._handle_evaluating(ppe_detections, t)

        elif self.state == State.GRANTED:
            self._handle_granted(t)

        elif self.state == State.DISPENSING:
            self._handle_dispensing(t)

    def consume_log(self) -> dict | None:
        """Return and clear the pending log entry (if any)."""
        log = self.pending_log
        self.pending_log = None
        return log

    # ------------------------------------------------------------------
    # Private state handlers
    # ------------------------------------------------------------------

    def _handle_idle(self, face_decision, face_name, face_staff_id, face_similarity, t):
        if face_decision == "AUTHORIZED":
            self.recognized_person     = face_name
            self.recognized_staff_id   = face_staff_id
            self.recognized_similarity = face_similarity
            self._history.clear()
            self.validated_set    = set()
            self._eval_start_time = t
            self.state = State.EVALUATING_PPE

    def _handle_evaluating(self, ppe_detections, t):
        # Accumulate detections into rolling history
        self._history.append(set(ppe_detections))

        # Temporal validation: count per-item occurrences
        counts: dict[str, int] = {}
        for frame_dets in self._history:
            for item in frame_dets:
                counts[item] = counts.get(item, 0) + 1

        self.validated_set = {
            cls for cls, cnt in counts.items()
            if cnt >= self.min_detect_count
        }

        missing = self.mandated_set - self.validated_set
        elapsed = t - self._eval_start_time
        self.hud_eval_remaining = max(0.0, self.eval_timeout - elapsed)
        self.hud_missing        = missing

        # Check if evaluation phase completed or early grant is enabled
        eval_finished = (elapsed >= self.eval_timeout)
        can_grant_early = (self.early_grant_allowed and not missing)

        if can_grant_early or (eval_finished and not missing):
            # ── Phase 2a: GRANTED (5 seconds) ──────────────────────────
            self.last_missing_ppe = set()
            self.pending_log = {
                'staff_id':      self.recognized_staff_id,
                'name':          self.recognized_person,
                'similarity':    self.recognized_similarity,
                'decision':      'AUTHORIZED',
                'ppe_status':    'COMPLIANT',
                'ppe_missing':   '',
                'validated_ppe': sorted(list(self.validated_set)),
            }
            self._granted_start_time   = t
            self.hud_granted_remaining = self.granted_hold_seconds
            self.state = State.GRANTED

        elif eval_finished and missing:
            # ── Phase 2b: DISPENSING (10 seconds) ──────────────────────
            self.last_missing_ppe = missing
            self.pending_log = {
                'staff_id':      self.recognized_staff_id,
                'name':          self.recognized_person,
                'similarity':    self.recognized_similarity,
                'decision':      'AUTHORIZED',
                'ppe_status':    'NON_COMPLIANT',
                'ppe_missing':   ','.join(sorted(missing)),
                'validated_ppe': sorted(list(self.validated_set)),
            }
            self._dispense_start_time   = t
            self.hud_dispense_remaining = self.dispense_hold_seconds
            self.state = State.DISPENSING

    def _handle_granted(self, t):
        self.hud_granted_remaining = max(
            0.0, self.granted_hold_seconds - (t - self._granted_start_time)
        )
        if t - self._granted_start_time >= self.granted_hold_seconds:
            self._reset()

    def _handle_dispensing(self, t):
        self.hud_dispense_remaining = max(
            0.0, self.dispense_hold_seconds - (t - self._dispense_start_time)
        )
        if t - self._dispense_start_time >= self.dispense_hold_seconds:
            self._reset()

    def _reset(self):
        self.state                  = State.IDLE
        self._history.clear()
        self.validated_set          = set()
        self.recognized_person      = None
        self.recognized_staff_id    = None
        self.recognized_similarity  = 0.0
        self.hud_eval_remaining     = 0.0
        self.hud_granted_remaining  = 0.0
        self.hud_dispense_remaining = 0.0
        self.hud_missing            = set()
