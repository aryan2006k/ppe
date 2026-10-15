"""Unified Entry Monitor — Face Recognition + Medical PPE Detection.

Single-camera loop that:
  1. Identifies the person via face recognition (background thread, same pattern
     as facerecg-main/app/entry_monitor.py).
  2. Once a person is AUTHORIZED, enters a PPE compliance evaluation window
     (YOLO Medical PPE model, temporal validation, same logic as
     step4_compliance_arbiter_medical_ppe.py).
  3. Draws face boxes + PPE boxes + HUD overlay on the same frame.
  4. Logs each outcome (identity + PPE status) to the shared SQLite database.

Usage
-----
    python unified_entry_monitor.py [--camera-index 0] [--threshold 0.62]

State Machine
-------------
    IDLE → (AUTHORIZED face) → EVALUATING_PPE
                                  │
                  (all PPE validated) → GRANTED → (5s hold) → IDLE
                  (timeout)           → VIOLATION            → IDLE
"""
import argparse
import os
import sys
import threading
import time

import cv2
try:
    import serial as _serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Arduino Serial Actuator
# ---------------------------------------------------------------------------

_arduino = None


def send_arduino(command: str) -> bool:
    """Send one newline-terminated command to the Arduino, if connected."""
    global _arduino
    if not _SERIAL_AVAILABLE or _arduino is None or not _arduino.is_open:
        print(f"[WARN] Arduino not connected; skipped command: {command}")
        return False
    try:
        _arduino.write((command + "\n").encode("utf-8"))
        _arduino.flush()
        print(f"[ARDUINO] Sent: {command}")
        return True
    except (_serial.SerialException, OSError) as exc:
        print(f"[WARN] Could not send '{command}' to Arduino: {exc}")
        return False


def connect_arduino(port: str | None = None, baud: int | None = None) -> None:
    """Try to open a serial connection to the Arduino.
    AI monitoring continues even if the Arduino is unavailable.
    """
    global _arduino
    if not _SERIAL_AVAILABLE:
        print("[WARN] pyserial not installed — Arduino control disabled.")
        return
    _port = port or cfg.ARDUINO_PORT
    _baud = baud or cfg.ARDUINO_BAUD
    try:
        _arduino = _serial.Serial(_port, _baud, timeout=1)
        time.sleep(2)  # Uno resets on serial open; wait for READY
        print(f"[INFO] Arduino connected on {_port} at {_baud} baud")
    except (_serial.SerialException, OSError) as exc:
        _arduino = None
        print(f"[WARN] Arduino unavailable on {_port}: {exc}")
        print("[WARN] Continuing without actuator control.")

# ---------------------------------------------------------------------------
# Path setup — import face recognition modules from facerecg-main/app
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACERECG_APP = os.path.join(PROJECT_ROOT, "facerecg-main", "app")
sys.path.insert(0, FACERECG_APP)
sys.path.insert(0, PROJECT_ROOT)

import face_engine
import db

import unified_config as cfg
from unified_state_machine import UnifiedStateMachine, State
from ppe_filter import filter_and_validate_ppe
from face_tracker import FaceTracker

# ---------------------------------------------------------------------------
# YOLO — loaded lazily to avoid import-time cost when camera isn't available
# ---------------------------------------------------------------------------
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(cfg.PPE_MODEL_PATH)
    return _yolo_model


# ---------------------------------------------------------------------------
# Face Detection Worker (background thread)
# ---------------------------------------------------------------------------

class _FaceWorker:
    """Runs detect → embed → match on the most recent frame in a background thread.

    Identical threading pattern to facerecg-main/app/entry_monitor.py's
    _DetectionWorker, so the video preview never freezes on CPU-bound inference.
    Uses FaceTracker to eliminate bounding box flickering and decision flip-flopping.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.tracker = FaceTracker(
            threshold=threshold,
            smoothing_alpha=getattr(cfg, "FACE_SMOOTHING_ALPHA", 0.7),
            max_lost_frames=getattr(cfg, "FACE_TRACK_MAX_LOST_FRAMES", 4),
            hysteresis_margin=getattr(cfg, "FACE_HYSTERESIS_MARGIN", 0.06),
            hysteresis_frames=getattr(cfg, "FACE_HYSTERESIS_FRAMES", 5),
        )
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._results_lock = threading.Lock()
        self.last_faces: list = []   # list of (x1, y1, x2, y2, MatchResult)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def submit_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id += 1

    def get_results(self) -> list:
        with self._results_lock:
            return list(self.last_faces)

    def _loop(self):
        processed_id = 0
        while not self._stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
            if frame is None or frame_id == processed_id:
                time.sleep(0.01)
                continue
            processed_id = frame_id

            small = cv2.resize(frame, None,
                               fx=cfg.DETECTION_DOWNSCALE, fy=cfg.DETECTION_DOWNSCALE)
            detections = face_engine.detect_faces(small)
            scale = 1.0 / cfg.DETECTION_DOWNSCALE

            results = []
            for det in detections:
                x1, y1, x2, y2 = [int(v * scale) for v in det["box"]]
                x1, y1 = max(x1, 0), max(y1, 0)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emb = face_engine.get_embedding(crop)
                m = face_engine.match(emb, threshold=self.threshold)
                results.append((x1, y1, x2, y2, m))

            # Apply temporal tracking, EMA coordinate smoothing, and hysteresis
            stabilized_results = self.tracker.update(results)

            with self._results_lock:
                self.last_faces = stabilized_results



# ---------------------------------------------------------------------------
# PPE Detection (main thread)
# ---------------------------------------------------------------------------

def _detect_ppe(frame) -> tuple[set, list]:
    """Run YOLO PPE detection on a frame with filtering and validation constraints.

    Returns
    -------
    detected_names : set of PPE class names validated in this frame
                     (gloves requires 2 non-overlapping boxes)
    filtered_boxes : list of (cls_id, conf, x1, y1, x2, y2) for rendering
                     (max 2 for gloves, max 1 for mask, coverall, goggles)
    """
    model = _get_yolo()
    results = model.predict(frame, verbose=False, conf=cfg.DEFAULT_CONF)
    raw_detections: list = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_detections.append((cls_id, conf, x1, y1, x2, y2))

    return filter_and_validate_ppe(
        raw_detections,
        class_names=cfg.CLASS_NAMES,
        class_conf=cfg.CLASS_CONF,
        default_conf=cfg.DEFAULT_CONF,
        max_boxes_per_class=cfg.MAX_BOXES_PER_CLASS,
        glove_iou_threshold=cfg.GLOVE_OVERLAP_IOU_THRESHOLD,
    )



# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

FACE_COLORS = {
    "AUTHORIZED": (0, 200, 0),    # green
    "DENIED":     (0, 0, 255),    # red
    "UNKNOWN":    (0, 165, 255),  # orange
}


def _draw_face_boxes(frame, face_results):
    for x1, y1, x2, y2, m in face_results:
        color = FACE_COLORS.get(m.decision, (200, 200, 200))
        sim_pct = min(max(m.similarity, 0.0), 1.0)

        # Compact label: name + sim only
        name_str = m.name or 'UNKNOWN'
        label = f"{name_str}  {m.similarity:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Slim confidence bar below box
        bar_h = 5
        bar_w = x2 - x1
        cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 2 + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + int(bar_w * sim_pct), y2 + 2 + bar_h), color, -1)
        tick_x = x1 + int(bar_w * cfg.MATCH_THRESHOLD)
        cv2.line(frame, (tick_x, y2), (tick_x, y2 + 2 + bar_h + 2), (255, 255, 255), 1)


def _draw_ppe_boxes(frame, ppe_boxes):
    for cls_id, conf, x1, y1, x2, y2 in ppe_boxes:
        color = cfg.PPE_BOX_COLORS.get(cls_id, (255, 255, 255))
        label = f"{cfg.CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# Panel layout constants
_PANEL_W     = 300    # width of the left info panel (pixels, on display frame)
_PANEL_ALPHA = 0.60   # opacity of the dark background
_ROW_H       = 40     # vertical spacing between labelled rows
_PAD_X       = 14     # left padding inside panel


def _panel_label(frame, text, row):
    y = 58 + row * _ROW_H
    cv2.putText(frame, text, (_PAD_X, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)


def _panel_value(frame, text, row, color, scale=0.64):
    y = 58 + row * _ROW_H + 18
    cv2.putText(frame, text, (_PAD_X, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_hud(frame, sm: UnifiedStateMachine):
    h, w = frame.shape[:2]
    pw = min(_PANEL_W, w)

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0, frame)
    cv2.line(frame, (pw, 0), (pw, h), (55, 55, 55), 1)

    # State chip
    if sm.state == State.IDLE:
        state_label, state_color = "IDLE", (140, 140, 140)
    elif sm.state == State.EVALUATING_PPE:
        state_label = f"SCANNING  {sm.hud_eval_remaining:.0f}s"
        state_color = (0, 160, 255)
    elif sm.state == State.GRANTED:
        state_label = f"GRANTED  {sm.hud_granted_remaining:.0f}s"
        state_color = (0, 210, 0)
    elif sm.state == State.DISPENSING:
        state_label = f"PPE MISSING  {sm.hud_dispense_remaining:.0f}s"
        state_color = (50, 50, 240)
    else:
        state_label, state_color = sm.state.value, (200, 200, 200)

    chip_y1, chip_y2 = 8, 42
    cv2.rectangle(frame, (_PAD_X - 4, chip_y1), (pw - _PAD_X, chip_y2), state_color, -1)
    lum = sum(state_color) / 3
    txt_col = (10, 10, 10) if lum > 130 else (240, 240, 240)
    cv2.putText(frame, state_label, (_PAD_X, chip_y2 - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_col, 2, cv2.LINE_AA)

    # Row 0: Person
    _panel_label(frame, "PERSON", 0)
    person_str = sm.recognized_person.upper() if sm.recognized_person else "---"
    person_col = (0, 210, 0) if sm.recognized_person else (90, 90, 90)
    _panel_value(frame, person_str, 0, person_col)

    # Row 1: Face confidence + bar
    _panel_label(frame, "FACE CONFIDENCE", 1)
    sim = sm.recognized_similarity
    sim_col = (0, 210, 0) if sim >= cfg.MATCH_THRESHOLD else (50, 80, 240)
    _panel_value(frame, f"{sim:.2f}   (min {cfg.MATCH_THRESHOLD:.2f})", 1, sim_col, scale=0.58)
    bar_y = 58 + 1 * _ROW_H + 20
    bx0, bx1 = _PAD_X, pw - _PAD_X
    bw = bx1 - bx0
    cv2.rectangle(frame, (bx0, bar_y), (bx1, bar_y + 6), (45, 45, 45), -1)
    cv2.rectangle(frame, (bx0, bar_y), (bx0 + int(bw * min(sim, 1.0)), bar_y + 6), sim_col, -1)
    tick = bx0 + int(bw * cfg.MATCH_THRESHOLD)
    cv2.line(frame, (tick, bar_y - 2), (tick, bar_y + 8), (210, 210, 210), 2)

    # Row 2: PPE status
    _panel_label(frame, "PPE STATUS", 2)
    if sm.state == State.EVALUATING_PPE:
        if sm.hud_missing:
            ppe_str = "Need: " + ', '.join(sorted(sm.hud_missing))
            ppe_col = (50, 80, 240)
        else:
            ppe_str = "All PPE OK \u2714"
            ppe_col = (0, 210, 0)
    elif sm.state == State.DISPENSING:
        ppe_str = "Collect: " + (', '.join(sorted(sm.last_missing_ppe)) if sm.last_missing_ppe else '')
        ppe_col = (50, 80, 240)
    elif sm.state == State.GRANTED:
        ppe_str = "Compliant \u2714"
        ppe_col = (0, 210, 0)
    else:
        ppe_str, ppe_col = "Waiting...", (90, 90, 90)
    _panel_value(frame, ppe_str, 2, ppe_col, scale=0.58)

    if sm.validated_set:
        val_y = 58 + 2 * _ROW_H + 34
        cv2.putText(frame, "OK: " + ', '.join(sorted(sm.validated_set)),
                    (_PAD_X, val_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 170, 110), 1, cv2.LINE_AA)

    # Footer
    cv2.putText(frame, "PPE CONTROL SYSTEM", (_PAD_X, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _ensure_snapshot_dirs():
    for d in ("AUTHORIZED", "DENIED", "UNKNOWN"):
        os.makedirs(os.path.join(cfg.ENTRY_SNAPSHOTS_DIR, d), exist_ok=True)


def _save_snapshot(
    frame,
    decision: str,
    key,
    face_box: tuple[int, int, int, int] | None = None,
    face_crop=None,
) -> str | None:
    crop_dir = os.path.join(cfg.ENTRY_SNAPSHOTS_DIR, decision)
    os.makedirs(crop_dir, exist_ok=True)
    path = os.path.join(crop_dir, f"{int(time.time())}_{key}.jpg")

    img_to_save = None
    if face_crop is not None and getattr(face_crop, "size", 0) > 0:
        img_to_save = face_crop
    elif face_box is not None and frame is not None:
        h, w = frame.shape[:2]
        x1 = max(0, int(face_box[0]))
        y1 = max(0, int(face_box[1]))
        x2 = min(w, int(face_box[2]))
        y2 = min(h, int(face_box[3]))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                img_to_save = crop

    if img_to_save is None:
        img_to_save = frame

    if img_to_save is not None and getattr(img_to_save, "size", 0) > 0:
        cv2.imwrite(path, img_to_save)
        return path
    return None



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(camera_index: int, threshold: float, cooldown: float) -> int:
    # Synchronize tunables with face_engine's config module
    import config as face_cfg
    face_cfg.DETECTOR_SCORE_THRESHOLD = cfg.DETECTOR_SCORE_THRESHOLD
    face_cfg.YUNET_NMS_THRESHOLD = cfg.YUNET_NMS_THRESHOLD
    face_cfg.MATCH_THRESHOLD = threshold
    face_cfg.DETECTION_DOWNSCALE = cfg.DETECTION_DOWNSCALE

    # Initialise DB + face models
    db.init_db()
    face_engine.load_all_embeddings()
    _ensure_snapshot_dirs()


    if not os.path.exists(cfg.PPE_MODEL_PATH):
        print(f"[ERROR] PPE model not found: {cfg.PPE_MODEL_PATH}")
        print("        Did step3_train_medical_ppe.py finish successfully?")
        return 1

    # Warm up face models first (loads torchvision fully on the main thread).
    # This MUST happen before any background thread touches facenet_pytorch / torchvision,
    # otherwise a race condition causes a circular-import InterpolationMode error.
    print("[INFO] Loading face recognition models…")
    face_engine.warm_up()

    # Pre-load YOLO (torchvision is already fully initialized above — safe for threads now).
    print("[INFO] Loading YOLO PPE model…")
    import torchvision  # noqa — belt-and-suspenders guard
    _get_yolo()

    # Connect Arduino after models are ready (non-fatal if unavailable).
    connect_arduino()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}", file=sys.stderr)
        return 1

    # Create a resizable named window and set it to the configured display size
    _WIN_TITLE = "PPE Entry Monitor  —  press Q to quit"
    dw = getattr(cfg, 'DISPLAY_WIDTH',  1280)
    dh = getattr(cfg, 'DISPLAY_HEIGHT', 720)
    cv2.namedWindow(_WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_TITLE, dw, dh)

    sm = UnifiedStateMachine(
        mandated_set=cfg.MANDATED_SET,
        history_len=cfg.HISTORY_LEN,
        min_detect_count=cfg.MIN_DETECT_COUNT,
        eval_timeout=cfg.EVAL_TIMEOUT,
        granted_hold_seconds=cfg.GRANTED_HOLD_SECONDS,
        dispense_hold_seconds=cfg.DISPENSE_HOLD_SECONDS,
        early_grant_allowed=getattr(cfg, "EARLY_GRANT_ALLOWED", False),
    )

    face_worker = _FaceWorker(threshold)
    face_worker.start()

    last_logged: dict = {}   # key → last log timestamp (debounce)
    last_face_crop = None
    prev_state = State.IDLE

    print("[INFO] Unified Entry Monitor running. Press 'q' or ESC to quit.")
    print(f"[INFO] Mandated PPE: {cfg.MANDATED_SET}")
    print(f"[INFO] Architecture: Mode 1 Eval = {cfg.EVAL_TIMEOUT}s | Mode 2 Grant = {cfg.GRANTED_HOLD_SECONDS}s | Dispense = {cfg.DISPENSE_HOLD_SECONDS}s")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # --- face recognition (result from last background pass) ---
            face_worker.submit_frame(frame)
            face_results = face_worker.get_results()

            # Pick the best (highest similarity) recognized face for state machine
            best_face_decision = "UNKNOWN"
            best_face_name = None
            best_face_staff_id = None
            best_face_sim = 0.0
            best_face_box = None

            for x1, y1, x2, y2, m in face_results:
                if m.decision == "AUTHORIZED" and m.similarity > best_face_sim:
                    best_face_decision = m.decision
                    best_face_name = m.name
                    best_face_staff_id = m.staff_id
                    best_face_sim = m.similarity
                    best_face_box = (x1, y1, x2, y2)

            # Maintain latest face crop for logging snapshot (only face bounding box)
            if best_face_box is not None:
                bx1, by1, bx2, by2 = best_face_box
                h, w = frame.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    last_face_crop = frame[by1:by2, bx1:bx2].copy()
            elif face_results and sm.state != State.IDLE:
                fx1, fy1, fx2, fy2 = face_results[0][:4]
                h, w = frame.shape[:2]
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(w, fx2), min(h, fy2)
                if fx2 > fx1 and fy2 > fy1:
                    last_face_crop = frame[fy1:fy2, fx1:fx2].copy()

            # --- PPE detection (main thread) ---
            ppe_names, ppe_boxes = _detect_ppe(frame)

            # --- update state machine ---
            sm.update(
                face_decision=best_face_decision,
                face_name=best_face_name,
                ppe_detections=ppe_names,
                face_staff_id=best_face_staff_id,
                face_similarity=best_face_sim,
            )

            # Check if state returned to IDLE from GRANTED or DISPENSING
            if prev_state in (State.GRANTED, State.DISPENSING) and sm.state == State.IDLE:
                print(f"[INFO] {prev_state.value} phase complete -> returning to IDLE.")
                send_arduino("RESET")
            prev_state = sm.state

            # --- handle pending log entry (fired upon transition to GRANTED or DISPENSING) ---
            log = sm.consume_log()
            if log:
                now = time.time()
                key = log.get('staff_id') or "UNKNOWN"
                if now - last_logged.get(key, 0) >= cooldown:
                    last_logged[key] = now
                    snapshot_path = _save_snapshot(
                        frame,
                        log['decision'],
                        key,
                        face_box=best_face_box,
                        face_crop=last_face_crop,
                    )
                    db.insert_entry_log(
                        staff_id=log['staff_id'],
                        matched_name=log['name'],
                        similarity=log['similarity'],
                        decision=log['decision'],
                        snapshot_path=snapshot_path,
                        ppe_status=log['ppe_status'],
                        ppe_missing=log.get('ppe_missing') or None,
                    )
                    val_str = ', '.join(log.get('validated_ppe') or []) or 'None'
                    if log['ppe_status'] == 'COMPLIANT':
                        print("\n" + "=" * 60)
                        print(f"[MODE 1: EVALUATION COMPLETE ({cfg.EVAL_TIMEOUT}s)]")
                        print(f"  Person:        {log['name']} (AUTHORIZED, similarity: {log['similarity']:.2f})")
                        print(f"  Validated PPE: {val_str}")
                        print(f"  Missing PPE:   None")
                        print(f"  Outcome:       ACCESS GRANTED")
                        print("-" * 60)
                        print(f"[MODE 2: ACCESS GRANTED ({cfg.GRANTED_HOLD_SECONDS}s)]")
                        print(f"  Green LED ON for {cfg.GRANTED_HOLD_SECONDS}s")
                        print("=" * 60 + "\n")
                        send_arduino("GRANT")   # Green LED ON for 5s
                    else:
                        missing = log.get('ppe_missing', '')
                        print("\n" + "=" * 60)
                        print(f"[MODE 1: EVALUATION COMPLETE ({cfg.EVAL_TIMEOUT}s)]")
                        print(f"  Person:        {log['name']} (AUTHORIZED, similarity: {log['similarity']:.2f})")
                        print(f"  Validated PPE: {val_str}")
                        print(f"  Missing PPE:   {missing}")
                        print(f"  Outcome:       NOT VALIDATED -> DISPENSING REQUIRED")
                        print("-" * 60)
                        print(f"[MODE 2: DISPENSING PHASE ({cfg.DISPENSE_HOLD_SECONDS}s)]")
                        print(f"  Red LED ON for {cfg.DISPENSE_HOLD_SECONDS}s")
                        send_arduino("DENY")    # Red LED ON for 10s
                        for item in missing.split(','):
                            item = item.strip().lower()
                            if item:
                                cmd = cfg.ARDUINO_DISPENSE_COMMANDS.get(item)
                                if cmd:
                                    print(f"  >> Opening PPE door: {cmd} (open for {cfg.DISPENSE_HOLD_SECONDS}s)")
                                    send_arduino(cmd)
                        print("=" * 60 + "\n")

            if sm.state == State.IDLE:
                last_face_crop = None

            # --- rendering on upscaled display frame ---
            dw = getattr(cfg, 'DISPLAY_WIDTH',  1280)
            dh = getattr(cfg, 'DISPLAY_HEIGHT', 720)
            display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
            _draw_face_boxes(display, [
                (int(x1 * dw / frame.shape[1]), int(y1 * dh / frame.shape[0]),
                 int(x2 * dw / frame.shape[1]), int(y2 * dh / frame.shape[0]), m)
                for x1, y1, x2, y2, m in face_results
            ])
            _draw_ppe_boxes(display, [
                (cid, cf,
                 int(bx1 * dw / frame.shape[1]), int(by1 * dh / frame.shape[0]),
                 int(bx2 * dw / frame.shape[1]), int(by2 * dh / frame.shape[0]))
                for cid, cf, bx1, by1, bx2, by2 in ppe_boxes
            ])
            _draw_hud(display, sm)

            cv2.imshow(_WIN_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        face_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        if _arduino is not None and _arduino.is_open:
            _arduino.close()
            print("[INFO] Arduino serial connection closed.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Face Recognition + PPE Detection")
    parser.add_argument("--camera-index", type=int, default=cfg.CAMERA_INDEX,
                        help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=cfg.MATCH_THRESHOLD,
                        help="Face match cosine similarity threshold (default: 0.62)")
    parser.add_argument("--cooldown", type=float, default=cfg.LOG_COOLDOWN_SECONDS,
                        help="Seconds between repeated log entries for the same person")
    parser.add_argument("--port", type=str, default=None,
                        help=f"Arduino COM port override (default: {cfg.ARDUINO_PORT} from config)")
    args = parser.parse_args()
    if args.port:
        cfg.ARDUINO_PORT = args.port
    sys.exit(run(args.camera_index, args.threshold, args.cooldown))
