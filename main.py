import argparse
import random
import threading
import time
from pathlib import Path
from typing import Optional, Union
import subprocess
import logging
import cv2

from logger import logger
from config import (
    LOCK_DURATION_SECONDS,
    CLEAN_CONFIRMATIONS,
    OVERRIDE_DEFAULT_FORCE_OPEN,
    CAMERA_SOURCE,
    CAP_PROP_BUFFERSIZE,
    VISION_EVERY_N_FRAMES,
    EVENT_END_MISSES,
    MODELS_DIR,
    TFOD_FROZEN_GRAPH,
    SHOW_PREVIEW,
)
from door_controller import lock_door, unlock_door, door_cleanup

import override  # reads hardware button state (falls back gracefully when no GPIO)
override_btn = override.init_override_button(27)

from vision.cat_finder_tfod import CatFinderTFOD
from vision.pipeline import VisionPipeline


# ── Door state machine ──────────────────────────────────────────────────────

_state_lock = threading.Lock()

door_locked: bool = True
lock_until_ts: Optional[float] = None
override_force_open: bool = OVERRIDE_DEFAULT_FORCE_OPEN

last_event_nr: Optional[int] = None
clean_hits_this_event: int = 0


def setup_ethernet_link_local(
    iface: str = "eth0",
    ip_addr: str = "169.254.1.2/16",
):
    logging.info("Setting up IP address for Ethernet...")
    try:
        subprocess.run(
            ["sudo", "ip", "addr", "replace", ip_addr, "dev", iface],
            check=True,
        )
        subprocess.run(
            ["sudo", "ip", "link", "set", iface, "up"],
            check=True,
        )
        logging.info(f"Ethernet {iface} configured with {ip_addr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to configure Ethernet {iface}: {e}")
        raise


def _apply_lock(reason: str):
    global door_locked
    lock_door()
    door_locked = True
    logger.info(f"Door -> LOCKED  ({reason})")


def _apply_unlock(reason: str):
    global door_locked
    unlock_door()
    door_locked = False
    logger.info(f"Door -> UNLOCKED ({reason})")


def door_decision_cb(decision: str, score=None, event_nr=None):
    global lock_until_ts, last_event_nr, clean_hits_this_event, override_force_open

    with _state_lock:
        override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN

        if override_force_open:
            _apply_unlock("override_force_open")
            return

        now = time.time()

        if lock_until_ts is not None and now < lock_until_ts:
            remaining = int(lock_until_ts - now)
            logger.info(f"  Prey lock active — {remaining}s remaining. Decision ignored.")
            return

        if lock_until_ts is not None and now >= lock_until_ts:
            lock_until_ts = None
            _apply_lock("prey_lock_expired")

        if event_nr is not None and event_nr != last_event_nr:
            last_event_nr = event_nr
            clean_hits_this_event = 0

        if decision == "prey":
            lock_until_ts = now + LOCK_DURATION_SECONDS
            clean_hits_this_event = 0
            _apply_lock(f"prey_detected — locked for {LOCK_DURATION_SECONDS}s")
            return

        if decision == "no_prey":
            clean_hits_this_event += 1
            logger.info(f"  Clean confirmation {clean_hits_this_event}/{CLEAN_CONFIRMATIONS}")
            if clean_hits_this_event >= CLEAN_CONFIRMATIONS:
                _apply_unlock("clean_confirmed")
            return

        logger.info("  DK — keeping previous door state.")


def timer_tick_forever(stop_event: threading.Event):
    global lock_until_ts, override_force_open

    while not stop_event.is_set():
        with _state_lock:
            override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN

            if override_force_open:
                if door_locked:
                    _apply_unlock("override_force_open_tick")
            else:
                now = time.time()
                if lock_until_ts is not None and now >= lock_until_ts:
                    lock_until_ts = None
                    _apply_lock("prey_lock_expired")
        time.sleep(0.5)


# ── Shared helpers ──────────────────────────────────────────────────────────

def _open_capture(source: Union[int, str]):
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_PROP_BUFFERSIZE)
    except Exception:
        pass
    try:
        from config import CAPTURE_WIDTH, CAPTURE_HEIGHT
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    except Exception:
        pass
    return cap


def _draw_preview(frame, det_box, face_box, label, window_title="Smart Cat Door"):
    display = frame.copy()
    if det_box is not None:
        (x1, y1), (x2, y2) = det_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if face_box is not None:
        (fx1, fy1), (fx2, fy2) = face_box
        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (255, 140, 0), 2)
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow(window_title, display)
    cv2.waitKey(1)


# ── Webcam / RTSP mode ──────────────────────────────────────────────────────

def run_vision_forever(stop_event: threading.Event, show_preview: bool):
    cat_finder = CatFinderTFOD(TFOD_FROZEN_GRAPH)
    pipeline = VisionPipeline(MODELS_DIR)

    cap = _open_capture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {CAMERA_SOURCE}")

    logger.info(f"Camera opened: {CAMERA_SOURCE}")

    frame_idx = 0
    event_nr = 0
    miss_streak = 0
    in_event = False

    last_det_box = None
    last_face_box = None
    last_label = "waiting..."

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("Camera read failed; retrying...")
            time.sleep(0.1)
            continue

        frame_idx += 1

        if show_preview:
            _draw_preview(frame, last_det_box, last_face_box, last_label)

        if frame_idx % VISION_EVERY_N_FRAMES != 0:
            continue

        det = cat_finder.detect(frame)

        if not det.found or det.box is None:
            miss_streak += 1
            if miss_streak == 1 and in_event:
                logger.info(f"  Frame {frame_idx}: cat lost (miss streak starting)")
            elif miss_streak % 30 == 0:
                logger.info(f"  Frame {frame_idx}: no cat detected (miss streak {miss_streak})")
            last_det_box = None
            last_face_box = None
            last_label = "no cat"
            if in_event and miss_streak >= EVENT_END_MISSES:
                logger.info(f"  Event #{event_nr} ended after {miss_streak} consecutive misses.")
                in_event = False
            if in_event:
                door_decision_cb("dk", score=None, event_nr=event_nr)
            continue

        # Cat found
        (x1, y1), (x2, y2) = det.box
        miss_streak = 0
        last_det_box = det.box

        if not in_event:
            event_nr += 1
            in_event = True
            logger.info(f">>> EVENT #{event_nr} START — cat detected")

        res = pipeline.analyze(frame, det.box, det.score)
        last_face_box = res.face_box

        prey_str  = {True: "PREY", False: "no_prey", None: "dk"}.get(res.prey, "?")
        conf_str  = f" conf={res.prey_conf:.2f}" if res.prey_conf is not None else ""
        face_str  = f"face={res.face_method or 'no'}"
        ff_str    = (f" ff={'OK' if res.ff_confirmed else 'REJECTED'} ({res.ff_score:.2f})"
                     if res.ff_score is not None else " ff=n/a")
        infer_str = f" tfod={det.inference_s*1000:.0f}ms"

        logger.info(
            f"  Frame {frame_idx:5d} | cat={det.score:.2f} [{x1},{y1},{x2},{y2}]"
            f" | {face_str}{ff_str} | pred={prey_str}{conf_str}{infer_str}"
        )

        last_label = f"event#{event_nr} {det.score:.2f} | {prey_str}{conf_str}"

        if res.prey is True:
            logger.info(f"  >>> PREY DETECTED — triggering door lock")
            door_decision_cb("prey", score=res.prey_conf, event_nr=event_nr)
        elif res.prey is False:
            door_decision_cb("no_prey", score=res.prey_conf, event_nr=event_nr)
        else:
            door_decision_cb("dk", score=None, event_nr=event_nr)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()


# ── Test-video mode ─────────────────────────────────────────────────────────

def run_test_videos(videos_dir: Path, show_preview: bool):
    """
    Runs the full vision pipeline over every video in
      <videos_dir>/with_prey/   (ground truth = prey)
      <videos_dir>/no_prey/     (ground truth = no_prey)

    Logs each detection with ground truth vs prediction, then prints
    per-video and overall accuracy summaries.  Does NOT operate the door.
    """
    cat_finder = CatFinderTFOD(TFOD_FROZEN_GRAPH)
    pipeline   = VisionPipeline(MODELS_DIR)

    prey_videos    = sorted((videos_dir / "with_prey").glob("*.mp4")) if (videos_dir / "with_prey").exists() else []
    no_prey_videos = sorted((videos_dir / "no_prey").glob("*.mp4"))  if (videos_dir / "no_prey").exists()  else []

    random.shuffle(prey_videos)
    random.shuffle(no_prey_videos)

    # Interleave: one prey, one no_prey, alternating, then append the leftovers
    video_entries: list[tuple[Path, str]] = []
    for p, n in zip(prey_videos, no_prey_videos):
        video_entries.append((p, "prey"))
        video_entries.append((n, "no_prey"))
    for p in prey_videos[len(no_prey_videos):]:
        video_entries.append((p, "prey"))
    for n in no_prey_videos[len(prey_videos):]:
        video_entries.append((n, "no_prey"))

    if not video_entries:
        logger.error(f"No .mp4 files found under {videos_dir}/with_prey/ or {videos_dir}/no_prey/")
        return

    logger.info(f"Found {len(video_entries)} test video(s) — {len(prey_videos)} prey, {len(no_prey_videos)} no_prey")

    total_frames_processed = 0
    total_cat_det   = 0
    total_correct   = 0
    total_wrong     = 0
    total_dk        = 0
    total_no_face   = 0

    for video_path, ground_truth in video_entries:
        sep = "=" * 70
        logger.info(sep)
        logger.info(f"VIDEO      : {video_path.name}")
        logger.info(f"GROUND TRUTH: {ground_truth.upper()}")
        logger.info(sep)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"  Cannot open {video_path} — skipping.")
            continue

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(f"  {total_video_frames} frames @ {fps:.1f} fps — analyzing every {VISION_EVERY_N_FRAMES} frame(s)")

        frame_idx      = 0
        v_processed    = 0
        v_cat_det      = 0
        v_correct      = 0
        v_wrong        = 0
        v_dk           = 0
        v_no_face      = 0
        aborted        = False

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1

            if frame_idx % VISION_EVERY_N_FRAMES != 0:
                if show_preview:
                    cv2.imshow(f"Test: {video_path.name}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        aborted = True
                        break
                continue

            v_processed += 1
            det = cat_finder.detect(frame)

            if not det.found or det.box is None:
                if show_preview:
                    _draw_preview(
                        frame, None, None,
                        f"gt={ground_truth} | no cat",
                        window_title=f"Test: {video_path.name}",
                    )
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        aborted = True
                        break
                continue

            v_cat_det += 1
            (x1, y1), (x2, y2) = det.box
            res = pipeline.analyze(frame, det.box, det.score)

            prey_str = {True: "PREY   ", False: "no_prey", None: "dk     "}.get(res.prey, "?      ")
            conf_str = f"conf={res.prey_conf:.2f}" if res.prey_conf is not None else "conf=n/a"
            face_str = f"face={res.face_method or 'no':4}"
            ff_str   = (f" ff={'OK      ' if res.ff_confirmed else 'REJECTED'} ({res.ff_score:.2f})"
                        if res.ff_score is not None else " ff=n/a        ")

            if not res.face:
                v_no_face += 1
                match_str = "NO_FACE"
            elif res.ff_confirmed is False:
                v_no_face += 1
                match_str = "FF_REJECTED"
            elif res.prey is None:
                v_dk += 1
                match_str = "DK     "
            elif (res.prey and ground_truth == "prey") or (not res.prey and ground_truth == "no_prey"):
                v_correct += 1
                match_str = "CORRECT"
            else:
                v_wrong += 1
                match_str = "WRONG  "

            gt_tag = f"[gt={ground_truth.upper()[:7]:7}]"
            logger.info(
                f"  {gt_tag} frame {frame_idx:5d}/{total_video_frames}"
                f" | cat={det.score:.2f} | {face_str}{ff_str} | pred={prey_str} {conf_str}"
                f" | {match_str}"
            )

            if show_preview:
                overlay = (
                    f"gt={ground_truth} | pred={'PREY' if res.prey else ('no_prey' if res.prey is False else 'dk')}"
                    f" | {match_str.strip()}"
                )
                _draw_preview(
                    frame, det.box, res.face_box, overlay,
                    window_title=f"Test: {video_path.name}",
                )
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    aborted = True
                    break

        cap.release()

        if aborted:
            logger.info("  (video aborted by user — Q key)")

        attempts = v_correct + v_wrong
        acc = v_correct / attempts * 100 if attempts else 0.0

        logger.info(f"  {'─'*60}")
        logger.info(f"  SUMMARY  {video_path.name}")
        logger.info(f"    Frames processed : {v_processed}")
        logger.info(f"    Cat detections   : {v_cat_det}")
        logger.info(f"    No face found    : {v_no_face}")
        logger.info(f"    CORRECT          : {v_correct}")
        logger.info(f"    WRONG            : {v_wrong}")
        logger.info(f"    DK (uncertain)   : {v_dk}")
        logger.info(f"    Accuracy (excl dk+no_face): {acc:.1f}%")
        logger.info(f"  {'─'*60}")

        total_frames_processed += v_processed
        total_cat_det  += v_cat_det
        total_correct  += v_correct
        total_wrong    += v_wrong
        total_dk       += v_dk
        total_no_face  += v_no_face

    if show_preview:
        cv2.destroyAllWindows()

    total_attempts = total_correct + total_wrong
    total_acc = total_correct / total_attempts * 100 if total_attempts else 0.0

    sep = "=" * 70
    logger.info(sep)
    logger.info("OVERALL RESULTS")
    logger.info(f"  Videos tested        : {len(video_entries)}")
    logger.info(f"  Frames processed     : {total_frames_processed}")
    logger.info(f"  Cat detections       : {total_cat_det}")
    logger.info(f"  No face found        : {total_no_face}")
    logger.info(f"  CORRECT predictions  : {total_correct}")
    logger.info(f"  WRONG predictions    : {total_wrong}")
    logger.info(f"  DK (uncertain)       : {total_dk}")
    logger.info(f"  Accuracy (excl dk+no_face): {total_acc:.1f}%")
    logger.info(sep)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart Cat Door — vision pipeline")
    parser.add_argument(
        "--mode",
        choices=["webcam", "videos"],
        default="webcam",
        help="webcam: live camera feed (default).  videos: run test videos and evaluate.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_videos",
        help="Root folder containing with_prey/ and no_prey/ subdirs (default: ./test_videos).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Suppress the OpenCV preview window.",
    )
    args = parser.parse_args()

    show_preview = SHOW_PREVIEW and not args.no_preview

    logger.info(f"Smart Cat Door starting — mode={args.mode} preview={show_preview}")

    if args.mode == "videos":
        run_test_videos(args.videos_dir, show_preview)
        return

    # ── Webcam / live mode ──
    # setup_ethernet_link_local()  # Pi only

    with _state_lock:
        _apply_lock("startup_default_locked")

    stop_event = threading.Event()

    vision_thread = threading.Thread(
        target=run_vision_forever, args=(stop_event, show_preview), daemon=True
    )
    vision_thread.start()
    logger.info("Vision thread started.")

    timer_thread = threading.Thread(target=timer_tick_forever, args=(stop_event,), daemon=True)
    timer_thread.start()
    logger.info("Timer thread started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Shutting down.")
    finally:
        stop_event.set()
        door_cleanup()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()
