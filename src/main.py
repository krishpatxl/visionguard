from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import load_settings
from src.heatmap import Heatmap
from src.logger import EventLogger
from src.analytics import BehaviorAnalyzer, Tripwire
from src.zones import Zone


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuard — tracking + behavior analytics")
    p.add_argument("--config", default="configs/settings.yaml", help="Path to YAML config")
    p.add_argument("--source", default=None, help="Override source: 0/1 or /path/to/video.mp4")
    return p.parse_args()


def open_source(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    s = str(source)
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    return cv2.VideoCapture(s)


def draw_zone(frame, z: Zone):
    cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), (0, 255, 255), 2)
    cv2.putText(frame, z.name, (z.x1, max(20, z.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw_tripwire(frame, tw: Tripwire):
    (ax, ay), (bx, by) = tw.a, tw.b
    cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 0), 2)
    cv2.putText(frame, tw.name, (int(ax) + 6, int(ay) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    args = parse_args()
    cfg = load_settings(args.config)

    out_dir = cfg.get("app", {}).get("output_dir", "outputs")
    events_file = cfg.get("app", {}).get("events_file", "events.jsonl")
    heatmap_file = cfg.get("app", {}).get("heatmap_file", "heatmap.png")

    video_cfg = cfg.get("video", {})
    default_source = video_cfg.get("default_source", 0)
    show_window = bool(video_cfg.get("show_window", True))

    model_cfg = cfg.get("model", {})
    weights = model_cfg.get("weights", "yolov8n.pt")
    conf = float(model_cfg.get("conf", 0.35))
    imgsz = int(model_cfg.get("imgsz", 640))

    analytics_cfg = cfg.get("analytics", {})
    loiter_seconds = float(analytics_cfg.get("loiter_seconds", 6.0))
    speed_spike = float(analytics_cfg.get("speed_spike_px_s", 900.0))
    err_window = float(analytics_cfg.get("erratic_window_s", 2.5))
    err_turns = int(analytics_cfg.get("erratic_turns_threshold", 4))
    cooldown = float(analytics_cfg.get("event_cooldown_seconds", 2.0))

    source = args.source if args.source is not None else default_source
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    yolo = YOLO(weights)
    logger = EventLogger(out_dir=out_dir, filename=events_file)

    heatmap = None
    last_frame = None

    # We'll build zone/tripwire once we know frame size
    zones = []
    tripwires = []
    analyzer = None

    print("VisionGuard running (press 'q' to quit)...")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Video ended or frame read failed.")
            break

        h, w = frame.shape[:2]
        now = time.time()
        last_frame = frame.copy()

        if heatmap is None:
            heatmap = Heatmap(width=w, height=h)

        # Initialize zone + tripwire after we know w/h
        if analyzer is None:
            # Zone from normalized settings
            zcfg = cfg.get("zones", {}).get("loiter_zone", {})
            if zcfg.get("enabled", True):
                z = Zone(
                    name="loiter_zone",
                    x1=int(float(zcfg.get("x1", 0.05)) * w),
                    y1=int(float(zcfg.get("y1", 0.10)) * h),
                    x2=int(float(zcfg.get("x2", 0.45)) * w),
                    y2=int(float(zcfg.get("y2", 0.55)) * h),
                )
                zones = [z]

            # Tripwire from normalized settings
            twcfg = cfg.get("tripwire", {})
            if twcfg.get("enabled", True):
                p1 = twcfg.get("p1", [0.55, 0.10])
                p2 = twcfg.get("p2", [0.55, 0.95])
                prox = float(twcfg.get("proximity_px", 40))

                tw = Tripwire(
                    name="tripwire",
                    a=(float(p1[0]) * w, float(p1[1]) * h),
                    b=(float(p2[0]) * w, float(p2[1]) * h),
                    proximity_px=prox,
                )
                tripwires = [tw]

            analyzer = BehaviorAnalyzer(
                zones=zones,
                loiter_seconds=loiter_seconds,
                speed_spike_threshold=speed_spike,
                erratic_window_seconds=err_window,
                erratic_turns_threshold=err_turns,
                event_cooldown_seconds=cooldown,
                tripwires=tripwires,
            )

        # Draw overlays
        for z in zones:
            draw_zone(frame, z)
        for tw in tripwires:
            draw_tripwire(frame, tw)

        # Run detection + tracking
        results = yolo.track(frame, conf=conf, imgsz=imgsz, persist=True, verbose=False)
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                heatmap.add(cx, cy, weight=1.0)

                analyzer.update(tid, (cx, cy), now=now)
                events = analyzer.check(tid, (cx, cy), now=now)

                for etype, payload in events:
                    logger.write(etype, payload)

                # bbox + ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if show_window:
            cv2.imshow("VisionGuard", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save heatmap at end
    if heatmap is not None and last_frame is not None:
        out_path = Path(out_dir) / heatmap_file
        heatmap.save_overlay(last_frame, out_path)
        print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    main()