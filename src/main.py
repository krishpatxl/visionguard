import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO

from configs.zones import ZONES
from src.analytics import BehaviorAnalyzer
from src.logger import EventLogger

# COCO class ids
TRACK_CLASS_IDS = {0, 2, 3, 5, 7}  # person, car, motorcycle, bus, truck

EVENT_BANNER_SECONDS = 2.0


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuard - real-time tracking + simple behavior events")
    p.add_argument("--source", default="0", help="0 for webcam, or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path (e.g. yolov8n.pt)")
    return p.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def draw_zones(frame):
    for z in ZONES:
        cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            z.name,
            (z.x1, max(20, z.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def draw_hud(frame, fps: float, track_count: int, zone_count: int, recent_events):
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Tracks: {track_count}",
        f"Zones: {zone_count}",
    ]

    y = 28
    for line in hud_lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 26

    # Event banner (short-lived)
    banner_y = y + 10
    now = time.time()
    for ts, text in list(recent_events):
        if now - ts <= EVENT_BANNER_SECONDS:
            cv2.putText(frame, text, (16, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            banner_y += 26


def main():
    args = parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}. Try --source 1 or check macOS camera permissions.")

    yolo = YOLO(args.model)
    analyzer = BehaviorAnalyzer(zones=ZONES)
    logger = EventLogger()

    recent_events = deque(maxlen=5)

    fps_clock = time.time()
    fps_counter = 0
    fps = 0.0

    print("VisionGuard running (press 'q' to quit)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        draw_zones(frame)

        # FPS calc (kept intentionally simple)
        fps_counter += 1
        if now - fps_clock >= 1.0:
            fps = fps_counter / (now - fps_clock)
            fps_clock = now
            fps_counter = 0

        results = yolo.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid, cls in zip(boxes, ids, classes):
                if cls not in TRACK_CLASS_IDS:
                    continue

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                analyzer.update(tid, (cx, cy), now=now)
                events = analyzer.check(tid, (cx, cy), now=now)

                # Trail (history line)
                st = analyzer.tracks[tid]
                pts = [p for (_, p) in st.history]
                for i in range(1, len(pts)):
                    x0, y0 = pts[i - 1]
                    x1_, y1_ = pts[i]
                    cv2.line(frame, (int(x0), int(y0)), (int(x1_), int(y1_)), (255, 255, 0), 2)

                # Box + ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Log + annotate events
                y_text = int(y2) + 18
                for event_type, payload in events:
                    logger.write(event_type, payload)
                    recent_events.append((now, f"{event_type} (ID {payload.get('track_id')})"))

                    cv2.putText(
                        frame,
                        event_type,
                        (int(x1), y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
                    y_text += 18

        draw_hud(frame, fps=fps, track_count=len(analyzer.tracks), zone_count=len(ZONES), recent_events=recent_events)

        cv2.imshow("VisionGuard", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()