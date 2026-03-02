from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from src.zones import Zone

Point = Tuple[float, float]


@dataclass
class TrackState:
    """Small per-track state: recent centers + zone timing + event cooldowns."""
    history: Deque[Tuple[float, Point]]
    last_zone: Optional[str] = None
    zone_enter_ts: Optional[float] = None
    last_event_ts: Dict[str, float] = None


class BehaviorAnalyzer:
    """
    Turns tracked centers into simple behavior events.
    Nothing fancy, but it’s fast and explainable.
    """

    def __init__(
        self,
        zones: List[Zone],
        history_seconds: float = 3.0,
        loiter_seconds: float = 3.0,
        speed_spike_threshold: float = 120.0,  # px/sec; tune to your cam
        erratic_turns_threshold: int = 6,
        erratic_window_seconds: float = 3.0,
        event_cooldown_seconds: float = 2.0,
    ):
        self.zones = zones
        self.history_seconds = history_seconds
        self.loiter_seconds = loiter_seconds
        self.speed_spike_threshold = speed_spike_threshold
        self.erratic_turns_threshold = erratic_turns_threshold
        self.erratic_window_seconds = erratic_window_seconds
        self.event_cooldown_seconds = event_cooldown_seconds

        self.tracks: Dict[int, TrackState] = defaultdict(
            lambda: TrackState(history=deque(), last_event_ts={})
        )

    def update(self, track_id: int, center: Point, now: Optional[float] = None) -> None:
        now = now or time.time()
        st = self.tracks[track_id]
        st.history.append((now, center))

        # time-bounded history
        while st.history and (now - st.history[0][0]) > self.history_seconds:
            st.history.popleft()

    def _zone_for(self, center: Point) -> Optional[str]:
        x, y = center
        for z in self.zones:
            if z.contains(x, y):
                return z.name
        return None

    def _speed_px_s(self, st: TrackState) -> Optional[float]:
        if len(st.history) < 2:
            return None
        t0, (x0, y0) = st.history[0]
        t1, (x1, y1) = st.history[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return math.hypot(x1 - x0, y1 - y0) / dt

    def _turns_in_window(self, st: TrackState, now: float) -> int:
        pts = [(t, p) for (t, p) in st.history if (now - t) <= self.erratic_window_seconds]
        if len(pts) < 4:
            return 0

        turns = 0
        prev_angle = None

        for i in range(1, len(pts)):
            (_, (x0, y0)) = pts[i - 1]
            (_, (x1, y1)) = pts[i]
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 and dy == 0:
                continue

            angle = math.atan2(dy, dx)
            if prev_angle is not None:
                diff = abs(angle - prev_angle)
                diff = min(diff, 2 * math.pi - diff)
                if diff > 0.7:  # ~40 degrees
                    turns += 1
            prev_angle = angle

        return turns

    def _allow_event(self, st: TrackState, event_type: str, now: float) -> bool:
        last = st.last_event_ts.get(event_type, 0.0)
        if now - last < self.event_cooldown_seconds:
            return False
        st.last_event_ts[event_type] = now
        return True

    def check(self, track_id: int, center: Point, now: Optional[float] = None):
        now = now or time.time()
        st = self.tracks[track_id]
        events = []

        zone = self._zone_for(center)

        # zone enter/exit
        if zone != st.last_zone:
            if st.last_zone and self._allow_event(st, "zone_exit", now):
                events.append(("zone_exit", {"track_id": track_id, "zone": st.last_zone}))

            if zone:
                st.zone_enter_ts = now
                if self._allow_event(st, "zone_enter", now):
                    events.append(("zone_enter", {"track_id": track_id, "zone": zone}))
            else:
                st.zone_enter_ts = None

            st.last_zone = zone

        # loitering
        if zone and st.zone_enter_ts and (now - st.zone_enter_ts) >= self.loiter_seconds:
            if self._allow_event(st, "loitering", now):
                events.append(
                    ("loitering", {
                        "track_id": track_id,
                        "zone": zone,
                        "seconds_in_zone": round(now - st.zone_enter_ts, 2),
                    })
                )

        # speed spike
        spd = self._speed_px_s(st)
        if spd is not None and spd >= self.speed_spike_threshold:
            if self._allow_event(st, "speed_spike", now):
                events.append(("speed_spike", {"track_id": track_id, "speed_px_s": round(spd, 2)}))

        # erratic motion
        turns = self._turns_in_window(st, now)
        if turns >= self.erratic_turns_threshold:
            if self._allow_event(st, "erratic_motion", now):
                events.append(("erratic_motion", {"track_id": track_id, "turns": turns}))

        return events