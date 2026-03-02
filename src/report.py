import json
from collections import Counter, defaultdict
from pathlib import Path

EVENTS_FILE = Path("outputs/events.jsonl")
OUT_FILE = Path("outputs/session_summary.txt")


def main():
    if not EVENTS_FILE.exists():
        print("No outputs/events.jsonl found. Run the app first.")
        return

    counts = Counter()
    by_track = defaultdict(Counter)

    with EVENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            etype = e.get("type", "unknown")
            tid = e.get("track_id", "unknown")
            counts[etype] += 1
            by_track[tid][etype] += 1

    lines = []
    lines.append("VisionGuard — Session Summary\n\n")
    lines.append("Event counts:\n")
    for k, v in counts.most_common():
        lines.append(f"  - {k}: {v}\n")

    lines.append("\nTop tracks by total events:\n")
    ranked = sorted(by_track.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:5]
    for tid, c in ranked:
        total = sum(c.values())
        breakdown = ", ".join([f"{k}:{v}" for k, v in c.most_common()])
        lines.append(f"  - ID {tid}: {total} ({breakdown})\n")

    OUT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()