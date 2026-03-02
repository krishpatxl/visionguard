import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

EVENTS_PATH = Path("outputs/events.jsonl")


def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip any partially-written lines
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Standardize columns we care about
    for col in ["type", "track_id", "zone"]:
        if col not in df.columns:
            df[col] = None

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    # human-readable time
    df["time"] = df["ts"].apply(lambda x: datetime.fromtimestamp(float(x)))
    df = df.sort_values("ts", ascending=False).reset_index(drop=True)

    return df


def _nice_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def main():
    st.set_page_config(page_title="VisionGuard Dashboard", page_icon="👁️", layout="wide")

    st.title("👁️ VisionGuard — Event Dashboard")
    st.caption("Reads structured events from outputs/events.jsonl and summarizes what the system detected.")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        if st.button("🔄 Refresh"):
            st.rerun()

        auto_refresh = st.toggle("Auto-refresh (every ~2s)", value=False)

        st.divider()
        st.write("**Data source**")
        st.code(str(EVENTS_PATH), language="text")

    df = _read_events(EVENTS_PATH)

    if df.empty:
        st.warning("No events found yet. Run the tracker first: `python -m src.main`")
        st.info("Once events are logged, refresh this page.")
        return

    # Filters
    all_types = sorted([t for t in df["type"].dropna().unique().tolist()])
    all_tracks = sorted([int(x) for x in df["track_id"].dropna().unique().tolist() if str(x).isdigit()])

    with st.sidebar:
        st.divider()
        st.subheader("Filters")

        selected_types = st.multiselect("Event types", options=all_types, default=all_types)
        selected_tracks = st.multiselect("Track IDs", options=all_tracks, default=[])

        # time range (based on df)
        min_time = df["time"].min()
        max_time = df["time"].max()

        time_range = st.slider(
            "Time range",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
        )

    filtered = df.copy()
    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]

    if selected_tracks:
        filtered = filtered[filtered["track_id"].isin(selected_tracks)]

    start_dt, end_dt = time_range
    filtered = filtered[(filtered["time"] >= start_dt) & (filtered["time"] <= end_dt)]

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events", int(len(df)))
    col2.metric("Filtered events", int(len(filtered)))
    col3.metric("Unique IDs", int(df["track_id"].nunique()))
    col4.metric("Event types", int(df["type"].nunique()))

    st.divider()

    # Charts row
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Events over time")

        # simple time bucketing (by minute)
        timeline = filtered.copy()
        timeline["minute"] = timeline["time"].dt.floor("min")
        counts = timeline.groupby(["minute", "type"]).size().reset_index(name="count")
        pivot = counts.pivot(index="minute", columns="type", values="count").fillna(0)

        st.line_chart(pivot)

    with right:
        st.subheader("Most active track IDs")

        top_tracks = (
            filtered.dropna(subset=["track_id"])
            .groupby("track_id")
            .size()
            .reset_index(name="events")
            .sort_values("events", ascending=False)
            .head(10)
        )
        st.bar_chart(top_tracks.set_index("track_id")["events"])

    st.divider()

    # Incident feed + details
    st.subheader("Incident feed")

    show_cols = ["time", "type", "track_id", "zone", "seconds_in_zone", "speed_px_s", "turns"]
    for c in show_cols:
        if c not in filtered.columns:
            filtered[c] = None

    feed = filtered[show_cols].copy()
    feed["time"] = feed["time"].apply(_nice_time)

    st.dataframe(feed, use_container_width=True, height=320)

    st.divider()

    # Downloadable session summary (quick + handy)
    st.subheader("Session summary")

    type_counts = filtered["type"].value_counts().reset_index()
    type_counts.columns = ["event_type", "count"]

    st.table(type_counts)

    summary_lines = []
    summary_lines.append("VisionGuard — Dashboard Summary\n")
    summary_lines.append(f"Generated: {_nice_time(datetime.now())}\n\n")
    summary_lines.append(f"Events in filter: {len(filtered)}\n")
    summary_lines.append(f"Unique IDs in filter: {filtered['track_id'].nunique()}\n\n")
    summary_lines.append("Event counts:\n")
    for _, row in type_counts.iterrows():
        summary_lines.append(f"  - {row['event_type']}: {row['count']}\n")

    st.download_button(
        label="⬇️ Download summary.txt",
        data="".join(summary_lines),
        file_name="visionguard_summary.txt",
        mime="text/plain",
    )

    # lightweight auto-refresh without extra deps
    if auto_refresh:
        st.caption("Auto-refresh on. Turn it off in the sidebar if it gets annoying.")
        st.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()