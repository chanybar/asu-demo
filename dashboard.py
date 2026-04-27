"""
dashboard.py — ASU Mobility Vision Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import sys
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from utils.congestion import CongestionTracker
from utils.overlay import draw_detections, draw_hud, build_heatmap

# ── Constants ────────────────────────────────────────────────────────────────
ASU_STREAM_URL   = "https://view.asu.edu/tempe/hayden"
FALLBACK_VIDEOS  = sorted((BASE_DIR.parent / "mobility_demo" / "CIS515-Project").glob("*.mp4"))
DEFAULT_MODEL    = BASE_DIR / "models" / "mobility_v1" / "weights" / "best.pt"
PRETRAINED_MODEL = "yolov8n.pt"
HEATMAP_HISTORY  = 60   # frames to accumulate for heatmap
LOG_DIR          = BASE_DIR / "logs"

STATUS_COLORS = {"LOW": "#40dc64", "MEDIUM": "#ffc81e", "HIGH": "#ff3b3b"}
STATUS_EMOJI  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASU Mobility Vision",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 50%, #0d1420 100%); }

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-number { font-size: 3rem; font-weight: 700; line-height: 1; }
.metric-label  { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }

.status-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    width: 100%;
    text-align: center;
}
.status-low    { background: rgba(64,220,100,0.15); color: #40dc64; border: 1px solid rgba(64,220,100,0.4); }
.status-medium { background: rgba(255,200,30,0.15);  color: #ffc81e; border: 1px solid rgba(255,200,30,0.4); }
.status-high   { background: rgba(255,59,59,0.15);   color: #ff3b3b; border: 1px solid rgba(255,59,59,0.4); }

.section-title {
    font-size: 0.75rem; font-weight: 600; color: #5b6a82;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;
}
.score-ring {
    font-size: 2.2rem; font-weight: 700; color: #a78bfa;
}
.stVideo, [data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running":         False,
        "model":           None,
        "tracker":         None,
        "congestion":      None,
        "frame_rgb":       None,
        "metrics":         {},
        "history_df":      pd.DataFrame(columns=["time", "score", "walkers", "wheeled"]),
        "det_history":     deque(maxlen=HEATMAP_HISTORY),
        "source":          "demo",
        "show_heatmap":    False,
        "show_track_ids":  True,
        "conf_threshold":  0.40,
        "stream_cap":      None,
        "video_cap":       None,
        "frame_idx":       0,
        "start_time":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


# ── Frame processor ──────────────────────────────────────────────────────────
def process_frame(frame_bgr: np.ndarray, model, conf: float, use_tracker: bool) -> tuple:
    """Run detection (+optional tracking) on a frame. Returns (annotated_rgb, walkers, wheeled, detections)."""
    results = model(frame_bgr, conf=conf, verbose=False)[0]

    raw_dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in (0, 1):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        raw_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cls_id": cls_id,
            "conf": float(box.conf[0].item()),
        })

    if use_tracker and st.session_state.tracker:
        tracked = st.session_state.tracker.update(raw_dets)
        display_dets = tracked
    else:
        display_dets = raw_dets

    walkers = sum(1 for d in display_dets if d.get("cls_id", 0) == 0)
    wheeled = sum(1 for d in display_dets if d.get("cls_id", 0) == 1)

    annotated = draw_detections(frame_bgr, display_dets,
                                show_conf=True,
                                show_track_ids=st.session_state.show_track_ids)
    return annotated, walkers, wheeled, display_dets


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    # Source selection
    st.markdown('<div class="section-title">Video Source</div>', unsafe_allow_html=True)
    source = st.radio(
        "Source",
        options=["🔴 Live Stream (ASU Hayden)", "📁 Demo Video (Local)", "📸 Demo Frames"],
        label_visibility="collapsed"
    )
    if "Live" in source:
        st.session_state.source = "live"
    elif "Video" in source:
        st.session_state.source = "video"
    else:
        st.session_state.source = "frames"

    # Model selection
    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Model",
        ["🏋️ Fine-tuned (after training)", "⚡ Pretrained YOLOv8n"],
        label_visibility="collapsed"
    )
    model_path = str(DEFAULT_MODEL) if ("Fine" in model_choice and DEFAULT_MODEL.exists()) else PRETRAINED_MODEL

    # Settings
    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
    st.session_state.conf_threshold = st.slider("Confidence", 0.20, 0.80, 0.40, 0.05)
    use_tracker = st.toggle("Enable SORT Tracking", value=True)
    st.session_state.show_heatmap   = st.toggle("Show Heatmap", value=False)
    st.session_state.show_track_ids = st.toggle("Show Track IDs", value=True)
    target_fps = st.slider("Target FPS", 1, 15, 5)

    st.markdown('<div class="section-title">Congestion Window</div>', unsafe_allow_html=True)
    window_sec = st.slider("Rolling Window (sec)", 10, 120, 60, 10)

    st.markdown("---")
    col_run, col_stop = st.columns(2)
    with col_run:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            # Load model
            try:
                st.session_state.model = load_model(model_path)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            # Init tracker
            if use_tracker:
                from utils.tracker import SORTTracker
                from utils.tracker import KalmanBoxTracker
                KalmanBoxTracker.count = 0
                st.session_state.tracker = SORTTracker()
            else:
                st.session_state.tracker = None

            # Init congestion tracker
            st.session_state.congestion = CongestionTracker(
                window_seconds=window_sec,
                log_dir=str(LOG_DIR),
            )
            st.session_state.running    = True
            st.session_state.start_time = time.time()
            st.session_state.frame_idx  = 0
            st.session_state.history_df = pd.DataFrame(columns=["time", "score", "walkers", "wheeled"])
            st.session_state.det_history.clear()

    with col_stop:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            if st.session_state.stream_cap:
                st.session_state.stream_cap.stop()
                st.session_state.stream_cap = None
            if st.session_state.video_cap:
                st.session_state.video_cap.release()
                st.session_state.video_cap = None

    # Log download
    log_files = sorted(LOG_DIR.glob("*.csv")) if LOG_DIR.exists() else []
    if log_files:
        st.markdown("---")
        st.markdown('<div class="section-title">Export Logs</div>', unsafe_allow_html=True)
        latest_log = log_files[-1]
        with open(latest_log, "rb") as f:
            st.download_button("⬇ Download CSV", f, file_name=latest_log.name,
                               mime="text/csv", use_container_width=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("## 🏫 ASU Mobility Vision — Real-Time Congestion Monitor")
st.markdown("*Pedestrian & Wheeled Mobility Detection · Hayden Library Zone*")

main_col, side_col = st.columns([3, 1], gap="large")

with main_col:
    video_placeholder = st.empty()

with side_col:
    st.markdown('<div class="section-title">Live Counts</div>', unsafe_allow_html=True)
    walker_metric  = st.empty()
    wheeled_metric = st.empty()
    score_metric   = st.empty()
    status_badge   = st.empty()
    st.markdown("---")
    st.markdown('<div class="section-title">Rolling Average</div>', unsafe_allow_html=True)
    rolling_metric = st.empty()
    predict_card   = st.empty()
    st.markdown("---")
    st.markdown('<div class="section-title">Session Info</div>', unsafe_allow_html=True)
    session_info   = st.empty()

st.markdown("---")
chart_col1, chart_col2 = st.columns([2, 1])
with chart_col1:
    st.markdown('<div class="section-title">Congestion Over Time</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
with chart_col2:
    st.markdown('<div class="section-title">Current Composition</div>', unsafe_allow_html=True)
    pie_placeholder = st.empty()


# ── Helper renderers ──────────────────────────────────────────────────────────
def render_metric(container, value: int, label: str, color: str):
    container.markdown(f"""
    <div class="metric-card">
        <div class="metric-number" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status(container, status: str):
    cls = f"status-{status.lower()}"
    emoji = STATUS_EMOJI.get(status, "⚪")
    container.markdown(f"""
    <div class="status-badge {cls}">{emoji} {status}</div>
    """, unsafe_allow_html=True)


def render_chart(container, df: pd.DataFrame):
    if df.empty or len(df) < 2:
        container.info("Waiting for data…")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["score"],
        mode="lines", name="Congestion Score",
        line=dict(color="#a78bfa", width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["walkers"],
        mode="lines", name="Walkers",
        line=dict(color="#40dc9f", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["wheeled"],
        mode="lines", name="Wheeled",
        line=dict(color="#ffb830", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=5,  line_dash="dash", line_color="rgba(64,220,100,0.4)",  annotation_text="Low")
    fig.add_hline(y=12, line_dash="dash", line_color="rgba(255,59,59,0.4)",   annotation_text="High")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(showgrid=False, title="Time (s)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        height=250,
    )
    container.plotly_chart(fig, use_container_width=True)


def render_pie(container, walkers: int, wheeled: int):
    if walkers + wheeled == 0:
        container.info("No detections yet.")
        return
    fig = go.Figure(go.Pie(
        labels=["Walkers", "Wheeled"],
        values=[walkers, wheeled],
        hole=0.55,
        marker=dict(colors=["#40dc9f", "#ffb830"]),
        textinfo="label+percent",
        insidetextorientation="radial",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=250,
    )
    container.plotly_chart(fig, use_container_width=True)


# ── Frame source helpers ──────────────────────────────────────────────────────
def get_next_frame_demo() -> np.ndarray | None:
    """Cycle through local MP4 fallback videos."""
    videos = FALLBACK_VIDEOS
    if not videos:
        return None
    if st.session_state.video_cap is None or not st.session_state.video_cap.isOpened():
        vid_idx = st.session_state.frame_idx % len(videos)
        st.session_state.video_cap = cv2.VideoCapture(str(videos[vid_idx]))

    cap = st.session_state.video_cap
    for _ in range(6):   # skip frames to hit target FPS
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            return frame
    return None


def get_next_frame_frames() -> np.ndarray | None:
    """Cycle through extracted JPEG frames."""
    frames_dir = BASE_DIR.parent / "mobility_demo" / "CIS515-Project" / "frames"
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        return None
    idx = st.session_state.frame_idx % len(files)
    img = cv2.imread(str(files[idx]))
    st.session_state.frame_idx += 1
    return img


def get_next_frame_live() -> np.ndarray | None:
    """Pull from the ASU live stream (or fall back to demo)."""
    cap = st.session_state.stream_cap
    if cap is None:
        from utils.stream import StreamCapture
        fallback = str(FALLBACK_VIDEOS[0]) if FALLBACK_VIDEOS else None
        cap = StreamCapture(ASU_STREAM_URL, fps_target=target_fps, fallback_video=fallback)
        cap.start()
        st.session_state.stream_cap = cap
    return cap.get_frame()


# ── Main loop ─────────────────────────────────────────────────────────────────
FRAME_DELAY = 1.0 / max(target_fps, 1)

if st.session_state.running:
    model = st.session_state.model
    cong  = st.session_state.congestion

    # Get frame
    source_mode = st.session_state.source
    if source_mode == "live":
        frame = get_next_frame_live()
        src_label = "LIVE · ASU HAYDEN"
    elif source_mode == "video":
        frame = get_next_frame_demo()
        src_label = "DEMO VIDEO"
        st.session_state.frame_idx += 1
    else:
        frame = get_next_frame_frames()
        src_label = "DEMO FRAMES"

    if frame is not None and model is not None:
        # Detect
        annotated_bgr, walkers, wheeled, dets = process_frame(
            frame, model,
            conf=st.session_state.conf_threshold,
            use_tracker=use_tracker,
        )

        # Update history for heatmap
        st.session_state.det_history.append(dets)

        # Heatmap overlay
        if st.session_state.show_heatmap and len(st.session_state.det_history) > 5:
            annotated_bgr = build_heatmap(annotated_bgr, list(st.session_state.det_history))

        # Update congestion
        metrics = cong.update(walkers, wheeled)

        # HUD overlay
        annotated_bgr = draw_hud(
            annotated_bgr,
            walkers=walkers,
            wheeled=wheeled,
            score=metrics["score"],
            status=metrics["status"],
            source_label=src_label,
            rolling_avg=metrics["rolling_avg_score"],
        )

        # Display frame
        frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update sidebar metrics
        render_metric(walker_metric,  walkers, "Walkers 🚶",  "#40dc9f")
        render_metric(wheeled_metric, wheeled, "Wheeled 🚲",  "#ffb830")
        score_metric.markdown(f"""
        <div class="metric-card">
            <div class="score-ring">{metrics['score']}</div>
            <div class="metric-label">Congestion Score</div>
        </div>
        """, unsafe_allow_html=True)
        render_status(status_badge, metrics["status"])

        rolling_metric.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.8rem;font-weight:700;color:#c4b5fd">{metrics['rolling_avg_score']}</div>
            <div class="metric-label">Rolling Avg ({window_sec}s)</div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction
        pred = cong.predict_congestion(lookahead_seconds=300)
        if pred["predicted_score"] is not None:
            trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(pred["trend"], "➡️")
            predict_card.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#5b6a82;text-transform:uppercase;letter-spacing:1px">5-min Forecast</div>
                <div style="font-size:1.5rem;font-weight:700;color:#f9a8d4">{pred['predicted_score']}</div>
                <div style="font-size:0.8rem;color:#8892a4">{trend_icon} {pred['trend'].capitalize()} · {pred['predicted_status']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Session info
        elapsed = int(time.time() - (st.session_state.start_time or time.time()))
        session_info.markdown(f"""
        <div style="font-size:0.82rem;color:#5b6a82;line-height:1.8">
            ⏱ {elapsed}s elapsed<br>
            🎞 Frame #{st.session_state.frame_idx}<br>
            📊 {metrics['window_size']} samples in window
        </div>
        """, unsafe_allow_html=True)

        # Append to history
        new_row = pd.DataFrame([{
            "time": metrics["elapsed_seconds"],
            "score": metrics["score"],
            "walkers": walkers,
            "wheeled": wheeled,
        }])
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row], ignore_index=True
        ).tail(500)

        # Charts
        render_chart(chart_placeholder, st.session_state.history_df)
        render_pie(pie_placeholder, walkers, wheeled)

        # Congestion alert
        if metrics["status"] == "HIGH":
            st.toast(f"🔴 HIGH congestion detected! Score: {metrics['score']}", icon="🚨")

        time.sleep(FRAME_DELAY)
        st.rerun()

    else:
        video_placeholder.warning("⚠️ Waiting for frame… Check video source or model.")
        time.sleep(1)
        st.rerun()

else:
    # Idle state
    video_placeholder.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(167,139,250,0.08), rgba(96,165,250,0.05));
        border: 1px dashed rgba(167,139,250,0.3);
        border-radius: 16px;
        padding: 60px;
        text-align: center;
        color: #5b6a82;
    ">
        <div style="font-size:4rem; margin-bottom:16px">🏫</div>
        <div style="font-size:1.3rem; font-weight:600; color:#8892a4; margin-bottom:8px">
            ASU Mobility Vision
        </div>
        <div style="font-size:0.95rem">
            Configure settings in the sidebar and press <strong style="color:#a78bfa">▶ Start</strong> to begin detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    walker_metric.markdown("""
    <div class="metric-card"><div class="metric-number" style="color:#40dc9f">—</div>
    <div class="metric-label">Walkers</div></div>""", unsafe_allow_html=True)
    wheeled_metric.markdown("""
    <div class="metric-card"><div class="metric-number" style="color:#ffb830">—</div>
    <div class="metric-label">Wheeled</div></div>""", unsafe_allow_html=True)
    score_metric.markdown("""
    <div class="metric-card"><div class="score-ring">—</div>
    <div class="metric-label">Congestion Score</div></div>""", unsafe_allow_html=True)
    status_badge.markdown("""
    <div class="status-badge" style="background:rgba(255,255,255,0.05);color:#5b6a82;
    border:1px solid rgba(255,255,255,0.08)">⚪ IDLE</div>""", unsafe_allow_html=True)
 
