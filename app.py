"""
Face Vitals Scanner - Streamlit app.

Estimates heart rate, HRV, respiration rate, an experimental stress index, and an
experimental SpO2 reading from ordinary webcam video using remote photoplethysmography
(rPPG). Camera access happens live in the browser via WebRTC (streamlit-webrtc);
video is processed in memory and never saved or uploaded anywhere.

NOT A MEDICAL DEVICE. Estimates only. See the in-app disclaimer.
"""

import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from vitals import analyze, resample_uniform

SCAN_SECONDS_DEFAULT = 25
TARGET_FS = 20  # samples/sec we resample onto for analysis (webcam delivers ~15-30fps)

LOGO_PATH = Path(__file__).parent / "assets" / "cds-group-wordmark.png"

# CDS brand tokens (from thecdsgroups.com): near-black canvas, indigo->purple->pink
# accent gradient, Cabinet Grotesk/General Sans type.
CDS_CSS = """
<style>
@import url('https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@500,700&f[]=general-sans@400,500,600&display=swap');

html, body, [class*="css"] {
    font-family: 'General Sans', ui-sans-serif, system-ui, -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(60% 50% at 15% 0%, rgba(105, 95, 241, 0.25), transparent 60%),
        radial-gradient(60% 50% at 85% 15%, rgba(239, 77, 180, 0.18), transparent 60%),
        #07070D;
}

.cds-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.04);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #A7A7B4;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.cds-title {
    font-family: 'Cabinet Grotesk', 'General Sans', ui-sans-serif, sans-serif;
    font-weight: 500;
    font-size: 2.6rem;
    line-height: 1.1;
    color: #FAFAFA;
    margin-bottom: 0.5rem;
}

.cds-title .accent {
    font-style: italic;
    background: linear-gradient(135deg, #695FF1, #BC5AED, #EF4DB4);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.cds-subtitle {
    color: #A7A7B4;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

div.stButton > button {
    border-radius: 9999px !important;
    font-weight: 600;
    border: none;
}

div.stButton > button[kind="primary"] {
    background: #FAFAFA !important;
    color: #07070D !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #EAEAEA !important;
    color: #07070D !important;
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# ICE servers for WebRTC NAT traversal. Google's public STUN server is enough
# for many networks; it is NOT enough for a lot of corporate/mobile networks or
# some cloud deployments, which need a TURN relay. openrelay.metered.ca offers a
# small free public TURN service that works for demos -- see README.md for how to
# swap in your own (e.g. Twilio) TURN credentials for a production deployment.
# ---------------------------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Canonical relative positions (fraction of face box width/height) for a stylized
# "scanning" mesh -- not real per-frame landmark detection (the Haar cascade only
# gives a bounding box), but a fixed rig mapped onto that box so the overlay reads
# like a face-mesh scanner instead of a static rectangle.
_MESH_RATIOS = {
    "forehead_l": (0.20, 0.15), "forehead_m": (0.50, 0.10), "forehead_r": (0.80, 0.15),
    "temple_l": (0.08, 0.32), "temple_r": (0.92, 0.32),
    "brow_l": (0.30, 0.34), "brow_r": (0.70, 0.34),
    "eye_l": (0.32, 0.42), "eye_r": (0.68, 0.42),
    "nose_bridge": (0.50, 0.44),
    "nose_l": (0.42, 0.56), "nose_r": (0.58, 0.56), "nose_tip": (0.50, 0.59),
    "cheek_l": (0.15, 0.56), "cheek_r": (0.85, 0.56),
    "mouth_l": (0.35, 0.76), "mouth_r": (0.65, 0.76), "mouth_c": (0.50, 0.79),
    "jaw_l": (0.20, 0.86), "jaw_r": (0.80, 0.86), "chin": (0.50, 0.93),
}
_MESH_EDGES = [
    ("forehead_l", "forehead_m"), ("forehead_m", "forehead_r"),
    ("forehead_l", "temple_l"), ("forehead_r", "temple_r"),
    ("forehead_l", "brow_l"), ("forehead_r", "brow_r"),
    ("forehead_m", "brow_l"), ("forehead_m", "brow_r"),
    ("temple_l", "eye_l"), ("temple_r", "eye_r"),
    ("brow_l", "eye_l"), ("brow_r", "eye_r"),
    ("brow_l", "nose_bridge"), ("brow_r", "nose_bridge"),
    ("eye_l", "nose_bridge"), ("eye_r", "nose_bridge"),
    ("eye_l", "cheek_l"), ("eye_r", "cheek_r"),
    ("temple_l", "cheek_l"), ("temple_r", "cheek_r"),
    ("nose_bridge", "nose_l"), ("nose_bridge", "nose_r"),
    ("nose_l", "nose_tip"), ("nose_r", "nose_tip"),
    ("nose_l", "cheek_l"), ("nose_r", "cheek_r"),
    ("nose_tip", "mouth_c"),
    ("nose_l", "mouth_l"), ("nose_r", "mouth_r"),
    ("cheek_l", "mouth_l"), ("cheek_r", "mouth_r"),
    ("cheek_l", "jaw_l"), ("cheek_r", "jaw_r"),
    ("mouth_l", "mouth_c"), ("mouth_r", "mouth_c"),
    ("mouth_l", "jaw_l"), ("mouth_r", "jaw_r"),
    ("mouth_c", "chin"), ("jaw_l", "chin"), ("jaw_r", "chin"),
    ("eye_l", "eye_r"),
    ("brow_l", "nose_r"), ("brow_r", "nose_l"),
    ("mouth_l", "nose_r"), ("mouth_r", "nose_l"),
]
_MESH_PHASE = {name: i * 0.7 for i, name in enumerate(_MESH_RATIOS)}


def _draw_scan_overlay(img, bbox, collecting: bool, t: float):
    """Draws an animated face-mesh + radar-sweep overlay over the detected face
    box: wobbling landmark dots joined by mesh lines, plus a rotating arc on the
    surrounding ring, so the scan reads as "live" rather than a frozen rectangle.
    """
    fx, fy, fw, fh = bbox
    accent = (237, 90, 188) if collecting else (255, 200, 150)  # BGR

    cx, cy = fx + fw / 2, fy + fh / 2
    axes = (int(fw * 0.62), int(fh * 0.68))
    cv2.ellipse(img, (int(cx), int(cy)), axes, 0, 0, 360, accent, 1, cv2.LINE_AA)
    sweep_deg = 70
    start = (t * 140) % 360
    cv2.ellipse(img, (int(cx), int(cy)), axes, 0, start, start + sweep_deg,
                (255, 255, 255), 1, cv2.LINE_AA)

    amp = 0.012 * fw
    pts = {}
    for name, (rx, ry) in _MESH_RATIOS.items():
        phase = _MESH_PHASE[name]
        wob_x = amp * np.sin(t * 2.2 + phase)
        wob_y = amp * np.cos(t * 2.6 + phase)
        pts[name] = (int(fx + rx * fw + wob_x), int(fy + ry * fh + wob_y))

    for a, b in _MESH_EDGES:
        cv2.line(img, pts[a], pts[b], accent, 1, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, 1, (255, 255, 255), -1, cv2.LINE_AA)


class VitalsProcessor:
    """Receives webcam frames on a background WebRTC thread, tracks the face,
    samples a forehead ROI, and buffers (timestamp, R, G, B) tuples.

    All shared state is guarded by `self.lock` since `recv()` runs on a different
    thread than the main Streamlit script.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.collecting = False
        self.scan_start = None
        self.scan_duration = SCAN_SECONDS_DEFAULT
        self.samples = []  # list of (t, r, g, b)
        self._last_bbox = None
        self._stale_frames = 0
        self._frame_count = 0

    def start_scan(self, duration: int):
        with self.lock:
            self.collecting = True
            self.scan_start = time.time()
            self.scan_duration = duration
            self.samples = []

    def snapshot(self):
        """Returns (elapsed, duration, samples_copy, done)."""
        with self.lock:
            if self.scan_start is None:
                return 0.0, self.scan_duration, [], False
            elapsed = time.time() - self.scan_start
            done = elapsed >= self.scan_duration
            if done:
                self.collecting = False
            return elapsed, self.scan_duration, list(self.samples), done

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Face detection is the most expensive step -- run it every 3rd frame and
        # reuse the last known box in between to keep the stream smooth.
        self._frame_count += 1
        bbox = self._last_bbox
        if self._frame_count % 3 == 0 or bbox is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = _FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) > 0:
                # pick the largest detected face
                bbox = max(faces, key=lambda f: f[2] * f[3])
                self._last_bbox = bbox
                self._stale_frames = 0
            else:
                self._stale_frames += 1
                if self._stale_frames > 15:  # ~0.5s of no detection -> give up
                    self._last_bbox = None
                    bbox = None

        if bbox is not None:
            fx, fy, fw, fh = bbox
            # Forehead ROI: upper-center band of the detected face box
            rx0 = fx + int(0.30 * fw)
            rx1 = fx + int(0.70 * fw)
            ry0 = fy + int(0.08 * fh)
            ry1 = fy + int(0.28 * fh)
            rx0, ry0 = max(0, rx0), max(0, ry0)
            rx1, ry1 = min(w, rx1), min(h, ry1)

            with self.lock:
                collecting = self.collecting
                scan_start = self.scan_start
                duration = self.scan_duration

            if collecting and rx1 > rx0 and ry1 > ry0:
                elapsed = time.time() - scan_start
                if elapsed <= duration:
                    roi = img[ry0:ry1, rx0:rx1]
                    b_mean, g_mean, r_mean = roi.reshape(-1, 3).mean(axis=0)
                    with self.lock:
                        self.samples.append((time.time(), float(r_mean), float(g_mean), float(b_mean)))

            # Draw overlay: animated scan mesh + ring (ROI itself stays invisible)
            _draw_scan_overlay(img, bbox, collecting, time.time())
        else:
            cv2.putText(img, "Align your face in view", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def tag_for(value, ranges):
    """ranges: list of (max_value, label). Returns the label for the first bucket
    whose max_value >= value."""
    for max_v, label in ranges:
        if value <= max_v:
            return label
    return ranges[-1][1]


def render_report(res: dict):
    st.subheader("Your Vitals Report")
    quality_color = {"Good": "🟢", "Fair": "🟡", "Poor": "🔴"}[res["quality"]]
    st.caption(f"{quality_color} Signal quality: {res['quality']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Heart Rate", f"{res['heart_rate']} bpm")
        st.caption(tag_for(res["heart_rate"], [(59, "Below typical resting range"),
                                                (100, "Typical resting range"),
                                                (999, "Above typical resting range")]))
    with c2:
        if res["rmssd"] is not None:
            st.metric("HRV (RMSSD)", f"{res['rmssd']:.0f} ms")
            st.caption(tag_for(res["rmssd"], [(20, "Low"), (50, "Moderate"), (9999, "High")]))
        else:
            st.metric("HRV (RMSSD)", "—")
            st.caption("Not enough clean signal")
    with c3:
        st.metric("Respiration", f"{res['respiration_rate']} br/min")
        st.caption(tag_for(res["respiration_rate"], [(11, "Below typical range"),
                                                       (20, "Typical range"),
                                                       (999, "Above typical range")]))
    with c4:
        if res["stress"] is not None:
            st.metric("Stress Index", f"{res['stress']}/100")
            st.caption(tag_for(res["stress"], [(33, "Low"), (66, "Moderate"), (100, "High")]))
        else:
            st.metric("Stress Index", "—")
            st.caption("Not enough clean signal")

    st.metric("Estimated SpO2 (experimental)", f"{res['spo2']}%")
    st.caption("Camera-based SpO2 is not clinically valid -- ordinary webcams lack "
               "the infrared channel real pulse oximeters use. Shown for illustration only.")

    st.line_chart(res["waveform"], height=180, use_container_width=True)


def main():
    st.set_page_config(page_title="Face Vitals Scanner", page_icon="💓", layout="centered")
    st.markdown(CDS_CSS, unsafe_allow_html=True)
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), size="large")

    st.markdown(
        '<div class="cds-eyebrow">✦ rPPG · webcam-based · experimental</div>'
        '<div class="cds-title">Face Vitals <span class="accent">Scanner</span></div>'
        '<div class="cds-subtitle">Camera-based wellness estimate — heart rate, HRV, '
        "respiration & stress</div>",
        unsafe_allow_html=True,
    )

    st.warning(
        "**Not a medical device.** This tool provides experimental, informational "
        "estimates only, using ordinary webcam video. It is not clinically validated "
        "and must not be used to diagnose or monitor any health condition. If you "
        "have health concerns, please consult a doctor.",
        icon="⚠️",
    )

    if "scan_key" not in st.session_state:
        st.session_state.scan_key = 0
    if "report" not in st.session_state:
        st.session_state.report = None
    if "scan_duration" not in st.session_state:
        st.session_state.scan_duration = SCAN_SECONDS_DEFAULT

    with st.expander("How it works", expanded=st.session_state.report is None):
        st.markdown(
            "1. Click **Start** below to enable your camera, then position your "
            "face so the green box locks on.\n"
            "2. Click **Start Scan** and hold still in good, even lighting for "
            "the countdown.\n"
            "3. The app analyzes subtle color changes in your skin caused by "
            "blood flow (remote photoplethysmography, or rPPG) to estimate your "
            "vitals. All processing happens on the fly — no video is saved."
        )
        st.session_state.scan_duration = st.slider(
            "Scan duration (seconds)", min_value=15, max_value=40,
            value=st.session_state.scan_duration, step=5,
        )

    ctx = webrtc_streamer(
        key=f"vitals-{st.session_state.scan_key}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VitalsProcessor,
        media_stream_constraints={"video": {"width": 480, "height": 480}, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("▶ Start Scan", type="primary", use_container_width=True):
                ctx.video_processor.start_scan(st.session_state.scan_duration)
                st.session_state.report = None
        with col_b:
            if st.button("↻ Scan Again", use_container_width=True):
                st.session_state.report = None
                st.session_state.scan_key += 1
                st.rerun()

        status = st.empty()
        progress = st.progress(0.0)

        _poll_scan(ctx, status, progress)

    elif st.session_state.report is None:
        st.info("Click **Start** on the video widget above to enable your camera.")

    if st.session_state.report is not None:
        render_report(st.session_state.report)


@st.fragment(run_every=0.5)
def _poll_scan(ctx, status, progress):
    """Polls the video processor's buffer roughly twice a second, updates the
    progress bar, and runs the analysis exactly once when the scan window ends."""
    if not ctx.video_processor:
        return
    elapsed, duration, samples, done = ctx.video_processor.snapshot()

    if elapsed == 0.0 and not samples:
        status.caption("Align your face, then click **Start Scan**.")
        progress.progress(0.0)
        return

    frac = min(1.0, elapsed / duration)
    progress.progress(frac)

    if not done:
        status.caption(f"Hold still… capturing signal ({elapsed:.0f}s / {duration}s)")
        return

    if st.session_state.report is not None:
        return  # already analyzed this scan

    status.caption("Analyzing your pulse signal…")
    try:
        t = np.array([s[0] for s in samples])
        r = np.array([s[1] for s in samples])
        g = np.array([s[2] for s in samples])
        b = np.array([s[3] for s in samples])
        t_u, r_u, g_u, b_u = resample_uniform(t, r, g, b, TARGET_FS)
        result = analyze(t_u, r_u, g_u, b_u, TARGET_FS)
        st.session_state.report = result
        status.caption("Done — see your report below.")
    except Exception as e:
        status.error(
            f"Couldn't produce a report from this scan ({e}). Try again with "
            "better lighting, and keep your face steady and centered."
        )
    st.rerun()


if __name__ == "__main__":
    main()
