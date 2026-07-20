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

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from vitals import analyze, resample_uniform

SCAN_SECONDS_DEFAULT = 25
TARGET_FS = 20  # samples/sec we resample onto for analysis (webcam delivers ~15-30fps)

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

            # Draw overlay: face box + ROI + status text
            color = (80, 209, 197) if collecting else (124, 156, 255)  # BGR-ish accent
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (255, 200, 0), 2)
            label = "Scanning..." if collecting else "Face detected - ready"
            cv2.putText(img, label, (fx, max(20, fy - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
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
    st.title("💓 Face Vitals Scanner")
    st.caption("Camera-based wellness estimate — heart rate, HRV, respiration & stress")

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
