# Face Vitals Scanner (Streamlit)

Estimates heart rate, HRV, respiration rate, an experimental stress index, and an
experimental SpO2 reading from webcam video using remote photoplethysmography (rPPG).
Camera access happens live in the browser via WebRTC — no video is saved or uploaded.

**Not a medical device.** Estimates only, not clinically validated. See the in-app
disclaimer.

## Files

- `app.py` — Streamlit UI, WebRTC video capture, OpenCV face tracking
- `vitals.py` — the signal-processing core (CHROM pulse extraction, FFT-based heart
  rate / respiration, HRV, stress & SpO2 heuristics). Kept separate so it can be
  unit-tested independently of the browser/camera stack.
- `requirements.txt` — Python dependencies
- `packages.txt` — apt packages Streamlit Community Cloud needs for OpenCV/av

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, click **Start** on the video widget to grant camera
access, align your face, then click **Start Scan**.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch, and `app.py`.
4. Deploy. Streamlit Cloud reads `requirements.txt` and `packages.txt` automatically.

Streamlit Cloud serves over HTTPS by default, which is required for browser camera
access (`getUserMedia`) to work at all.

### About the TURN server (important for reliable camera streaming)

`streamlit-webrtc` streams your camera from the browser to the server over WebRTC.
That requires NAT traversal: a STUN server is often enough on an open home network, but
many corporate networks, mobile networks, and cloud hosts need a **TURN** relay or the
connection will fail to establish ("stuck on connecting").

`app.py` ships with a small free public TURN service (openrelay.metered.ca) as a
fallback alongside Google's STUN server, so the demo works out of the box for most
people. For a production deployment, replace it with your own TURN credentials
(e.g. [Twilio's Network Traversal Service](https://www.twilio.com/docs/stun-turn),
or a self-hosted [coturn](https://github.com/coturn/coturn)) — free public TURN
services can be slow or rate-limited. Update the `RTC_CONFIGURATION` dict near the
top of `app.py`, or better, load credentials from `st.secrets` so they aren't
committed to the repo.

## How it works

1. OpenCV's Haar cascade detects your face in each frame; a forehead region is
   sampled for its average red/green/blue pixel values.
2. Over a ~25 second scan, those RGB traces are combined using the **CHROM** method
   (de Haan & Jeanne, 2013), which is more robust to lighting and small motion than
   using a single color channel.
3. Heart rate = the dominant frequency of that combined pulse signal (via FFT) in
   the 42–240 bpm range.
4. HRV (RMSSD/SDNN) comes from beat-to-beat peak timing, but is only shown when
   there's enough clean signal to trust it — camera-based beat timing is noisier
   than an ECG, so a shaky reading is reported as "not enough signal" rather than
   as a confident wrong number.
5. Respiration rate = the dominant low-frequency (6–30 breaths/min) modulation of
   the green channel.
6. Stress index and SpO2 are explicitly-labeled heuristic estimates, not clinical
   measurements — true SpO2 requires an infrared sensor that ordinary webcams don't have.

## Validation

`vitals.py`'s math was checked against synthetic signals with known heart rate and
respiration rate injected under varying noise levels — heart rate and respiration
rate estimates land within about 1 bpm on average, and the HRV/quality gating
correctly suppresses unreliable readings under high-noise conditions. This confirms
the algorithm is implemented correctly; it does not guarantee accuracy against real
human skin/lighting, which varies far more than any synthetic test.
