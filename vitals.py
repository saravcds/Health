"""
Core signal-processing for camera-based (rPPG) vitals estimation.

Kept separate from app.py so the math can be unit-tested with synthetic
signals independent of Streamlit / OpenCV / WebRTC.

Method summary
--------------
- Face is located each frame (Haar cascade); a forehead ROI is sampled.
- CHROM method (de Haan & Jeanne, 2013) combines the R/G/B channel traces
  into a single pulse signal that is more robust to lighting/motion than
  any single channel alone.
- Heart rate = dominant frequency of the pulse signal's FFT in the
  0.7-4.0 Hz band (42-240 bpm).
- HRV (RMSSD/SDNN) = derived from beat-to-beat peak timing, but only
  reported when enough beats are found AND the result is physiologically
  plausible -- camera-based beat timing is noisier than ECG, so an
  implausible reading is treated as low-confidence rather than shown.
- Respiration rate = dominant frequency of the low-frequency modulation
  of the (non-CHROM) green channel in the 0.1-0.5 Hz band (6-30 breaths/min).
- Stress index and SpO2 are heuristic, explicitly experimental estimates,
  not clinically validated values.

None of this is a medical device. It is a wellness/informational demo.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

HR_BAND = (0.7, 4.0)       # Hz -> 42-240 bpm
RESP_BAND = (0.1, 0.5)     # Hz -> 6-30 breaths/min


def resample_uniform(t: np.ndarray, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float):
    """Linearly interpolate irregularly-timed webcam samples onto a uniform time grid
    at rate `fs`. Webcam frame arrival times jitter slightly even at a nominal frame
    rate, and the FFT-based analysis assumes uniform sampling."""
    t = np.asarray(t, dtype=float)
    duration = t[-1] - t[0]
    n = int(duration * fs)
    if n < 2:
        raise ValueError("insufficient samples for resampling")
    t_uniform = t[0] + np.arange(n) / fs
    r_u = np.interp(t_uniform, t, r)
    g_u = np.interp(t_uniform, t, g)
    b_u = np.interp(t_uniform, t, b)
    return t_uniform, r_u, g_u, b_u


def moving_average_detrend(x: np.ndarray, window: int) -> np.ndarray:
    """Subtract a centered moving average to remove slow drift (lighting, motion)."""
    window = max(1, int(window))
    n = len(x)
    cumsum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        out[i] = x[i] - (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def bandpass(signal: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    """FFT brick-wall bandpass. Applies a Tukey window first to reduce spectral
    leakage from analyzing a short, non-periodic segment (important: without this,
    beat-to-beat timing used for HRV becomes noisy even on a clean signal)."""
    n = len(signal)
    win = tukey(n, alpha=0.2)
    spec = np.fft.rfft(signal * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec[(freqs < lo_hz) | (freqs > hi_hz)] = 0
    return np.fft.irfft(spec, n=n)


def dominant_frequency(signal: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> dict:
    """Return the strongest frequency component within [lo_hz, hi_hz], plus a
    signal-to-noise proxy (peak magnitude / total in-band magnitude)."""
    n = len(signal)
    win = tukey(n, alpha=0.2)
    centered = signal - np.mean(signal)
    spec = np.abs(np.fft.rfft(centered * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band_mask):
        return {"freq_hz": 0.0, "magnitude": 0.0, "band_energy": 0.0}
    band_freqs = freqs[band_mask]
    band_mags = spec[band_mask]
    best_idx = int(np.argmax(band_mags))
    return {
        "freq_hz": float(band_freqs[best_idx]),
        "magnitude": float(band_mags[best_idx]),
        "band_energy": float(np.sum(band_mags)),
    }


def chrom_pulse(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> np.ndarray:
    """CHROM pulse extraction. Channels are normalized by dividing by their mean
    (DC level) only -- NOT full z-scoring. Z-scoring would rescale each channel to
    unit variance and destroy the relative pulsatility ratios between channels that
    CHROM's fixed coefficients (3,-2 / 1.5,1,-1.5) depend on."""
    win_samples = round(fs * 2)  # ~2s detrend window
    r_mean, g_mean, b_mean = np.mean(r) or 1, np.mean(g) or 1, np.mean(b) or 1
    rd = moving_average_detrend(r, win_samples) / r_mean
    gd = moving_average_detrend(g, win_samples) / g_mean
    bd = moving_average_detrend(b, win_samples) / b_mean

    xs = 3 * rd - 2 * gd
    ys = 1.5 * rd + gd - 1.5 * bd

    xf = bandpass(xs, fs, *HR_BAND)
    yf = bandpass(ys, fs, *HR_BAND)
    alpha = (np.std(xf) / np.std(yf)) if np.std(yf) > 1e-6 else 1.0
    return xf - alpha * yf


def analyze(t: np.ndarray, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> dict:
    """Run the full pipeline on a uniformly-sampled window of ROI color traces.

    Parameters
    ----------
    t, r, g, b : 1-D arrays of equal length, uniformly sampled at rate `fs` (Hz).
    """
    n = len(r)
    if n < fs * 8:
        raise ValueError("insufficient samples for analysis (need >= 8s of data)")

    # Motion/quality proxy: frame-to-frame jitter on the raw green channel
    jitter = float(np.mean(np.abs(np.diff(g))))

    pulse = chrom_pulse(r, g, b, fs)

    hr_est = dominant_frequency(pulse, fs, *HR_BAND)
    heart_rate = round(hr_est["freq_hz"] * 60)

    # Beat-to-beat peaks for HRV. Minimum spacing is derived from the already-detected
    # heart rate so noise-driven sub-cycle ripple isn't miscounted as extra beats.
    max_bpm = min(220, heart_rate * 1.6) if heart_rate > 0 else 200
    min_dist = max(1, round(fs * 60 / max_bpm))
    threshold = np.std(pulse) * 0.35
    peaks, _ = find_peaks(pulse, distance=min_dist, height=threshold)

    ibis = np.diff(peaks) / fs * 1000.0  # ms
    ibis = ibis[(ibis >= 300) & (ibis <= 1500)]

    rmssd = sdnn = None
    if len(ibis) >= 8:
        diffs = np.diff(ibis)
        rmssd_candidate = float(np.sqrt(np.mean(diffs ** 2)))
        sdnn_candidate = float(np.std(ibis))
        # Camera-derived beat timing is noisier than ECG; treat implausible spreads
        # (uncommon in a healthy resting adult) as low-confidence rather than showing them.
        if rmssd_candidate <= 150 and sdnn_candidate <= 150:
            rmssd, sdnn = rmssd_candidate, sdnn_candidate

    # Respiration: low-frequency modulation of the raw (non-CHROM) green channel
    g_for_resp = moving_average_detrend(g, round(fs * 8))
    resp_est = dominant_frequency(g_for_resp, fs, *RESP_BAND)
    respiration_rate = round(resp_est["freq_hz"] * 60)

    # Stress index heuristic (NOT clinical): lower HRV + elevated HR -> higher score
    stress = None
    if rmssd is not None:
        hrv_component = np.clip(1 - (rmssd / 80), 0, 1)  # 80ms treated as a relaxed reference
        hr_component = np.clip((heart_rate - 60) / 60, 0, 1)
        stress = round(float(np.clip(0.7 * hrv_component + 0.3 * hr_component, 0, 1) * 100))

    # Experimental SpO2 heuristic: ratio-of-ratios using red & blue AC/DC as an
    # unvalidated proxy for red/infrared. Ordinary RGB webcams cannot measure true
    # SpO2 (that requires an infrared channel) -- this is illustrative only.
    win_samples = round(fs * 2)

    def ac_dc(raw, detrended_band):
        return np.std(detrended_band) / (np.mean(raw) or 1)

    red_ac_dc = ac_dc(r, bandpass(moving_average_detrend(r, win_samples), fs, *HR_BAND))
    blue_ac_dc = ac_dc(b, bandpass(moving_average_detrend(b, win_samples), fs, *HR_BAND))
    ratio = (red_ac_dc / blue_ac_dc) if blue_ac_dc > 1e-9 else 1.0
    spo2 = int(np.clip(round(110 - 17 * ratio), 90, 100))

    # Signal quality
    snr = (hr_est["magnitude"] / hr_est["band_energy"]) if hr_est["band_energy"] > 0 else 0
    if snr > 0.35 and jitter < 6:
        quality = "Good"
    elif snr > 0.18 and jitter < 12:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "heart_rate": heart_rate,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "respiration_rate": respiration_rate,
        "stress": stress,
        "spo2": spo2,
        "quality": quality,
        "waveform": pulse,
        "fs": fs,
    }
