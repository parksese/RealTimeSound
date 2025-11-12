# ==============================================================
# Stand-alone Broadband Noise Generator (ready for integration)
# ==============================================================

import numpy as np
from scipy.signal import butter, sosfilt

def get_one_third_octave_bands():
    """Return array of [f_low, f_center, f_high] for standard 1/3-octave bands."""
    center = np.array([
        12.5, 16, 20, 25, 31.5, 40, 50, 62.5, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
        4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])
    lower = center / (2 ** (1/6))
    upper = center * (2 ** (1/6))
    return np.vstack((lower, center, upper)).T


def synthesize_broadband_block(
    one_third_freqs,
    spl_one_third,
    fs,
    n_samples,
    seed=None,
):
    """
    Generate broadband acoustic pressure block from 1/3-octave spectrum.

    Parameters
    ----------
    one_third_freqs : array-like
        Center frequencies of 1/3-oct bands [Hz]
    spl_one_third : array-like
        Corresponding SPLs [dB re 20 µPa]
    fs : int
        Sample rate [Hz]
    n_samples : int
        Number of samples in output block
    seed : int or None
        RNG seed for repeatability

    Returns
    -------
    pressure : ndarray (n_samples,)
        Synthesized acoustic pressure [Pa]
    """

    rng = np.random.default_rng(seed)
    p0 = 20e-6
    t = np.arange(n_samples) / fs
    pressure = np.zeros_like(t, dtype=np.float32)

    for fc, spl in zip(one_third_freqs, spl_one_third):
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        if f_high >= fs / 2:
            continue

        # white noise -> band-pass
        white = rng.standard_normal(len(t)).astype(np.float32)
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        band_noise = sosfilt(sos, white)

        # scale RMS to SPL target
        p_rms_target = p0 * 10 ** (spl / 20.0)
        p_rms_current = np.sqrt(np.mean(band_noise ** 2)) + 1e-12
        band_noise *= (p_rms_target / p_rms_current)

        pressure += band_noise

    return pressure


def compute_rotor_broadband_spectrum(rpm, thrust_n, cl_bar):
    """
    Very simple empirical rotor broadband SPL spectrum
    (placeholder — replace with your physical model later).
    Returns 1/3-oct band center frequencies and SPL values.
    """
    bands = get_one_third_octave_bands()
    f_peak = 240 * np.log10(thrust_n + 1e-3) + 2.448 * (rpm * 2*np.pi/60 * 3.048 / 334.3) + 942.0
    # find closest band
    idx = np.argmin(np.abs(bands[:, 1] - f_peak))
    f_center = bands[idx, 1]
    spl_deltas = np.array([-29.0, -24.5, -19.5, -15.3, -11.7, -7.5,
                           -11.5, -12.1, -16.5, -17.0, -21.8, -26.4, -30.0])
    one_third_freqs = [f_center * (2 ** i) for i in range(-5, 8)]
    # scaling with Mach^3 and lift coefficient term (Lawson-type logic)
    M = (rpm * 2*np.pi/60 * 3.048) / 334.3
    term1 = 20 * np.log10(M ** 3.0)
    term2 = 130.0  # baseline offset (arbitrary reference)
    if cl_bar <= 0.48:
        f_cl = 10.0 * np.log10(cl_bar / 0.4)
    else:
        f_cl = 0.8 + 80.0 * np.log10(cl_bar / 0.48)
    spl_one_third = term1 + term2 + f_cl + spl_deltas
    return np.array(one_third_freqs), np.array(spl_one_third)


# ==============================================================
# Stand-alone test (press Ctrl+C to stop)
# ==============================================================

if __name__ == "__main__":
    import sounddevice as sd
    import time

    fs = 22050
    block_size = 4096

    # example: single rotor, constant parameters
    rpm = 470
    thrust_n = 1500.0
    cl_bar = 0.45

    one_third_freqs, spl_one_third = compute_rotor_broadband_spectrum(rpm, thrust_n, cl_bar)

    def callback(outdata, frames, time_info, status):
        pressure = synthesize_broadband_block(one_third_freqs, spl_one_third, fs, frames)
        # simple stereo duplication
        outdata[:] = np.column_stack((pressure, pressure)).astype(np.float32)

    print("Playing broadband rotor noise — press Ctrl+C to stop.")
    with sd.OutputStream(
        channels=2,
        samplerate=fs,
        blocksize=block_size,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)