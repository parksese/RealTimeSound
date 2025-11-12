# ============================================================
#  Standalone Broadband Noise Test + Plot (ready for merge)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq

# ------------------ Helper functions ------------------
def get_one_third_octave_bands():
    center = np.array([
        12.5, 16, 20, 25, 31.5, 40, 50, 62.5, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
        4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])
    lower = center / (2 ** (1/6))
    upper = center * (2 ** (1/6))
    return np.vstack((lower, center, upper)).T


def synthesize_broadband_block(one_third_freqs, spl_one_third, fs, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    p0 = 20e-6
    t = np.arange(n_samples) / fs
    pressure = np.zeros_like(t, dtype=np.float32)

    for fc, spl in zip(one_third_freqs, spl_one_third):
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        if f_high >= fs / 2:
            continue

        white = rng.standard_normal(len(t)).astype(np.float32)
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        band_noise = sosfilt(sos, white)

        # Scale RMS
        p_rms_target = p0 * 10 ** (spl / 20.0)
        p_rms_current = np.sqrt(np.mean(band_noise ** 2)) + 1e-12
        band_noise *= (p_rms_target / p_rms_current)
        pressure += band_noise

    return t, pressure


def compute_rotor_broadband_spectrum(rpm, thrust_n, cl_bar):
    bands = get_one_third_octave_bands()
    f_peak = 240 * np.log10(thrust_n + 1e-3) + 2.448 * (rpm * 2*np.pi/60 * 3.048 / 334.3) + 942.0
    idx = np.argmin(np.abs(bands[:, 1] - f_peak))
    f_center = bands[idx, 1]
    spl_deltas = np.array([-29.0, -24.5, -19.5, -15.3, -11.7, -7.5,
                           -11.5, -12.1, -16.5, -17.0, -21.8, -26.4, -30.0])
    one_third_freqs = [f_center * (2 ** i) for i in range(-5, 8)]
    M = (rpm * 2*np.pi/60 * 3.048) / 334.3
    term1 = 20 * np.log10(M ** 3.0)
    term2 = 130.0
    if cl_bar <= 0.48:
        f_cl = 10.0 * np.log10(cl_bar / 0.4)
    else:
        f_cl = 0.8 + 80.0 * np.log10(cl_bar / 0.48)
    spl_one_third = term1 + term2 + f_cl + spl_deltas
    return np.array(one_third_freqs), np.array(spl_one_third)


def compute_spl_db(pressure_signal, pref=20e-6):
    p_rms = np.sqrt(np.mean(pressure_signal ** 2))
    return 20 * np.log10(p_rms / pref + 1e-20)


# ------------------ Test + Plot ------------------
if __name__ == "__main__":
    fs = 22050
    duration = 2.0
    n_samples = int(fs * duration)

    # Example parameters
    rpm = 470
    thrust_n = 1500.0
    cl_bar = 0.45

    # Compute 1/3-octave spectrum and synthesize pressure
    one_third_freqs, spl_one_third = compute_rotor_broadband_spectrum(rpm, thrust_n, cl_bar)
    t, pressure = synthesize_broadband_block(one_third_freqs, spl_one_third, fs, n_samples)

    # Compute FFT and OASPL
    freqs = rfftfreq(n_samples, 1/fs)
    spectrum = np.abs(rfft(pressure))
    oaspl = compute_spl_db(pressure)
    print(f"Overall SPL (OASPL): {oaspl:.2f} dB re 20µPa")

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)

    # --- Time domain ---
    axs[0].plot(t, pressure, lw=0.7)
    axs[0].set_title("Broadband Pressure Time Series")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Pressure [Pa]")
    axs[0].grid(True)

    # --- Frequency domain ---
    axs[1].semilogx(freqs, 20*np.log10(spectrum/np.max(spectrum) + 1e-12))
    axs[1].set_xlim(20, fs/2)
    axs[1].set_ylim(-60, 0)
    axs[1].set_title("Normalized Spectrum")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Level [dB]")
    axs[1].grid(True, which="both", ls="--", alpha=0.6)

    # --- SPL bar chart per band ---
    axs[2].bar(one_third_freqs, spl_one_third, width=np.array(one_third_freqs)*0.2, align='center')
    axs[2].set_xscale('log')
    axs[2].set_xlim(50, 10000)
    axs[2].set_ylim(min(spl_one_third)-5, max(spl_one_third)+5)
    axs[2].set_title(f"1/3-Octave Spectrum (OASPL = {oaspl:.1f} dB)")
    axs[2].set_xlabel("Center Frequency [Hz]")
    axs[2].set_ylabel("SPL [dB]")
    axs[2].grid(True, which="both", ls="--", alpha=0.6)

    plt.show()