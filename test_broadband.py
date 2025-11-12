# ============================================================
#  Real-time Broadband Noise Streaming + Live Plot
# ============================================================

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq
import threading
import time
from collections import deque

# ------------------ Constants ------------------
p0 = 20e-6
fs = 22050
blocksize = 4096
channels = 2
duration = 2.0        # seconds for time window in plot
N_window = int(fs * duration)

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


def synthesize_broadband_block(one_third_freqs, spl_one_third, fs, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    pressure = np.zeros(n_samples, dtype=np.float32)
    for fc, spl in zip(one_third_freqs, spl_one_third):
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        if f_high >= fs/2:
            continue
        white = rng.standard_normal(n_samples).astype(np.float32)
        sos = butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        band_noise = sosfilt(sos, white)
        p_rms_target = p0 * 10 ** (spl / 20.0)
        p_rms_current = np.sqrt(np.mean(band_noise**2)) + 1e-12
        band_noise *= (p_rms_target / p_rms_current)
        pressure += band_noise
    return pressure


def compute_spl_db(p):
    return 20 * np.log10(np.sqrt(np.mean(p**2)) / p0 + 1e-20)


# ------------------ Live Data Buffers ------------------
wave_buffer = deque(maxlen=N_window)
freqs = rfftfreq(blocksize, 1/fs)
spl_spectrum = np.zeros_like(freqs)
spl_1_3 = None
one_third_freqs = None
oaspl = 0.0
lock = threading.Lock()

# ------------------ Parameters ------------------
rpm = 470
thrust_n = 1500.0
cl_bar = 0.45

# Precompute broadband spectrum
one_third_freqs, spl_1_3 = compute_rotor_broadband_spectrum(rpm, thrust_n, cl_bar)

# ------------------ Audio Callback ------------------
def audio_callback(outdata, frames, time_info, status):
    global oaspl, spl_spectrum
    if status:
        print(status)
    pressure = synthesize_broadband_block(one_third_freqs, spl_1_3, fs, frames)
    # compute SPL for this block
    oaspl = compute_spl_db(pressure)

    # FFT spectrum for display
    spectrum = np.abs(rfft(pressure)) / frames
    psd = (spectrum ** 2) * 2 / (fs / frames)
    spl_spectrum = 10 * np.log10(psd / (p0 ** 2) + 1e-30)

    # store latest data for plotting
    with lock:
        wave_buffer.extend(pressure)

    # stereo output
    outdata[:] = np.column_stack((pressure, pressure))


# ------------------ Plotting Thread ------------------
def start_plotting():
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)
    axs[0].set_title("Broadband Pressure Time Series")
    axs[1].set_title("SPL Spectrum (FFT)")
    axs[2].set_title("1/3-Octave Spectrum")

    line_wave, = axs[0].plot([], [], lw=0.7, color='tab:blue')
    line_fft, = axs[1].semilogx([], [], lw=1.0, color='tab:blue')
    bars = axs[2].bar(one_third_freqs, spl_1_3, width=np.array(one_third_freqs)*0.25, color='tab:orange', alpha=0.8)

    axs[0].set_xlim(0, duration)
    axs[0].set_ylim(-0.2, 0.2)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Pressure [Pa]")
    axs[0].grid(True)

    axs[1].set_xlim(20, fs/2)
    axs[1].set_ylim(40, 140)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("SPL [dB re 20µPa]")
    axs[1].grid(True, which='both', ls='--', alpha=0.6)

    axs[2].set_xscale('log')
    axs[2].set_xlim(50, 10000)
    axs[2].set_ylim(40, 140)
    axs[2].set_xlabel("Center Frequency [Hz]")
    axs[2].set_ylabel("SPL [dB re 20µPa]")
    axs[2].grid(True, which='both', ls='--', alpha=0.6)

    def update(frame):
        with lock:
            wave = np.array(wave_buffer)
            spl_fft = spl_spectrum.copy()
        if len(wave) > 0:
            t_wave = np.linspace(0, duration, len(wave))
            line_wave.set_data(t_wave, wave)
            axs[0].set_ylim(wave.min()*1.2, wave.max()*1.2)
        line_fft.set_data(freqs, spl_fft)
        axs[1].set_ylim(min(spl_fft)-5, max(spl_fft)+5)
        axs[2].set_title(f"1/3-Octave Spectrum (OASPL = {oaspl:.1f} dB)")
        return [line_wave, line_fft] + list(bars)

    ani = FuncAnimation(fig, update, interval=200, blit=False)
    plt.show()


# ------------------ Main ------------------
def main():
    print("Starting broadband noise streaming + plotting...")
    threading.Thread(target=start_plotting, daemon=True).start()
    with sd.OutputStream(
        channels=channels,
        samplerate=fs,
        blocksize=blocksize,
        callback=audio_callback,
    ):
        while plt.get_fignums():
            time.sleep(0.1)

if __name__ == "__main__":
    main()