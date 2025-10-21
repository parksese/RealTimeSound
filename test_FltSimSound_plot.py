import numpy as np
import threading
import time
import sounddevice as sd
import matplotlib.pyplot as plt

# ---------------- Constants ----------------
pi = np.pi
AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 1024
AUDIO_CHANNELS = 2
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048
SPEED_OF_SOUND = 334.3
SCALING_FACTOR = 5.0
VOLUME_RAMP_SPEED = 0.02
UDP_TIMEOUT = 1.0

# ---------------- State Variables ----------------
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
volume_gain = 0.0
udp_received = threading.Event()

# fake last_state to replace UDP
last_state = {
    "spd": 20.0,
    "aos": 5.0,
    "aoa": 3.0,
    "tilt": np.zeros(NUMBER_OF_ROTORS),
    "rpm": np.full(NUMBER_OF_ROTORS, 2000.0),
    "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)]),
    "last_update_time": time.time()
}

state_lock = threading.Lock()

# ---------------- Simplified audio callback ----------------
def audio_callback(outdata, frames, time_info, status):
    global rpm_filtered, volume_gain
    if status:
        print(status)

    dt = 1.0 / AUDIO_SAMPLE_RATE
    out_buffer = np.zeros((frames, AUDIO_CHANNELS))
    t0 = time.time()

    with state_lock:
        spd = last_state["spd"]
        aos = last_state["aos"]
        rpm_target = last_state["rpm"].copy()
        tilt_all = last_state["tilt"].copy()
        azimuth_all = last_state["azimuth"]

    # --- Smooth volume gain ---
    time_since_last = t0 - last_state["last_update_time"]
    target_gain = 0.0 if time_since_last > UDP_TIMEOUT else 1.0
    volume_gain += (target_gain - volume_gain) * VOLUME_RAMP_SPEED

    for rotor_id in range(NUMBER_OF_ROTORS):
        rpm_filtered[rotor_id] += 0.02 * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        omega = rpm_filtered[rotor_id] * 2 * np.pi / 60
        if rpm_filtered[rotor_id] < 1e-2:
            continue

        n = np.arange(frames)
        azimuth_all[rotor_id] += omega * dt
        p = 0.001 * np.sin(2 * np.pi * 500 * n * dt + azimuth_all[rotor_id])
        out_buffer[:, 0] += p
        out_buffer[:, 1] += p

    out_buffer *= SCALING_FACTOR * volume_gain
    outdata[:] = out_buffer

    # store for plotting
    with state_lock:
        global waveform_latest
        waveform_latest = out_buffer[:, 0].copy()
        global current_values
        current_values = {
            "t": time.time(),
            "spd": spd,
            "tilt": tilt_all[0],
            "rpm": rpm_filtered[0],
            "aos": aos
        }

# ---------------- Real-time plot thread ----------------
def plot_thread():
    global waveform_latest, current_values
    waveform_latest = np.zeros(AUDIO_BLOCK_SIZE)
    current_values = {"t": 0, "spd": 0, "tilt": 0, "rpm": 0, "aos": 0}

    history = {"t": [], "spd": [], "tilt": [], "rpm": [], "aos": []}

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()
    labels = ["spd", "tilt", "rpm", "aos"]
    lines = {}
    for i, label in enumerate(labels):
        lines[label], = axs[i].plot([], [], label=label)
        axs[i].set_title(label)
        axs[i].set_xlim(0, 10)
        axs[i].set_ylim(-10, 100)
        axs[i].legend()

    # bottom-right panel for waveform
    fig_wave, ax_wave = plt.subplots()
    line_wave, = ax_wave.plot([], [])
    ax_wave.set_title("Waveform (recent block)")
    ax_wave.set_ylim(-1, 1)

    start_time = time.time()
    while True:
        with state_lock:
            t = current_values["t"] - start_time
            for k in ["spd", "tilt", "rpm", "aos"]:
                history[k].append(current_values[k])
            history["t"].append(t)

        if len(history["t"]) > 200:
            for key in history:
                history[key] = history[key][-200:]

        # update variable plots
        for label in labels:
            lines[label].set_data(history["t"], history[label])
            axs[labels.index(label)].set_xlim(max(0, t - 10), t)
        plt.pause(0.001)

        # update waveform
        ax_wave.cla()
        ax_wave.set_title("Waveform (recent block)")
        ax_wave.plot(waveform_latest)
        ax_wave.set_ylim(-1, 1)
        plt.pause(0.05)

# ---------------- Main ----------------
def main():
    # start plotting thread
    threading.Thread(target=plot_thread, daemon=True).start()

    print("Starting audio stream...")
    with sd.OutputStream(channels=AUDIO_CHANNELS,
                         samplerate=AUDIO_SAMPLE_RATE,
                         blocksize=AUDIO_BLOCK_SIZE,
                         callback=audio_callback):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()