import numpy as np
import sounddevice as sd
import threading
import time
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation

# ---------------- Constants ----------------
pi = np.pi
SPEED_OF_SOUND = 334.3
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BLOCK_SIZE = 1024
SCALING_FACTOR = 50.0
VOLUME_RAMP_SPEED = 0.01

latest_out_buffer = None

# ---------------- Observer & geometry ----------------
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398], [-2.554, 3.962, 0.398],
                        [4.101, -3.962, 0.667], [4.101, 3.962, 0.667]])
rotor_center = tilt_center + np.array([-0.188, 0, 1.430])
rotor_direction = np.array([-1, 1, -1, 1])

# ---------------- Simulated constant inputs ----------------
TEST_INPUT = {
    "spd": 67.0,
    "aoa": 0.0,
    "aos": 0.0,
    "rpm": np.array([168, 168, 168, 168], dtype=float),
    "coll": np.array([50, 50, 50, 50], dtype=float),
    "tilt": np.array([0, 0, 0, 0], dtype=float)
}

# ---------------- State variables ----------------
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
coll_filtered = np.zeros(NUMBER_OF_ROTORS)
azimuth = np.array([2*pi*(sid%NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)])
volume_gain = 0.0

# ---------------- Plot / window parameters ----------------
PARAM_WINDOW_SEC = 5.0          # seconds for spd, tilt, rpm, aos rolling plot
WAVEFORM_WINDOW_SEC = 5.0       # seconds for waveform rolling plot
PLOT_UPDATE_INTERVAL = 0.2      # seconds

# Buffers for plotting
param_len = int(PARAM_WINDOW_SEC * AUDIO_SAMPLE_RATE / AUDIO_BLOCK_SIZE)
wave_len = int(WAVEFORM_WINDOW_SEC * AUDIO_SAMPLE_RATE)
plot_data = {
    "time": deque(maxlen=param_len),
    "spd": deque(maxlen=param_len),
    "tilt": deque(maxlen=param_len),
    "rpm": deque(maxlen=param_len),
    "aos": deque(maxlen=param_len),
    "wave": deque(maxlen=wave_len)
}

# ---------------- Audio Callback ----------------
def audio_callback(outdata, frames, time_info, status):
    global rpm_filtered, coll_filtered, azimuth, volume_gain
    global latest_out_buffer
    
    if status:
        print(status)
    
    out_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)
    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)
    
    spd = TEST_INPUT["spd"]
    aos = TEST_INPUT["aos"]
    aoa = TEST_INPUT["aoa"]
    tilt = TEST_INPUT["tilt"]
    rpm_target = TEST_INPUT["rpm"]
    coll_target = TEST_INPUT["collective"]
    
    volume_gain += (1.0 - volume_gain) * VOLUME_RAMP_SPEED

    for rotor_id in range(NUMBER_OF_ROTORS):
        rpm_filtered[rotor_id] += 0.02 * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        coll_filtered[rotor_id] += 0.02 * (coll_target[rotor_id] - coll_filtered[rotor_id])
        omega = rpm_filtered[rotor_id] * 2 * np.pi / 60
        if rpm_filtered[rotor_id] < 1e-2:
            continue
        c = lookup.get_coefficients(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        a0, a1, b1, a2, b2 = c["a0"], 
        tilt_rad = 
        trans_tilt = 

        # Simple sinusoidal lift model for demonstration
        for blade in range(NUMBER_OF_BLADES):
            source_id = rotor_id * NUMBER_OF_BLADES + blade
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n
            azimuth[source_id] = az_block[-1]


            if rotor_direction[rotor_id] == 1:
                        L = (a0)
            else:
                        L = (a0)

            # Source position (before tilt)
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = rotor_center[rotor_id][2] * np.ones_like(x)
            source_position = np.stack((x, y, z), axis=1)
            
            r = observer_position - source_position
            rmag = np.linalg.norm(r, axis=1)

            # Mach vector
            M = 
            Mi = 
            Mi = 

            # Force vector

            # Dot products

            # Pressure
            p_near = 
            # subtract
            p_near -= np.mean(p_near)

            out_buffer[:, 0] += p_near #* volume_gain
            out_buffer[:, 1] += p_near #* volume_gain

    out_buffer *= SCALING_FACTOR
    outdata[:] = out_buffer

    # store last few seconds for plotting
    latest_out_buffer = np.copy(out_buffer)
    plot_data["wave"].extend(out_buffer[:,0])

    # update param buffers
    plot_data["spd"].append(spd)
    plot_data["tilt"].append(tilt[0])
    plot_data["rpm"].append(np.mean(rpm_filtered))
    plot_data["aos"].append(aos)
    plot_data["time"].append(time.time() % PARAM_WINDOW_SEC)

# ---------------- Plotting ----------------
def start_plots():
    fig1, axs1 = plt.subplots(2,2, figsize=(10,6))
    axs1[0,0].set_title("Speed"); axs1[0,1].set_title("Tilt")
    axs1[1,0].set_title("RPM");   axs1[1,1].set_title("AOS")
    lines1 = [axs1[0,0].plot([],[])[0],
              axs1[0,1].plot([],[])[0],
              axs1[1,0].plot([],[])[0],
              axs1[1,1].plot([],[])[0]]

    fig2, axs2 = plt.subplots(2,1, figsize=(10,6))
    axs2[0].set_title("Waveform"); axs2[1].set_title("FFT")
    line_wave = axs2[0].plot([],[])[0]
    line_fft = axs2[1].plot([],[])[0]

    def update(frame):
        t_vals = list(plot_data["time"])
        for k, key in enumerate(["spd","tilt","rpm","aos"]):
            lines1[k].set_data(t_vals, list(plot_data[key]))
            axs1[k//2,k%2].set_xlim(max(0, t_vals[0]), t_vals[-1]+1e-6)
            axs1[k//2,k%2].set_ylim(min(plot_data[key])-1, max(plot_data[key])+1)
        axs1[0,0].set_ylim(-10, 100)        
        axs1[0,1].set_ylim(-10, 100)
        axs1[1,0].set_ylim(0,500)
        axs1[1,1].set_ylim(-20,20)

        # waveform
        wave = np.array(plot_data["wave"])
        N = len(wave)
        t_wave = np.linspace(0, WAVEFORM_WINDOW_SEC, N)
        line_wave.set_data(t_wave, wave)
        axs2[0].set_xlim(0, WAVEFORM_WINDOW_SEC)
        axs2[0].set_ylim(np.min(wave), np.max(wave)+1e-6)

        # FFT
        if N>0:
            freqs = np.fft.rfftfreq(N, 1/AUDIO_SAMPLE_RATE)
            spectrum = np.abs(np.fft.rfft(wave))
            line_fft.set_data(freqs, 20*np.log10(spectrum+1e-12))
            axs2[1].set_xlim(20, AUDIO_SAMPLE_RATE/2)
            axs2[1].set_ylim(np.min(20*np.log10(spectrum+1e-12)), np.max(20*np.log10(spectrum+1e-12))+1)
        return lines1+ [line_wave, line_fft]

    ani1 = FuncAnimation(fig1, update, interval=int(PLOT_UPDATE_INTERVAL*1000))
    ani2 = FuncAnimation(fig2, update, interval=int(PLOT_UPDATE_INTERVAL*1000))
    plt.show()

# ---------------- Main ----------------
def main():
    print("Starting sound stream...")
    with sd.OutputStream(
        channels=AUDIO_CHANNELS,
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    ):
        start_plots()

if __name__ == "__main__":
    main()