import numpy as np
import sounddevice as sd
import threading
import time
import matplotlib
matplotlib.use("TkAgg")  # safer on Windows on many setups
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.io import wavfile

# ================== Constants ==================
pi = np.pi
SPEED_OF_SOUND = 334.3
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048

AUDIO_SAMPLE_RATE = 22050
AUDIO_CHANNELS = 2
AUDIO_BLOCK_SIZE = 4096

ROTOR_SCALING_FACTOR = 5.0      # rotor noise overall gain
WIND_BASE_GAIN = 0.3            # base scaling for wind noise (tune this)
WIND_FADE_SAMPLES = 4096        # crossfade length for loop

VOLUME_RAMP_SPEED = 0.01

volume_gain = 0.0

# ================== Observer & geometry ==================
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([
    [-2.554, -3.962, 0.398],
    [-2.554,  3.962, 0.398],
    [ 4.101, -3.962, 0.667],
    [ 4.101,  3.962, 0.667]
])
rotor_center = tilt_center + np.array([-0.188, 0, 1.430])
rotor_direction = np.array([-1, 1, -1, 1])

# ================== Simulated test input ==================
TEST_INPUT = {
    "rpm": np.array([467, 467, 467, 467], dtype=float),
    "coll": np.array([50, 50, 50, 50], dtype=float),
    "tilt": np.array([70, 70, 70, 70], dtype=float),
    "spd": 46.9,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([
        2 * pi * (sid % NUMBER_OF_BLADES) / NUMBER_OF_BLADES
        for sid in range(NUMBER_OF_SOURCES)
    ]),
    "last_update_time": time.time(),
}

# ================== Filter setup ==================
F_LOW = 20
F_HIGH = 1000
FILTER_ORDER = 6
FILTER_ENABLED = False
sos = butter(FILTER_ORDER, [F_LOW, F_HIGH], btype='bandstop',
             fs=AUDIO_SAMPLE_RATE, output='sos')
zi = np.zeros((sos.shape[0], 2))
filter_lock = threading.Lock()

# ================== Plot buffers ==================
PARAM_WINDOW_SEC = 2.0
WAVEFORM_WINDOW_SEC = 2.0
PLOT_UPDATE_INTERVAL = 0.2

param_len = int(PARAM_WINDOW_SEC * AUDIO_SAMPLE_RATE / AUDIO_BLOCK_SIZE)
wave_len = int(WAVEFORM_WINDOW_SEC * AUDIO_SAMPLE_RATE)

plot_data = {
    "time": deque(maxlen=param_len),
    "spd": deque(maxlen=param_len),
    "tilt": deque(maxlen=param_len),
    "coll": deque(maxlen=param_len),
    "rpm": deque(maxlen=param_len),
    "aoa": deque(maxlen=param_len),
    "aos": deque(maxlen=param_len),
    "wave": deque(maxlen=wave_len),
    "block_time": deque(maxlen=param_len),
    "block_mean": deque(maxlen=param_len),
    "spl_db": deque(maxlen=param_len),
}

# ================== State variables ==================
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
coll_filtered = np.zeros(NUMBER_OF_ROTORS)
azimuth = np.array([
    2 * pi * (sid % NUMBER_OF_BLADES) / NUMBER_OF_BLADES
    for sid in range(NUMBER_OF_SOURCES)
])

# ================== TableLookup ==================
class TableLookup:
    def __init__(self):
        # (Same table content as in your latest version)
        table = np.array([
           #  spd tilt aoa aos coll   a0         a1      b1        a2       b2
            [ 0.0, 90, -24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90, -16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  -8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   0,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 5.0, 90, -24,   0, 15,  2019.10, -159.62, 330.10,    45.05,    2.67],
            [ 5.0, 90, -16,   0, 15,  2018.86, -159.24, 373.12,    58.96,  -16.87],
            [ 5.0, 90,  -8,   0, 15,  2043.73, -183.79, 398.20,    70.20,   -7.37],
            [ 5.0, 90,   0,   0, 15,  2065.45, -203.84, 402.33,    67.18,   -1.95],
            [ 5.0, 90,   8,   0, 15,  2076.82, -218.69, 404.37,    73.34,    6.24],
            [ 5.0, 90,  16,   0, 15,  2083.48, -224.29, 403.09,    91.48,   -3.60],
            [ 5.0, 90,  24,   0, 15,  2097.72, -228.09, 374.76,   100.52,   -1.42],
            [46.9, 70, -24,   0, 15,    89.87, -251.46, 117.25, -302.04, -123.60],
            [46.9, 70, -16,   0, 15,   649.22, -349.74, 468.23, -404.10, -169.95],
            [46.9, 70,  -8,   0, 15,  1261.38, -463.84, 909.35, -475.91, -220.62],
            [46.9, 70,   0,   0, 15,  1873.06, -549.95, 279.70, -528.09, -251.81],
            [46.9, 70,   8,   0, 15,  2352.58, -503.41, 464.50, -501.43, -145.07],
            [46.9, 70,  16,   0, 15,  2666.80, -272.75, 490.22, -435.59,   19.22],
            [46.9, 70,  24,   0, 15,  2857.03,  -49.74, 296.69, -280.00,   39.44],
            [49.4, 45, -24,   0, 20,  -591.37, -144.36,  594.41,  -98.45,  -41.34],
            [49.4, 45, -16,   0, 20,  -247.41, -200.25,  893.25, -203.29,  -65.24],
            [49.4, 45,  -8,   0, 20,   199.35, -261.73, 1241.71, -311.33,  -96.86],
            [49.4, 45,   0,   0, 20,   716.33, -341.13, 1650.76, -425.16, -134.87],
            [49.4, 45,   8,   0, 20,  1300.63, -440.39, 2116.61, -530.24, -174.72],
            [49.4, 45,  16,   0, 20,  1874.38, -479.41, 2488.53, -568.15, -162.83],
            [49.4, 45,  24,   0, 20,  2267.01, -357.56, 2575.59, -456.40,   26.17],
            [54.0, 20, -24,   0, 35,  -455.03,   30.27, -144.29,   -6.21,   -0.45],
            [54.0, 20, -16,   0, 35,  -455.03,  -30.50,  144.19,   -6.24,   -0.35],
            [54.0, 20,  -8,   0, 35,  -315.31,  -86.77,  413.03,  -71.63,  -10.37],
            [54.0, 20,   0,   0, 35,   -86.97,  -46.00,  758.33, -167.55,   62.44],
            [54.0, 20,   8,   0, 35,   189.60, -177.95, 1052.89, -308.95,  -52.48],
            [54.0, 20,  16,   0, 35,   533.64, -227.19, 1428.15, -471.48,  -80.31],
            [54.0, 20,  24,   0, 35,   874.65, -227.72, 1687.72, -499.41,  -48.05],
            [67.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28],
            [67.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [67.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            [67.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            [67.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [67.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [67.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46],
            [85.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28],
            [85.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [85.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            [85.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            [85.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [85.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [85.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46]
        ])

        table_vfm = np.array([
           #  spd tilt aoa  aos coll    a0        a1      b1        a2       b2
            [ 0.0, 90,   0,   0, 10, 1365.50,    0.81,     3.88,    0.15,    2.63],
            [ 0.0, 90,   0,   0, 15, 2111.16,    3.24,    -2.43,    1.31,   -1.43],
            [ 0.0, 90,   0,   0, 20, 2638.02,    0.78,    -0.23,   -0.10,   -0.06],
            [ 5.0, 90,   0,   0, 10, 1359.52, -195.00,   241.41,   84.96,   -0.46],
            [ 5.0, 90,   0,   0, 15, 2065.45, -203.84,   402.33,   67.18,   -1.95],
            [ 5.0, 90,   0,   0, 20, 2620.15,  -64.57,   429.97,   32.50,   -2.66],
        ])

        names = ["a0", "a1", "b1", "a2", "b2"]

        # vfm grid
        self.spd_vals_vfm = np.unique(table_vfm[:, 0])
        self.coll_vals_vfm = np.unique(table_vfm[:, 4])
        n_spd_vfm = len(self.spd_vals_vfm)
        n_coll_vfm = len(self.coll_vals_vfm)
        self.coeffs_vfm = {}
        for j, name in enumerate(names, start=5):
            grid_vfm = table_vfm[:, j].reshape(n_spd_vfm, n_coll_vfm)
            self.coeffs_vfm[name] = grid_vfm

        # tfm/ffm grid
        self.spd_vals = np.unique(table[:, 0])
        self.aoa_vals = np.unique(table[:, 2])
        n_spd = len(self.spd_vals)
        n_aoa = len(self.aoa_vals)
        self.coeffs = {}
        for j, name in enumerate(names, start=5):
            grid = table[:, j].reshape(n_spd, n_aoa)
            self.coeffs[name] = grid

    def _bilinear_interp(self, var1_vals, var2_vals, grid, var1, var2):
        i = np.searchsorted(var1_vals, var1) - 1
        j = np.searchsorted(var2_vals, var2) - 1
        i = np.clip(i, 0, len(var1_vals) - 2)
        j = np.clip(j, 0, len(var2_vals) - 2)
        x0, x1 = var1_vals[i], var1_vals[i+1]
        y0, y1 = var2_vals[j], var2_vals[j+1]
        q11 = grid[i,   j]
        q21 = grid[i+1, j]
        q12 = grid[i,   j+1]
        q22 = grid[i+1, j+1]
        tx = (var1 - x0) / (x1 - x0) if x1 > x0 else 0.0
        ty = (var2 - y0) / (y1 - y0) if y1 > y0 else 0.0
        return (q11*(1-tx)*(1-ty) + q21*tx*(1-ty) +
                q12*(1-tx)*ty       + q22*tx*ty)

    def get_coefficients(self, spd, aoa, coll, aos=None):
        coeffs_out = {}
        if spd <= 5.0:
            for name, grid in self.coeffs_vfm.items():
                coeffs_out[name] = self._bilinear_interp(
                    self.spd_vals_vfm, self.coll_vals_vfm,
                    grid, spd, coll
                )
        else:
            for name, grid in self.coeffs.items():
                coeffs_out[name] = self._bilinear_interp(
                    self.spd_vals, self.aoa_vals,
                    grid, spd, aoa
                )
        return coeffs_out

lookup = TableLookup()

# ================== Wind loop streamer ==================
class WindLoopStreamer:
    def __init__(self, filename, target_sr, fade_samples=4096):
        self.enabled = False
        try:
            sr, data = wavfile.read(filename)
        except FileNotFoundError:
            print(f"[Wind] '{filename}' not found, wind disabled.")
            return

        # to float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)

        # mono to stereo
        if data.ndim == 1:
            data = np.stack((data, data), axis=-1)

        # resample if needed using np.interp
        if sr != target_sr:
            print(f"[Wind] Resampling {filename} {sr} -> {target_sr} Hz")
            n_old = data.shape[0]
            n_new = int(round(n_old * target_sr / sr))
            x_old = np.linspace(0.0, 1.0, n_old, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
            data = np.column_stack([
                np.interp(x_new, x_old, data[:, ch])
                for ch in range(data.shape[1])
            ]).astype(np.float32)

        # choose loop region (avoid edges)
        total = data.shape[0]
        if total < 4 * fade_samples:
            # tiny file: loop almost all
            loop_start = 0
            loop_end = max(total - fade_samples - 1, 1)
        else:
            margin = total // 10
            loop_start = margin
            loop_end = total - margin - fade_samples - 1
            if loop_end <= loop_start:
                loop_start = 0
                loop_end = max(total - fade_samples - 1, 1)

        self.data = data
        self.loop_start = int(loop_start)
        self.loop_end = int(loop_end)
        self.fade_len = int(min(fade_samples, self.loop_end - self.loop_start))
        if self.fade_len < 1:
            self.fade_len = 1

        self.idx = self.loop_start
        self.enabled = True

        print(f"[Wind] Loaded. total={total}, "
              f"loop_start={self.loop_start}, loop_end={self.loop_end}, "
              f"fade_len={self.fade_len}")

    def get_block(self, frames):
        if not self.enabled:
            return np.zeros((frames, 2), dtype=np.float32)

        out = np.empty((frames, 2), dtype=np.float32)

        for i in range(frames):
            if self.idx < self.loop_end:
                # normal region
                out[i] = self.data[self.idx]
            else:
                # in or after loop_end: crossfade region
                k = self.idx - self.loop_end  # 0,1,2,...
                if k < self.fade_len:
                    end_pos = self.loop_end - self.fade_len + k
                    start_pos = self.loop_start + k
                    # safety clamp
                    end_pos = np.clip(end_pos, 0, self.data.shape[0]-1)
                    start_pos = np.clip(start_pos, 0, self.data.shape[0]-1)
                    alpha = k / float(self.fade_len)
                    out[i] = (1.0 - alpha) * self.data[end_pos] + alpha * self.data[start_pos]
                else:
                    # finished crossfade, wrap inside loop
                    rel = k - self.fade_len
                    self.idx = self.loop_start + (rel % (self.loop_end - self.loop_start))
                    out[i] = self.data[self.idx]

            self.idx += 1

            # hard wrap safety
            if self.idx >= self.loop_end + self.fade_len:
                self.idx = self.loop_start

        return out

# create global wind streamer
wind_streamer = WindLoopStreamer("wind.wav", AUDIO_SAMPLE_RATE, fade_samples=WIND_FADE_SAMPLES)

# ================== SPL helper ==================
def compute_spl_db(pressure_signal, pref=20e-6):
    rms = np.sqrt(np.mean(pressure_signal**2))
    if rms < 1e-12:
        return -np.inf
    return 20.0 * np.log10(rms / pref)

# ================== Audio Callback ==================
def audio_callback(outdata, frames, time_info, status):
    global rpm_filtered, coll_filtered, azimuth, volume_gain

    if status:
        print(status)

    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)

    spd = TEST_INPUT["spd"]
    aos = TEST_INPUT["aos"]
    aoa = TEST_INPUT["aoa"]
    tilt = TEST_INPUT["tilt"]
    rpm_target = TEST_INPUT["rpm"]
    coll_target = TEST_INPUT["coll"]

    # smooth volume (if you later want fade-in/out)
    volume_gain += (1.0 - volume_gain) * VOLUME_RAMP_SPEED

    # ---------- Rotor noise ----------
    rotor_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)

    for rotor_id in range(NUMBER_OF_ROTORS):
        rpm_filtered[rotor_id] += 0.02 * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        coll_filtered[rotor_id] += 0.02 * (coll_target[rotor_id] - coll_filtered[rotor_id])

        omega = rotor_direction[rotor_id] * rpm_filtered[rotor_id] * 2.0 * pi / 60.0
        if rpm_filtered[rotor_id] < 1e-2:
            continue

        coeffs = lookup.get_coefficients(spd=spd, aoa=aoa, coll=coll_filtered[rotor_id])
        a0 = coeffs["a0"]
        a1 = coeffs["a1"]
        b1 = coeffs["b1"]
        a2 = coeffs["a2"]
        b2 = coeffs["b2"]

        tilt_rad = np.radians(90.0 - tilt[rotor_id])  # 90: VFM, 0: FFM
        aos_rad = np.radians(aos)

        trans_tilt = np.array([
            [np.cos(tilt_rad), 0.0, -np.sin(tilt_rad)],
            [0.0,             1.0,  0.0],
            [np.sin(tilt_rad), 0.0,  np.cos(tilt_rad)]
        ])

        for blade in range(NUMBER_OF_BLADES):
            source_id = rotor_id * NUMBER_OF_BLADES + blade
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n
            azimuth[source_id] = az_block[-1]

            # periodic lift
            phase = np.abs(az_block - aos_rad * rotor_direction[rotor_id])
            L = (a0
                 + a1 * np.cos(phase)
                 + b1 * np.sin(phase)
                 + a2 * np.cos(2.0 * phase)
                 + b2 * np.sin(2.0 * phase))

            # source positions on tilted disk
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = rotor_center[rotor_id][2] * np.ones_like(x)
            source_pos = np.stack((x, y, z), axis=1)
            source_pos = tilt_center[rotor_id] + (source_pos - tilt_center[rotor_id]) @ trans_tilt.T

            r = observer_position - source_pos
            rmag = np.linalg.norm(r, axis=1)

            M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
            Mi = np.stack((-M * np.sin(az_block),
                           M * np.cos(az_block),
                           np.zeros_like(az_block)), axis=1)
            Mi = Mi @ trans_tilt.T

            Fi = np.stack((np.zeros_like(az_block),
                           np.zeros_like(az_block),
                           L), axis=1)
            Fi = Fi @ trans_tilt.T

            Mr = np.sum(r * Mi, axis=1) / rmag
            Fr = np.sum(r * Fi, axis=1) / rmag

            p_near = (0.25 / pi) * (
                (1.0 / (1.0 - Mr)**2) / (rmag**2)
                * (Fr * (1.0 - M**2) / (1.0 - Mr) - np.sum(Fi * Mi, axis=1))
            )

            rotor_buffer[:, 0] += p_near
            rotor_buffer[:, 1] += p_near

    # dynamic offset from rotor only
    if np.any(rpm_filtered > 0):
        T_rev = 60.0 / np.mean(rpm_filtered[rpm_filtered > 0])
    else:
        T_rev = 0.1
    T_window = T_rev / NUMBER_OF_BLADES
    N_window = int(T_window * AUDIO_SAMPLE_RATE)

    if 0 < N_window < frames:
        offset_dynamic = float(np.mean(rotor_buffer[:N_window, 0]))
    else:
        offset_dynamic = float(np.mean(rotor_buffer[:, 0]))

    plot_data["block_mean"].append(offset_dynamic)
    t_now = time.time()
    plot_data["block_time"].append(t_now)

    rotor_buffer -= offset_dynamic
    rotor_buffer *= ROTOR_SCALING_FACTOR

    # ---------- Wind noise ----------
    if wind_streamer.enabled:
        # simple mapping: more speed -> more wind
        spd_norm = max(0.0, min(1.0, spd / 67.0))
        wind_gain = WIND_BASE_GAIN * (0.2 + 0.8 * spd_norm)
        wind_block = wind_streamer.get_block(frames)
        mix_buffer = rotor_buffer + wind_gain * wind_block
    else:
        mix_buffer = rotor_buffer

    # ---------- Optional filter ----------
    with filter_lock:
        if FILTER_ENABLED:
            mix_buffer[:, 0], zi[:, :] = sosfilt(sos, mix_buffer[:, 0], zi=zi)
            mix_buffer[:, 1], _ = sosfilt(sos, mix_buffer[:, 1], zi=zi)

    # Prevent insane clipping (light limiter)
    peak = np.max(np.abs(mix_buffer))
    if peak > 1.0:
        mix_buffer /= peak

    outdata[:] = mix_buffer.astype(np.float32)

    # ---------- Logging to plots ----------
    plot_data["wave"].extend(mix_buffer[:, 0])
    plot_data["spd"].append(spd)
    plot_data["tilt"].append(TEST_INPUT["tilt"][0])
    plot_data["coll"].append(coll_filtered[0])
    plot_data["rpm"].append(float(np.mean(rpm_filtered)))
    plot_data["aoa"].append(aoa)
    plot_data["aos"].append(aos)
    plot_data["time"].append(time.time() % PARAM_WINDOW_SEC)

    spl_db = compute_spl_db(mix_buffer[:, 0])
    plot_data["spl_db"].append(spl_db)

# ================== Transition thread ==================
def simulate_transition():
    start_time = time.time()
    while True:
        t = time.time() - start_time
        transition_duration = 60.0
        phase = (t % (2 * transition_duration)) / transition_duration
        if phase > 1.0:
            phase = 2.0 - phase

        TEST_INPUT["spd"]  = 67.0 * phase
        TEST_INPUT["aoa"]  = 10.0 * np.sin(2 * np.pi * 10)  # (effectively constant, but keeping your line)
        TEST_INPUT["tilt"] = np.array([90.0 * (1.0 - phase)] * 4)
        TEST_INPUT["rpm"]  = np.array([477.5 - 309.5 * phase] * 4)
        TEST_INPUT["coll"] = np.array([15.0 + 35.0 * phase] * 4)
        time.sleep(0.05)

# ================== Plotting ==================
def start_plots():
    fig1, axs1 = plt.subplots(2, 3, figsize=(10, 8))
    fig1.subplots_adjust(wspace=0.4, hspace=0.5)
    axs1[0,0].set_title("Speed")
    axs1[0,1].set_title("Tilt")
    axs1[0,2].set_title("Coll")
    axs1[1,0].set_title("RPM")
    axs1[1,1].set_title("AOA")
    axs1[1,2].set_title("AOS")

    lines1 = [
        axs1[0,0].plot([], [])[0],
        axs1[0,1].plot([], [])[0],
        axs1[0,2].plot([], [])[0],
        axs1[1,0].plot([], [])[0],
        axs1[1,1].plot([], [])[0],
        axs1[1,2].plot([], [])[0],
    ]

    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 8))
    fig2.subplots_adjust(hspace=0.5)
    axs2[0].set_title("Waveform")
    axs2[1].set_title("FFT")
    axs2[2].set_title("Segment Mean Pressure")
    axs2[3].set_title("Block SPL (dB re 20µPa)")

    line_wave = axs2[0].plot([], [])[0]
    line_fft  = axs2[1].plot([], [])[0]
    line_mean = axs2[2].plot([], [])[0]
    line_spl  = axs2[3].plot([], [])[0]

    def update(frame):
        t_vals = list(plot_data["time"])
        if len(t_vals) < 2:
            return []

        # params
        keys = ["spd", "tilt", "coll", "rpm", "aoa", "aos"]
        for k, key in enumerate(keys):
            y = list(plot_data[key])
            if len(y) < 2:
                continue
            lines1[k].set_data(t_vals, y)
            r = k // 3
            c = k % 3
            axs1[r, c].set_xlim(min(t_vals), max(t_vals) + 1e-6)

        axs1[0,0].set_ylabel('Speed (m/s)')
        axs1[0,0].set_ylim(-10, 100)
        axs1[0,0].grid(True, axis='y', linestyle='--', alpha=0.7)

        axs1[0,1].set_ylabel('Tilt (deg)')
        axs1[0,1].set_ylim(-10, 100)
        axs1[0,1].grid(True, axis='y', linestyle='--', alpha=0.7)

        axs1[0,2].set_ylabel('Coll (deg)')
        axs1[0,2].set_ylim(-10, 70)
        axs1[0,2].grid(True, axis='y', linestyle='--', alpha=0.7)

        axs1[1,0].set_ylabel('RPM')
        axs1[1,0].set_ylim(0, 500)
        axs1[1,0].grid(True, axis='y', linestyle='--', alpha=0.7)

        axs1[1,1].set_ylabel('AoA (deg)')
        axs1[1,1].set_ylim(-20, 20)
        axs1[1,1].grid(True, axis='y', linestyle='--', alpha=0.7)

        axs1[1,2].set_ylabel('AoS (deg)')
        axs1[1,2].set_ylim(-20, 20)
        axs1[1,2].grid(True, axis='y', linestyle='--', alpha=0.7)

        # waveform
        wave = np.array(plot_data["wave"])
        N = len(wave)
        if N > 0:
            t_wave = np.linspace(0, WAVEFORM_WINDOW_SEC, N)
            line_wave.set_data(t_wave, wave)
            axs2[0].set_xlim(0, WAVEFORM_WINDOW_SEC)
            axs2[0].set_ylim(-12*ROTOR_SCALING_FACTOR, 12*ROTOR_SCALING_FACTOR)

            # FFT (low-freq region for visualization)
            freqs = np.fft.rfftfreq(N, 1.0 / AUDIO_SAMPLE_RATE)
            spectrum = np.abs(np.fft.rfft(wave))
            spec_db = 20.0 * np.log10(spectrum + 1e-12)
            line_fft.set_data(freqs, spec_db)
            axs2[1].set_xlim(0, 100)
            axs2[1].set_ylim(np.max(spec_db) - 80, np.max(spec_db) + 5)

        # mean
        if len(plot_data["block_mean"]) > 1:
            t_b = list(plot_data["block_time"])
            m_b = list(plot_data["block_mean"])
            line_mean.set_data(t_b, m_b)
            axs2[2].set_xlim(min(t_b), max(t_b) + 1e-6)
            axs2[2].set_ylim(min(m_b) - 1, max(m_b) + 1)

        # SPL
        if len(plot_data["spl_db"]) > 1:
            t_spl = list(plot_data["block_time"])[-len(plot_data["spl_db"]):]
            spl_vals = list(plot_data["spl_db"])
            line_spl.set_data(t_spl, spl_vals)
            axs2[3].set_xlim(min(t_spl), max(t_spl) + 1e-6)
            axs2[3].set_ylim(min(spl_vals) - 2, max(spl_vals) + 2)
            axs2[3].grid(True, axis='y', linestyle='--', alpha=0.7)

        return lines1 + [line_wave, line_fft, line_mean, line_spl]

    FuncAnimation(fig1, update, interval=int(PLOT_UPDATE_INTERVAL * 1000),
                  cache_frame_data=False, save_count=100)
    FuncAnimation(fig2, update, interval=int(PLOT_UPDATE_INTERVAL * 1000),
                  cache_frame_data=False, save_count=100)
    plt.show()

# ================== Main ==================
def main():
    threading.Thread(target=simulate_transition, daemon=True).start()
    print("Starting sound stream (rotor + wind)...")
    with sd.OutputStream(
        channels=AUDIO_CHANNELS,
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=AUDIO_BLOCK_SIZE,
        dtype='float32',
        callback=audio_callback
    ):
        start_plots()

if __name__ == "__main__":
    main()