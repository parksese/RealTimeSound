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
plot_done = False

# Observer and geometry setup
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398],    # FL                        
                        [-2.554, 3.962, 0.398],     # FR                        
                        [4.101, -3.962, 0.667],     # AL    
                        [4.101, 3.962, 0.667]])     # AR                        
rotor_center = tilt_center + np.array([-0.188, 0, 1.430]) # based on VFM mode
rotor_direction = np.array([-1, 1, -1, 1])  # 1: Counter-Clockwise, 2: Clockwise

time_window = 2.0  # seconds
plot_buffer = np.zeros((int(AUDIO_SAMPLE_RATE * time_window), AUDIO_CHANNELS), dtype=np.float32)


# ---------------- Simulated Constant Inputs ----------------
TEST_INPUT = {
    "spd": 67.0,
    "aoa": 0.0,
    "aos": 0.0,
    "rpm": np.array([200, 200, 200, 200], dtype=float),
    "collective": np.array([50, 50, 50, 50], dtype=float),
    "tilt": np.array([0, 0, 0, 0], dtype=float)
}

# ---------------- State Variables ----------------
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
coll_filtered = np.zeros(NUMBER_OF_ROTORS)
azimuth = np.array([2*pi*(sid%NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)])
volume_gain = 0.0

# ---------------- Observer & Geometry ----------------
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398], [-2.554, 3.962, 0.398],
                        [-4.101, -3.962, 0.667], [4.101, -3.962, 0.667]])
prop_center = tilt_center + np.array([-0.188, 0, 1.430])

# ---------------- Lookup Table ----------------
class TableLookup:
    def __init__(self):
        table = np.array([
           # spd  aoa aos coll    a0        a1      b1        a2       b2
            [0.00, 0, -10, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43], # case1 50.0 rad/s = 477.5 rpm
            [0.00, 0,  -5, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [0.00, 0,   0, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],  
            [0.00, 0,   5, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [0.00, 0,  10, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [5.00, 0, -10, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], # case2 50.0 rad/s = 477.5 rpm
            [5.00, 0,  -5, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
            [5.00, 0,   0, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], 
            [5.00, 0,   5, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
            [5.00, 0,  10, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
            [46.9, 0, -10, 15, 1896.92, -117.84, 2352.80, -582.05,  -43.35], # case3 48.9 rad/s = 467 rpm
            [46.9, 0,  -5, 15, 1880.13, -334.72, 2323.99, -567.27, -149.37],
            [46.9, 0,   0, 15, 1873.06, -549.95, 2279.70, -528.09, -251.81],
            [46.9, 0,   5, 15, 1880.49, -755.70, 2220.03, -469.80, -344.33],
            [46.9, 0,  10, 15, 1895.88, -963.69, 2153.17, -400.84, -424.70],
            [49.4, 0, -10, 20,  776.86,   70.29, 1736.77, -449.34,   79.58], # case4 47.1 rad/s = 450 rpm
            [49.4, 0,  -5, 20,  729.57, -141.32, 1699.35, -449.35,  -21.83],
            [49.4, 0,   0, 20,  716.33, -341.13, 1650.76, -425.16, -134.87], 
            [49.4, 0,   5, 20,  731.30, -550.92, 1617.01, -387.08, -230.59],
            [49.4, 0,  10, 20,  777.19, -752.89, 1567.50, -316.74, -328.63],
            [54.0, 0, -10, 35,  -32.84,  240.33,  791.73, -146.91,  147.89], # case5 30.05 rad/s = 287 rpm
            [54.0, 0,  -5, 35,  -86.97,  -46.00,  758.33, -167.55,   62.44],
            [54.0, 0,   0, 35, -104.95, -139.91,  722.07, -168.47,  -24.68], 
            [54.0, 0,   5, 35,  -86.99, -323.71,  687.28, -142.86, -107.53],
            [54.0, 0,  10, 35,  -29.40, -506.64,  652.07,  -92.83, -185.67],
            [67.0, 1, -10, 60, 153.86,  457.98,    47.82,   80.31,    4.25], # case6 17.6 rad/s = 168 rpm
            [67.0, 1,  -5, 60,  80.76,  229.69,   24.315,   19.71,    0.99],
            [67.0, 1,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
            [67.0, 1,   5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
            [67.0, 1,  10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25],
            [85.0, 0, -10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25], # case7 17.6 rad/s = 168 rpm
            [85.0, 0,  -5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
            [85.0, 0,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
            [85.0, 0,   5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
            [85.0, 0,  10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25],
        ])

        table_vfm = np.array([
           # spd  aoa aos coll    a0        a1      b1        a2       b2
            [0.00, 0,   0, 10, 1365.50,    0.81,   3.88,     0.15,    2.63], # case1 50.0 rad/s = 477.5 rpm
            [0.00, 0,   0, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43], 
            [0.00, 0,   0, 20, 2638.02,    0.78,  -0.23,    -0.10,   -0.06],
            [5.00, 0,   0, 10, 1359.52, -195.00, 241.41,    84.96,   -0.46], # case2 50.0 rad/s = 477.5 rpm
            [5.00, 0,   0, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], 
            [5.00, 0,   0, 20, 2620.15,  -64.57, 429.97,    32.50,   -2.66], 
            ])

        
        names = ["a0", "a1", "b1", "a2", "b2"]
        # vfm
        self.spd_vals_vfm = np.unique(table_vfm[:, 0])  # extract unique sorted speed and coll grids
        self.coll_vals_vfm = np.unique(table_vfm[:, 3])
        n_spd_vfm = len(self.spd_vals_vfm)
        n_coll_vfm = len(self.coll_vals_vfm)
        # build fast interpolators for each coefficient
        self.coeffs_vfm = {}
        for j, name in enumerate(names, start=4):       # build fast interpolators for each coefficient
            grid_vfm = table_vfm[:, j].reshape(n_spd_vfm, n_coll_vfm)
            self.coeffs_vfm[name] = grid_vfm
        
        # print(self.coeffs_vfm)

        # tfm, ffm
        self.spd_vals = np.unique(table[:, 0])          # extract unique sorted speed and aos grids
        self.aos_vals = np.unique(table[:, 2])
        n_spd = len(self.spd_vals)
        n_aos = len(self.aos_vals)
        # build fast interpolators for each coefficient
        self.coeffs = {}
        for j, name in enumerate(names, start=4):       # build fast interpolators for each coefficient
            grid = table[:, j].reshape(n_spd, n_aos)
            self.coeffs[name] = grid
        # print(self.coeffs)

    def _bilinear_interp(self, var1_vals, var2_vals, grid, var1, var2):
        """
        manual bilinear interpolation of grid at (var1, var2)
        grid: 2d array indexed as [i_spd, j_aos]
        """
        # find indices
        i = np.searchsorted(var1_vals, var1) - 1
        j = np.searchsorted(var2_vals, var2) - 1
        i = np.clip(i, 0, len(var1_vals) - 2)
        j = np.clip(j, 0, len(var2_vals) - 2)
        # grid corners
        x0, x1 = var1_vals[i], var1_vals[i+1]
        y0, y1 = var2_vals[j], var2_vals[j+1]
        q11 = grid[i  , j  ]
        q21 = grid[i+1, j  ]
        q12 = grid[i  , j+1]
        q22 = grid[i+1, j+1]        
        # normalize distances
        tx = (var1-x0) / (x1-x0) if x1 > x0 else 0
        ty = (var2-y0) / (y1-y0) if y1 > y0 else 0
        return (q11*(1-tx)*(1-ty) + q21*tx*(1-ty) + q12*(1-tx)*ty + q22*tx*ty)

    def get_coefficients(self, spd, aos, coll, aoa=None):
        """
        get coefficients for (spd, aos)
        for vfm (spd <=5), only interpolate in spd, aos=0 row        
        """
        coeffs_out = {}
        if spd <= 5: # vfm
            for name, grid in self.coeffs_vfm.items():
                coeffs_out[name] = self._bilinear_interp(self.spd_vals_vfm, self.coll_vals_vfm, grid, spd, coll)
        else: # tfm, ffm
            for name, grid in self.coeffs.items():
                coeffs_out[name] = self._bilinear_interp(self.spd_vals, self.aos_vals, grid, spd, aos)                
        return coeffs_out

lookup = TableLookup()

# ---------------- Buffers for Plotting ----------------
plot_len = 200
plot_data = {
    "time": deque(maxlen=plot_len),
    "spd": deque(maxlen=plot_len),
    "tilt": deque(maxlen=plot_len),
    "coll": deque(maxlen=plot_len),
    "aos": deque(maxlen=plot_len),
    "wave": deque(maxlen=plot_len*AUDIO_BLOCK_SIZE)
}
start_time = time.time()

# ---------------- Audio Callback ----------------
def audio_callback(outdata, frames, time_info, status):
    """Vectorized real-time audio callback"""
    global rpm_filtered, coll_filtered, azimuth, volume_gain
    global latest_out_buffer, plot_done
     
    if status:
        print(status)
           
    out_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)
    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)

    # Simulated constant inputs
    spd = TEST_INPUT["spd"]
    aos = TEST_INPUT["aos"]
    aoa = TEST_INPUT["aoa"]
    tilt = TEST_INPUT["tilt"]
    rpm_target = TEST_INPUT["rpm"]
    coll_target = TEST_INPUT["collective"]

    # Smooth volume gain
    target_gain = 1.0
    volume_gain += (target_gain - volume_gain) * VOLUME_RAMP_SPEED

    for rotor_id in range(NUMBER_OF_ROTORS):
        # Smooth inputs
        rpm_filtered[rotor_id] += 0.02 * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        coll_filtered[rotor_id] += 0.02 * (coll_target[rotor_id] - coll_filtered[rotor_id])
        omega = rpm_filtered[rotor_id] * 2 * np.pi / 60
        if rpm_filtered[rotor_id] < 1e-2:
            continue

        # Get coefficients (same for all samples in block)    
        c = lookup.get_coefficients(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        a0, a1, b1, a2, b2 = c["a0"], c["a1"], c["b1"], c["a2"], c["b2"]
        # Tilt matrix  (precompute once per rotor)
        tilt_rad = np.radians(90 - tilt[rotor_id]) # tilt: 90 for VFM, 0 for FFM
        trans_tilt = np.array([
            [np.cos(tilt_rad), 0, -np.sin(tilt_rad)],  # need to double-check the transformation !!!
            [0, 1, 0],
            [np.sin(tilt_rad), 0, np.cos(tilt_rad)]
        ])

        for blade in range(NUMBER_OF_BLADES):
            source_id = rotor_id * NUMBER_OF_BLADES + blade
            # Azimuth evolution for this blade over block
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n

            # cos_az = np.cos(az_block)
            # sin_az = np.sin(az_block)
            # cos_2az = np.cos(2*az_block)
            # sin_2az = np.sin(2*az_block)

            # Update azimuth state (for continuity to next block)
            azimuth[source_id] = az_block[-1]

            # Lift (periodic loading)
            if rotor_direction[rotor_id] == 1:  # Counter-clockwise
                L = (a0 + a1 * np.cos(abs(az_block)) + b1 * np.sin(abs(az_block)) + a2 * np.cos(abs(2*az_block)) + b2 * np.sin(abs(2*az_block)))
            else:  # Clockwise
                L = (a0 + a1 * np.cos(abs(az_block)-2.0*aos) + b1 * np.sin(abs(az_block)-2.0*aos) + a2 * np.cos(abs(2*az_block)-2.0*aos) + b2 * np.sin(abs(2*az_block)-2.0*aos))

            # Source position (before tilt)
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block) 
            z = rotor_center[rotor_id][2] * np.ones_like(x)
            source_position = np.stack((x, y, z), axis=1) # (frames, 3)
            # Apply tilt about tilt_center            
            source_position = tilt_center[rotor_id] + (source_position - tilt_center[rotor_id]) @ trans_tilt.T 
            
            # Observer vector
            r = observer_position - source_position
            rmag = np.linalg.norm(r, axis=1)

            # Mach vector 
            M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
            Mi = np.stack((-M * np.sin(az_block), M * np.cos(az_block), np.zeros_like(az_block)), axis=1)
            Mi = (Mi @ trans_tilt.T)    # Apply tilt            

            # Force vector
            Fi = np.stack((np.zeros_like(az_block),
                           np.zeros_like(az_block),
                           L), axis=1)
            Fi = (Fi @ trans_tilt.T)    # Apply tilt

            # Dot products
            Mr = np.sum(r * Mi, axis=1) / rmag
            Fr = np.sum(r * Fi, axis=1) / rmag

            # Pressure (vectorized form, near-field only for speed)
            p_near = (0.25 / pi) * (
                1 / (1 - Mr) ** 2 / rmag**2
                * (Fr * (1 - M**2) / (1 - Mr) - np.sum(Fi * Mi, axis=1))
            )
            out_buffer[:, 0] += p_near * volume_gain
            out_buffer[:, 1] += p_near * volume_gain

    out_buffer *= SCALING_FACTOR
    outdata[:] = out_buffer
    # outdata[:] = out_buffer.reshape(-1,1)
    
    # Capture one buffer for later plotting
    if latest_out_buffer is None:
        latest_out_buffer = np.copy(out_buffer)



# ---------------- Plot Thread ----------------
def plot_waveform_and_fft(signal, sample_rate):
    
    if signal.ndim > 1:
        signal = signal[:, 0]  # Use first channel for plotting
    
    N = len(signal)
    t = np.arange(N) / sample_rate

    # FFT
    freqs = np.fft.rfftfreq(N, 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Waveform
    axs[0].plot(t, signal)
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")

    # FFT (in dB)
    axs[1].semilogx(freqs, 20 * np.log10(spectrum + 1e-12))
    axs[1].set_title("FFT Spectrum")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Magnitude [dB]")
    axs[1].set_xlim(20, sample_rate / 2)
    axs[1].grid(True, which='both', ls='--')

    plt.tight_layout()
    plt.show(block=True)
    
# ---------------- Main ----------------
def main():
    global latest_out_buffer

    print("Starting sound stream...")
    with sd.OutputStream(
        channels=AUDIO_CHANNELS,
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    ):
        # Wait until we get one buffer
        timeout = 5.0
        start_time = time.time()
        while latest_out_buffer is None and time.time() - start_time < timeout:
            time.sleep(0.05)

        # Plot result
        if latest_out_buffer is not None:
            print("Buffer captured — plotting waveform and FFT...")
            plot_waveform_and_fft(latest_out_buffer, AUDIO_SAMPLE_RATE)
        else:
            print("No buffer captured — callback may not have run?")

        
if __name__ == "__main__":
    main()
