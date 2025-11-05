import numpy as np
import threading
import time
import socket
import struct
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")  # safer on Windows; avoids WinError 6 from Qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque


# ---------------- Constants ----------------
pi = np.pi
SPEED_OF_SOUND = 334.3
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048

AUDIO_SAMPLE_RATE = 22050 #44100
AUDIO_CHANNELS = 2
# AUDIO_BLOCK_SIZE = 1024  # Smaller block size for lower latency
AUDIO_BLOCK_SIZE = 4096  # Smaller block size for lower latency
SCALING_FACTOR = 5.0 #1 # 0.5
VOLUME_RAMP_SPEED = 0.01
volume_gain = 0.0
smoothing_factor = 1 #50 #0.005

rpm_filter = 0.0
rpm_target = 0.0
volume_ramp = 0.0 # current ramp level (0 to 1)

UDP_TIMEOUT = 1.0

# ---------------- Plot / window parameters ----------------
PARAM_WINDOW_SEC = 2.0          # seconds for spd, tilt, rpm, aos rolling plot
WAVEFORM_WINDOW_SEC = 2.0       # seconds for waveform rolling plot
PLOT_UPDATE_INTERVAL = 0.2      # seconds

# Buffers for plotting
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
    "block_mean": deque(maxlen=param_len)
}

# ---------------- State variables ----------------
# Keep the last known state for real-time calculation
last_state = {
    "rpm": np.zeros(NUMBER_OF_ROTORS),
    "coll": np.zeros(NUMBER_OF_ROTORS),
    "tilt": np.zeros(NUMBER_OF_ROTORS),
    "spd": 0.0,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
                         for sid in range(NUMBER_OF_SOURCES)]),
    "last_update_time": time.time()
}
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
coll_filtered = np.zeros(NUMBER_OF_ROTORS)

state_lock = threading.Lock()
udp_received = threading.Event()

# ---------------- Observer & geometry ----------------
# Observer and geometry setup
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398],    # FL                        
                        [-2.554, 3.962, 0.398],     # FR                        
                        [4.101, -3.962, 0.667],     # AL    
                        [4.101, 3.962, 0.667]])     # AR                        
rotor_center = tilt_center + np.array([-0.188, 0, 1.430]) # based on VFM mode
rotor_direction = np.array([-1, 1, -1, 1])  # 1: Counter-Clockwise, 2: Clockwise

# ---------------- UDP Setup ----------------
# SERVER_IP = "127.0.0.1" # Local test only
# SERVER_IP = "192.168.100.50"  # MFS
SERVER_IP = "192.168.100.15"  # FFS
SERVER_PORT = 1700
RecvBufferSize = 10240
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

# ---------------- TableLookup ----------------
class TableLookup:
    def __init__(self):
        table = np.array([
           #  spd tilt aoa aos coll   a0         a1      b1        a2       b2
            [ 0.0, 90, -24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43], # case1 50.0 rad/s = 477.5 rpm           
            [ 0.0, 90, -16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  -8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   0,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 5.0, 90, -24,   0, 15,  2019.10, -159.62, 330.10,    45.05,    2.67], # case2 50.0 rad/s = 477.5 rpm
            [ 5.0, 90, -16,   0, 15,  2018.86, -159.24, 373.12,    58.96,  -16.87],
            [ 5.0, 90,  -8,   0, 15,  2043.73, -183.79, 398.20,    70.20,   -7.37],
            [ 5.0, 90,   0,   0, 15,  2065.45, -203.84, 402.33,    67.18,   -1.95],
            [ 5.0, 90,   8,   0, 15,  2076.82, -218.69, 404.37,    73.34,    6.24],
            [ 5.0, 90,  16,   0, 15,  2083.48, -224.29, 403.09,    91.48,   -3.60],
            [ 5.0, 90,  24,   0, 15,  2097.72, -228.09, 374.76,   100.52,   -1.42],         
            [46.9, 70, -24,   0, 15,    89.87, -251.46, 1117.25, -302.04, -123.60], # case3 48.9 rad/s = 467 rpm
            [46.9, 70, -16,   0, 15,   649.22, -349.74, 1468.23, -404.10, -169.95],          
            [46.9, 70,  -8,   0, 15,  1261.38, -463.84, 1909.35, -475.91, -220.62],
            [46.9, 70,   0,   0, 15,  1873.06, -549.95, 2279.70, -528.09, -251.81],
            [46.9, 70,   8,   0, 15,  2352.58, -503.41, 2464.50, -501.43, -145.07],
            [46.9, 70,  16,   0, 15,  2666.80, -272.75, 2490.22, -435.59,   19.22],
            [46.9, 70,  24,   0, 15,  2857.03,  -49.74, 2296.69, -280.00,   39.44],                   
            [49.4, 45, -24,   0, 20,  -591.37, -144.36,  594.41,  -98.45,  -41.34], # case4 47.1 rad/s = 450 rpm
            [49.4, 45, -16,   0, 20,  -247.41, -200.25,  893.25, -203.29,  -65.24],
            [49.4, 45,  -8,   0, 20,   199.35, -261.73, 1241.71, -311.33,  -96.86],
            [49.4, 45,   0,   0, 20,   716.33, -341.13, 1650.76, -425.16, -134.87],
            [49.4, 45,   8,   0, 20,  1300.63, -440.39, 2116.61, -530.24, -174.72],
            [49.4, 45,  16,   0, 20,  1874.38, -479.41, 2488.53, -568.15, -162.83],
            [49.4, 45,  24,   0, 20,  2267.01, -357.56, 2575.59, -456.40,   26.17],     
            [54.0, 20, -24,   0, 35,  -455.03,   30.27, -144.29,   -6.21,   -0.45], # case5 30.05 rad/s = 287 rpm
            [54.0, 20, -16,   0, 35,  -455.03,  -30.50,  144.19,   -6.24,   -0.35],
            [54.0, 20,  -8,   0, 35,  -315.31,  -86.77,  413.03,  -71.63,  -10.37],
            [54.0, 20,   0,   0, 35,   -86.97,  -46.00,  758.33, -167.55,   62.44],
            [54.0, 20,   8,   0, 35,   189.60, -177.95, 1052.89, -308.95,  -52.48],        
            [54.0, 20,  16,   0, 35,   533.64, -227.19, 1428.15, -471.48,  -80.31],
            [54.0, 20,  24,   0, 35,   874.65, -227.72, 1687.72, -499.41,  -48.05],
            [67.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28], # case6 17.6 rad/s = 168 rpm
            [67.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [67.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            # [67.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            [67.0,  0,   0,   0, 60,   159.69,    0.00,     0.00,    0.00,    0.00],

            [67.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [67.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [67.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46],
            [85.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28], # case7 17.6 rad/s = 168 rpm
            [85.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [85.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            # [85.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            [85.0,  0,   0,   0, 60,   159.69,    0.00,     0.00,    0.00,    0.00],
            
            [85.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [85.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [85.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46]
        ])

        table_vfm = np.array([
           #  spd tilt aoa  aos coll    a0        a1      b1        a2       b2
            [ 0.0, 90,   0,   0, 10, 1365.50,    0.81,     3.88,    0.15,    2.63], # case1 50.0 rad/s = 477.5 rpm
            [ 0.0, 90,   0,   0, 15, 2111.16,    3.24,    -2.43,    1.31,   -1.43], 
            [ 0.0, 90,   0,   0, 20, 2638.02,    0.78,    -0.23,   -0.10,   -0.06],
            [ 5.0, 90,   0,   0, 10, 1359.52, -195.00,   241.41,   84.96,   -0.46], # case2 50.0 rad/s = 477.5 rpm
            [ 5.0, 90,   0,   0, 15, 2065.45, -203.84,   402.33,   67.18,   -1.95], 
            [ 5.0, 90,   0,   0, 20, 2620.15,  -64.57,   429.97,   32.50,   -2.66], 
        ])
        
        names = ["a0", "a1", "b1", "a2", "b2"]
        # vfm
        self.spd_vals_vfm = np.unique(table_vfm[:, 0])  # extract unique sorted speed and coll grids
        self.coll_vals_vfm = np.unique(table_vfm[:, 4])
        n_spd_vfm = len(self.spd_vals_vfm)
        n_coll_vfm = len(self.coll_vals_vfm)
        # build fast interpolators for each coefficient
        self.coeffs_vfm = {}
        for j, name in enumerate(names, start=5):       # build fast interpolators for each coefficient
            grid_vfm = table_vfm[:, j].reshape(n_spd_vfm, n_coll_vfm)
            self.coeffs_vfm[name] = grid_vfm
        
        # print(self.coeffs_vfm)

        # tfm, ffm
        self.spd_vals = np.unique(table[:, 0])          # extract unique sorted speed and aos grids
        self.aoa_vals = np.unique(table[:, 2])        
        n_spd = len(self.spd_vals)
        n_aoa = len(self.aoa_vals)
        # build fast interpolators for each coefficient
        self.coeffs = {}
        for j, name in enumerate(names, start=5):       # build fast interpolators for each coefficient
            # grid = table[:, j].reshape(n_spd, n_aos)
            grid = table[:, j].reshape(n_spd, n_aoa)
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

    def get_coefficients(self, spd, aoa, coll, aos=None):
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
                coeffs_out[name] = self._bilinear_interp(self.spd_vals, self.aoa_vals, grid, spd, aoa)                
        return coeffs_out



lookup = TableLookup()

# ---------------- UDP Listener ----------------
def update_state_from_udp():
    """Update state from UDP data"""
    while True:
        try:
            fmt = '<ii3f4f4f4d' # no padding
            expected_size = struct.calcsize(fmt)
            
            data, _ = sock.recvfrom(RecvBufferSize)
            print(data)
            
            if len(data) == expected_size:                
                count, size, spd_kn, aoa, aos, *values = struct.unpack(fmt, data)
                with state_lock:
                    spd = spd_kn * 0.514444
                    last_state["spd"] = spd
                    last_state["aoa"] = aoa
                    last_state["aos"] = aos
                    last_state["rpm"][:] = values[:NUMBER_OF_ROTORS]
                    last_state["coll"][:] = values[NUMBER_OF_ROTORS:2*NUMBER_OF_ROTORS]
                    last_state["tilt"][:] = values[2*NUMBER_OF_ROTORS:3*NUMBER_OF_ROTORS]
                    last_state["last_update_time"] = time.time()
                udp_received.set()

                # debug use only
                print(f"  count= {count}")
                print(f"  size = {size}")                    
                print(f"  spd  = {spd:.1f} m/s")
                print(f"  aoa  = {aoa:.1f} deg")
                print(f"  aos  = {aos:.1f} deg")
                print(f"  rpm  = {values[:NUMBER_OF_ROTORS]} rpm")
                print(f"  coll = {values[NUMBER_OF_ROTORS:2*NUMBER_OF_ROTORS]} deg")
                print(f"  tilt = {values[2*NUMBER_OF_ROTORS:3*NUMBER_OF_ROTORS]} deg\n")
                

            else:
                print(f"Wrong packet size: got {len(data)}, expected {expected_size}")
        
        except Exception as e:
            print(f"UDP error: {e}")
            time.sleep(0.01)


# ---------------- Audio Callback ----------------
def compute_spl_db(pressure_signal, pref=20e-6):
    """Compute SPL in dB from pressure signal array"""
    rms_pressure = np.sqrt(np.mean(pressure_signal**2))
    if rms_pressure < 1e-12:
        return -np.inf
    spl_db = 20 * np.log10(rms_pressure / pref)
    return spl_db


# ---------------- Audio Callback ----------------
def audio_callback(outdata, frames, time_info, status):
    """Vectorized real-time audio callback"""

    global rpm_filtered, coll_filtered, volume_ramp, volume_gain, smoothing_factor

    if status:
        print(status)

    start_time = time.perf_counter() # timing diagnotic (comment out for normal run)

    if not udp_received.is_set(): # no UDP data yet
        outdata.fill(0)
        # print("no udp data yet")
        return

    time_since_last = start_time - last_state["last_update_time"]
    if time_since_last > UDP_TIMEOUT:
        # Fade out when no fresh UDP data
        target_gain = 0.0
    else:
        # Fade in when data is live
        target_gain = 1.0

    # Smoothly move toward target gain (prevent clicks)
    volume_gain += (target_gain - volume_gain) * VOLUME_RAMP_SPEED

    if volume_gain < 1e-3:
        outdata[:] = 0.0
        print("check if volume gain is too small")
        return

    # Prepare output buffer
    out_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)
    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)  # vector of [0, 1, ..., frames-1]

    # Copy current state once (avoid lock inside heavy loop)
    with state_lock:
        spd = last_state["spd"]
        aoa = last_state["aoa"]
        aos = last_state["aos"]
        # rpm = last_state["rpm"]
        rpm_target = last_state["rpm"].copy() # added 10-13-2025
        coll_target = last_state["coll"].copy() # added 10-13-2025   
        # coll = last_state["coll"].copy()
        tilt = last_state["tilt"].copy()
        azimuth = last_state["azimuth"].copy()
        last_state["azimuth"][:] = azimuth  # will be updated below

    for rotor_id in range(NUMBER_OF_ROTORS):          
        # omega = rotor_direction[rotor_id] * rpm[rotor_id] * 2 * pi / 60
        # domega_dtau = 0.0  # could be updated from UDP if needed       
        
        # filter/smoothing 10-13-2025
        rpm_filtered[rotor_id] += smoothing_factor * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        coll_filtered[rotor_id] += smoothing_factor * (coll_target[rotor_id] - coll_filtered[rotor_id])
        omega = rotor_direction[rotor_id] * rpm_filtered[rotor_id] * 2 * pi / 60 # rad/s
        domega_dtau = 0.0  # could be updated from UDP if needed       
        # fade out when rpm -> 0
        if rpm_filtered[rotor_id] < 1e-2:
            print("rpm_filtered too small?")
            continue

        # Get coefficients (same for all samples in block)        
        c = lookup.get_coefficients(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        a0, a1, b1, a2, b2 = c["a0"], c["a1"], c["b1"], c["a2"], c["b2"]

        # Tilt matrix  (precompute once per rotor)
        tilt_rad = np.radians(90 - tilt[rotor_id]) # tilt: 90 for VFM, 0 for FFM
        aos_rad = np.radians(aos)
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
            # Update azimuth state (for continuity to next block)
            azimuth[source_id] = az_block[-1]

            # # Lift (periodic loading)
            # if rotor_direction[rotor_id] == 1:  # Counter-clockwise
            #     L = (a0 + a1 * np.cos(abs(az_block)) + b1 * np.sin(abs(az_block)) + a2 * np.cos(abs(2*az_block)) + b2 * np.sin(abs(2*az_block)))
            # else:  # Clockwise
            #     L = (a0 + a1 * np.cos(abs(az_block)-2.0*aos_rad) + b1 * np.sin(abs(az_block)-2.0*aos_rad) + a2 * np.cos(abs(2*az_block)-2.0*aos) + b2 * np.sin(abs(2*az_block)-2.0*aos))

            # Lift (periodic loading)  # corrected 2025-10-23
            L = (a0 
            + a1 * np.cos(abs(az_block-aos_rad*rotor_direction[rotor_id])) 
            + b1 * np.sin(abs(az_block-aos_rad*rotor_direction[rotor_id])) 
            + a2 * np.cos(abs(2*(az_block-aos_rad*rotor_direction[rotor_id]))) 
            + b2 * np.sin(abs(2*(az_block-aos_rad*rotor_direction[rotor_id]))))

            # Source position (before tilt)
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = rotor_center[rotor_id][2] * np.ones_like(x)
            source_position = np.stack((x, y, z), axis=1)  # (frames, 3)
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

            # Accumulate into output buffer (stereo: duplicate channels)
            out_buffer[:, 0] += p_near
            out_buffer[:, 1] += p_near            

    # Save updated azimuth back
    with state_lock:
        last_state["azimuth"][:] = azimuth

   
    # --- compute dynamic offset (mean) using partial mean over one-fifth revolution ---
    # period per revolution (s)
    T_rev = 60.0 / np.mean(rpm_filtered[rpm_filtered > 0]) if np.any(rpm_filtered > 0) else 0.1
    # choose window length = one-fifth revolution
    T_window = T_rev / NUMBER_OF_BLADES
    N_window = int(T_window * AUDIO_SAMPLE_RATE)

    # ensure window smaller than available samples
    if N_window > 0 and N_window < len(out_buffer):
        offset_dynamic = np.mean(out_buffer[:N_window, 0])
    else:
        offset_dynamic = np.mean(out_buffer[:, 0])

    block_mean = offset_dynamic
    plot_data["block_mean"].append(block_mean)
    # plot_data["block_time"].append(len(plot_data["block_time"]) * AUDIO_BLOCK_SIZE / AUDIO_SAMPLE_RATE)
    t_now = time.time()
    plot_data["block_time"].append(t_now)   

    # apply offset correction (instead of static TEST_INPUT["offset"])
    out_buffer -= offset_dynamic
    
    # print(" SCALING")
    # print(f"  Before: MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)
    out_buffer *= SCALING_FACTOR 
    # out_buffer *= volume_gain
    # print(f"  After : MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)   

    # --- compute SPL for monitoring ---
    spl_db = compute_spl_db(out_buffer[:,0])
    # print(f"SPL: {spl_db:.2f} dB, Offset applied: {offset_dynamic:.1f} Pa")
    plot_data.setdefault("spl_db", deque(maxlen=param_len))
    plot_data["spl_db"].append(spl_db)
    
    outdata[:] = out_buffer
    
    # store last few seconds for plotting
    # latest_out_buffer = np.copy(out_buffer)
    
    plot_data["wave"].extend(out_buffer[:,0])

    # update param buffers
    plot_data["spd"].append(spd)
    plot_data["tilt"].append(tilt[0])
    plot_data["coll"].append(coll_filtered[0])
    plot_data["rpm"].append(np.mean(rpm_filtered))
    plot_data["aoa"].append(aoa)
    plot_data["aos"].append(aos)
    plot_data["time"].append(time.time() % PARAM_WINDOW_SEC)
    
    # debug to identify why the left channel sound is quieter than the right channel sound
    # stereo=np.column_stack([p_near.astype(np.float32), p_near.astype(np.float32)])
    # outdata[:]=stereo

    # timing diagnotic
    # elapsed = (time.perf_counter()-start_time) * 1000.0 # ms
    # budget = frames / AUDIO_SAMPLE_RATE * 1000.0 # ms
    # if elapsed > budget:
    #     print(f" Callback overran: {elapsed:.2f} ms > {budget:.2f} ms")
    # else:
    #     print(f" Callback OK: {elapsed:.2f} ms < {budget:.2f} ms")


# ---------------- Plotting ----------------
def start_plots():
    fig1, axs1 = plt.subplots(2, 3, figsize=(10,6))
    axs1[0,0].set_title("Speed"); axs1[0,1].set_title("Tilt"); axs1[0,2].set_title("Coll");
    axs1[1,0].set_title("RPM");   axs1[1,1].set_title("AOA");  axs1[1,2].set_title("AOS")
    lines1 = [axs1[0,0].plot([],[])[0],
              axs1[0,1].plot([],[])[0],
              axs1[0,2].plot([],[])[0],
              axs1[1,0].plot([],[])[0],
              axs1[1,1].plot([],[])[0],
              axs1[1,2].plot([],[])[0]]

    fig2, axs2 = plt.subplots(4, 1, figsize=(10,8))
    axs2[0].set_title("Waveform")
    axs2[1].set_title("FFT")
    axs2[2].set_title("Segment Mean Pressure")
    axs2[3].set_title("Block SPL (dB re 20µPa)")

    line_wave = axs2[0].plot([],[])[0]
    line_fft  = axs2[1].plot([],[])[0]
    line_mean = axs2[2].plot([],[])[0]
    line_spl  = axs2[3].plot([],[])[0]


    def update(frame):
        t_vals = list(plot_data["time"])
        # for k, key in enumerate(["spd","tilt","rpm","aos"]):
        #     lines1[k].set_data(t_vals, list(plot_data[key]))
        #     axs1[k//2,k%2].set_xlim(max(0, t_vals[0]), t_vals[-1]+1e-6)
        #     # axs1[k//2,k%2].set_ylim(min(plot_data[key]), max(plot_data[key]))
        
        # corrected 2025-10-22: to avoid list index out-of-range error
        if len(t_vals) < 2:
            return [] # skip until we have data
        for k, key in enumerate(["spd","tilt","coll","rpm","aoa","aos"]):
            y_vals = list(plot_data[key])
            if len(y_vals) < 2:
                continue
            lines1[k].set_data(t_vals, y_vals)
            try:
                axs1[k//3,k%3].set_xlim(max(0, t_vals[0]), t_vals[-1]+1e-6)
            except IndexError:
                # This can happen if axs1 indexing doesn't match grid shape
                continue
        
        axs1[0,0].set_ylabel('Speed (m/s)')
        axs1[0,0].set_ylim(-10, 100)
        axs1[0,0].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[0,1].set_ylabel('Tilt (deg)')
        axs1[0,1].set_ylim(-10, 100)
        axs1[0,1].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[0,2].set_ylabel('Coll (deg)')
        axs1[0,2].set_ylim(-10, 70)
        axs1[0,2].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[1,0].set_ylabel('Rotating Speed (rpm)')
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
        t_wave = np.linspace(0, WAVEFORM_WINDOW_SEC, N)
        line_wave.set_data(t_wave, wave)
        axs2[0].set_xlim(0, WAVEFORM_WINDOW_SEC)
        # axs2[0].set_ylim(np.min(wave)-0.1*np.abs(np.min(wave)), np.max(wave)+0.1*np.abs(np.min(wave)))
        axs2[0].set_ylim(-25*SCALING_FACTOR, 25*SCALING_FACTOR)

        # FFT
        if N > 0:
            freqs = np.fft.rfftfreq(N, 1/AUDIO_SAMPLE_RATE)
            spectrum = np.abs(np.fft.rfft(wave))
            line_fft.set_data(freqs, 20*np.log10(spectrum+1e-12))
            # axs2[1].set_xlim(20, AUDIO_SAMPLE_RATE/2)
            axs2[1].set_xlim(0, 100)
            # axs2[1].set_ylim(np.min(20*np.log10(spectrum+1e-12)), np.max(20*np.log10(spectrum+1e-12))+1)
            axs2[1].set_ylim(40, 120)

        if len(plot_data["block_mean"]) > 0:
            # t_vals = list(plot_data["time"])[-len(plot_data["block_mean"]):]
            t_vals = list(plot_data["block_time"])
            mean_vals = list(plot_data["block_mean"])
            line_mean.set_data(t_vals, mean_vals)
            axs2[2].set_xlim(min(t_vals), max(t_vals) + 1e-6)
            axs2[2].set_ylim(min(mean_vals)-1, max(mean_vals)+1)

        if "spl_db" in plot_data and len(plot_data["spl_db"]) > 0:
            t_spl = list(plot_data["block_time"])[-len(plot_data["spl_db"]):]
            spl_vals = list(plot_data["spl_db"])
            # line_spl.set_data(t_spl, spl_vals)
            line_spl.set_data(t_spl[::2], spl_vals[::2])
            axs2[3].set_xlim(min(t_spl), max(t_spl)+1e-6)
            axs2[3].set_ylim(min(spl_vals)-2, max(spl_vals)+2)
            axs2[3].grid(True, axis='y', linestyle='--', alpha=0.7)

        return lines1 + [line_wave, line_fft, line_mean, line_spl]

    ani1 = FuncAnimation(fig1, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False, save_count=100)
    ani2 = FuncAnimation(fig2, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False, save_count=100)
        
    plt.show()


# ---------------- Main ----------------
def main():
    try:
        # Start UDP listener
        udp_thread = threading.Thread(target=update_state_from_udp, daemon=True)
        udp_thread.start()

        print("\nStarting real-time audio stream...")
        print(f"  Sample rate: {AUDIO_SAMPLE_RATE} Hz")
        print(f"  Block size : {AUDIO_BLOCK_SIZE} samples")
        print(f"  Channels   : {AUDIO_CHANNELS}")

        # Start audio stream
        with sd.OutputStream(channels=AUDIO_CHANNELS,
                             samplerate=AUDIO_SAMPLE_RATE,
                             blocksize=AUDIO_BLOCK_SIZE,
                             callback=audio_callback):
            start_plots()

            print("Audio stream is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    
