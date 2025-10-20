import numpy as np
import threading
import time
import socket
import struct
import sounddevice as sd

# ---------------- Constants ----------------
pi = np.pi
SPEED_OF_SOUND = 334.3
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BLOCK_SIZE = 1024  # Smaller block size for lower latency
# AUDIO_BLOCK_SIZE = 4096  # Smaller block size for lower latency
SCALING_FACTOR = 1 #10.0 #0.02 #0.008



rpm_filter = 0.0
rpm_target = 0.0
volume_ramp = 0.0 # current ramp level (0 to 1)

UDP_TIMEOUT = 1.0
VOLUME_RAMP_SPEED=0.05
volume_gain = 0.0
smoothing_factor = 0.005

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

# Observer and geometry setup
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398],    # FL                        
                        [-2.554, 3.962, 0.398],     # FR                        
                        [4.101, -3.962, 0.667],     # AL    
                        [4.101, 3.962, 0.667]])     # AR                        
rotor_center = tilt_center + np.array([-0.188, 0, 1.430]) # based on VFM mode
rotor_direction = np.array([-1, 1, -1, 1])  # 1: Counter-Clockwise, 2: Clockwise

# ---------------- UDP Setup ----------------
# SERVER_IP = "127.0.0.1"
SERVER_IP = "192.168.100.50"
SERVER_PORT = 1700
RecvBufferSize = 10240
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

# ---------------- TableLookup ----------------
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
            # [67.0, 1,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
            [67.0, 1,   0, 60, 108.69,    0.00,     0.00,    0.00,    0.00], 
            [67.0, 1,   5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
            [67.0, 1,  10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25],
            [85.0, 0, -10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25], # case7 17.6 rad/s = 168 rpm
            [85.0, 0,  -5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
            # [85.0, 0,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
            [85.0, 0,   0, 60, 108.69,    0.00,     0.00,    0.00,    0.00], 
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

# ---------------- UDP Listener ----------------
def update_state_from_udp():
    """Update state from UDP data"""
    while True:
        try:
            fmt = '<ii3f4f4f4d' # no padding
            expected_size = struct.calcsize(fmt)
            
            data, _ = sock.recvfrom(RecvBufferSize)
            
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
                # print(f"  aoa  = {aoa:.1f} deg")
                # print(f"  aos  = {aos:.1f} deg")
                print(f"  rpm  = {values[:NUMBER_OF_ROTORS]} rpm")
                # print(f"  coll = {values[NUMBER_OF_ROTORS:2*NUMBER_OF_ROTORS]} deg")
                print(f"  tilt = {values[2*NUMBER_OF_ROTORS:3*NUMBER_OF_ROTORS]} deg\n")
                

            else:
                print(f"Wrong packet size: got {len(data)}, expected {expected_size}")
        
        except Exception as e:
            print(f"UDP error: {e}")
            time.sleep(0.01)

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
        omega = rpm_filtered[rotor_id] * 2 * pi / 60
        domega_dtau = 0.0  # could be updated from UDP if needed       
        # fade out when rpm -> 0
        if rpm_filtered[rotor_id] < 1e-2:
            print("rpm_filtered too small?")
            continue

        # Get coefficients (same for all samples in block)    
        c = lookup.get_coefficients(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        # print(f"spd={spd}, coll={coll_filtered}")
        # print(f"c={c}")
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
            
            # subtract mean offset ??? temporary
            p_near -= np.mean(p_near)

            # if rotor_id ==0 and blade ==0:
            #     print(p_near)


            # Accumulate into output buffer (stereo: duplicate channels)
            out_buffer[:, 0] += p_near
            out_buffer[:, 1] += p_near
            

    # Save updated azimuth back
    with state_lock:
        last_state["azimuth"][:] = azimuth

    # # Apply scaling 
    # print(" SCALING")
    # print(f"  Before: MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)
    out_buffer *= SCALING_FACTOR * volume_gain
    # print(f"  After : MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)   

    # Assign to output
    # print(out_buffer[:,0].mean(), out_buffer[:,1].mean())
    # print(out_buffer[:,0].min(), out_buffer[:,1].max())
    outdata[:] = out_buffer
    
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
            print("Audio stream is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    

