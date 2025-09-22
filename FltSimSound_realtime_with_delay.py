import numpy as np
import threading
import time
import socket
import struct
import queue
import sounddevice as sd
from collections import deque

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
SCALING_FACTOR = 5.0

# Maximum expected delay (based on maximum distance and speed of sound)
MAX_DISTANCE = 50.0  # meters - adjust based on your simulation space
MAX_DELAY_SAMPLES = int(MAX_DISTANCE / SPEED_OF_SOUND * AUDIO_SAMPLE_RATE)

# Keep history of states for retarded time calculations
HISTORY_DURATION = MAX_DISTANCE / SPEED_OF_SOUND  # seconds of history to keep
HISTORY_SAMPLES = int(HISTORY_DURATION * AUDIO_SAMPLE_RATE)

class StateHistory:
    def __init__(self):
        self.times = deque(maxlen=HISTORY_SAMPLES)
        self.rpms = deque(maxlen=HISTORY_SAMPLES)
        self.collectives = deque(maxlen=HISTORY_SAMPLES)
        self.tilts = deque(maxlen=HISTORY_SAMPLES)
        self.spds = deque(maxlen=HISTORY_SAMPLES)
        self.aoas = deque(maxlen=HISTORY_SAMPLES)
        self.aoss = deque(maxlen=HISTORY_SAMPLES)
        self.azimuths = {sid: deque(maxlen=HISTORY_SAMPLES) for sid in range(NUMBER_OF_SOURCES)}
        
    def append(self, state, t):
        self.times.append(t)
        self.rpms.append(state["rpm"].copy())
        self.collectives.append(state["collective"].copy())
        self.tilts.append(state["tilt"].copy())
        self.spds.append(state["spd"])
        self.aoas.append(state["aoa"])
        self.aoss.append(state["aos"])
        for sid in range(NUMBER_OF_SOURCES):
            self.azimuths[sid].append(state["azimuth"][sid])
            
    def get_interpolated_state(self, t):
        """Get state at time t using interpolation"""
        if not self.times or t < self.times[0] or t > self.times[-1]:
            return None
            
        # Find the two closest times
        idx = np.searchsorted(self.times, t)
        if idx == 0:
            idx = 1
        t0, t1 = self.times[idx-1], self.times[idx]
        alpha = (t - t0) / (t1 - t0)
        
        # Interpolate all state variables
        state = {}
        state["rpm"] = np.array(self.rpms[idx-1]) * (1-alpha) + np.array(self.rpms[idx]) * alpha
        state["collective"] = np.array(self.collectives[idx-1]) * (1-alpha) + np.array(self.collectives[idx]) * alpha
        state["tilt"] = np.array(self.tilts[idx-1]) * (1-alpha) + np.array(self.tilts[idx]) * alpha
        state["spd"] = self.spds[idx-1] * (1-alpha) + self.spds[idx] * alpha
        state["aoa"] = self.aoas[idx-1] * (1-alpha) + self.aoas[idx] * alpha
        state["aos"] = self.aoss[idx-1] * (1-alpha) + self.aoss[idx] * alpha
        state["azimuth"] = np.array([
            self.azimuths[sid][idx-1] * (1-alpha) + self.azimuths[sid][idx] * alpha
            for sid in range(NUMBER_OF_SOURCES)
        ])
        return state

# Current state and history
current_state = {
    "rpm": np.zeros(NUMBER_OF_ROTORS),
    "collective": np.zeros(NUMBER_OF_ROTORS),
    "tilt": np.zeros(NUMBER_OF_ROTORS),
    "spd": 0.0,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([2*pi*(sid%NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)])
}
state_history = StateHistory()
state_lock = threading.Lock()

# Observer and geometry setup
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398], [-2.554, 3.962, 0.398], 
                       [-4.101,-3.962, 0.667], [4.101, -3.962, 0.667]])
prop_center = tilt_center + np.array([-0.188, 0, 1.430])

# UDP Setup
SERVER_IP = "127.0.0.1"
SERVER_PORT = 1700
RecvBufferSize = 10240
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

class TableLookup:
    def __init__(self, table=None):
        self.spd_vals = np.array([0.0, 5.0, 10.0, 46.9, 49.4, 54.0, 67.0])
        self.aos_vals = np.array([0, 5, 10])
        self.coeffs = {name: np.ones((len(self.spd_vals), len(self.aos_vals))) 
                      for name in ["a0","a1","b1","a2","b2"]}

    def get_coefficients(self, spd, aos):
        return {name: 1.0 for name in ["a0","a1","b1","a2","b2"]}

lookup = TableLookup()

def update_state_from_udp():
    """Update state from UDP data and maintain history"""
    while True:
        try:
            data, _ = sock.recvfrom(RecvBufferSize)
            if len(data) == struct.calcsize('iiddd4f4f4d'):
                _, _, spd, aoa, aos, *values = struct.unpack('iiddd4f4f4d', data)
                
                with state_lock:
                    current_state["spd"] = spd
                    current_state["aoa"] = aoa
                    current_state["aos"] = aos
                    current_state["rpm"][:] = values[:NUMBER_OF_ROTORS]
                    current_state["collective"][:] = values[NUMBER_OF_ROTORS:2*NUMBER_OF_ROTORS]
                    current_state["tilt"][:] = values[2*NUMBER_OF_ROTORS:3*NUMBER_OF_ROTORS]
                    
                    # Update azimuth based on RPM
                    dt = 1.0 / AUDIO_SAMPLE_RATE
                    for sid in range(NUMBER_OF_SOURCES):
                        rotor_id = sid // NUMBER_OF_BLADES
                        omega = current_state["rpm"][rotor_id] * 2*pi/60
                        current_state["azimuth"][sid] += omega * dt
                    
                    # Add current state to history
                    state_history.append(current_state, time.time())
                    
        except Exception as e:
            print(f"UDP error: {e}")
            time.sleep(0.01)

def calculate_pressure_with_delay(source_id, observer_time):
    """Calculate acoustic pressure considering retarded time"""
    rotor_id = source_id // NUMBER_OF_BLADES
    
    # Get source position at current time (approximation for distance calculation)
    source_pos = prop_center[rotor_id]  # Base position
    r = observer_position - source_pos
    rmag = np.linalg.norm(r)
    
    # Calculate retarded time
    retarded_time = observer_time - rmag / SPEED_OF_SOUND
    
    # Get interpolated state at retarded time
    state = state_history.get_interpolated_state(retarded_time)
    if state is None:
        return 0.0
    
    # Get state variables at retarded time
    rpm = state["rpm"][rotor_id]
    omega = rpm * 2*pi/60
    az = state["azimuth"][source_id]
    
    # Update source position with accurate azimuth
    source_pos = prop_center[rotor_id] + np.array([
        ROTOR_RADIUS * np.cos(az),
        ROTOR_RADIUS * np.sin(az),
        0.
    ])
    
    # Apply tilt transformation
    tilt = state["tilt"][rotor_id]
    tilt_rad = np.radians(90 - tilt)
    trans_tilt = np.array([
        [np.cos(tilt_rad), 0, -np.sin(tilt_rad)],
        [0, 1, 0],
        [np.sin(tilt_rad), 0, np.cos(tilt_rad)]
    ])
    source_pos = tilt_center[rotor_id] + np.dot(trans_tilt, source_pos - tilt_center[rotor_id])
    
    # Recalculate observer-source vector with accurate position
    r = observer_position - source_pos
    rmag = np.linalg.norm(r)
    
    # Get coefficients for acoustic calculation
    c = lookup.get_coefficients(state["spd"], state["aos"])
    
    # Calculate acoustic terms
    M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
    L = (c["a0"] + c["a1"]*np.cos(az) + c["b1"]*np.sin(az) + 
         c["a2"]*np.cos(2*az) + c["b2"]*np.sin(2*az))
    D = 0
    
    Mi = np.array([-M*np.sin(az), M*np.cos(az), 0])
    Mi = np.dot(trans_tilt, Mi)  # Apply tilt to Mach vector
    Fi = np.array([D*np.sin(az), -D*np.cos(az), L])
    Fi = np.dot(trans_tilt, Fi)  # Apply tilt to force vector
    
    Mr = np.dot(r, Mi)/rmag
    Fr = np.dot(r, Fi)/rmag
    
    # Calculate pressure (with both near and far field terms)
    p_near = 0.25/pi * (1/(1-Mr)**2/rmag**2 * (Fr*(1-M**2)/(1-Mr) - np.dot(Fi,Mi)))
    p = p_near  # Simplified for real-time, but could add far-field term if needed
    
    return p

def audio_callback(outdata, frames, time_info, status):
    """Real-time audio callback with proper time delay handling"""
    if status:
        print(status)
    
    current_time = time.time()
    out_buffer = np.zeros((frames, AUDIO_CHANNELS))
    
    # Calculate pressures for each sample in the buffer
    for i in range(frames):
        sample_time = current_time + i/AUDIO_SAMPLE_RATE
        pressure = sum(calculate_pressure_with_delay(sid, sample_time) 
                      for sid in range(NUMBER_OF_SOURCES))
        out_buffer[i] = [pressure, pressure] * SCALING_FACTOR
    
    outdata[:] = out_buffer

def main():
    try:
        udp_thread = threading.Thread(target=update_state_from_udp, daemon=True)
        udp_thread.start()
        
        print("\nStarting real-time audio stream with retarded time...")
        print(f"Sample rate: {AUDIO_SAMPLE_RATE} Hz")
        print(f"Block size: {AUDIO_BLOCK_SIZE} samples")
        print(f"Maximum delay: {MAX_DELAY_SAMPLES/AUDIO_SAMPLE_RATE*1000:.1f} ms")
        
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
