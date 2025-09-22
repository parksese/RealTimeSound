import numpy as np
import threading
import time
import socket
import struct
import queue
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
SCALING_FACTOR = 5.0

# Keep the last known state for real-time calculation
last_state = {
    "rpm": np.zeros(NUMBER_OF_ROTORS),
    "collective": np.zeros(NUMBER_OF_ROTORS),
    "tilt": np.zeros(NUMBER_OF_ROTORS),
    "spd": 0.0,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([2*pi*(sid%NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)]),
    "last_update_time": time.time()
}
state_lock = threading.Lock()

# Observer and geometry setup remains the same
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398], [-2.554, 3.962, 0.398], 
                       [-4.101,-3.962, 0.667], [4.101, -3.962, 0.667]])
prop_center = tilt_center + np.array([-0.188, 0, 1.430])

# ---------------- UDP Setup ----------------
SERVER_IP = "127.0.0.1"
SERVER_PORT = 1700
RecvBufferSize = 10240
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

# ---------------- TableLookup remains the same ----------------
class TableLookup:
    def __init__(self, table=None):
        self.spd_vals = np.array([0.0, 5.0, 10.0, 46.9, 49.4, 54.0, 67.0])
        self.aos_vals = np.array([0, 5, 10])
        self.coeffs = {name: np.ones((len(self.spd_vals), len(self.aos_vals))) 
                      for name in ["a0","a1","b1","a2","b2"]}

    def get_coefficients(self, spd, aos):
        return {name: 1.0 for name in ["a0","a1","b1","a2","b2"]}  # Simplified for testing

lookup = TableLookup()

def update_state_from_udp():
    """Update state from UDP data"""
    while True:
        try:
            data, _ = sock.recvfrom(RecvBufferSize)
            if len(data) == struct.calcsize('iiddd4f4f4d'):
                _, _, spd, aoa, aos, *values = struct.unpack('iiddd4f4f4d', data)
                
                with state_lock:
                    last_state["spd"] = spd
                    last_state["aoa"] = aoa
                    last_state["aos"] = aos
                    last_state["rpm"][:] = values[:NUMBER_OF_ROTORS]
                    last_state["collective"][:] = values[NUMBER_OF_ROTORS:2*NUMBER_OF_ROTORS]
                    last_state["tilt"][:] = values[2*NUMBER_OF_ROTORS:3*NUMBER_OF_ROTORS]
                    last_state["last_update_time"] = time.time()
                    
        except Exception as e:
            print(f"UDP error: {e}")
            time.sleep(0.01)

def calculate_pressure_realtime(source_id, current_time):
    """Calculate acoustic pressure for a single source in real-time"""
    with state_lock:
        dt = current_time - last_state["last_update_time"]
        rotor_id = source_id // NUMBER_OF_BLADES
        
        # Get current state
        rpm = last_state["rpm"][rotor_id]
        omega = rpm * 2*pi/60
        
        # Update azimuth in real-time
        az = last_state["azimuth"][source_id] + omega * dt
        last_state["azimuth"][source_id] = az
        
        # Get coefficients
        c = lookup.get_coefficients(last_state["spd"], last_state["aos"])
        
        # Calculate source position
        source_pos = prop_center[rotor_id] + np.array([
            ROTOR_RADIUS * np.cos(az),
            ROTOR_RADIUS * np.sin(az),
            0.
        ])
        
        # Calculate acoustic terms
        M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
        L = (c["a0"] + c["a1"]*np.cos(az) + c["b1"]*np.sin(az) + 
             c["a2"]*np.cos(2*az) + c["b2"]*np.sin(2*az))
        D = 0
        
        Mi = np.array([-M*np.sin(az), M*np.cos(az), 0])
        Fi = np.array([D*np.sin(az), -D*np.cos(az), L])
        
        r = observer_position - source_pos
        rmag = np.linalg.norm(r)
        Mr = np.dot(r, Mi)/rmag
        Fr = np.dot(r, Fi)/rmag
        
        # Simplified pressure calculation for real-time
        p_near = 0.25/pi * (1/(1-Mr)**2/rmag**2 * (Fr*(1-M**2)/(1-Mr) - np.dot(Fi,Mi)))
        p = p_near  # Simplified to just near-field for real-time performance
        
        return p

def audio_callback(outdata, frames, time_info, status):
    """Real-time audio callback"""
    if status:
        print(status)
    
    current_time = time.time()
    out_buffer = np.zeros((frames, AUDIO_CHANNELS))
    
    # Calculate pressure for each source in real-time
    for source_id in range(NUMBER_OF_SOURCES):
        p = calculate_pressure_realtime(source_id, current_time)
        out_buffer += p
    
    # Apply scaling
    out_buffer *= SCALING_FACTOR
    
    # Assign to output buffer
    outdata[:] = out_buffer

def main():
    try:
        # Start UDP listener thread
        udp_thread = threading.Thread(target=update_state_from_udp, daemon=True)
        udp_thread.start()
        
        print("\nStarting real-time audio stream...")
        print(f"Sample rate: {AUDIO_SAMPLE_RATE} Hz")
        print(f"Block size: {AUDIO_BLOCK_SIZE} samples")
        print(f"Channels: {AUDIO_CHANNELS}")
        
        # Start real-time audio stream
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
