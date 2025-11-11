import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ---------------- Load wind.wav ----------------
WIND_FILE = "wind.wav"
sr, wind_data = wavfile.read(WIND_FILE)

# Convert to float32 in [-1, 1]
if wind_data.dtype == np.int16:
    wind_data = wind_data.astype(np.float32) / 32768.0
elif wind_data.dtype == np.int32:
    wind_data = wind_data.astype(np.float32) / 2147483648.0
elif wind_data.dtype == np.uint8:
    wind_data = (wind_data.astype(np.float32) - 128) / 128.0
else:
    wind_data = wind_data.astype(np.float32)

# Convert mono → stereo if needed
if wind_data.ndim == 1:
    wind_data = np.stack((wind_data, wind_data), axis=-1)

num_samples = len(wind_data)
sample_rate = sr
print(f"Loaded wind.wav: {num_samples} samples at {sample_rate} Hz")

# ---------------- Stream callback ----------------
wind_idx = 0  # playback index

def audio_callback(outdata, frames, time_info, status):
    global wind_idx
    if status:
        print(status)

    # Prepare output buffer
    out = np.zeros((frames, 2), dtype=np.float32)
    remaining = frames
    start = 0

    # Loop wind.wav seamlessly
    while remaining > 0:
        chunk = min(remaining, num_samples - wind_idx)
        out[start:start+chunk] = wind_data[wind_idx:wind_idx+chunk]
        wind_idx = (wind_idx + chunk) % num_samples
        remaining -= chunk
        start += chunk

    outdata[:] = out  # send to output

# ---------------- Start streaming ----------------
with sd.OutputStream(
    samplerate=sample_rate,
    channels=2,
    dtype='float32',
    callback=audio_callback,
    blocksize=1024
):
    print("Playing wind sound continuously... Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped.")