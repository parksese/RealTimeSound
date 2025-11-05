import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import time

# -----------------------------------------
# User-configurable parameters
AUDIO_RATE = 22050        # your existing sample rate
BLOCK = 1024              # callback block size
# -----------------------------------------

# ------------ Load wind.wav --------------
fs, wind = wavfile.read("wind.wav")   # shape: (N,) or (N,2)

# Convert to float32 (normalize)
if wind.dtype == np.int16:
    wind = wind.astype(np.float32) / 32768.0
elif wind.dtype == np.int32:
    wind = wind.astype(np.float32) / 2147483648.0
else:
    wind = wind.astype(np.float32)

# Stereo -> mono
if wind.ndim == 2:
    wind = wind.mean(axis=1)

# Resample if needed
if fs != AUDIO_RATE:
    Nout = int(len(wind) * AUDIO_RATE / fs)
    wind = resample(wind, Nout)

# Loop index
play_index = 0
wind_len = len(wind)

# Simulated variable
wind_speed = 0.0

# ---------------- Callback ----------------
def callback(outdata, frames, time_info, status):
    global play_index, wind_speed

    # Simulate speed increasing
    wind_speed += 0.05
    if wind_speed > 60:
        wind_speed = 0.0

    # Gain mapping
    gain = np.clip(wind_speed / 60, 0.0, 1.0)  # normalized [0,1]

    # Allocate output
    block = np.zeros(frames, dtype=np.float32)

    # Build looped block
    for i in range(frames):
        block[i] = wind[play_index] * gain
        play_index = (play_index + 1) % wind_len

    # Stereo mix
    outdata[:,0] = block
    outdata[:,1] = block

# ---------------- Main ----------------
def main():
    print("Playing wind noise test...")
    with sd.OutputStream(channels=2,
                         samplerate=AUDIO_RATE,
                         blocksize=BLOCK,
                         callback=callback):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()