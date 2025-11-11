import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample

# ---------------- Configuration ----------------
WIND_FILE = "wind.wav"
AUDIO_SAMPLE_RATE = 44100
WIND_GAIN = 0.6         # independent gain scaling factor
CROSSFADE_MS = 20       # crossfade length in milliseconds

# ---------------- Load & preprocess ----------------
def load_wind_noise():
    """Load and preprocess wind.wav for seamless streaming."""
    sr, wind_data = wavfile.read(WIND_FILE)

    # Convert to float32 [-1, 1]
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

    # Resample to match simulation rate
    if sr != AUDIO_SAMPLE_RATE:
        print(f"Resampling wind.wav from {sr} → {AUDIO_SAMPLE_RATE} Hz")
        new_len = int(len(wind_data) * AUDIO_SAMPLE_RATE / sr)
        wind_data = resample(wind_data, new_len, axis=0)

    # Apply crossfade at loop edges (gentle blending)
    fade_len = int(CROSSFADE_MS * AUDIO_SAMPLE_RATE / 1000)
    fade = np.linspace(0, 1, fade_len)
    for ch in range(wind_data.shape[1]):
        wind_data[:fade_len, ch] *= fade        # fade-in
        wind_data[-fade_len:, ch] *= fade[::-1] # fade-out

    # Normalize & apply gain
    wind_data /= np.max(np.abs(wind_data))
    wind_data *= WIND_GAIN

    return wind_data

# ---------------- Seamless loop class ----------------
class WindStreamer:
    def __init__(self, wind_data, sample_rate=44100, crossfade_ms=50):
        self.wind_data = wind_data
        self.num_samples = len(wind_data)
        self.idx = 0
        self.sample_rate = sample_rate

        # fade settings
        self.fade_len = int(crossfade_ms * sample_rate / 1000)
        self.fade_in = np.linspace(0, 1, self.fade_len)
        self.fade_out = np.linspace(1, 0, self.fade_len)

        # pre-buffer overlap region (used when looping)
        self.overlap_buf = np.zeros((self.fade_len, 2), dtype=np.float32)

    def get_block(self, frames: int) -> np.ndarray:
        """Return the next continuous block with seamless crossfaded looping."""
        out = np.zeros((frames, 2), dtype=np.float32)
        remaining = frames
        start = 0

        while remaining > 0:
            # if remaining samples fit before loop boundary
            chunk = min(remaining, self.num_samples - self.idx)
            out[start:start+chunk] = self.wind_data[self.idx:self.idx+chunk]
            self.idx += chunk
            start += chunk
            remaining -= chunk

            # if we reached the end, blend overlap
            if self.idx >= self.num_samples:
                overlap = min(self.fade_len, frames)
                # fetch both tail and head
                tail = self.wind_data[-overlap:]
                head = self.wind_data[:overlap]
                # overlap-add crossfade
                blended = tail * self.fade_out[:overlap,None] + head * self.fade_in[:overlap,None]
                out[start-overlap:start] = blended
                self.idx = overlap  # continue from beginning
        return out

# ---------------- Testing standalone playback ----------------
if __name__ == "__main__":
    wind_data = load_wind_noise()
    streamer = WindStreamer(wind_data)

    def callback(outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:] = streamer.get_block(frames)

    print("Playing wind noise continuously... Press Ctrl+C to stop.")
    with sd.OutputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=2,
        dtype='float32',
        callback=callback,
        blocksize=1024
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")