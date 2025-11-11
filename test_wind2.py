import numpy as np
import sounddevice as sd
from scipy.io import wavfile, resample

# ============================================================
# Configuration
# ============================================================
WIND_FILE = "wind.wav"
AUDIO_SAMPLE_RATE = 44100
WIND_GAIN = 0.7          # overall loudness
CROSSFADE_MS = 50        # crossfade duration in milliseconds
RANDOMIZE_SEGMENT = True # adds realistic non-repeating texture

# ============================================================
# Utility: Load and preprocess wind.wav
# ============================================================
def load_wind_noise():
    """Load wind.wav and normalize / resample / fade edges."""
    sr, data = wavfile.read(WIND_FILE)

    # --- Normalize to float32 [-1, 1] ---
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    # --- Convert mono to stereo if needed ---
    if data.ndim == 1:
        data = np.stack((data, data), axis=-1)

    # --- Resample if necessary ---
    if sr != AUDIO_SAMPLE_RATE:
        print(f"Resampling {sr} → {AUDIO_SAMPLE_RATE} Hz")
        new_len = int(len(data) * AUDIO_SAMPLE_RATE / sr)
        data = resample(data, new_len, axis=0)

    # --- Edge fade (avoid clicks at file ends) ---
    fade_len = int(CROSSFADE_MS * AUDIO_SAMPLE_RATE / 1000)
    fade = np.linspace(0, 1, fade_len)
    for ch in range(data.shape[1]):
        data[:fade_len, ch] *= fade        # fade-in
        data[-fade_len:, ch] *= fade[::-1] # fade-out

    # --- Normalize amplitude ---
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak

    data *= WIND_GAIN
    return data

# ============================================================
# WindStreamer Class
# ============================================================
class WindStreamer:
    """Continuously loop wind noise with smooth random crossfades."""
    def __init__(self, data, sample_rate=AUDIO_SAMPLE_RATE, crossfade_ms=50):
        self.data = data
        self.len = len(data)
        self.idx = 0
        self.sample_rate = sample_rate
        self.fade_len = int(crossfade_ms * sample_rate / 1000)
        self.fade_in = np.linspace(0, 1, self.fade_len)
        self.fade_out = np.linspace(1, 0, self.fade_len)
        self.next_offset = np.random.randint(0, self.len - self.fade_len)

    def get_block(self, frames):
        out = np.zeros((frames, 2), dtype=np.float32)
        start = 0
        while start < frames:
            remain = frames - start
            chunk = min(remain, self.len - self.idx)
            out[start:start+chunk] = self.data[self.idx:self.idx+chunk]
            self.idx += chunk
            start += chunk

            # --- If reached end, crossfade to a new random segment ---
            if self.idx >= self.len:
                overlap = self.fade_len
                tail = self.data[-overlap:]
                new_head_start = self.next_offset
                head = self.data[new_head_start:new_head_start+overlap]
                blend = tail*self.fade_out[:,None] + head*self.fade_in[:,None]
                out[start-overlap:start] = blend
                # jump to new random segment
                self.idx = new_head_start + overlap
                self.next_offset = np.random.randint(0, self.len - self.fade_len)
        return out

# ============================================================
# Stand-alone Playback Test
# ============================================================
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