import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ============================================================
# Configuration
# ============================================================
WIND_FILE = "wind.wav"
AUDIO_SAMPLE_RATE = 22050
AUDIO_BLOCK_SIZE = 4096
WIND_GAIN = 0.7          # overall loudness
CROSSFADE_MS = 30        # crossfade duration in ms
RANDOMIZE_SEGMENT = True # pick random segment to loop for realism

# ============================================================
# Load and prepare the wind sample
# ============================================================
def load_wind_noise():
    sr, data = wavfile.read(WIND_FILE)

    # --- Convert to float32 in [-1, 1] ---
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)

    # --- Mono → stereo if needed ---
    if data.ndim == 1:
        data = np.stack((data, data), axis=-1)

    # --- Resample using interpolation ---
    if sr != AUDIO_SAMPLE_RATE:
        print(f"Interpolating from {sr} → {AUDIO_SAMPLE_RATE} Hz")
        ratio = AUDIO_SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, new_len)
        data = np.column_stack([
            np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])
        ])

    # --- Fade edges to prevent clicks ---
    fade_len = int(CROSSFADE_MS * AUDIO_SAMPLE_RATE / 1000)
    fade = np.linspace(0, 1, fade_len)
    for ch in range(data.shape[1]):
        data[:fade_len, ch] *= fade
        data[-fade_len:, ch] *= fade[::-1]

    # --- Normalize and apply gain ---
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    data *= WIND_GAIN
    return data

# ============================================================
# Continuous cross-fading wind streamer
# ============================================================
class WindStreamer:
    def __init__(self, data, sample_rate=AUDIO_SAMPLE_RATE, crossfade_ms=30):
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

            # --- Crossfade when reaching end of loop ---
            if self.idx >= self.len:
                overlap = min(self.fade_len, start)
                tail = self.data[-overlap:]
                head_start = self.next_offset
                head = self.data[head_start:head_start+overlap]
                blend = tail * self.fade_out[:overlap, None] + head * self.fade_in[:overlap, None]
                out[start-overlap:start] = blend

                self.idx = head_start + overlap
                self.next_offset = np.random.randint(0, self.len - self.fade_len)
        return out

# ============================================================
# Playback test
# ============================================================
if __name__ == "__main__":
    wind_data = load_wind_noise()
    streamer = WindStreamer(wind_data)

    def callback(outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:] = streamer.get_block(frames)

    print("Playing continuous wind noise (interpolated)... Press Ctrl+C to stop.")
    with sd.OutputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=2,
        dtype="float32",
        callback=callback,
        blocksize=AUDIO_BLOCK_SIZE,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")