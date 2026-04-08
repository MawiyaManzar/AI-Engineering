import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
MODEL_SIZE = "small"

# =========================
# LOAD MODEL
# =========================
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",          # change to "cuda" if GPU
    compute_type="int8"
)

# =========================
# AUDIO QUEUE
# =========================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# =========================
# RECORDING THREAD
# =========================
def record_audio():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        print("🎤 Listening...")
        while True:
            sd.sleep(1000)

# =========================
# TRANSCRIPTION THREAD
# =========================
def transcribe_audio():
    buffer = []

    while True:
        data = audio_queue.get()
        buffer.append(data)

        # Convert buffer to chunk
        if len(buffer) * len(data) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_chunk = np.concatenate(buffer, axis=0).flatten()
            buffer = []

            segments, _ = model.transcribe(
                audio_chunk,
                beam_size=1   # low latency
            )

            for segment in segments:
                print("📝", segment.text)

# =========================
# RUN
# =========================
threading.Thread(target=record_audio).start()
threading.Thread(target=transcribe_audio).start()