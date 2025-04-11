import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
from TTS.api import TTS
from pydub import AudioSegment
import simpleaudio as sa

# === CONFIGURATION ===
SAMPLE_RATE = 16000
DURATION = 6  # seconds
VOICE_SAMPLE_PATH = r"C:\Users\DELL 3410\Projects\fast_app\interview\mohan-v.wav"
OUTPUT_PATH = "cloned-voice.wav"
TEXT_TO_CLONE = "Artificial Intelligence, or AI, is revolutionizing our world by enabling machines to think, learn, and make decisions. From healthcare to finance, its applications are transforming industries while raising critical ethical concerns about privacy and fairness"

# === STEP 1: Record voice (only if not already saved) ===
def record_voice_if_needed():
    if os.path.exists(VOICE_SAMPLE_PATH):
        print(f"📁 Using existing voice sample: {VOICE_SAMPLE_PATH}")
        return 0.0
    
    print("🎤 Recording voice sample... Speak clearly.")
    start = time.time()
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    wav.write(VOICE_SAMPLE_PATH, SAMPLE_RATE, recording)
    duration = time.time() - start
    print(f"✅ Voice sample saved to: {VOICE_SAMPLE_PATH} ({duration:.2f}s)")
    return duration

# === STEP 2: Synthesize speech ===
def synthesize_voice():
    print("🧠 Loading XTTS-v2 model and generating speech...")
    start = time.time()
    
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

    tts.tts_to_file(
        text=TEXT_TO_CLONE,
        speaker_wav=VOICE_SAMPLE_PATH,
        language="en",
        file_path=OUTPUT_PATH
    )
    duration = time.time() - start
    print(f"✅ Audio generated and saved to: {OUTPUT_PATH} ({duration:.2f}s)")
    return duration

# === STEP 3: Play audio ===
def play_audio(file_path):
    print("🔊 Playing generated voice...")
    start = time.time()
    sound = AudioSegment.from_wav(file_path)
    play_obj = sa.play_buffer(
        sound.raw_data,
        num_channels=sound.channels,
        bytes_per_sample=sound.sample_width,
        sample_rate=sound.frame_rate
    )
    play_obj.wait_done()
    duration = time.time() - start
    print(f"🔁 Playback finished ({duration:.2f}s)")
    return duration

# === MAIN FLOW ===
if __name__ == "__main__":
    total_start = time.time()

    rec_time = record_voice_if_needed()
    synth_time = synthesize_voice()
    play_time = play_audio(OUTPUT_PATH)

    total_time = time.time() - total_start

    print("\n=== Summary ===")
    print(f"🎙️ Recording Time  : {rec_time:.2f}s")
    print(f"🧠 Synthesis Time  : {synth_time:.2f}s")
    print(f"🔊 Playback Time   : {play_time:.2f}s")
    print(f"⏱️ Total Time      : {total_time:.2f}s")
