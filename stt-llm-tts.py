import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from groq import Groq
import wave
import time
import threading
import torch
import tempfile
from pydub import AudioSegment
import simpleaudio as sa


# Choose one of these TTS options:
# Option 1: Pyttsx3 - Fast but lower quality
USE_PYTTSX3 = False
if USE_PYTTSX3:
    import pyttsx3
    
# Option 2: gTTS - Google's TTS (requires internet)
USE_GTTS = False 
if USE_GTTS:
    from gtts import gTTS
    import io
    from pydub import AudioSegment
    import simpleaudio as sa


# Option 3: Edge-TTS - Microsoft Edge's TTS (requires internet but fast streaming)
USE_EDGE_TTS = True
if USE_EDGE_TTS:
    import asyncio
    import edge_tts

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Audio parameters for recording
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1         # Mono
BLOCK_SIZE = 4096    # Larger for faster processing
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0

# Initialize TTS engine if using pyttsx3
if USE_PYTTSX3:
    tts_engine = pyttsx3.init()
    # Speed up the rate (default is 200)
    tts_engine.setProperty('rate', 225)
    # Get available voices and use a better one if available
    voices = tts_engine.getProperty('voices')
    # Try to find a female voice
    for voice in voices:
        if "female" in voice.name.lower():
            tts_engine.setProperty('voice', voice.id)
            break

def record_audio():
    """Record audio until silence is detected"""
    print("Listening... (speak now)")
    
    audio_chunks = []
    silence_start = None
    recording = True
    
    def audio_callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata**2))
        audio_chunks.append(indata.copy())
        
        nonlocal silence_start, recording
        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                recording = False
                raise sd.CallbackStop()
        else:
            silence_start = None

    try:
        # Use device 5 as in the original code
        input_device = 5  # Microphone Array
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
            device=input_device
        ):
            while recording:
                sd.sleep(100)
    except Exception as e:
        print(f"Recording error: {e}")
        return None

    if not audio_chunks:
        return None
            
    # Concatenate audio chunks
    audio = np.concatenate(audio_chunks, axis=0)
    if len(audio) / SAMPLE_RATE < 0.5:  # Discard if shorter than 0.5s
        return None
    
    # Save to temp file
    temp_file = "temp_input.wav"
    wavfile.write(temp_file, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return temp_file

def transcribe_audio(audio_file):
    """Transcribe audio using Groq Whisper"""
    try:
        start_time = time.time()
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
        print(f"Transcription time: {time.time() - start_time:.2f}s")
        return transcription.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def generate_response(text, history):
    """Generate a response using Groq Llama"""
    start_time = time.time()
    
    # Build messages with conversation history for context
    messages = [
        {"role": "system", "content": "You're an expert interviewer conducting a technical interview. Keep your questions concise and related to ml."}
    ]
    
    # Add limited conversation history
    for exchange in history[-2:]:
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})
        
    # Add current user message
    messages.append({"role": "user", "content": text})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fastest model
            messages=messages,
            temperature=0.7,
            max_tokens=75  # Keep responses shorter
        )
        print(f"Response generation time: {time.time() - start_time:.2f}s")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Response error: {e}")
        return "Could you repeat that, please?"

def text_to_speech_pyttsx3(text):
    """Convert text to speech using pyttsx3 (fast local TTS)"""
    start_time = time.time()
    tts_engine.say(text)
    tts_engine.runAndWait()
    print(f"TTS time: {time.time() - start_time:.2f}s")

def text_to_speech_gtts(text):
    """Convert text to speech using Google TTS"""
    start_time = time.time()
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        temp_file = "temp_output.mp3"
        tts.save(temp_file)

        # Play using pydub + simpleaudio
        from pydub import AudioSegment
        import simpleaudio as sa

        sound = AudioSegment.from_file(temp_file, format="mp3")
        play_obj = sa.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        play_obj.wait_done()

        os.remove(temp_file)
        print(f"TTS time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"TTS error: {e}")


async def text_to_speech_edge(text):
    """Convert text to speech using Microsoft Edge TTS (streaming)"""
    start_time = time.time()
    try:
        # Use a fast voice - you can change this to other voices
        voice = "en-US-AriaNeural"
        
        # Create communication
        communicate = edge_tts.Communicate(text, voice)
        
        # Save to file and play
        temp_file = "temp_output.mp3"
        await communicate.save(temp_file)
        
        # Replace playsound(temp_file)
        sound = AudioSegment.from_file(temp_file, format="mp3")
        play_obj = sa.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        play_obj.wait_done()

        
        # Clean up
        os.remove(temp_file)
        print(f"TTS time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Edge TTS error: {e}")

def text_to_speech(text):
    """Convert text to speech using the selected method"""
    if USE_PYTTSX3:
        text_to_speech_pyttsx3(text)
    elif USE_GTTS:
        text_to_speech_gtts(text)
    elif USE_EDGE_TTS:
        asyncio.run(text_to_speech_edge(text))

def main():
    """Main interview loop"""
    print("=== Fast Interview Assistant Started ===")
    print("Press Ctrl+C to exit")
    
    history = []
    
    while True:
        try:
            # Record audio
            start_time = time.time()
            audio_file = record_audio()
            if not audio_file:
                continue
            
            # Transcribe audio
            transcription = transcribe_audio(audio_file)
            os.remove(audio_file)
            
            if not transcription:
                continue
                
            print(f"You: {transcription}")
            
            # Generate response
            response = generate_response(transcription, history)
            print(f"Assistant: {response}")
            
            # Update history
            history.append({"user": transcription, "assistant": response})
            if len(history) > 5:
                history.pop(0)
            
            # Convert to speech
            text_to_speech(response)
            
            # Show total processing time
            print(f"Total time: {time.time() - start_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n=== Interview Assistant Stopped ===")
            break

if __name__ == "__main__":
    main()