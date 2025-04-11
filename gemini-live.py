import asyncio
import base64
import io
import traceback
import os
import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=GEMINI_API_KEY)

CONFIG = types.LiveConnectConfig(
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

pya = pyaudio.PyAudio()

# Job Description Prompt
JD_PROMPT = """
You are an AI interviewer conducting a live interview for the following job position:

**Job Title:** Junior AI Developer  
**Job Type:** Full-Time  
**Location:** Remote  
**Responsibilities:**  
- Develop and maintain AI models for real-time applications.  
- Collaborate with cross-functional teams to integrate AI solutions.  
- Analyze data and optimize algorithms for performance.  
- Ensure ethical AI practices and compliance with regulations.  
- Participate in code reviews and debugging sessions.  

**Requirements:**  
- Bachelor’s degree in Computer Science, Engineering, or related field.  
- Proficiency in Python and familiarity with AI frameworks (e.g., TensorFlow, PyTorch).  
- Understanding of machine learning concepts and real-time systems.  
- Strong problem-solving skills and attention to detail.  
- Excellent communication and teamwork abilities.  

**Preferred Skills:**  
- Experience with computer vision or audio processing.  
- Knowledge of cloud platforms (e.g., AWS, Google Cloud).  
- Prior work with APIs and live data streams.  

Your task is to:
1. Ask the candidate a series of interview questions based on this job description.
2. Monitor the candidate via camera to detect potential cheating (e.g., looking away excessively, unusual movements).
3. Provide audio-based feedback or warnings if cheating is suspected.
4. Proceed with the next question after receiving an audio response from the candidate.

Start the interview by greeting the candidate and asking the first question: 
"Tell me about yourself and how your background aligns with the Junior AI Developer role."
"""

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.interview_started = False

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(3.0)
            await self.out_queue.put(frame)
        cap.release()

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if text := response.text:
                    print(text, end="")

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def start_interview(self):
        if not self.interview_started:
            await self.session.send(input=JD_PROMPT, end_of_turn=True)
            self.interview_started = True

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.start_interview())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await asyncio.Future()  # Run indefinitely until manually stopped

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())