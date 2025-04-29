# 🧠 Convo-AI: Voice-Based Interview Suite

This repository provides two powerful AI-based systems for real-time interviews and testing using voice:

1. 🎙️ **Gemini Live Interview Assistant** — Fully real-time system using Google Gemini for voice+video interviews  
2. 🧪 **STT–LLM–TTS Pipeline** — Voice-based interview assistant using OpenAI, Groq Whisper, and local or cloud TTS options  

---

## 📁 Contents

| File | Description |
|------|-------------|
| `main.py` | Gemini-based real-time streaming interview assistant |
| `stt--llm-tts.py` | Local voice-based interview system with STT → GPT → TTS |
| `requirements.txt` | All dependencies for both systems |
| `README.md` | You're reading it! 👋 |

---

## ✅ Features

### Gemini Live Interview (`main.py`)
- Real-time AI voice assistant (powered by Gemini)
- Camera or screen capture as context
- Live audio streaming and TTS playback
- Smart job description prompt
- Follow-ups, silence detection, real-time conversation

### STT–LLM–TTS System (`stt--llm-tts.py`)
- Records your voice → transcribes using **Groq Whisper**
- Passes transcription to **OpenAI GPT-4o-mini**
- Detects off-topic answers and course-corrects
- Converts AI response back to speech using:
  - ✅ Edge TTS *(default)*
  - ✅ pyttsx3 *(offline option)*
  - ✅ Google TTS *(optional)*

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### 📦 `requirements.txt`

```
google-generativeai
openai
pymongo
pyodbc
numpy
sounddevice
scipy
opencv-python
pyaudio
pillow
mss
gtts
pydub
simpleaudio
edge-tts
```

---

## 🔐 Setup Environment Variables

```bash
# For Gemini
export GEMINI_API_KEY=your_google_gemini_key

# For OpenAI
export OPENAI_API_KEY=your_openai_key
```

---

## ▶️ Usage

### 🎤 Run Gemini Live Interview Assistant

```bash
python main.py --mode camera
```

Modes:
- `camera` → capture webcam
- `screen` → capture screen
- `none` → audio-only

---

### 🧪 Run STT–LLM–TTS Voice Interview

```bash
python stt--llm-tts.py
```

✅ Will ask up to 10 questions  
✅ Detects silence to stop recording  
✅ Stores Q/A in MongoDB or SQL Server if configured  

---

## ⚙️ Notes

- Requires working **microphone** and optionally a **camera**
- On Windows, for `pyaudio`, install via:

```bash
pip install pipwin
pipwin install pyaudio
```

- Groq's Whisper is assumed to be available (`groq` Python SDK required for `transcribe_audio_groq`)
- Replace `fetch_transcript()` with your MongoDB logic for live data

---

## 📈 Roadmap

- [ ] Auto scoring system
- [ ] Resume parsing and matching
- [ ] Proctoring: Eye/face detection (YOLO/MediaPipe)
- [ ] Live dashboard for admins

---

## 📜 License

MIT License.  
Free for commercial and academic use. Credit appreciated! 🙌

---

Would you like me to also:
- Export this into a downloadable `README.md` file?
- Split the two apps into separate folders with their own configs?

Let me know!
