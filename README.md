### 🧠 Convo-AI: Voice-Based Interview Suite

This project contains 3 AI-powered tools for real-time voice cloning, interviewing, and monitoring:

## 📂 Project Files

- `xtts-1.py` – Clone your voice using Coqui XTTS-v2
- `sttts.py` – Voice interview assistant using Groq (Whisper + LLM + TTS)
- `gemini-live.py` – Real-time AI interviewer with webcam + audio via Google Gemini Live API

---

## 🔧 Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (For XTTS) Download model
python -m TTS.utils.download --model_name "tts_models/multilingual/multi-dataset/xtts_v2"
```

---

## ▶️ Usage

### 🗣️ Voice Cloner (XTTS)
```bash
python xtts-1.py
```

### 🧑‍💻 Interview Assistant (Groq LLM + TTS)
```bash
python sttts.py
```

### 🎥 Live AI Interviewer (Gemini + Webcam)
```bash
python gemini-live.py --mode camera
```

---

## 🔑 Environment Variables

Set these before running:
- `GROQ_API_KEY` – for Whisper + LLM (Groq)
- `GEMINI_API_KEY` – for Gemini Live API

---

## ✅ Requirements

- Python 3.10 or 3.11
- torch==2.5.1, torchaudio==2.5.1
- Microphone, speakers, (optional: webcam)
```
"# convo-ai" 
