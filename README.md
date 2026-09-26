# Musa (zarvish_v4.py)

A macOS voice assistant with wake-word detection, built with a two-tier architecture — a lightweight Tier 1 listener that stays idle in the background, and a full Tier 2 assistant that activates on wake-word.

## Features
- Wake-word detection to trigger the assistant
- Speech-to-text via Whisper
- LLM-powered responses using Ollama/LLaMA
- Menu bar status icon (rumps)
- Voice output via pyttsx3
- Fuzzy app-name matching for launching apps
- Intent routing to bypass LLM for direct commands
- Power commands with voice confirmation
- Auto-start on login (macOS LaunchAgent)
- Screen wake detection

## Tech Stack
- Python
- Whisper (speech-to-text)
- Ollama / LLaMA (LLM)
- rumps (macOS menu bar)
- pyttsx3 (text-to-speech)

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python zarvish_v4.py`

## License
MIT
