import os
import sys

# Ensure script directory is in sys.path BEFORE any local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Force UTF-8 Encoding for Console Output
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import re
import difflib
import tempfile
import subprocess
import urllib.parse
import threading
import traceback
import concurrent.futures
from contextlib import contextmanager


def safe_print(text):
    """Safely prints text supporting Devanagari Hindi characters without encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', errors='replace').decode('utf-8'))

from datetime import datetime, timedelta
from collections import deque

import requests
import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np

# Sentinel file check BEFORE loading heavy dependencies
MANUAL_QUIT_SENTINEL = os.path.expanduser("~/.musa_manual_quit")

def check_manual_quit_sentinel():
    """Checks and removes manual quit sentinel file if present."""
    if os.path.exists(MANUAL_QUIT_SENTINEL):
        try:
            os.remove(MANUAL_QUIT_SENTINEL)
            print(f"🛑 [Startup Sentinel Check]: Manual quit sentinel ('{MANUAL_QUIT_SENTINEL}') detected.")
            print("   User explicitly clicked 'Quit Musa'. Exiting cleanly without relaunching.")
        except Exception as e:
            print(f"⚠️ Error removing sentinel file: {e}")
        return True
    return False

if check_manual_quit_sentinel():
    sys.exit(0)


# Extensible System Control Module
try:
    from system_control import (
        execute_command as execute_system_control,
        resolve_app_name,
        is_protected_app,
        system_controller,
        WEBSITE_MAP, APP_MAP, FOLDER_MAP
    )
except Exception as e:
    print(f"⚠️ system_control import error: {e}")
    execute_system_control = None
    resolve_app_name = None
    system_controller = None
    is_protected_app = lambda x: False
    WEBSITE_MAP, APP_MAP, FOLDER_MAP = {}, {}, {}

# Rumps menu bar framework import
try:
    import rumps
except ImportError:
    rumps = None

# Optional pyttsx3 import
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# ==========================================
# CONFIGURATION & GLOBAL CONSTANTS
# ==========================================

IDLE_TIMEOUT_SECONDS = 90     # Default 90s idle timeout for Tier 2 before returning to Tier 1
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TIMEOUT = 45           # 45 seconds request timeout for LLM generation
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
SAMPLE_RATE = 16000           # 16kHz sample rate required by Whisper
MIN_AUDIO_RMS = float(os.environ.get("MUSA_ENERGY_THRESHOLD", 0.005)) # Baseline audio RMS threshold
ENERGY_THRESHOLD = MIN_AUDIO_RMS
DYNAMIC_ENERGY_THRESHOLD = os.environ.get("MUSA_DYNAMIC_ENERGY", "true").lower() == "true"
DYNAMIC_ENERGY_RATIO = float(os.environ.get("MUSA_DYNAMIC_RATIO", 1.4)) # Multiplier on ambient noise floor
PAUSE_THRESHOLD = float(os.environ.get("MUSA_PAUSE_THRESHOLD", 0.2))    # VAD silence padding in seconds
VAD_FRAME_MS = int(os.environ.get("MUSA_VAD_FRAME_MS", 20))             # VAD chunk window (ms)
WAKE_WORD_SENSITIVITY = int(os.environ.get("MUSA_WAKE_SENSITIVITY", 6))  # 1 (loose/sensitive) to 10 (strict)
WAKE_WORD_COOLDOWN = float(os.environ.get("MUSA_WAKE_COOLDOWN", 1.5))   # Cooldown after non-matching audio (sec)
WAKE_WORD_LOG_FILE = os.path.expanduser("~/.musa/wakeword_debug.log")   # Wake-word debug log
SOFTWARE_GAIN = 4.5           # 4.5x software input gain multiplier
INPUT_DEVICE_INDEX = None     # Audio device index override
CALIBRATION_DURATION = 2.0   # Mic ambient noise floor calibration duration
LISTEN_DURATION = 6           # Command recording window duration (sec)
TRANSCRIPTION_TIMEOUT = 15.0  # Max seconds allowed for Whisper STT before timeout
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

MUSA_REMINDERS_FILE = os.path.expanduser("~/musa_reminders.json")
MUSA_NOTES_FILE = os.path.expanduser("~/musa_notes.txt")
MACROS_FILE = os.path.expanduser("~/.musa/macros.json")
INITIAL_PROMPT = "Musa Hinglish voice assistant commands: open Finder, open Camera, open Gallery, open YouTube, open Chrome, open VS Code, close game, band karo, kholo, volume 50, sleep, restart"

# Voice & Tone Preferences
PREFERRED_DEEP_VOICE = "Rishi"
PREFERRED_HINDI_VOICE = "Lekha"
SPEECH_RATE = "160"

# State trackers & synchronization flags
consecutive_low_count = 0
has_shown_mic_hint = False
is_speaking = False
is_running = True
selected_female_voice_id = None

pending_confirmation_action = None
last_active_app = None

# ==========================================
# MENU BAR APP & STATE INDICATOR (rumps)
# ==========================================
STATE_IDLE = "IDLE"
STATE_LISTENING = "LISTENING"
STATE_PROCESSING = "PROCESSING"
STATE_SPEAKING = "SPEAKING"
STATE_ERROR = "ERROR"

STATE_TITLES = {
    STATE_IDLE: "⚪ Musa (Idle)",
    STATE_LISTENING: "🟢 Musa (Listening...)",
    STATE_PROCESSING: "⚡ Musa (Processing...)",
    STATE_SPEAKING: "🔊 Musa (Speaking...)",
    STATE_ERROR: "🔴 Musa (Error / Resetting...)"
}

current_musa_state = STATE_IDLE
rumps_app_instance = None

# Processing Diagnostics & Watchdog Trackers
current_processing_step = "None"
current_processing_start_time = 0.0
current_processing_thread_name = "None"
MAX_PROCESSING_WATCHDOG_SECONDS = 35.0

SETTINGS_FILE = os.path.expanduser("~/.musa/settings.json")

DEFAULT_SETTINGS = {
    "wake_word_sensitivity": 6,          # 4 (Low/Sensitive), 6 (Medium/Balanced), 8 (High/Strict)
    "voice_speed": "Normal",             # "Slow", "Normal", "Fast"
    "speech_rate": 165,                  # numeric speech rate (130, 165, 200)
    "microphone_device_index": None,     # device index (None = System Default)
    "microphone_device_name": "System Default"
}


def load_settings():
    """
    Loads saved settings from ~/.musa/settings.json and synchronizes
    shared global variables (WAKE_WORD_SENSITIVITY, SPEECH_RATE, INPUT_DEVICE_INDEX).
    """
    global WAKE_WORD_SENSITIVITY, SPEECH_RATE, INPUT_DEVICE_INDEX
    settings = dict(DEFAULT_SETTINGS)
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
            safe_print(f"⚙️ [Settings Loaded]: {settings}")
    except Exception as e:
        safe_print(f"⚠️ Error loading settings from {SETTINGS_FILE}: {e}")

    # Synchronize shared config variables
    WAKE_WORD_SENSITIVITY = int(settings.get("wake_word_sensitivity", 6))
    SPEECH_RATE = str(settings.get("speech_rate", 165))
    dev_idx = settings.get("microphone_device_index")
    if dev_idx is not None:
        try:
            dev_idx = int(dev_idx)
            devs = sd.query_devices()
            if 0 <= dev_idx < len(devs) and devs[dev_idx].get("max_input_channels", 0) > 0:
                INPUT_DEVICE_INDEX = dev_idx
            else:
                INPUT_DEVICE_INDEX = None
        except Exception:
            INPUT_DEVICE_INDEX = None
    else:
        INPUT_DEVICE_INDEX = None

    return settings


def save_settings(new_settings):
    """Saves updated settings dictionary to ~/.musa/settings.json."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        current = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except Exception:
                current = {}
        current.update(new_settings)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        safe_print(f"💾 [Settings Saved]: {new_settings} -> {SETTINGS_FILE}")
    except Exception as e:
        safe_print(f"⚠️ Error saving settings to {SETTINGS_FILE}: {e}")

# Initialize settings from ~/.musa/settings.json
load_settings()


def show_mac_notification(message, title="Musa Assistant", sound="Glass"):
    """Displays a native macOS system notification banner using osascript (non-blocking)."""
    def notify():
        try:
            sound_clause = f' sound name "{sound}"' if sound else ""
            cmd = [
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"{sound_clause}'
            ]
            subprocess.run(cmd, check=False, capture_output=True, timeout=5)
        except Exception:
            pass
    threading.Thread(target=notify, daemon=True).start()


def set_musa_state(state):
    """Updates Musa's status dynamically on macOS Menu Bar & Notification Center."""
    global current_musa_state, current_processing_start_time, current_processing_thread_name
    old_state = current_musa_state
    current_musa_state = state
    if state == STATE_PROCESSING:
        current_processing_start_time = time.time()
        current_processing_thread_name = threading.current_thread().name
    title = STATE_TITLES.get(state, "⚪ Musa")
    safe_print(f"📌 [Menu Bar Status]: {title}")

    if rumps_app_instance and hasattr(rumps_app_instance, 'title'):
        try:
            rumps_app_instance.title = title
            if hasattr(rumps_app_instance, 'status_item'):
                rumps_app_instance.status_item.title = f"Status: {state.capitalize()}"
        except Exception:
            pass

    if old_state != state and state == STATE_IDLE:
        show_mac_notification("Musa: Idle", title="Musa Assistant", sound=None)


def log_processing_step(step_desc):
    """Logs current processing step with thread name, line context, and timestamp."""
    global current_processing_step
    current_processing_step = step_desc
    th_name = threading.current_thread().name
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    safe_print(f"⚡ [Processing Step] [{ts}] [{th_name}]: {step_desc}")


@contextmanager
def acquire_timeout(lock, timeout=5.0, lock_name="lock"):
    """Context manager for acquiring a threading.Lock with a strict timeout to prevent deadlocks."""
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        safe_print(f"⚠️ [Lock Timeout]: Failed to acquire '{lock_name}' within {timeout}s!")
        raise TimeoutError(f"Failed to acquire '{lock_name}' within {timeout}s")
    try:
        yield
    finally:
        lock.release()


def start_processing_watchdog():
    """
    Daemon thread that monitors Musa's processing state.
    If stuck in STATE_PROCESSING for > MAX_PROCESSING_WATCHDOG_SECONDS,
    logs the stuck step and thread, and forcefully resets state to IDLE.
    """
    def watchdog_loop():
        global current_musa_state
        safe_print("🛡️ [Watchdog]: Musa processing state watchdog active.")
        while is_running:
            try:
                if current_musa_state == STATE_PROCESSING and current_processing_start_time > 0:
                    elapsed = time.time() - current_processing_start_time
                    if elapsed >= MAX_PROCESSING_WATCHDOG_SECONDS:
                        safe_print(f"\n🚨 [WATCHDOG TIMEOUT]: Musa has been stuck in PROCESSING for {elapsed:.1f}s!")
                        safe_print(f"🚨 [WATCHDOG DETAIL]: Stuck at step '{current_processing_step}' on thread '{current_processing_thread_name}'!")
                        safe_print("🚨 [WATCHDOG ACTION]: Forcefully resetting Menu Bar status to IDLE...")
                        show_mac_notification(
                            f"Processing timed out ({int(elapsed)}s at {current_processing_step}). Reset to Idle.",
                            title="⚠️ Musa Watchdog Alert",
                            sound="Basso"
                        )
                        set_musa_state(STATE_ERROR)
                        time.sleep(1.0)
                        set_musa_state(STATE_IDLE)
            except Exception as e:
                safe_print(f"⚠️ Watchdog loop error: {e}")
            time.sleep(2.0)

    wd_thread = threading.Thread(target=watchdog_loop, daemon=True, name="MusaProcessingWatchdog")
    wd_thread.start()


def create_manual_quit_sentinel():
    """Writes a sentinel file indicating Musa was deliberately quit by the user."""
    try:
        with open(MANUAL_QUIT_SENTINEL, "w", encoding="utf-8") as f:
            f.write(f"Manual quit at {datetime.now().isoformat()}\n")
        safe_print(f"📝 [Sentinel File]: Created manual quit sentinel at {MANUAL_QUIT_SENTINEL}")
    except Exception as e:
        safe_print(f"⚠️ [Sentinel Error]: Could not create sentinel file: {e}")


def quit_musa_assistant(speak_message=True):
    """Cleanly stops Musa assistant, background threads, writes manual quit sentinel, and exits app with code 0."""
    global is_running
    is_running = False
    safe_print("\n👋 Shutting down Musa Voice Assistant...")

    if speak_message:
        speak("Alvida! Musa ko band kiya ja raha hai.")

    create_manual_quit_sentinel()
    time.sleep(0.3)
    safe_print("👋 Terminating Musa process...")

    if rumps:
        try:
            rumps.quit_application()
        except Exception:
            pass

    sys.exit(0)


class MusaMenuBarApp(rumps.App if rumps else object):
    """
    Lightweight macOS Menu Bar status app using rumps framework
    with interactive Settings submenu (Sensitivity, Voice Speed, Mic Selection).
    """
    def __init__(self):
        if not rumps:
            return

        super(MusaMenuBarApp, self).__init__("⚪ Musa (Idle)", quit_button=None)
        self.status_item = rumps.MenuItem("Status: Idle")
        self.status_item.set_callback(None)

        # Build dynamic Settings submenu
        self.settings_menu = self.build_settings_menu()

        self.menu = [
            self.status_item,
            None,
            self.settings_menu,
            None,
            rumps.MenuItem("Quit Musa", callback=self.quit_app)
        ]

    def build_settings_menu(self):
        """Constructs the Settings submenu with live checkmarked radio options."""
        settings_root = rumps.MenuItem("Settings")

        # 1. Wake-word Sensitivity Submenu
        self.sens_menu = rumps.MenuItem("Wake-word Sensitivity")
        self.sens_items = {}
        sens_options = [
            ("Low (Sensitive)", 4),
            ("Medium (Balanced)", 6),
            ("High (Strict)", 8)
        ]
        current_sens = WAKE_WORD_SENSITIVITY
        closest_sens = min(sens_options, key=lambda opt: abs(opt[1] - current_sens))[1]

        for label, val in sens_options:
            item = rumps.MenuItem(label, callback=self.on_change_sensitivity)
            item.val = val
            item.state = 1 if val == closest_sens else 0
            self.sens_items[val] = item
            self.sens_menu.add(item)

        # 2. Voice Speed Submenu
        self.speed_menu = rumps.MenuItem("Voice Speed")
        self.speed_items = {}
        speed_options = [
            ("Slow", 130),
            ("Normal", 165),
            ("Fast", 200)
        ]
        current_rate = int(SPEECH_RATE) if str(SPEECH_RATE).isdigit() else 165
        closest_rate = min(speed_options, key=lambda opt: abs(opt[1] - current_rate))[1]

        for label, rate in speed_options:
            item = rumps.MenuItem(label, callback=self.on_change_voice_speed)
            item.rate = rate
            item.label = label
            item.state = 1 if rate == closest_rate else 0
            self.speed_items[rate] = item
            self.speed_menu.add(item)

        # 3. Microphone Selection Submenu
        self.mic_menu = rumps.MenuItem("Microphone")
        self.mic_items = []

        default_mic_item = rumps.MenuItem("System Default", callback=self.on_change_mic)
        default_mic_item.dev_idx = None
        default_mic_item.dev_name = "System Default"
        default_mic_item.state = 1 if INPUT_DEVICE_INDEX is None else 0
        self.mic_items.append(default_mic_item)
        self.mic_menu.add(default_mic_item)

        try:
            devs = sd.query_devices()
            for idx, dev in enumerate(devs):
                if dev.get("max_input_channels", 0) > 0:
                    d_name = dev.get("name", f"Device {idx}")
                    label = f"[{idx}] {d_name}"
                    m_item = rumps.MenuItem(label, callback=self.on_change_mic)
                    m_item.dev_idx = idx
                    m_item.dev_name = d_name
                    m_item.state = 1 if INPUT_DEVICE_INDEX == idx else 0
                    self.mic_items.append(m_item)
                    self.mic_menu.add(m_item)
        except Exception as e:
            safe_print(f"⚠️ Error querying microphones for menu: {e}")

        settings_root.add(self.sens_menu)
        settings_root.add(self.speed_menu)
        settings_root.add(self.mic_menu)

        return settings_root

    def on_change_sensitivity(self, sender):
        """Updates wake-word sensitivity immediately and persists to config."""
        global WAKE_WORD_SENSITIVITY
        val = getattr(sender, "val", 6)
        WAKE_WORD_SENSITIVITY = val
        for item in self.sens_items.values():
            item.state = 1 if item is sender else 0
        save_settings({"wake_word_sensitivity": val})
        safe_print(f"🎛️ [Settings]: Wake-word sensitivity updated to {val} ({sender.title})")

    def on_change_voice_speed(self, sender):
        """Updates voice speech rate immediately for both say and pyttsx3."""
        global SPEECH_RATE
        rate = getattr(sender, "rate", 165)
        label = getattr(sender, "label", "Normal")
        SPEECH_RATE = str(rate)
        if pyttsx3:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', rate)
            except Exception:
                pass
        for item in self.speed_items.values():
            item.state = 1 if item is sender else 0
        save_settings({"voice_speed": label, "speech_rate": rate})
        safe_print(f"🗣️ [Settings]: Voice speed updated to {label} ({rate} wpm)")

    def on_change_mic(self, sender):
        """Selects the audio input microphone immediately and persists to config."""
        global INPUT_DEVICE_INDEX
        dev_idx = getattr(sender, "dev_idx", None)
        dev_name = getattr(sender, "dev_name", "System Default")
        INPUT_DEVICE_INDEX = dev_idx
        for item in self.mic_items:
            item.state = 1 if item is sender else 0
        save_settings({"microphone_device_index": dev_idx, "microphone_device_name": dev_name})
        safe_print(f"🎤 [Settings]: Audio input device set to {dev_name} (Index: {dev_idx})")

    def quit_app(self, _):
        safe_print("\n👋 Menu Bar Quit clicked. Shutting down Musa...")
        quit_musa_assistant(speak_message=True)



# System Prompt for Ollama Conversational Queries
conversation_history = [
    {
        "role": "system",
        "content": (
            "Tum Musa ho — ek friendly, smart aur highly capable Hinglish voice assistant for macOS users. "
            "Tumhara role user ki daily Mac productivity, system controls, questions aur tasks me help karna hai. "
            "Aapka jawab warm, natural, concise (1-2 short sentences max) aur Hinglish me hona chahiye. "
            "Nakli robot phrasing ya 'hahaha' jaisa artificial natak mat karo. Seedha, helpful aur accurate jawab do."
        )
    }
]

# Short-term rolling conversation context (in-memory only, last 3 turns)
short_term_context = deque(maxlen=3)


def add_to_context(command, action):
    """
    Maintains a rolling memory of the last 3 user commands and Musa's actions/responses.
    Stored strictly in-memory (deque maxlen=3).
    """
    global last_active_app
    if not command or not action:
        return
    clean_action = str(action).strip()
    if len(clean_action) > 120:
        clean_action = clean_action[:117] + "..."
    short_term_context.append({
        "command": str(command).strip(),
        "action": clean_action
    })

    # Keep last_active_app up-to-date if an app was identified
    if resolve_app_name:
        detected = resolve_app_name(command) or resolve_app_name(clean_action)
        if detected:
            last_active_app = detected


def get_context_prompt():
    """
    Formats the last 3 exchanges into a concise, lightweight plain-text string
    to prepend to the LLM prompt so the model can resolve pronouns/references
    (e.g., 'isko band kar', 'wapas kholo') without latency overhead.
    """
    if not short_term_context:
        return ""

    lines = ["Recent context:"]
    for item in short_term_context:
        lines.append(f"- User: \"{item['command']}\" -> Musa: \"{item['action']}\"")
    return "\n".join(lines) + "\n\n"


def build_prompt(command):
    """
    Builds the final prompt for the LLM by prepending recent conversation
    context to the incoming voice command.
    """
    context_prefix = get_context_prompt()
    if context_prefix:
        return f"{context_prefix}Current Command: {command}"
    return command






# Whisper Model Setup (Lazy Loaded)
command_model = None

def get_whisper_model():
    """Lazily loads and returns the Whisper STT model instance."""
    global command_model, WHISPER_MODEL_NAME
    if command_model is None:
        safe_print("\n" + "="*65)
        safe_print(" 🚀 Initializing Whisper Model...")
        safe_print("="*65)
        try:
            safe_print(f"Loading Whisper '{WHISPER_MODEL_NAME}' model for optimized Hinglish performance...")
            command_model = whisper.load_model(WHISPER_MODEL_NAME)
            safe_print(f"✅ Whisper '{WHISPER_MODEL_NAME}' model loaded successfully!")
        except Exception as e:
            safe_print(f"⚠️ {WHISPER_MODEL_NAME} model load error ({e}), falling back to 'base' model...")
            WHISPER_MODEL_NAME = "base"
            command_model = whisper.load_model("base")
            safe_print("✅ Whisper 'base' model loaded successfully!")
    return command_model

# ==========================================
# 0. DIAGNOSTICS & SYSTEM CHECKS ON STARTUP
# ==========================================

def safe_sd_wait(timeout=10.0):
    """Waits for sounddevice stream with explicit timeout to prevent hanging on line 359 waiter.acquire()."""
    cb = getattr(sd, '_last_callback', None)
    if cb and hasattr(cb, 'event'):
        finished = cb.event.wait(timeout=timeout)
        if not finished:
            safe_print(f"⚠️ [Audio Timeout]: sounddevice recording timed out after {timeout}s! Aborting stream...")
            try:
                sd.stop()
            except Exception:
                pass
            return False
    try:
        sd.wait()
        return True
    except Exception as e:
        safe_print(f"⚠️ sounddevice wait exception: {e}")
        try:
            sd.stop()
        except Exception:
            pass
        return False


def calibrate_microphone(duration=CALIBRATION_DURATION):
    """
    Records ambient silence at startup to calculate baseline noise floor
    and dynamically sets MIN_AUDIO_RMS (bounded between 0.002 and 0.008 max).
    """
    global MIN_AUDIO_RMS
    safe_print(f"\n🎛️ [Mic Calibration] Measuring ambient silence ({duration:.1f}s)... Please remain quiet.")
    try:
        rec_kwargs = {"samplerate": SAMPLE_RATE, "channels": 1, "dtype": 'float32'}
        if INPUT_DEVICE_INDEX is not None:
            rec_kwargs["device"] = INPUT_DEVICE_INDEX

        recording = sd.rec(int(duration * SAMPLE_RATE), **rec_kwargs)
        safe_sd_wait(timeout=duration + 3.0)

        if recording is not None and len(recording) > 0:
            recording = np.clip(recording * SOFTWARE_GAIN, -1.0, 1.0)
            ambient_rms = float(np.sqrt(np.mean(np.square(recording))))
            if DYNAMIC_ENERGY_THRESHOLD:
                dynamic_threshold = float(np.clip(ambient_rms * DYNAMIC_ENERGY_RATIO, 0.002, 0.015))
                MIN_AUDIO_RMS = dynamic_threshold
            else:
                MIN_AUDIO_RMS = ENERGY_THRESHOLD

            safe_print(f"📊 [Mic Calibration] Ambient Noise Floor (RMS): {ambient_rms:.5f}")
            safe_print(f"🎯 [Mic Calibration] Audio Threshold set to: {MIN_AUDIO_RMS:.5f} (Gain: {SOFTWARE_GAIN}x, Dynamic: {DYNAMIC_ENERGY_THRESHOLD})")
    except Exception as e:
        safe_print(f"⚠️ [Mic Calibration Warning]: {e}. Using default threshold {MIN_AUDIO_RMS:.5f}")


def test_ollama_connection():
    """
    Tests Ollama API connectivity at startup, resolves model tags (e.g. 'llama3.2' -> 'llama3.2:latest'),
    and logs active LLM status.
    """
    global MODEL, OLLAMA_URL
    urls_to_test = [
        f"{OLLAMA_BASE_URL}/api/tags",
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags"
    ]

    for test_url in urls_to_test:
        try:
            safe_print(f"📡 [Ollama Diagnostic]: Ping -> {test_url}...")
            res = requests.get(test_url, timeout=5)
            if res.status_code == 200:
                base_endpoint = test_url.replace("/api/tags", "")
                OLLAMA_URL = f"{base_endpoint}/api/chat"

                data = res.json()
                models_data = data.get("models", [])
                model_names = [m.get("name", "") for m in models_data]

                safe_print(f"✅ [Ollama Connected]: Endpoint active at {base_endpoint}")
                safe_print(f"📦 [Ollama Installed Models]: {model_names}")

                # Auto-resolve model tag (e.g. 'llama3.2' -> 'llama3.2:latest')
                if MODEL not in model_names:
                    for m_name in model_names:
                        if m_name.startswith(MODEL + ":") or m_name == MODEL:
                            safe_print(f"🏷️ [Model Auto-Resolve]: Resolved '{MODEL}' -> '{m_name}'")
                            MODEL = m_name
                            break

                safe_print(f"🎯 [Active Ollama Model]: '{MODEL}' (Request URL: {OLLAMA_URL})")
                return True
        except Exception as e:
            safe_print(f"⚠️ [Ollama Diagnostic Warning]: {test_url} unreachable: {e}")

    safe_print(f"❌ [Ollama Connectivity Failure]: Could not connect to Ollama at {OLLAMA_BASE_URL}")
    safe_print("💡 Make sure Ollama is running (`ollama serve` or check 127.0.0.1:11434).")
    return False


def check_battery_and_power_mode():
    """Checks and logs macOS battery level and Low Power Mode status at startup."""
    safe_print("\n🔋 Checking Battery & macOS Power Mode Status...")
    safe_print("-" * 60)
    try:
        batt_out = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=5).strip()
        pm_out = subprocess.check_output(["pmset", "-g"], text=True, timeout=5).strip()

        battery_pct = "Unknown"
        power_source = "Unknown"

        match = re.search(r"(\d+)%", batt_out)
        if match:
            battery_pct = f"{match.group(1)}%"

        if "AC Power" in batt_out:
            power_source = "AC Power (Plugged in)"
        elif "Battery Power" in batt_out:
            power_source = "Battery Power"

        low_power_mode = "Disabled (0)"
        for line in pm_out.splitlines():
            if "lowpowermode" in line.lower():
                val = line.strip().split()[-1]
                low_power_mode = f"Enabled (1)" if val == "1" else "Disabled (0)"
                break

        safe_print(f"   • Power Source   : {power_source}")
        safe_print(f"   • Battery Level  : {battery_pct}")
        safe_print(f"   • Low Power Mode : {low_power_mode}")

        if battery_pct != "Unknown":
            pct_val = int(battery_pct.replace("%", ""))
            if pct_val <= 15:
                safe_print(f"⚠️ [LOW BATTERY WARNING]: Battery is at {battery_pct}!")
                safe_print("   macOS energy management may throttle background processes,")
                safe_print("   coalesce timers, or drop low-priority audio streams under low battery.")

    except Exception as e:
        safe_print(f"⚠️ Battery diagnostic check error: {e}")


def run_startup_diagnostics():
    """Runs startup checks for System Audio Volume, Microphones, Sound Devices, and TTS Engines."""
    check_battery_and_power_mode()
    safe_print("\n🔍 Running Startup Diagnostic Checks...")
    safe_print("-" * 60)

    try:
        vol_level = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"], text=True, timeout=5).strip()
        vol_info = subprocess.check_output(["osascript", "-e", "get volume settings"], text=True, timeout=5).strip()
        safe_print(f"📊 [System Volume] Level: {vol_level}% | Settings: {vol_info}")

        if "output muted:true" in vol_info or int(vol_level) == 0:
            safe_print("⚠️ System audio output is muted/0%! Setting volume to 85%...")
            subprocess.run(["osascript", "-e", "set volume output muted false"], check=False, timeout=5)
            subprocess.run(["osascript", "-e", "set volume output volume 85"], check=False, timeout=5)
    except Exception as e:
        safe_print(f"⚠️ Volume check error: {e}")

    try:
        devs = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        active_in_idx = INPUT_DEVICE_INDEX if INPUT_DEVICE_INDEX is not None else default_in
        active_in_name = devs[active_in_idx]['name'] if active_in_idx >= 0 and active_in_idx < len(devs) else "Unknown"

        safe_print(f"🎤 [Active Input Mic]      : [{active_in_idx}] {active_in_name} (Gain: {SOFTWARE_GAIN}x)")
        safe_print(f"🔊 [Default Output Speaker] : [{default_out}] {devs[default_out]['name']}")
    except Exception as e:
        safe_print(f"⚠️ SoundDevice query error: {e}")

    calibrate_microphone()

    safe_print("\n🚀 Pre-warming Whisper Model...")
    try:
        get_whisper_model()
    except Exception as e:
        safe_print(f"⚠️ Whisper pre-warm error: {e}")

    safe_print("\n🧠 Testing Ollama LLM API Connectivity...")
    test_ollama_connection()

    safe_print("\n🧪 Testing Primary Voice Output (macOS native 'say')...")
    say_worked = speak_mac_say("नमस्ते, मूsa वॉइस सिस्टम तैयार है.", voice="Lekha")
    if say_worked:
        safe_print("✅ Primary TTS Engine (macOS native voices) is WORKING 100%!")

    safe_print("-" * 60)
    start_reminder_scheduler()
    start_processing_watchdog()
    safe_print("✅ Startup Diagnostics Complete!\n")

# ==========================================
# 1. AUDIO RECORDING, VAD & SYNCHRONIZED TTS
# ==========================================

def clean_text_for_speech(text):
    if not text:
        return ""
    cleaned = re.sub(r'[*_`#~]', '', text)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def speak_mac_say(text, voice="Lekha", rate=SPEECH_RATE):
    clean_txt = clean_text_for_speech(text)
    if not clean_txt:
        return False

    voices_to_try = [voice, "Rishi", "Lekha", "Samantha", None]
    for v in voices_to_try:
        try:
            cmd = ["say", "-r", rate]
            if v:
                cmd.extend(["-v", v])
            cmd.append(clean_txt)

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                return True
        except Exception as e:
            safe_print(f"⚠️ macOS 'say' exception (voice={v}): {e}")

    return False


def speak_pyttsx3(text, voice_id=None):
    if not pyttsx3:
        return False
    clean_txt = clean_text_for_speech(text)
    if not clean_txt:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        if voice_id:
            engine.setProperty('voice', voice_id)
        engine.say(clean_txt)
        engine.runAndWait()
        engine.stop()
        del engine
        return True
    except Exception as e:
        safe_print(f"⚠️ pyttsx3 exception: {e}")
        return False


def speak(text):
    """Synchronized TTS Output Function. Releases mic stream & forces volume unmute."""
    global is_speaking
    if not text:
        return

    prev_state = current_musa_state
    set_musa_state(STATE_SPEAKING)
    is_speaking = True
    safe_print(f"\n🔊 [Speaking]: '{text}'")

    try:
        try:
            sd.stop()
        except Exception:
            pass

        try:
            vol_info = subprocess.check_output(["osascript", "-e", "get volume settings"], text=True, timeout=5)
            if "output muted:true" in vol_info or "output volume:0" in vol_info:
                safe_print("🔊 [Auto-Unmute]: Unmuting speaker and setting volume to 85%...")
                subprocess.run(["osascript", "-e", "set volume output muted false"], check=False, timeout=5)
                subprocess.run(["osascript", "-e", "set volume output volume 85"], check=False, timeout=5)
        except Exception:
            pass

        has_devanagari = bool(re.search(r'[ऀ-ॿ]', text))
        target_voice = PREFERRED_HINDI_VOICE if has_devanagari else PREFERRED_DEEP_VOICE

        safe_print(f"🎙️ [TTS Voice Selected]: '{target_voice}' (Rate: {SPEECH_RATE}, Devanagari: {has_devanagari})")
        success = speak_mac_say(text, voice=target_voice, rate=SPEECH_RATE)

        if not success:
            safe_print("⚠️ Primary macOS 'say' failed. Attempting pyttsx3 fallback...")
            success = speak_pyttsx3(text, selected_female_voice_id)

        if not success:
            safe_print(f"❌ [TTS FAILURE]: Could not play audio output for text: '{text}'")
    finally:
        is_speaking = False
        time.sleep(0.4)
        target_restore_state = STATE_IDLE if prev_state == STATE_PROCESSING else prev_state
        set_musa_state(target_restore_state)


def is_audio_valid(recording, threshold=None):
    if threshold is None:
        threshold = MIN_AUDIO_RMS
    if recording is None or len(recording) == 0:
        return False, 0.0
    rms = float(np.sqrt(np.mean(np.square(recording))))
    return rms >= threshold, rms


def apply_vad_trimming(recording, sample_rate=SAMPLE_RATE, frame_duration_ms=VAD_FRAME_MS, threshold=None):
    """
    Voice Activity Detection (VAD) using NumPy RMS energy chunking.
    Trims silence with 200ms speech padding so word endings/beginnings aren't cut off.
    Returns: (trimmed_recording, total_duration_sec, speech_duration_sec, speech_detected: bool)
    """
    if threshold is None:
        threshold = MIN_AUDIO_RMS

    if recording is None or len(recording) == 0:
        return recording, 0.0, 0.0, False

    total_duration = len(recording) / sample_rate
    frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))
    num_frames = len(recording) // frame_samples

    speech_mask = []
    for i in range(num_frames):
        frame = recording[i * frame_samples : (i + 1) * frame_samples]
        f_rms = float(np.sqrt(np.mean(np.square(frame))))
        speech_mask.append(f_rms >= threshold)

    if not any(speech_mask):
        return recording, total_duration, 0.0, False

    first_speech = speech_mask.index(True)
    last_speech = len(speech_mask) - 1 - speech_mask[::-1].index(True)

    padding_frames = max(1, int((PAUSE_THRESHOLD * 1000.0) / frame_duration_ms))
    start_frame = max(0, first_speech - padding_frames)
    end_frame = min(num_frames, last_speech + 1 + padding_frames)

    start_sample = start_frame * frame_samples
    end_sample = min(len(recording), end_frame * frame_samples)

    trimmed = recording[start_sample:end_sample]
    speech_duration = len(trimmed) / sample_rate

    safe_print(f"🎙️ [VAD Trimming]: Recorded Total = {total_duration:.2f}s | Speech Detected = {speech_duration:.2f}s (Silence Trimmed: {total_duration - speech_duration:.2f}s)")
    return trimmed, total_duration, speech_duration, True


def record_clip(seconds, min_threshold=None):
    """Fresh Audio Recording Function with Gain, VAD Trimming & 16-bit PCM WAV Output."""
    global consecutive_low_count, has_shown_mic_hint, is_speaking

    if min_threshold is None:
        min_threshold = MIN_AUDIO_RMS

    while is_speaking:
        time.sleep(0.1)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            recording = None
            start_time = time.time()

            rec_kwargs = {"samplerate": SAMPLE_RATE, "channels": 1, "dtype": 'float32'}
            if INPUT_DEVICE_INDEX is not None:
                rec_kwargs["device"] = INPUT_DEVICE_INDEX

            recording = sd.rec(int(seconds * SAMPLE_RATE), **rec_kwargs)
            wait_ok = safe_sd_wait(timeout=seconds + 4.0)
            if not wait_ok:
                safe_print(f"⚠️ [Record Error]: Audio stream wait timed out after {seconds + 4.0}s.")
                return None, 0.0

            elapsed = time.time() - start_time

            if recording is not None and len(recording) > 0:
                recording = np.clip(recording * SOFTWARE_GAIN, -1.0, 1.0)

            valid, rms = is_audio_valid(recording, min_threshold)

            try:
                target_dev = INPUT_DEVICE_INDEX if INPUT_DEVICE_INDEX is not None else sd.default.device[0]
                active_dev_name = sd.query_devices(target_dev)['name']
            except Exception:
                active_dev_name = "Default Mic"

            if not valid:
                consecutive_low_count += 1
                if consecutive_low_count >= 4 and (consecutive_low_count % 4 == 0):
                    safe_print(f"⚠️ [Low Audio Level] Mic '{active_dev_name}' level ({rms:.5f}) below threshold ({min_threshold:.5f}) ({consecutive_low_count}x).")
                return None, rms

            consecutive_low_count = 0
            ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            safe_print(f"🎤 [{ts_str}] [{active_dev_name}] Mic Energy (RMS): {rms:.5f} (Recorded: {elapsed:.2f}s, Threshold: {min_threshold:.5f})")

            # Apply VAD Trimming
            trimmed_audio, total_dur, speech_dur, speech_found = apply_vad_trimming(recording, SAMPLE_RATE, threshold=min_threshold)

            # Peak Normalization to 0.9 peak
            audio_level = abs(trimmed_audio).max()
            if audio_level > 0.0001:
                trimmed_audio = trimmed_audio / audio_level * 0.9

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="musa_mic_")
            os.close(tmp_fd)

            # Write 16-bit PCM integer WAV for 100% Whisper compatibility
            audio_int16 = (trimmed_audio * 32767.0).astype(np.int16)
            wavfile.write(tmp_path, SAMPLE_RATE, audio_int16)
            recording = None

            return tmp_path, rms
        except Exception as e:
            safe_print(f"\n⚠️ Recording Error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)

    return None, 0.0

# ==========================================
# 2. TRANSCRIPTION, VAD & WAKE WORD
# ==========================================

def is_repetitive_text(text):
    if not text:
        return False
    clean = text.strip()
    words = clean.split()
    if len(words) >= 4:
        half = len(words) // 2
        first_half = ' '.join(words[:half])
        second_half = ' '.join(words[half:half*2])
        if first_half.lower() == second_half.lower():
            return True
    return False


def is_text_valid_script(text):
    if not text:
        return False
    forbidden_pattern = re.compile(r'[가-힣一-鿿぀-ヿ؀-ۿЀ-ӿ]')
    if forbidden_pattern.search(text):
        safe_print(f"⚠️ [Sanity Filter] Discarded non-target script hallucination: '{text}'")
        return False
    return True


def _do_whisper_transcribe(audio_path):
    """Internal helper function to run Whisper STT transcription with strict parameters."""
    model = get_whisper_model()
    return model.transcribe(
        audio_path,
        language="en",
        initial_prompt=INITIAL_PROMPT,
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        logprob_threshold=-0.8,
        compression_ratio_threshold=2.2
    )


def transcribe(audio_path):
    """Transcribes audio using Whisper model with timeout, RAM tracking, logprob confidence filtering."""
    if not audio_path or not os.path.exists(audio_path):
        return ""

    try:
        import psutil
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        safe_print(f"🧠 [Memory Check]: RAM usage before STT: {mem_mb:.1f} MB (Model: '{WHISPER_MODEL_NAME}')")
    except Exception:
        pass

    try:
        safe_print(f"⚡ [STT Processing]: Transcribing {os.path.basename(audio_path)} (Timeout limit: {TRANSCRIPTION_TIMEOUT}s)...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_do_whisper_transcribe, audio_path)
            result = future.result(timeout=TRANSCRIPTION_TIMEOUT)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        segments = result.get("segments", [])
        valid_segments = []

        if segments:
            for seg in segments:
                avg_logprob = seg.get("avg_logprob", -1.0)
                no_speech_prob = seg.get("no_speech_prob", 0.0)
                seg_text = seg.get("text", "").strip()

                safe_print(f"📊 [Whisper Segment Confidence]: Logprob = {avg_logprob:.2f}, NoSpeechProb = {no_speech_prob:.2f} -> '{seg_text}'")

                if avg_logprob >= -0.90 and no_speech_prob <= 0.50:
                    valid_segments.append(seg_text)
                else:
                    safe_print(f"⚠️ [Low Confidence Filter]: Discarded low-confidence segment '{seg_text}' (Logprob: {avg_logprob:.2f})")

            text = " ".join(valid_segments).strip()
        else:
            text = result.get("text", "").strip()

        if is_repetitive_text(text):
            safe_print(f"⚠️ [Repetition Filter] Discarded repetitive hallucination: '{text}'")
            return ""

        if not is_text_valid_script(text):
            return ""

        return text
    except concurrent.futures.TimeoutError:
        safe_print(f"\n⏱️ [STT TIMEOUT]: Whisper transcription exceeded {TRANSCRIPTION_TIMEOUT}s limit on model '{WHISPER_MODEL_NAME}'. Aborting.")
        return ""
    except Exception as e:
        safe_print(f"\n❌ [Transcribe Error]: {e}")
        safe_print(traceback.format_exc())
        return ""
    finally:
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass


def listen(duration=LISTEN_DURATION, max_stt_retries=2):
    """Listens to mic and transcribes speech with retry logic."""
    set_musa_state(STATE_LISTENING)
    safe_print(f"\n🎤 Musa sun rahi hai... ({duration}s bolo)")
    time.sleep(0.3)

    for attempt in range(max_stt_retries):
        tmp_path, rms = record_clip(duration, min_threshold=None)

        if tmp_path is None:
            set_musa_state(STATE_IDLE)
            return ""

        set_musa_state(STATE_PROCESSING)
        try:
            text = transcribe(tmp_path)
            if text:
                safe_print(f"👂 [Suna / Transcribed]: '{text}'")
                return text
            else:
                if attempt < max_stt_retries - 1:
                    safe_print(f"🔄 [STT Retry {attempt+1}/{max_stt_retries}]: Transcription empty/low confidence. Retrying listen...")
                    set_musa_state(STATE_LISTENING)
                    time.sleep(0.2)
        finally:
            if current_musa_state == STATE_PROCESSING:
                set_musa_state(STATE_IDLE)

    return ""


def get_sensitivity_score_threshold(sensitivity=None):
    """Maps sensitivity 1-10 to fuzzy score threshold (1=0.55 loose, 6=0.73 balanced, 10=0.88 strict)."""
    if sensitivity is None:
        sensitivity = WAKE_WORD_SENSITIVITY
    s = max(1, min(10, int(sensitivity)))
    return 0.50 + (s / 10.0) * 0.38


def log_wakeword_attempt(text, rms, best_var, score, req_thresh, verdict):
    """Logs wake-word attempt details with energy and confidence score to ~/.musa/wakeword_debug.log."""
    try:
        os.makedirs(os.path.dirname(WAKE_WORD_LOG_FILE), exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = (
            f"[{ts}] Verdict: {verdict:<8} | Score: {score:.2f} (Req: {req_thresh:.2f}) "
            f"| RMS: {rms:.5f} | BestVar: {str(best_var):<8} | Heard: \"{text}\"\n"
        )
        with open(WAKE_WORD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        safe_print(f"⚠️ Wake-word debug log error: {e}")


def detect_wake_word(text, sensitivity=None):
    """
    Dual-Script Fuzzy Wake Word Detector with Configurable Sensitivity (1-10).
    - Checks exact whole-word matches against phonetically tuned variants.
    - Applies difflib sequence matching with exclusion filtering for partial phonetics.
    - Word boundary checks prevent false positives on substrings (e.g., 'samosa', 'famous').
    Returns: (is_match: bool, best_variant: str, score: float, required_threshold: float)
    """
    if not text:
        return False, None, 0.0, 0.0

    if sensitivity is None:
        sensitivity = WAKE_WORD_SENSITIVITY

    req_threshold = get_sensitivity_score_threshold(sensitivity)
    clean_text = text.strip()
    all_variants = [
        "musa", "mosa", "moosa", "mussa", "mausa", "musah", "mousa", "musaa", "mossa",
        "मुसा", "मुस्टा", "मूसा", "मुस्सा", "मौसा"
    ]
    exclusions = ["music", "museum", "muscle", "must", "samosa", "famous", "amusing", "mouse"]

    words = re.findall(r'\w+', clean_text, flags=re.UNICODE)

    # 1. Exact whole-word match
    for word in words:
        w_lower = word.lower()
        if w_lower in exclusions:
            continue
        for variant in all_variants:
            if w_lower == variant.lower():
                return True, variant, 1.0, req_threshold

    # 2. Fuzzy sequence matching with threshold governed by sensitivity
    best_variant = None
    best_score = 0.0

    for word in words:
        w_lower = word.lower()
        if w_lower in exclusions or len(w_lower) < 3:
            continue
        for variant in all_variants:
            ratio = difflib.SequenceMatcher(None, w_lower, variant.lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_variant = variant

    is_match = best_score >= req_threshold
    return is_match, best_variant, best_score, req_threshold


def wait_for_wake_word():
    """Continuous wake word detection loop."""
    set_musa_state(STATE_IDLE)
    safe_print("\n👂 Musa active hai... ('Musa' / 'मुसा' bolkar bulao)")
    while True:
        tmp_path, rms = record_clip(2.5, min_threshold=None)

        if tmp_path is None:
            time.sleep(0.1)
            continue

        set_musa_state(STATE_PROCESSING)
        try:
            text = transcribe(tmp_path)
            if text:
                is_match, best_var, score, req_thresh = detect_wake_word(text, sensitivity=WAKE_WORD_SENSITIVITY)
                verdict = "MATCH" if is_match else "REJECT"
                safe_print(f"👂 [Whisper Heard]: '{text}' -> Best Match: '{best_var}' (Score: {score:.2f}, Req: {req_thresh:.2f}) [{verdict}]")
                log_wakeword_attempt(text, rms, best_var, score, req_thresh, verdict)

                if is_match:
                    safe_print(f"✅ Wake word ('Musa') matched! (Variant: '{best_var}', Score: {score:.2f})")
                    show_mac_notification("Musa is listening...", title="Musa Assistant", sound="Glass")
                    speak("Haan ji, boliye?")
                    time.sleep(0.2)
                    return
                else:
                    safe_print(f"⏳ [Cooldown]: Non-matching voice heard. Waiting {WAKE_WORD_COOLDOWN}s before listening again...")
                    time.sleep(WAKE_WORD_COOLDOWN)
        finally:
            set_musa_state(STATE_IDLE)

# ==========================================
# 3. STRICT PRE-LLM ACTION INTENT ROUTER
# ==========================================

def resolve_context_references(text):
    """Resolves context pronouns ('usko', 'ise', 'is app') to last active app."""
    global last_active_app
    if not text or not last_active_app:
        return text

    lower = text.lower()
    pronouns = ["usko", "ise", "is app", "us app", "that app", "this app"]

    for pronoun in pronouns:
        if pronoun in lower:
            resolved = re.sub(r'' + re.escape(pronoun) + r'', last_active_app, text, flags=re.IGNORECASE)
            safe_print(f"🔗 [Context Resolution]: Resolved '{pronoun}' -> '{last_active_app}' in '{text}'")
            return resolved

    return text


def parse_multistep_command(text):
    """Splits compound multi-step commands on conjunctions and phrase delimiters."""
    if not text:
        return []
    delimiters = r'\s*(?:,| aur | and | then | phir | uske baad )\s*'
    parts = [p.strip() for p in re.split(delimiters, text, flags=re.IGNORECASE) if p.strip()]
    return parts if parts else [text]


# ==========================================
# 3. MACRO / CUSTOM COMMAND SYSTEM
# ==========================================

DEFAULT_MACROS = {
    "start coding session": [
        "open VS Code",
        "open Terminal",
        "open Chrome to github.com"
    ],
    "meeting mode": [
        "open Zoom",
        "open Notes",
        "mute"
    ],
    "shutdown routine": [
        "close Google Chrome",
        "close Visual Studio Code",
        "sleep"
    ]
}

_macros_cache = {}
_macros_mtime = 0


def ensure_default_macros_file():
    """Ensures ~/.musa directory and macros.json exist with pre-filled examples."""
    try:
        macro_dir = os.path.dirname(MACROS_FILE)
        os.makedirs(macro_dir, exist_ok=True)
        if not os.path.exists(MACROS_FILE):
            with open(MACROS_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MACROS, f, indent=2)
            safe_print(f"📝 [Macros Initialized]: Created default macros config at {MACROS_FILE}")
    except Exception as e:
        safe_print(f"⚠️ Error creating macros config: {e}")


def load_macros():
    """
    Loads macros from ~/.musa/macros.json with mtime caching for instant
    hot-reloads without editing code or restarting Musa.
    """
    global _macros_cache, _macros_mtime
    ensure_default_macros_file()
    try:
        if os.path.exists(MACROS_FILE):
            mtime = os.path.getmtime(MACROS_FILE)
            if mtime != _macros_mtime:
                with open(MACROS_FILE, "r", encoding="utf-8") as f:
                    _macros_cache = json.load(f)
                _macros_mtime = mtime
                safe_print(f"🔄 [Macros Loaded]: Loaded {len(_macros_cache)} macros from {MACROS_FILE}")
            return _macros_cache
    except Exception as e:
        safe_print(f"⚠️ Error loading macros: {e}")
    return _macros_cache or DEFAULT_MACROS


def find_matching_macro(query, macros, threshold=0.60):
    """
    Fuzzy matches the query string against macro triggers using exact match,
    substring containment, token overlap, and difflib similarity.
    """
    if not query or not macros:
        return None, []

    q = query.lower().strip()
    q_words = set(re.findall(r'\w+', q))
    best_trigger = None
    best_score = 0.0

    for trigger, actions in macros.items():
        t = trigger.lower().strip()
        t_words = set(re.findall(r'\w+', t))

        # 1. Exact match
        if q == t:
            return trigger, actions

        # 2. Substring containment
        if t in q:
            score = 0.80 + 0.20 * (len(t) / len(q))
            if score > best_score:
                best_score = score
                best_trigger = trigger
        elif q in t:
            score = 0.70 + 0.30 * (len(q) / len(t))
            if score > best_score:
                best_score = score
                best_trigger = trigger

        # 3. Token overlap (e.g., 'start coding' in 'start coding session')
        if t_words:
            overlap = len(t_words & q_words) / len(t_words)
            if overlap >= 0.66:
                score = 0.70 + 0.25 * overlap
                if score > best_score:
                    best_score = score
                    best_trigger = trigger

        # 4. difflib SequenceMatcher similarity
        sim = difflib.SequenceMatcher(None, q, t).ratio()
        if sim > best_score:
            best_score = sim
            best_trigger = trigger

    if best_trigger and best_score >= threshold:
        return best_trigger, macros[best_trigger]

    return None, []


def execute_single_action(action_str):
    """
    Executes a single action by reusing existing zarvish_v4 & system_control handlers.
    Supports app opening, website navigation, media, volume, and system controls.
    """
    if not action_str:
        return None

    clean_act = action_str.strip()
    lower = clean_act.lower()

    # 1. URL / Website opening pattern (e.g. "open Chrome to github.com", "open https://github.com")
    url_match = re.search(r'https?://\S+', clean_act)
    if url_match and system_controller:
        return system_controller.open_website(url_match.group(0))

    if any(lower.startswith(p) for p in ["open website ", "website ", "browse "]) and system_controller:
        target = re.sub(r'^(open\s+)?(website|browse)\s+', '', clean_act, flags=re.IGNORECASE).strip()
        return system_controller.open_website(target)

    if " to " in lower and ("chrome" in lower or "browser" in lower or "safari" in lower) and system_controller:
        target = clean_act.split(" to ", 1)[1].strip()
        return system_controller.open_website(target)

    # 2. Try standard handle_command (with allow_macros=False to prevent recursive loops)
    res = handle_command(clean_act, allow_macros=False)
    if res:
        return res

    # 3. Fallback to open_app directly if it was just an app name
    if system_controller and hasattr(system_controller, "open_app"):
        return system_controller.open_app(clean_act)

    return f"Action '{clean_act}' executed."


def execute_macro(macro_name, actions, delay=0.8):
    """
    Executes all actions for a recognized macro in sequence
    with a small delay between each so apps don't fight for window focus.
    """
    safe_print(f"\n🚀 [Macro Executing]: '{macro_name}' ({len(actions)} actions)")
    executed_count = 0
    for i, action in enumerate(actions, 1):
        safe_print(f"   ▶ [{i}/{len(actions)}]: {action}")
        try:
            execute_single_action(action)
            executed_count += 1
        except Exception as e:
            safe_print(f"   ⚠️ Action '{action}' error: {e}")
        time.sleep(delay)

    return f"{macro_name.title()} start kar diya, {executed_count} actions complete ho gaye."


def check_and_run_macro(query):
    """
    Checks if query matches any macro in ~/.musa/macros.json via fuzzy matching.
    If matched, executes the sequence of actions and returns confirmation string.
    """
    macros = load_macros()
    matched_trigger, actions = find_matching_macro(query, macros)
    if matched_trigger and actions:
        return execute_macro(matched_trigger, actions, delay=0.8)
    return None

# ==========================================
# 4. REMINDER & NOTIFICATION SUBSYSTEM
# ==========================================

# In-memory list of dicts: {"time": datetime, "task": str, "triggered": bool}
pending_reminders = []
reminders_lock = threading.Lock()
reminder_scheduler_thread = None


def parse_reminder_command(text):
    """
    Parses voice commands for reminders supporting:
    - Relative time: 'in 10 minutes', 'in 1 hour', 'aadhe ghante mein', '15 minute mein', '2 ghante me'
    - Absolute time: '5 baje', '5:30 baje', 'at 6:30 pm', 'at 5', 'shaam 7 baje', 'subah 8 baje'
    Returns (target_datetime, task_string) or (None, None) if not a reminder command.
    """
    if not text:
        return None, None

    now = datetime.now()
    lower = text.lower().strip()

    # Must contain reminder trigger keyword
    reminder_keywords = [
        "yaad dila", "yaad dilana", "yaad dilao", "remind me", "reminder", "set a reminder"
    ]
    if not any(k in lower for k in reminder_keywords):
        return None, None

    target_time = None
    time_phrase_span = None

    # 1. Relative Hinglish/English special phrases
    half_hour_match = re.search(r'(?:in\s+)?(aadhe ghante|aadha ghanta|half an?\s*hour)(?:\s*(?:mein|me))?', lower)
    dedh_hour_match = re.search(r'(?:in\s+)?(dedh ghante|derh ghante|1\.5\s*hours?)(?:\s*(?:mein|me))?', lower)

    if half_hour_match:
        target_time = now + timedelta(minutes=30)
        time_phrase_span = half_hour_match.span()
    elif dedh_hour_match:
        target_time = now + timedelta(minutes=90)
        time_phrase_span = dedh_hour_match.span()
    else:
        # Standard relative: 'in X minutes/hours' or 'X minute/ghante mein'
        rel_match = re.search(r'(?:in\s+)?(\d+(?:\.\d+)?)\s*(minutes?|mins?|minute|m|hours?|hrs?|ghante?|ghanta)\s*(?:mein|me)?', lower)
        if rel_match:
            val = float(rel_match.group(1))
            unit = rel_match.group(2)
            if any(u in unit for u in ['hour', 'hr', 'ghant']):
                target_time = now + timedelta(hours=val)
            else:
                target_time = now + timedelta(minutes=val)
            time_phrase_span = rel_match.span()

    # 2. Absolute time: '5 baje', '5:30 baje', 'at 6:30 pm', 'shaam 7 baje', 'subah 8 baje'
    if not target_time:
        abs_match = re.search(r"(?:at\s+)?(?:(shaam|subah|raat|dopahar)\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje|o\x27?clock)?", lower)
        if abs_match and (abs_match.group(1) or abs_match.group(4) or 'at ' in lower[max(0, abs_match.start()-4):abs_match.start()+1]):
            period_hi = abs_match.group(1)
            hour = int(abs_match.group(2))
            minute = int(abs_match.group(3)) if abs_match.group(3) else 0
            period_en = abs_match.group(4)

            is_pm = False
            if period_en == 'pm' or period_hi in ['shaam', 'raat']:
                is_pm = True
            elif period_en == 'am' or period_hi == 'subah':
                is_pm = False
            elif period_hi == 'dopahar':
                is_pm = True if hour != 12 else False
            elif hour <= 7 and (period_en == 'baje' or not period_en):
                # Ambiguous clock time (e.g., '5 baje') defaults to upcoming afternoon/evening
                is_pm = True

            if is_pm and hour < 12:
                hour += 12
            elif not is_pm and hour == 12 and (period_en == 'am' or period_hi == 'subah'):
                hour = 0

            candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                # If target time has already passed today, set for tomorrow
                candidate += timedelta(days=1)
            target_time = candidate
            time_phrase_span = abs_match.span()

    if not target_time or not time_phrase_span:
        return None, None

    # Extract clean task by removing time expression and reminder triggers
    s, e = time_phrase_span
    raw_task = text[:s] + " " + text[e:]

    task_res = raw_task
    phrases_to_remove = [
        r"\b(mujhe|tum|please)?\s*(yaad\s+dilana|yaad\s+dila\s+do|yaad\s+dila\s+dena|yaad\s+dila\s+dijiye|yaad\s+dila|yaad\s+dilao)\s*(ki)?\b",
        r"\b(please\s+)?(remind\s+me\s+to|remind\s+me\s+that|remind\s+me\s+about|remind\s+me|reminder\s+for|reminder)\b",
        r"^\s*(ki|to|that|for)\s+"
    ]
    for p in phrases_to_remove:
        task_res = re.sub(p, " ", task_res, flags=re.IGNORECASE)
    clean_task = re.sub(r"\s+", " ", task_res).strip(" .,:")

    if not clean_task:
        clean_task = "Reminder"

    return target_time, clean_task


def trigger_reminder(reminder):
    """
    Fires a native macOS notification banner via osascript and announces
    the reminder aloud using Musa's speech synthesis engine.
    """
    task = reminder.get("task", "Reminder")
    time_str = reminder["time"].strftime("%I:%M %p")
    safe_print(f"\n🔔 [Reminder Triggered]: \x27{task}\x27 (Scheduled for {time_str})")

    # 1. Native macOS Notification banner
    notification_msg = f"{task}"
    show_mac_notification(notification_msg, title="⏰ Musa Reminder", sound="Glass")

    # 2. Vocal announcement via existing TTS
    vocal_msg = f"Dhyan dijiye! Aapka reminder hai: {task}."
    def announce():
        try:
            speak(vocal_msg)
        except Exception as e:
            safe_print(f"⚠️ Reminder vocal announcement error: {e}")

    threading.Thread(target=announce, daemon=True).start()


def reminder_scheduler_loop(interval=10):
    """
    Background daemon loop that checks every 10-15 seconds for due reminders.
    Runs continuously and independently of voice listening cycles.
    """
    safe_print(f"⏰ [Reminder Scheduler]: Background daemon started (polling every {interval}s).")
    while is_running:
        now = datetime.now()
        due_reminders = []
        with acquire_timeout(reminders_lock, timeout=5.0, lock_name="reminders_lock"):
            for rem in pending_reminders:
                if not rem["triggered"] and now >= rem["time"]:
                    rem["triggered"] = True
                    due_reminders.append(rem)

        for due in due_reminders:
            try:
                trigger_reminder(due)
            except Exception as e:
                safe_print(f"⚠️ Error firing reminder: {e}")

        time.sleep(interval)


def start_reminder_scheduler():
    """Starts the background reminder scheduler daemon thread if not already running."""
    global reminder_scheduler_thread
    if reminder_scheduler_thread is None or not reminder_scheduler_thread.is_alive():
        reminder_scheduler_thread = threading.Thread(
            target=reminder_scheduler_loop,
            args=(10,),
            daemon=True,
            name="MusaReminderScheduler"
        )
        reminder_scheduler_thread.start()


def add_reminder(target_time, task):
    """Stores pending reminder in memory as {time, task, triggered: False}."""
    reminder = {
        "time": target_time,
        "task": task,
        "triggered": False
    }
    with acquire_timeout(reminders_lock, timeout=5.0, lock_name="reminders_lock"):
        pending_reminders.append(reminder)
    start_reminder_scheduler()
    return reminder


def handle_command(text, allow_macros=True):
    """
    STRICT PRE-LLM INTENT ROUTER (Order of Precedence):
    0. Macro / Custom Command System (Fuzzy matched from ~/.musa/macros.json)
    1. Musa App Termination Triggers
    2. Pending Confirmations & Security Guardrails
    3. Mac Power Controls (Shutdown, Restart, Sleep, Log out)
    4. System Control Intent Module (App Open/Close with Safety Exclusions, Media, Volume, Brightness, Wi-Fi, Search)
    5. Returns None to trigger Ollama LLM fallback
    """
    global pending_confirmation_action, last_active_app
    lower = text.lower().strip()

    # 0. Macro Trigger Check
    if allow_macros:
        macro_res = check_and_run_macro(text)
        if macro_res:
            return macro_res

    # 1. Musa App Quit Trigger
    if any(phrase in lower for phrase in [
        "quit musa", "exit musa", "musa band karo", "band karo musa", 
        "musa ko band karo", "turn off musa", "musa quit", "musa exit"
    ]):
        quit_musa_assistant(speak_message=True)
        return "Musa band ho rahi hai."

    # 2. Pending Confirmations
    if pending_confirmation_action:
        action_info = pending_confirmation_action
        pending_confirmation_action = None
        if any(w in lower for w in ["haan", "yes", "pakka", "confirm", "kardo", "karo", "sure", "yeah", "sach me"]):
            cmd = action_info.get("command")
            action_name = action_info.get("action", "action")
            confirm_msg = action_info.get("msg", f"Theek hai, system {action_name} ho raha hai.")
            speak(confirm_msg)
            time.sleep(0.5)
            if cmd:
                safe_print(f"⚡ [Executing Power Command]: {cmd}")
                subprocess.run(cmd, shell=True, check=False, timeout=5)
            return f"System {action_name} executed."
        else:
            action_name = action_info.get("action", "action")
            return f"Theek hai, {action_name} cancel kar diya."

    # 3. Security Guardrails
    if any(w in lower for w in ["unlock", "password", "passcode", "login", "sudo", "admin"]):
        return "Main security reasons se password, unlock ya admin settings handle nahi kar sakti."

    # 4. Mac Power Controls
    if any(w in lower for w in ["shutdown", "shut down", "laptop shutdown", "mac shutdown", "system shutdown"]):
        pending_confirmation_action = {
            "action": "shutdown",
            "command": 'osascript -e "tell app \"System Events\" to shut down"',
            "msg": "Theek hai, system shutdown ho raha hai. Alvida!"
        }
        return "Kya aap sach me shutdown karna chahte hain? Haan ya Na?"

    if any(w in lower for w in ["restart", "reboot", "laptop restart", "mac restart", "system restart"]):
        pending_confirmation_action = {
            "action": "restart",
            "command": 'osascript -e "tell app \"System Events\" to restart"',
            "msg": "Theek hai, system restart ho raha hai."
        }
        return "Kya aap sach me restart karna chahte hain? Haan ya Na?"

    if any(w in lower for w in ["sleep", "so jao system", "laptop sleep", "mac sleep", "sleep mode"]):
        speak("System sleep mode mein ja raha hai.")
        time.sleep(0.5)
        subprocess.run(["pmset", "sleepnow"], check=False, timeout=5)
        return "System sleep mode mein chala gaya."

    if any(w in lower for w in ["log out", "logout", "log out mac", "sign out", "user logout"]):
        speak("System log out ho raha hai.")
        time.sleep(0.5)
        subprocess.run(["osascript", "-e", 'tell app "System Events" to log out'], check=False, timeout=5)
        return "System log out ho raha hai."

    # 5. Extensible System Control Module Execution
    if execute_system_control:
        ctrl_res = execute_system_control(text)
        if ctrl_res:
            return ctrl_res

    # 6. Smart Reminders (Relative & Absolute Time)
    rem_time, rem_task = parse_reminder_command(text)
    if rem_time and rem_task:
        add_reminder(rem_time, rem_task)
        time_display = rem_time.strftime("%I:%M %p")
        if rem_time.date() > datetime.now().date():
            time_display += f" ({rem_time.strftime('%d %b')})"
        return f"Theek hai, main aapko {time_display} par '{rem_task}' yaad dila doongi."

    # 7. Check Pending Reminders
    if any(phrase in lower for phrase in ["pending reminders", "reminder dikhao", "kya reminder hai", "show reminders", "list reminders"]):
        with acquire_timeout(reminders_lock, timeout=5.0, lock_name="reminders_lock"):
            active = [r for r in pending_reminders if not r["triggered"]]
        if not active:
            return "Abhi koi pending reminder nahi hai."
        lines = [f"{r['task']} ({r['time'].strftime('%I:%M %p')})" for r in active]
        return "Aapke pending reminders hain: " + ", ".join(lines)

    # 8. Direct Notes (un-timed quick notes)
    if "note likho" in lower or "note banao" in lower or "yaad dila" in lower:
        for trigger in ["note likho ki", "note likho", "note banao", "yaad dilana ki", "yaad dila do"]:
            if trigger in lower:
                idx = lower.find(trigger) + len(trigger)
                n_text = text[idx:].strip(" .,:")
                if n_text:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    with open(MUSA_NOTES_FILE, "a", encoding="utf-8") as f:
                        f.write(f"[{timestamp}] {n_text}\n")
                    return f"Note save kar liya: {n_text}"
        return "Kya note likhna hai, thoda clearly bataiye."

    # 7. Time & Date Actions
    if "time" in lower or "samay" in lower or "kitne baj" in lower:
        return f"Abhi {datetime.now().strftime('%I:%M %p')} baj rahe hain."

    if "date" in lower or "tareekh" in lower or "aaj kya din" in lower:
        return f"Aaj {datetime.now().strftime('%d %B, %Y')} hai."

    return None


def chat_with_ollama(prompt):
    """Queries Ollama model for Conversational Intents with timeout and specific exception handling."""
    full_prompt = build_prompt(prompt)
    conversation_history.append({"role": "user", "content": full_prompt})

    payload = {
        "model": MODEL,
        "messages": conversation_history,
        "stream": False
    }

    safe_print(f"💬 [Ollama Query]: Sending prompt to {OLLAMA_URL} (Model: '{MODEL}', Timeout: {OLLAMA_TIMEOUT}s)...")

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=(5.0, OLLAMA_TIMEOUT))

        if response.status_code == 200:
            data = response.json()
            reply = ""
            if "message" in data and "content" in data["message"]:
                reply = data["message"]["content"].strip()
            elif "response" in data:
                reply = data["response"].strip()

            if reply:
                conversation_history.append({"role": "assistant", "content": reply})
                return reply
            else:
                safe_print(f"⚠️ [Ollama Error]: Empty response payload received: {data}")
                return "Sorry, Ollama se khali reply aaya."
        else:
            safe_print(f"❌ [Ollama HTTP Error]: Status {response.status_code} - {response.text}")
            return f"Ollama HTTP error {response.status_code} aaya."

    except requests.exceptions.ConnectionError as e:
        safe_print(f"❌ [Ollama Connection Error]: Could not connect to {OLLAMA_URL} - {e}")
        safe_print("💡 Make sure Ollama is running (`ollama serve` or check 127.0.0.1:11434).")
        return "Ollama server se connection establish nahi ho pa raha. Make sure Ollama running ho."
    except requests.exceptions.Timeout as e:
        safe_print(f"❌ [Ollama Timeout Error]: Request to {OLLAMA_URL} timed out after {OLLAMA_TIMEOUT}s - {e}")
        return "Ollama se response aane mein zyada samay laga."
    except json.JSONDecodeError as e:
        safe_print(f"❌ [Ollama JSON Error]: Response parsing failed - {e}")
        return "Ollama response parsing mein error aaya."
    except Exception as e:
        safe_print(f"❌ [Ollama Exception]: Unexpected error - {e}")
        safe_print(traceback.format_exc())
        return "Ollama processing mein error aaya."

# ==========================================
# 4. CONVERSATION LOOP & INTENT SEPARATION
# ==========================================
# 4. TIER 1 & TIER 2 ASSISTANT EXECUTION LOOPS
# ==========================================

def voice_assistant_loop_tier2():
    """
    Tier 2 Voice Assistant Loop with Idle Timeout.
    Monitors user activity and exits cleanly after IDLE_TIMEOUT_SECONDS of inactivity.
    """
    global is_running
    last_activity_time = time.time()

    show_mac_notification("Musa is listening...", title="Musa Assistant", sound="Glass")
    speak("Haan ji, boliye?")

    safe_print(f"\n⚡ [Tier 2 Active]: Listening for commands. Idle timeout is set to {IDLE_TIMEOUT_SECONDS}s.")

    while is_running:
        idle_elapsed = time.time() - last_activity_time
        if idle_elapsed >= IDLE_TIMEOUT_SECONDS:
            safe_print(f"\n⏱️ [Tier 2 Idle Timeout]: No follow-up commands for {int(idle_elapsed)}s (threshold: {IDLE_TIMEOUT_SECONDS}s).")
            safe_print("[Tier 2] Idle timeout reached, returning to lightweight mode...")
            if rumps and rumps_app_instance:
                try:
                    rumps.quit_application()
                except Exception:
                    pass
            os._exit(42)  # Exit code 42 indicates clean return to Tier 1 listener

        set_musa_state(STATE_LISTENING)
        safe_print(f"\n🎤 Musa sun rahi hai... bolo (Idle timeout in {int(IDLE_TIMEOUT_SECONDS - idle_elapsed)}s)")

        tmp_path, rms = record_clip(LISTEN_DURATION, min_threshold=None)

        if tmp_path is None:
            set_musa_state(STATE_IDLE)
            time.sleep(0.2)
            continue

        set_musa_state(STATE_PROCESSING)
        try:
            log_processing_step("Step 1/5: Transcribing speech audio with Whisper...")
            user_input = transcribe(tmp_path)

            if user_input:
                last_activity_time = time.time()
                log_processing_step(f"Step 2/5: Transcribed input: '{user_input}'")
                safe_print(f"Tumne kaha: {user_input}")

                lower_input = user_input.lower().strip()

                if any(phrase in lower_input for phrase in ["so jao", "chup ho jao", "bye", "go to sleep", "sleep mode"]):
                    safe_print("Musa: Theek hai, main so rahi hoon. Jab zaroorat ho 'Musa' bolkar bulana.")
                    speak("Theek hai, main so rahi hoon. Jab zaroorat ho, Musa bolkar bulana!")
                    safe_print("[Tier 2] Idle timeout reached, returning to lightweight mode...")
                    if rumps and rumps_app_instance:
                        try:
                            rumps.quit_application()
                        except Exception:
                            pass
                    os._exit(42)

                if any(phrase in lower_input for phrase in ["quit musa", "exit musa", "musa band karo", "band karo musa", "turn off musa"]):
                    quit_musa_assistant(speak_message=True)
                    return

                sub_commands = parse_multistep_command(user_input)
                replies = []
                for sub_cmd in sub_commands:
                    resolved_cmd = resolve_context_references(sub_cmd)
                    if any(phrase in resolved_cmd.lower() for phrase in ["quit musa", "exit musa", "musa band karo", "band karo musa"]):
                        quit_musa_assistant(speak_message=True)
                        return

                    log_processing_step(f"Step 3/5: Routing sub-command: '{resolved_cmd}'")
                    system_reply = handle_command(resolved_cmd)
                    if system_reply:
                        replies.append(system_reply)
                        add_to_context(resolved_cmd, system_reply)
                    else:
                        log_processing_step(f"Step 4/5: Querying Ollama LLM ({MODEL})...")
                        reply = chat_with_ollama(resolved_cmd)
                        if reply:
                            replies.append(reply)
                            add_to_context(resolved_cmd, reply)

                if replies:
                    log_processing_step("Step 5/5: Synthesizing voice response...")
                    speak(" ".join(replies))
        except Exception as e:
            safe_print(f"❌ [Processing Error]: {e}\n{traceback.format_exc()}")
            set_musa_state(STATE_ERROR)
            time.sleep(1.0)
        finally:
            log_processing_step("Processing pipeline complete. Resetting state to IDLE.")
            set_musa_state(STATE_IDLE)


def run_tier2_assistant():
    """
    TIER 2 — Full Assistant Entry Point.
    Loads menu bar icon (rumps), Ollama diagnostics, system wake observers, and handles multi-turn audio interaction.
    """
    global rumps_app_instance

    if check_manual_quit_sentinel():
        sys.exit(0)

    run_startup_diagnostics()
    start_wake_observer()

    safe_print("=" * 65)
    safe_print(" 👧 Musa Voice Assistant [Tier 2 Full Assistant Active]")
    safe_print("=" * 65)

    assistant_thread = threading.Thread(target=voice_assistant_loop_tier2, daemon=True)
    assistant_thread.start()

    if rumps:
        rumps_app_instance = MusaMenuBarApp()
        rumps_app_instance.run()
    else:
        safe_print("⚠️ 'rumps' library not found. Running in CLI mode only.")
        while is_running and assistant_thread.is_alive():
            assistant_thread.join(timeout=1.0)


def run_tier1_listener():
    """
    TIER 1 — Lightweight Wake-Word Listener.
    Continuously listens for wake word "Musa" using existing scoring/sensitivity logic.
    No menu bar icon, no Ollama LLM load, minimal RAM/CPU usage.
    """
    if check_manual_quit_sentinel():
        sys.exit(0)

    safe_print("=" * 65)
    safe_print(" 👧 Musa Voice Assistant [Tier 1 Lightweight Wake-Word Listener]")
    safe_print("=" * 65)

    try:
        calibrate_microphone(duration=1.5)
    except Exception as e:
        safe_print(f"⚠️ Mic calibration error in Tier 1: {e}")

    while is_running:
        if os.path.exists(MANUAL_QUIT_SENTINEL):
            safe_print("🛑 [Tier 1] Manual quit sentinel detected. Exiting Tier 1.")
            break

        safe_print("\n[Tier 1] Listening for wake word...")

        tmp_path, rms = record_clip(2.5, min_threshold=None)
        if tmp_path is None:
            time.sleep(0.1)
            continue

        text = transcribe(tmp_path)
        if text:
            is_match, best_var, score, req_thresh = detect_wake_word(text, sensitivity=WAKE_WORD_SENSITIVITY)
            verdict = "MATCH" if is_match else "REJECT"
            safe_print(f"👂 [Whisper Heard]: '{text}' -> Best Match: '{best_var}' (Score: {score:.2f}, Req: {req_thresh:.2f}) [{verdict}]")
            log_wakeword_attempt(text, rms, best_var, score, req_thresh, verdict)

            if is_match:
                safe_print(f"✅ Wake word ('Musa') matched! (Variant: '{best_var}', Score: {score:.2f})")
                safe_print("\n[Tier 2] Musa activated, loading full assistant...")

                script_path = os.path.abspath(__file__)
                cmd = [sys.executable, script_path, "--tier2"]

                try:
                    res = subprocess.run(cmd, check=False)
                    exit_code = res.returncode
                    safe_print(f"📌 [Tier 2 Process Exited]: Exit Code {exit_code}")

                    if os.path.exists(MANUAL_QUIT_SENTINEL):
                        safe_print("🛑 [Tier 1] Manual quit sentinel detected. Exiting Musa completely.")
                        sys.exit(0)

                    safe_print("\n[Tier 2] Idle timeout reached, returning to lightweight mode...")
                except Exception as e:
                    safe_print(f"⚠️ Error launching Tier 2 process: {e}")
                    time.sleep(1)
            else:
                safe_print(f"⏳ [Cooldown]: Non-matching voice heard. Waiting {WAKE_WORD_COOLDOWN}s before listening again...")
                time.sleep(WAKE_WORD_COOLDOWN)

        time.sleep(0.1)


# ==========================================
# MACOS SYSTEM WAKE & UNLOCK EVENT OBSERVER
# ==========================================
try:
    from AppKit import NSWorkspace, NSWorkspaceDidWakeNotification, NSWorkspaceSessionDidBecomeActiveNotification
    from Foundation import NSObject, NSDistributedNotificationCenter, NSRunLoop
    HAS_PYOBJC_APPKIT = True
except ImportError:
    HAS_PYOBJC_APPKIT = False


def on_system_wake_event():
    global consecutive_low_count, has_shown_mic_hint
    safe_print("\n☀️ [macOS Wake Event Detected]: Re-initializing audio input stream & resetting Musa state...")
    consecutive_low_count = 0
    has_shown_mic_hint = False

    try:
        calibrate_microphone(duration=1.0)
    except Exception as e:
        safe_print(f"⚠️ Mic re-calibration on wake error: {e}")

    set_musa_state(STATE_IDLE)
    safe_print("✅ Musa is awake and ready to listen for wake word!")


if HAS_PYOBJC_APPKIT:
    class MusaWakeObserver(NSObject):
        def handleWakeNotification_(self, notification):
            safe_print("\n☀️ [NSWorkspace Notification]: System woke from sleep.")
            on_system_wake_event()

        def handleUnlockNotification_(self, notification):
            safe_print("\n🔓 [NSWorkspace Notification]: Screen unlocked / Session active.")
            on_system_wake_event()


def start_wake_observer():
    if not HAS_PYOBJC_APPKIT:
        safe_print("⚠️ PyObjC not available for system wake detection.")
        return

    def observer_thread_target():
        try:
            observer = MusaWakeObserver.alloc().init()
            nc = NSWorkspace.sharedWorkspace().notificationCenter()
            nc.addObserver_selector_name_object_(observer, "handleWakeNotification:", NSWorkspaceDidWakeNotification, None)
            nc.addObserver_selector_name_object_(observer, "handleUnlockNotification:", NSWorkspaceSessionDidBecomeActiveNotification, None)

            dnc = NSDistributedNotificationCenter.defaultCenter()
            dnc.addObserver_selector_name_object_(observer, "handleUnlockNotification:", "com.apple.screenIsUnlocked", None)

            safe_print("✅ macOS System Wake & Unlock Notification Observer active!")
            NSRunLoop.currentRunLoop().run()
        except Exception as e:
            safe_print(f"⚠️ Wake observer error: {e}")

    observer_thread = threading.Thread(target=observer_thread_target, daemon=True)
    observer_thread.start()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Musa Voice Assistant — Two-Tier Startup Model")
    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 lightweight wake-word listener (default)")
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 full assistant")

    args = parser.parse_args()

    if args.tier2:
        run_tier2_assistant()
    else:
        run_tier1_listener()


if __name__ == "__main__":
    main()
