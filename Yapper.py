import sys
import os
import time
import wave
import json
import threading
import subprocess
import msvcrt
import numpy as np
import pyaudio
import keyboard
import speech_recognition as sr
# import winsound  # Zastapione przez QSoundEffect dla kontroli glosnosci
from PIL import Image
import pystray
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QRadioButton,
    QButtonGroup, QTextEdit, QFrame, QSplitter, QMessageBox,
    QStatusBar, QSystemTrayIcon, QMenu, QSizePolicy, QListView,
    QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QUrl
from PyQt6.QtGui import QIcon, QFont, QAction
from PyQt6.QtMultimedia import QSoundEffect

# --- Dotenv dla klucza API ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Vertex AI / Gemini API (OAuth2) ---
VERTEX_AVAILABLE = False
VERTEX_UNAVAILABLE_REASON = ""
VERTEX_MODEL_ID = ""
vertex_model = None
user_credentials = None
single_instance_handle = None

def load_vertex_config():
    """Wczytuje konfiguracje Vertex AI z settings.json."""
    config = {
        "project_id": "",
        "location": "global",
        "client_secret_file": "",
        "model_id": "gemini-3.1-flash-lite"
    }

    if os.path.exists("settings.json"):
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
                vertex_config = settings.get("vertex_ai", {})
                config["project_id"] = vertex_config.get("project_id", "")
                config["location"] = vertex_config.get("location", "global")
                config["client_secret_file"] = vertex_config.get("client_secret_file", "")
                config["model_id"] = vertex_config.get("model_id", config["model_id"])
        except Exception:
            pass

    # Fallback do zmiennych srodowiskowych
    if not config["project_id"]:
        config["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not config["client_secret_file"]:
        config["client_secret_file"] = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "")
    env_model_id = os.getenv("GOOGLE_VERTEX_MODEL", "")
    if env_model_id:
        config["model_id"] = env_model_id
    if not config["model_id"]:
        config["model_id"] = GEMINI_MODEL

    return config

def setup_vertex_ai():
    """Konfiguruje Vertex AI z OAuth2 credentials."""
    global VERTEX_AVAILABLE, VERTEX_UNAVAILABLE_REASON, VERTEX_MODEL_ID, vertex_model, user_credentials

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        import vertexai
        from vertexai.generative_models import GenerativeModel
        import pickle

        vertex_config = load_vertex_config()
        PROJECT_ID = vertex_config["project_id"]
        LOCATION = vertex_config["location"]
        CLIENT_SECRET_FILE = vertex_config["client_secret_file"]
        MODEL_ID = vertex_config["model_id"]

        if not PROJECT_ID or not CLIENT_SECRET_FILE:
            VERTEX_UNAVAILABLE_REASON = "Brak konfiguracji w settings.json (project_id lub client_secret_file)"
            print(f"UWAGA: {VERTEX_UNAVAILABLE_REASON}")
            return False

        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        TOKEN_FILE = "vertex_token.pickle"

        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                user_credentials = pickle.load(token)

        if not user_credentials or not user_credentials.valid:
            if user_credentials and user_credentials.expired and user_credentials.refresh_token:
                try:
                    user_credentials.refresh(Request())
                except Exception as e:
                    log_message(f"Token Vertex AI niewazny, wymuszam ponowne logowanie: {e}")
                    user_credentials = None
                    try:
                        os.remove(TOKEN_FILE)
                    except OSError:
                        pass

            if not user_credentials or not user_credentials.valid:
                client_secret_path = CLIENT_SECRET_FILE
                if not os.path.exists(client_secret_path):
                    client_secret_path = resource_path(CLIENT_SECRET_FILE)

                if not os.path.exists(client_secret_path):
                    VERTEX_UNAVAILABLE_REASON = f"Brak pliku client_secret: {CLIENT_SECRET_FILE}"
                    print(f"UWAGA: {VERTEX_UNAVAILABLE_REASON}")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                user_credentials = flow.run_local_server(port=0)

            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(user_credentials, token)

        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=user_credentials)
        vertex_model = GenerativeModel(MODEL_ID)

        VERTEX_AVAILABLE = True
        VERTEX_MODEL_ID = MODEL_ID
        log_message(f"Vertex AI skonfigurowany (model: {MODEL_ID})")
        print(f"Vertex AI skonfigurowany (projekt: {PROJECT_ID}, model: {MODEL_ID})")
        return True

    except Exception as e:
        VERTEX_UNAVAILABLE_REASON = f"Błąd: {e}"
        print(f"UWAGA: Nie można skonfigurować Vertex AI: {e}")
        VERTEX_AVAILABLE = False
        return False

# --- Vosk Import (opcjonalny) ---
vosk_model = None
VOSK_AVAILABLE = False
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    print("UWAGA: Vosk nie jest zainstalowany. Tryb offline niedostępny.")

# --- Konfiguracja ---
LOG_FILE_NAME = "yapper_log.txt"
SETTINGS_FILE = "settings.json"  # Plik z zapisanymi ustawieniami
SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
DELAY_AFTER_KEY_RELEASE_MS = 500
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
MIN_RECORDING_SECONDS = 1.0

USE_ENSEMBLE_FOR_POLISH = True
VOSK_MODEL_PATH = None

USE_GEMINI = True
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"
GEMINI_CHUNK_SECONDS = 30
GEMINI_CHUNK_PAUSE_SECONDS = 0.35
GEMINI_MP3_BITRATE_KBPS = 64
DEFAULT_GOOGLE_FALLBACK = False
BENCHMARK_MODE = False

DEFAULT_CUSTOM_RULES = """# Przykładowe reguły (usuń # żeby aktywować):
# - Usuń wszystkie "yyy", "eee", "hmm"
# - Zamień "nowa linia" na znak nowej linii
# - Zamień "kropka" na "."
# - Zamień "przecinek" na ","
# - Formatuj jako lista punktowana"""

UI_LANGUAGE = "pl"

TEXT = {
    "pl": {
        "window_title": "Yapper",
        "model_group": "Model transkrypcji",
        "model_gemini": "Gemini (Vertex AI)",
        "model_google": "Google Speech API",
        "model_vosk": "Vosk (offline)",
        "model_status_checking": "Sprawdzanie dostępności...",
        "gui_language_group": "Język GUI",
        "language_group": "Język",
        "language_polish": "Polski",
        "language_english": "English",
        "rules_group": "Reguły Gemini",
        "rules_help": "Wpisz reguły (jedna na linię). Linie z # są ignorowane.",
        "sounds_group": "Dźwięki",
        "sounds_enable": "Włącz dźwięki PTT",
        "volume_label": "Głośność:",
        "status_group": "Status",
        "status_initializing": "Inicjalizacja...",
        "last_message": "Ostatnia wiadomość:",
        "save_config": "Zapisz konfigurację",
        "save_config_tooltip": "Zapisz wszystkie ustawienia do pliku",
        "channel_group": "Kanał {channel_id}",
        "mic_label": "Mikrofon:",
        "ptt_label": "PTT:",
        "ready": "Gotowy.",
        "recording": "Nagrywanie...",
        "processing": "Przetwarzanie...",
        "select_microphone": "BŁĄD: Wybierz mikrofon!",
        "no_audio": "Nie nagrano dźwięku.",
        "audio_conversion_error": "Błąd konwersji audio.",
        "too_short": "Nagrano zbyt krótko.",
        "too_little_data": "Nagrano zbyt mało danych.",
        "mic_not_configured": "BŁĄD: Mikrofon nie jest skonfigurowany.",
        "stream_open_error": "BŁĄD: Nie można otworzyć strumienia audio.",
        "recognized": "Rozpoznano [{system}]: {text}",
        "gemini_limit": "Limit Gemini dla tego nagrania. Tekst nie został skopiowany.",
        "recognition_failed": "Nie udało się rozpoznać mowy.",
        "processing_error": "Błąd przetwarzania: {error}",
        "settings_saved": "Ustawienia zapisane do {settings_file}",
        "settings_save_error": "Błąd zapisu ustawień: {error}",
        "settings_loaded": "Wczytano zapisane ustawienia",
        "ptt_key_prompt": "Dla kanału {channel_id} wciśnij nowy klawisz PTT (ESC, aby anulować)...",
        "ptt_key_cancelled": "Anulowano zmianę klawisza dla kanału {channel_id}.",
        "ptt_instruction": "Kanał 1 ('{key1}'): Nagrywaj | Kanał 2 ('{key2}'): Nagrywaj\nPuść, aby przetworzyć. ESC chowa do zasobnika.",
        "py_audio_error_title": "Błąd PyAudio",
        "py_audio_error_body": "Nie można zainicjalizować PyAudio: {error}\nProgram nie może działać.",
        "audio_init_error": "Błąd inicjalizacji audio. Zamykanie...",
        "tray_show": "Pokaż",
        "tray_exit": "Wyjdź",
        "model_status_gemini_ok": "Gemini: OK",
        "model_status_gemini_unavailable": "Gemini: niedostępny ({reason})",
        "model_status_google_ok": "Google: OK",
        "model_status_vosk_ok": "Vosk: OK",
        "model_status_vosk_unavailable": "Vosk: niedostępny",
    },
    "en": {
        "window_title": "Yapper",
        "model_group": "Transcription model",
        "model_gemini": "Gemini (Vertex AI)",
        "model_google": "Google Speech API",
        "model_vosk": "Vosk (offline)",
        "model_status_checking": "Checking availability...",
        "gui_language_group": "GUI language",
        "language_group": "Language",
        "language_polish": "Polish",
        "language_english": "English",
        "rules_group": "Gemini rules",
        "rules_help": "Enter rules, one per line. Lines starting with # are ignored.",
        "sounds_group": "Sounds",
        "sounds_enable": "Enable PTT sounds",
        "volume_label": "Volume:",
        "status_group": "Status",
        "status_initializing": "Initializing...",
        "last_message": "Last message:",
        "save_config": "Save config",
        "save_config_tooltip": "Save all settings to file",
        "channel_group": "Channel {channel_id}",
        "mic_label": "Microphone:",
        "ptt_label": "PTT:",
        "ready": "Ready.",
        "recording": "Recording...",
        "processing": "Processing...",
        "select_microphone": "ERROR: Select a microphone!",
        "no_audio": "No audio recorded.",
        "audio_conversion_error": "Audio conversion error.",
        "too_short": "Recording too short.",
        "too_little_data": "Not enough audio data.",
        "mic_not_configured": "ERROR: Microphone is not configured.",
        "stream_open_error": "ERROR: Cannot open audio stream.",
        "recognized": "Recognized [{system}]: {text}",
        "gemini_limit": "Gemini limit for this recording. Text was not copied.",
        "recognition_failed": "Could not recognize speech.",
        "processing_error": "Processing error: {error}",
        "settings_saved": "Settings saved to {settings_file}",
        "settings_save_error": "Settings save error: {error}",
        "settings_loaded": "Saved settings loaded",
        "ptt_key_prompt": "For channel {channel_id}, press a new PTT key (ESC to cancel)...",
        "ptt_key_cancelled": "Cancelled key change for channel {channel_id}.",
        "ptt_instruction": "Channel 1 ('{key1}'): record | Channel 2 ('{key2}'): record\nRelease to process. ESC hides to tray.",
        "py_audio_error_title": "PyAudio error",
        "py_audio_error_body": "Cannot initialize PyAudio: {error}\nThe program cannot run.",
        "audio_init_error": "Audio initialization error. Closing...",
        "tray_show": "Show",
        "tray_exit": "Exit",
        "model_status_gemini_ok": "Gemini: OK",
        "model_status_gemini_unavailable": "Gemini: unavailable ({reason})",
        "model_status_google_ok": "Google: OK",
        "model_status_vosk_ok": "Vosk: OK",
        "model_status_vosk_unavailable": "Vosk: unavailable",
    },
}


def txt(key):
    return TEXT[UI_LANGUAGE][key]


def fmt(key, **kwargs):
    return txt(key).format(**kwargs)

custom_rules_text = ""

# --- Zasoby Globalne ---
pyaudio_instance = None
recognizer = sr.Recognizer()
stop_program_event = threading.Event()
tray_icon = None


def acquire_single_instance_lock():
    """Zapobiega uruchomieniu kilku instancji aplikacji naraz."""
    global single_instance_handle
    if os.name != "nt":
        return True

    lock_path = os.path.join(os.path.expanduser("~"), ".yapper.lock")
    handle = open(lock_path, "a+b")
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return False

    single_instance_handle = handle
    return True


def initialize_vosk_model():
    """Inicjalizuje model Vosk dla polskiego (lazy loading)."""
    global vosk_model, VOSK_AVAILABLE

    if not VOSK_AVAILABLE:
        return None

    if vosk_model is not None:
        return vosk_model

    try:
        log_message("Inicjalizacja modelu Vosk dla polskiego...")
        if VOSK_MODEL_PATH and os.path.exists(VOSK_MODEL_PATH):
            vosk_model = VoskModel(VOSK_MODEL_PATH)
        else:
            vosk_model = VoskModel(lang="pl")
        log_message("Model Vosk zaladowany pomyslnie.")
        return vosk_model
    except Exception as e:
        log_message(f"BLAD: Nie mozna zaladowac modelu Vosk: {e}")
        VOSK_AVAILABLE = False
        return None


def transcribe_with_vosk(audio_data_bytes, sample_rate):
    """Transkrypcja za pomoca Vosk."""
    global vosk_model

    if not VOSK_AVAILABLE:
        return {'text': '', 'confidence': 0, 'system': 'Vosk', 'error': 'Vosk niedostępny'}

    model = initialize_vosk_model()
    if model is None:
        return {'text': '', 'confidence': 0, 'system': 'Vosk', 'error': 'Model nie zaladowany'}

    try:
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)

        chunk_size = 4000
        for i in range(0, len(audio_data_bytes), chunk_size):
            chunk = audio_data_bytes[i:i + chunk_size]
            rec.AcceptWaveform(chunk)

        result = json.loads(rec.FinalResult())
        text = result.get('text', '')

        confidence = 0.85
        if 'result' in result and result['result']:
            word_confs = [w.get('conf', 0.85) for w in result['result']]
            if word_confs:
                confidence = sum(word_confs) / len(word_confs)

        return {
            'text': text,
            'confidence': confidence,
            'system': 'Vosk'
        }
    except Exception as e:
        log_message(f"Blad Vosk: {e}")
        return {'text': '', 'confidence': 0, 'system': 'Vosk', 'error': str(e)}


def transcribe_with_google(audio_data_obj, lang_code):
    """Transkrypcja za pomoca Google Speech API."""
    try:
        text = recognizer.recognize_google(audio_data_obj, language=lang_code)
        return {
            'text': text,
            'confidence': 0.92,
            'system': 'Google'
        }
    except sr.UnknownValueError:
        return {'text': '', 'confidence': 0, 'system': 'Google', 'error': 'UnknownValueError'}
    except sr.RequestError as e:
        return {'text': '', 'confidence': 0, 'system': 'Google', 'error': str(e)}
    except Exception as e:
        return {'text': '', 'confidence': 0, 'system': 'Google', 'error': str(e)}


def build_wav_data(audio_bytes, sample_rate):
    """Opakowuje surowe mono PCM 16-bit w kontener WAV."""
    import io

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    wav_buffer.seek(0)
    return wav_buffer.read()


def build_mp3_data(audio_bytes, sample_rate):
    """Koduje surowe mono PCM 16-bit do MP3 bez zewnetrznego ffmpeg."""
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(GEMINI_MP3_BITRATE_KBPS)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(5)
    return bytes(encoder.encode(audio_bytes) + encoder.flush())


def build_gemini_audio_payload(audio_bytes, sample_rate):
    """Buduje payload audio dla Gemini, preferujac MP3 i cofajac sie do WAV."""
    try:
        mp3_data = build_mp3_data(audio_bytes, sample_rate)
        if mp3_data:
            return mp3_data, "audio/mp3"
    except Exception as e:
        log_message(f"Gemini: MP3 encode failed, fallback do WAV: {e}")

    return build_wav_data(audio_bytes, sample_rate), "audio/wav"


def is_quota_error(error_text):
    """Rozpoznaje bledy limitow Vertex AI."""
    normalized = (error_text or "").lower()
    return "429" in normalized or "resource has been exhausted" in normalized or "quota" in normalized


def get_gemini_model(model_id=None):
    """Zwraca obiekt modelu Gemini dla aktualnego albo wskazanego modelu."""
    if model_id is None or model_id == VERTEX_MODEL_ID:
        return vertex_model

    from vertexai.generative_models import GenerativeModel
    return GenerativeModel(model_id)


def generate_gemini_content(audio_bytes, sample_rate, prompt, model_id=None):
    """Wysyla pojedynczy fragment audio do Gemini."""
    from vertexai.generative_models import Part

    audio_data, mime_type = build_gemini_audio_payload(audio_bytes, sample_rate)
    audio_part = Part.from_data(audio_data, mime_type=mime_type)
    model = get_gemini_model(model_id)
    return model.generate_content([audio_part, prompt])


def split_audio_chunks(audio_bytes, sample_rate, chunk_seconds):
    """Dzieli mono PCM 16-bit na rowne fragmenty czasowe."""
    bytes_per_second = sample_rate * 2
    chunk_size = max(bytes_per_second, bytes_per_second * chunk_seconds)
    for start in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[start:start + chunk_size]


def clean_transcript_text(text):
    """Usuwa proste opakowanie cytatami z odpowiedzi modelu."""
    text = (text or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.strip()


def transcribe_with_gemini(audio_bytes, sample_rate, lang_code="pl-PL"):
    """Transkrypcja za pomoca Gemini przez Vertex AI."""
    global vertex_model, VERTEX_AVAILABLE, custom_rules_text

    if not VERTEX_AVAILABLE or vertex_model is None:
        return {'text': '', 'confidence': 0, 'system': 'Gemini', 'error': 'Vertex AI niedostępny'}

    try:
        start_time = time.time()
        lang_name = "polskim" if lang_code.startswith("pl") else "angielskim"

        prompt = f"""Jesteś precyzyjnym systemem transkrypcji mowy.
Przetranskrybuj dokładnie to, co słyszysz w nagraniu audio.
Język: {lang_name}.
WAŻNE: Zwróć TYLKO tekst transkrypcji, bez żadnych dodatkowych komentarzy, wyjaśnień ani formatowania.
Jeśli nie słyszysz żadnej mowy, zwróć pusty tekst."""

        active_rules = [line.strip() for line in custom_rules_text.split('\n')
                       if line.strip() and not line.strip().startswith('#')]

        if active_rules:
            rules_text = '\n'.join(f"- {rule}" for rule in active_rules)
            prompt += f"""

DODATKOWE REGUŁY PRZETWARZANIA (zastosuj je do transkrypcji):
{rules_text}"""

        duration_seconds = len(audio_bytes) / max(sample_rate * 2, 1)
        used_model_id = VERTEX_MODEL_ID or GEMINI_MODEL

        if duration_seconds > GEMINI_CHUNK_SECONDS:
            chunk_count = (len(audio_bytes) + (sample_rate * 2 * GEMINI_CHUNK_SECONDS) - 1) // (sample_rate * 2 * GEMINI_CHUNK_SECONDS)
            log_message(
                f"Gemini: dlugie audio {duration_seconds:.1f}s, "
                f"dzielenie na {chunk_count} fragmentow po {GEMINI_CHUNK_SECONDS}s, format MP3"
            )
            partial_texts = []

            for index, chunk in enumerate(split_audio_chunks(audio_bytes, sample_rate, GEMINI_CHUNK_SECONDS), start=1):
                try:
                    log_message(f"Gemini chunk {index}/{chunk_count}: model {used_model_id}")
                    response = generate_gemini_content(chunk, sample_rate, prompt, used_model_id)
                except Exception as chunk_error:
                    error_text = str(chunk_error)
                    if used_model_id != GEMINI_FALLBACK_MODEL and is_quota_error(error_text):
                        log_message(
                            f"Gemini chunk {index}/{chunk_count}: quota na {used_model_id}, "
                            f"retry tego samego fragmentu przez {GEMINI_FALLBACK_MODEL}"
                        )
                        used_model_id = GEMINI_FALLBACK_MODEL
                        response = generate_gemini_content(chunk, sample_rate, prompt, used_model_id)
                    else:
                        raise

                chunk_text = response.text.strip() if response.text else ''
                if chunk_text:
                    partial_texts.append(clean_transcript_text(chunk_text))
                time.sleep(GEMINI_CHUNK_PAUSE_SECONDS)

            elapsed_time = time.time() - start_time
            return {
                'text': " ".join(partial_texts).strip(),
                'confidence': 0.95,
                'system': 'Gemini',
                'model_id': used_model_id,
                'latency': elapsed_time
            }

        try:
            log_message(f"Gemini: krotkie audio {duration_seconds:.1f}s, format MP3, model {used_model_id}")
            response = generate_gemini_content(audio_bytes, sample_rate, prompt, used_model_id)
        except Exception as e:
            error_text = str(e)
            if used_model_id != GEMINI_FALLBACK_MODEL and is_quota_error(error_text):
                log_message(f"Gemini quota na {used_model_id}, retry tego samego audio przez {GEMINI_FALLBACK_MODEL}")
                used_model_id = GEMINI_FALLBACK_MODEL
                response = generate_gemini_content(audio_bytes, sample_rate, prompt, used_model_id)
            else:
                raise

        elapsed_time = time.time() - start_time
        text = response.text.strip() if response.text else ''
        text = clean_transcript_text(text)

        return {
            'text': text,
            'confidence': 0.95,
            'system': 'Gemini',
            'model_id': used_model_id,
            'latency': elapsed_time
        }

    except Exception as e:
        log_message(f"Blad Gemini: {e}")
        return {'text': '', 'confidence': 0, 'system': 'Gemini', 'error': str(e)}


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    try:
        with open(LOG_FILE_NAME, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"BLAD ZAPISU LOGA: {e}")


class SignalBridge(QObject):
    """Most do komunikacji miedzy watkami a GUI Qt."""
    status_changed = pyqtSignal(str)
    processed_model_changed = pyqtSignal(str)
    clipboard_write_requested = pyqtSignal(str, str, str, str)
    ready_signal = pyqtSignal()
    stop_recording_ch1 = pyqtSignal()  # Sygnał do zatrzymania nagrywania kanału 1
    stop_recording_ch2 = pyqtSignal()  # Sygnał do zatrzymania nagrywania kanału 2
class PttChannel:
    """Zarzadza jednym kanalem Push-to-Talk."""

    def __init__(self, channel_id, app, initial_ptt_key, initial_lang, target_mic_name_start=""):
        self.id = channel_id
        self.app = app
        self.ptt_activation_key = initial_ptt_key
        self.ptt_key_is_keypad = True if initial_ptt_key.isdigit() else False
        self.current_lang_code = initial_lang

        self.mic_details = {"index": None, "name": "Nie wybrano", "sample_rate": 16000, "channels": 1,
                            "sample_width": 2}
        self.target_mic_name_start = target_mic_name_start.lower()

        self.is_recording_active = False
        self.ptt_key_pressed = False
        self.pyaudio_stream = None
        self.audio_frames = []
        self.current_recording_actual_channels = 1

        # Elementy GUI (Qt)
        self.lang_combo = None
        self.mic_combo = None
        self.ptt_button = None

    def update_status(self, message):
        self.app.update_status(f"[Kanal {self.id}] {message}")

    def start_recording(self):
        # Ignoruj jesli juz nagrywamy lub program sie zamyka
        if stop_program_event.is_set() or self.is_recording_active or self.app.is_any_recording_active:
            return

        if self.mic_details.get("index") is None:
            self.update_status("BLAD: Wybierz mikrofon!")
            return

        self.ptt_key_pressed = True
        self.is_recording_active = True
        self.app.is_any_recording_active = True
        
        # Dzwiek startu
        self.app._play_sound("press")
        
        self.update_status("Nagrywanie...")
        threading.Thread(target=self.record_audio_loop, daemon=True).start()

    def stop_recording_and_process(self):
        # Natychmiast resetuj flage klawisza
        if not self.ptt_key_pressed:
            return
        
        self.ptt_key_pressed = False
        
        if stop_program_event.is_set() or not self.is_recording_active:
            return

        # Wyslij sygnal do glownego watku Qt
        if self.id == "1":
            self.app.signal_bridge.stop_recording_ch1.emit()
        else:
            self.app.signal_bridge.stop_recording_ch2.emit()

    def _actual_stop_and_process(self):
        if not self.is_recording_active:
            return

        # Zatrzymaj nagrywanie
        self.is_recording_active = False
        self.app.is_any_recording_active = False

        # Dzwiek stopu
        self.app._play_sound("release")

        # Poczekaj az watek nagrywania sie zakonczy
        time.sleep(0.15)

        self.update_status("Przetwarzanie...")

        if not self.audio_frames:
            self.update_status("Nie nagrano dzwieku.")
            self.app.set_ready_status()
            return

        recorded_audio_data = b''.join(self.audio_frames)
        self.audio_frames = []

        log_message(f"CH {self.id}: Zakonczono nagrywanie. Rozmiar: {len(recorded_audio_data)} bajtow.")

        if self.current_recording_actual_channels > 1:
            log_message(f"CH {self.id}: Konwersja do mono.")
            try:
                audio_array = np.frombuffer(recorded_audio_data, dtype=np.int16)
                audio_mono = audio_array[::self.current_recording_actual_channels]
                recorded_audio_data = audio_mono.tobytes()
            except Exception as e:
                log_message(f"CH {self.id}: Blad konwersji do mono: {e}")
                self.update_status("Blad konwersji audio.")
                self.app.set_ready_status()
                return

        recording_filename = f"last_recording_ch{self.id}.wav"
        if SAVE_LAST_RECORDING or PLAY_LAST_RECORDING:
            self.save_and_play_audio(recorded_audio_data, recording_filename)

        if len(recorded_audio_data) < 400:
            self.update_status("Nagrano zbyt malo danych.")
            self.app.set_ready_status()
            return

        # Przetwarzanie w osobnym watku
        threading.Thread(target=self._process_audio, args=(recorded_audio_data,), daemon=True).start()

    def _process_audio(self, recorded_audio_data):
        """Przetwarzanie audio w osobnym watku."""
        try:
            audio_data_obj = sr.AudioData(recorded_audio_data, self.mic_details['sample_rate'],
                                          self.mic_details['sample_width'])
            
            text = None
            used_system = None
            
            selected = self.app.get_selected_model()
            log_message(f"CH {self.id}: Wybrany model: {selected}")
            
            # --- GEMINI ---
            if selected == "gemini" and VERTEX_AVAILABLE:
                self.update_status("Gemini...")
                gemini_start = time.time()
                gemini_result = transcribe_with_gemini(
                    recorded_audio_data, 
                    self.mic_details['sample_rate'],
                    self.current_lang_code
                )
                gemini_time = time.time() - gemini_start
                text = gemini_result.get('text', '')
                
                if text:
                    log_message(f"CH {self.id}: Gemini OK w {gemini_time:.2f}s")
                    used_system = "Gemini"
                else:
                    error = gemini_result.get('error', 'Brak tekstu')
                    log_message(f"CH {self.id}: Gemini nie rozpoznal: {error}")
            
            # --- GOOGLE ---
            elif selected == "google":
                self.update_status("Google Speech...")
                try:
                    google_start = time.time()
                    text = recognizer.recognize_google(audio_data_obj, language=self.current_lang_code)
                    google_time = time.time() - google_start
                    log_message(f"CH {self.id}: Google OK w {google_time:.2f}s")
                    used_system = "Google"
                except sr.UnknownValueError:
                    log_message(f"CH {self.id}: Google nie rozpoznal mowy")
                except sr.RequestError as e:
                    log_message(f"CH {self.id}: Google API blad: {e}")
            
            # --- VOSK ---
            elif selected == "vosk" and VOSK_AVAILABLE:
                self.update_status("Vosk (offline)...")
                vosk_start = time.time()
                vosk_result = transcribe_with_vosk(recorded_audio_data, self.mic_details['sample_rate'])
                vosk_time = time.time() - vosk_start
                text = vosk_result.get('text', '')
                
                if text:
                    log_message(f"CH {self.id}: Vosk OK w {vosk_time:.2f}s")
                    used_system = "Vosk"
            
            # --- FALLBACK ---
            if not text and selected != "google":
                self.update_status("Fallback: Google Speech...")
                try:
                    google_start = time.time()
                    text = recognizer.recognize_google(audio_data_obj, language=self.current_lang_code)
                    google_time = time.time() - google_start
                    log_message(f"CH {self.id}: Google fallback OK w {google_time:.2f}s")
                    used_system = "Google (fallback)"
                except sr.UnknownValueError:
                    log_message(f"CH {self.id}: Google fallback nie rozpoznal")
                except sr.RequestError as e:
                    log_message(f"CH {self.id}: Google fallback blad: {e}")
            
            # --- WYNIK ---
            if text:
                self.update_status(f"Rozpoznano [{used_system}]: {text}")
                pyperclip.copy(text)
            else:
                self.update_status("Nie udalo sie rozpoznac mowy.")
                log_message(f"CH {self.id}: Wszystkie systemy zawiodly")
                    
        except Exception as e:
            self.update_status(f"Blad przetwarzania: {e}")
            log_message(f"CH {self.id}: Blad podczas przetwarzania: {e}")

        self.app.set_ready_status()

    def record_audio_loop(self):
        global pyaudio_instance

        if self.mic_details.get("index") is None:
            self.update_status("BLAD: Mikrofon nie jest skonfigurowany.")
            self.is_recording_active = False
            return

        stream_opened_successfully = False
        for target_ch in [1, 2, self.mic_details['channels']]:
            if stream_opened_successfully: break
            if target_ch == self.mic_details['channels'] and target_ch in [1, 2]:
                continue

            try:
                log_message(f"CH {self.id}: Proba otwarcia strumienia z {target_ch} kanalem/ami...")
                self.pyaudio_stream = pyaudio_instance.open(
                    format=AUDIO_FORMAT, channels=target_ch,
                    rate=self.mic_details['sample_rate'], input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    input_device_index=self.mic_details['index'])
                self.current_recording_actual_channels = target_ch
                stream_opened_successfully = True
                log_message(f"CH {self.id}: SUKCES. Strumien otwarty.")
            except Exception as e:
                log_message(f"CH {self.id}: BLAD otwarcia strumienia: {e}")

        if not stream_opened_successfully:
            self.update_status(f"BLAD: Nie mozna otworzyc strumienia audio.")
            self.is_recording_active = False
            self.app.is_any_recording_active = False
            return

        self.audio_frames = []
        while self.is_recording_active and not stop_program_event.is_set():
            try:
                data = self.pyaudio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.audio_frames.append(data)
            except IOError:
                pass
            except Exception:
                self.is_recording_active = False
                break

        if self.pyaudio_stream:
            try:
                if self.pyaudio_stream.is_active(): self.pyaudio_stream.stop_stream()
                self.pyaudio_stream.close()
            except Exception:
                pass
        self.pyaudio_stream = None
        log_message(f"CH {self.id}: Petla nagrywania zakonczona.")

    def save_and_play_audio(self, audio_data_bytes, filename):
        channels_to_save = 1
        filepath = os.path.join(os.getcwd(), filename)
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(channels_to_save)
                wf.setsampwidth(self.mic_details['sample_width'])
                wf.setframerate(self.mic_details['sample_rate'])
                wf.writeframes(audio_data_bytes)

            if PLAY_LAST_RECORDING:
                if os.name == 'nt':
                    os.startfile(filepath)
                else:
                    subprocess.call(["open" if sys.platform == "darwin" else "xdg-open", filepath])
        except Exception as e:
            log_message(f"CH {self.id}: Blad zapisu pliku: {e}")


class SpeechToClipboardApp(QMainWindow):
    """Glowne okno aplikacji PyQt6."""
    
    def __init__(self):
        super().__init__()
        
        self.signal_bridge = SignalBridge()
        self.signal_bridge.status_changed.connect(self._handle_signal)
        self.signal_bridge.ready_signal.connect(self._set_ready)
        
        self.channel_being_configured = None
        self.is_any_recording_active = False
        self.all_input_mics_details = []
        self.selected_model = "gemini"
        self.sounds_enabled = True
        self.sound_volume = 50  # Domyslna glosnosc 50%
        
        self.press_sound = QSoundEffect()
        self.release_sound = QSoundEffect()
        self._init_sounds()
        
        self.ptt_channels = {
            "1": PttChannel(channel_id="1", app=self, initial_ptt_key="5", initial_lang="pl-PL",
                            target_mic_name_start="Voicemeeter Out B1"),
            "2": PttChannel(channel_id="2", app=self, initial_ptt_key="6", initial_lang="pl-PL",
                            target_mic_name_start="CABLE Output")
        }
        
        # Podlacz sygnaly stop_recording do kanalow
        self.signal_bridge.stop_recording_ch1.connect(
            lambda: QTimer.singleShot(DELAY_AFTER_KEY_RELEASE_MS, self.ptt_channels["1"]._actual_stop_and_process)
        )
        self.signal_bridge.stop_recording_ch2.connect(
            lambda: QTimer.singleShot(DELAY_AFTER_KEY_RELEASE_MS, self.ptt_channels["2"]._actual_stop_and_process)
        )
        
        self.init_ui()
        
    def init_ui(self):
        """Inicjalizacja interfejsu uzytkownika."""
        self.setWindowTitle("Yapper")
        self.setMinimumSize(650, 550)
        
        # Ikona okna
        try:
            icon_path = resource_path("_internal/wafflin.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        
        # Glowny widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # --- Model transkrypcji ---
        model_group = QGroupBox("Model transkrypcji")
        model_layout = QVBoxLayout(model_group)
        
        models_row = QHBoxLayout()
        self.model_button_group = QButtonGroup(self)
        
        # Gemini - domyslnie wylaczony, wlaczony w run() po inicjalizacji Vertex AI
        self.gemini_radio = QRadioButton("Gemini (Vertex AI)")
        self.gemini_radio.setEnabled(False)  # Wlaczony pozniej w run()
        self.gemini_radio.toggled.connect(lambda checked: self._on_model_change("gemini") if checked else None)
        self.model_button_group.addButton(self.gemini_radio)
        models_row.addWidget(self.gemini_radio)
        
        # Google
        self.google_radio = QRadioButton("Google Speech API")
        self.google_radio.toggled.connect(lambda checked: self._on_model_change("google") if checked else None)
        self.model_button_group.addButton(self.google_radio)
        models_row.addWidget(self.google_radio)
        
        # Vosk
        self.vosk_radio = QRadioButton("Vosk (offline)")
        self.vosk_radio.setEnabled(VOSK_AVAILABLE)
        self.vosk_radio.toggled.connect(lambda checked: self._on_model_change("vosk") if checked else None)
        self.model_button_group.addButton(self.vosk_radio)
        models_row.addWidget(self.vosk_radio)
        
        models_row.addStretch()
        model_layout.addLayout(models_row)
        
        # Status dostepnosci - bedzie aktualizowany w run()
        self.model_status_label = QLabel("Sprawdzanie dostepnosci...")
        self.model_status_label.setStyleSheet("color: #7a7aaa; font-size: 11px;")
        model_layout.addWidget(self.model_status_label)
        
        # Domyslnie Google dopoki Vertex AI nie zostanie sprawdzony
        self.google_radio.setChecked(True)
        self.selected_model = "google"
        
        main_layout.addWidget(model_group)
        
        # --- Kanaly PTT ---
        for ch_id in self.ptt_channels:
            channel_group = self._create_channel_ui(ch_id)
            main_layout.addWidget(channel_group)
        
        # --- Custom Rules ---
        rules_group = QGroupBox("Custom Rules (Gemini)")
        rules_layout = QVBoxLayout(rules_group)
        rules_layout.setSpacing(4)
        rules_layout.setContentsMargins(8, 6, 8, 8)
        
        help_label = QLabel("Wpisz reguly (jedna na linie). Linie z # sa ignorowane.")
        help_label.setStyleSheet("color: #7a7aaa; font-size: 11px;")
        rules_layout.addWidget(help_label)
        
        self.rules_text = QTextEdit()
        self.rules_text.setPlainText(DEFAULT_CUSTOM_RULES)
        self.rules_text.setMinimumHeight(90)
        self.rules_text.setMaximumHeight(110)
        self.rules_text.textChanged.connect(self._update_custom_rules)
        rules_layout.addWidget(self.rules_text)
        
        main_layout.addWidget(rules_group)
        
        # --- Dolny rzad (Dzwieki i Status) ---
        bottom_row_layout = QHBoxLayout()
        
        # --- Dzwieki ---
        sounds_group = QGroupBox("Dzwieki")
        sounds_layout = QHBoxLayout(sounds_group)
        sounds_layout.setContentsMargins(8, 6, 8, 8)
        
        self.sounds_checkbox = QCheckBox("Wlacz dzwieki PTT")
        self.sounds_checkbox.setChecked(True)
        self.sounds_checkbox.toggled.connect(self._toggle_sounds)
        self.sounds_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #5a5a8a;
                background: #2a2a45;
            }
            QCheckBox::indicator:checked {
                background: #6c6cff;
                border: 1px solid #6c6cff;
                image: url(:/qt-project.org/styles/commonstyle/images/standardbutton-yes-16.png);
            }
            QCheckBox::indicator:hover {
                border: 1px solid #7c7cff;
            }
        """)
        sounds_layout.addWidget(self.sounds_checkbox)
        
        # Suwak glosnosci
        vol_label = QLabel("Glosnosc:")
        vol_label.setStyleSheet("color: #d0d0d0; margin-left: 20px;")
        sounds_layout.addWidget(vol_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(self.sound_volume)
        self.volume_slider.setFixedWidth(120)  # Mniejszy suwak
        self.volume_slider.valueChanged.connect(self._on_volume_change)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3a3a5c;
                height: 8px;
                background: #2a2a45;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6c6cff;
                border: 1px solid #6c6cff;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #7c7cff;
            }
        """)
        sounds_layout.addWidget(self.volume_slider)
        
        # Etykieta z procentami
        self.volume_percent_label = QLabel(f"{self.sound_volume}%")
        self.volume_percent_label.setStyleSheet("color: #a8b4ff; font-weight: bold; min-width: 35px; margin-left: 5px;")
        sounds_layout.addWidget(self.volume_percent_label)
        
        sounds_layout.addStretch()  # Wyrownanie do lewej
        
        bottom_row_layout.addWidget(sounds_group, 1)
        
        # --- Status i Zapisz ---
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)
        status_layout.setContentsMargins(8, 6, 8, 8)
        
        self.status_label = QLabel("Inicjalizacja...")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label, 1)
        
        # Przycisk zapisu konfiguracji
        save_config_btn = QPushButton("Zapisz konfiguracje")
        save_config_btn.setFixedWidth(150)
        save_config_btn.setToolTip("Zapisz wszystkie ustawienia do pliku")
        save_config_btn.clicked.connect(self._save_settings)
        status_layout.addWidget(save_config_btn)
        
        bottom_row_layout.addWidget(status_group, 1)
        
        main_layout.addLayout(bottom_row_layout)
        
        # --- Instrukcje PTT ---
        self.ptt_instruction_label = QLabel()
        self.ptt_instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ptt_instruction_label.setStyleSheet("""
            font-size: 12px;
            padding: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #252542, stop:1 #1f1f38);
            border: 1px solid #3a3a5c;
            border-radius: 8px;
            color: #b0b0d0;
        """)
        main_layout.addWidget(self.ptt_instruction_label)
        
        # Rozciagnij reszte
        main_layout.addStretch()
        
        # Inicjalizacja custom rules
        self._update_custom_rules()
        
    def _create_channel_ui(self, channel_id):
        """Tworzy UI dla kanalu PTT."""
        channel = self.ptt_channels[channel_id]
        
        group = QGroupBox(f"Kanal {channel_id}")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 6, 10, 6)
        
        # Jezyk
        lang_label = QLabel("Jezyk:")
        layout.addWidget(lang_label)
        
        channel.lang_combo = QComboBox()
        channel.lang_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Usuniecie scrollbara z listy rozwijanej
        lang_view = QListView()
        lang_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lang_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        channel.lang_combo.setView(lang_view)
        
        # Próba naprawy czarnego tła przy zaokrąglonych rogach
        try:
            # Pobieramy kontener (QComboBoxPrivateContainer)
            container = lang_view.parentWidget()
            if container:
                container.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
                container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        except Exception:
            pass
        
        channel.lang_combo.addItem("Polski", "pl-PL")
        channel.lang_combo.addItem("Angielski", "en-US")
        channel.lang_combo.setFixedWidth(90)
        # Wymuszenie szerokosci popupu takiej samej jak combobox
        lang_view.setFixedWidth(90)
        for i in range(channel.lang_combo.count()):
            if channel.lang_combo.itemData(i) == channel.current_lang_code:
                channel.lang_combo.setCurrentIndex(i)
                break
        channel.lang_combo.currentIndexChanged.connect(
            lambda idx, ch=channel: self._on_language_change(ch, ch.lang_combo.itemData(idx))
        )
        layout.addWidget(channel.lang_combo)
        
        # Mikrofon
        mic_label = QLabel("Mikrofon:")
        layout.addWidget(mic_label)
        
        channel.mic_combo = QComboBox()
        channel.mic_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Usuniecie scrollbara z listy rozwijanej
        mic_view = QListView()
        mic_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mic_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        channel.mic_combo.setView(mic_view)
        
        channel.mic_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        channel.mic_combo.currentIndexChanged.connect(
            lambda idx, ch=channel: self._on_mic_select(ch)
        )
        layout.addWidget(channel.mic_combo, 1)  # stretch factor 1
        
        # Klawisz PTT
        ptt_label = QLabel("PTT:")
        layout.addWidget(ptt_label)
        
        channel.ptt_button = QPushButton(channel.ptt_activation_key.upper())
        channel.ptt_button.setFixedWidth(70)
        channel.ptt_button.clicked.connect(
            lambda checked, ch_id=channel_id: self._activate_ptt_key_setting(ch_id)
        )
        layout.addWidget(channel.ptt_button)
        
        return group
    
    def _on_model_change(self, model):
        """Zmiana modelu transkrypcji."""
        self.selected_model = model
        log_message(f"Zmieniono model na: {model}")
    
    def get_selected_model(self):
        """Zwraca wybrany model."""
        return self.selected_model
    
    def _on_language_change(self, channel, lang_code):
        """Zmiana jezyka dla kanalu."""
        channel.current_lang_code = lang_code
        self.update_status(f"CH {channel.id}: Jezyk zmieniony na {lang_code}")
    
    def _on_mic_select(self, channel):
        """Wybor mikrofonu dla kanalu."""
        idx = channel.mic_combo.currentIndex()
        if idx >= 0 and idx < len(self.all_input_mics_details):
            mic = self.all_input_mics_details[idx]
            channel.mic_details.update(mic)
            self.update_status(f"CH {channel.id}: Zmieniono mikrofon na {mic['name']}")
    
    def _activate_ptt_key_setting(self, channel_id):
        """Aktywacja trybu ustawiania klawisza PTT."""
        self.channel_being_configured = self.ptt_channels[channel_id]
        self.update_status(f"Dla kanalu {channel_id} wcisnij nowy klawisz PTT (ESC by anulowac)...")
    
    def _set_preset(self, preset_text):
        """Ustawia preset custom rules."""
        self.rules_text.setPlainText(preset_text)
    
    def _toggle_sounds(self, checked):
        """Wlacza/wylacza dzwieki."""
        self.sounds_enabled = checked
        log_message(f"Dzwieki {'wlaczone' if checked else 'wylaczone'}")

    def _on_volume_change(self, value):
        """Zmiana glosnosci."""
        self.sound_volume = value
        if hasattr(self, 'volume_percent_label'):
            self.volume_percent_label.setText(f"{value}%")
        self._update_volume()

    def _init_sounds(self):
        """Inicjalizacja dzwiekow."""
        try:
            press_path = os.path.join(os.getcwd(), "sounds", "press.wav")
            release_path = os.path.join(os.getcwd(), "sounds", "release.wav")
            
            if os.path.exists(press_path):
                self.press_sound.setSource(QUrl.fromLocalFile(press_path))
            if os.path.exists(release_path):
                self.release_sound.setSource(QUrl.fromLocalFile(release_path))
                
            self._update_volume()
        except Exception as e:
            log_message(f"Blad inicjalizacji dzwiekow: {e}")

    def _update_volume(self):
        """Aktualizacja glosnosci efektow."""
        vol = self.sound_volume / 100.0
        self.press_sound.setVolume(vol)
        self.release_sound.setVolume(vol)

    def _play_sound(self, sound_type):
        """Odtwarza dzwiek (press/release)."""
        if not self.sounds_enabled:
            return
            
        try:
            if sound_type == "press":
                if self.press_sound.status() == QSoundEffect.Status.Ready:
                    self.press_sound.play()
            else:
                if self.release_sound.status() == QSoundEffect.Status.Ready:
                    self.release_sound.play()
        except Exception as e:
            log_message(f"Blad odtwarzania dzwieku: {e}")

    def _save_settings(self):
        """Zapisuje ustawienia do pliku."""
        try:
            # Wczytaj istniejace ustawienia vertex_ai jesli istnieja
            existing_vertex_config = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                        existing_vertex_config = existing.get("vertex_ai", {})
                except Exception:
                    pass
            
            # Jesli nie ma konfiguracji vertex_ai, stworz domyslna
            if not existing_vertex_config:
                existing_vertex_config = {
                    "project_id": "",
                    "location": "us-central1",
                    "client_secret_file": ""
                }
            
            settings = {
                "model": self.selected_model,
                "custom_rules": self.rules_text.toPlainText(),
                "sounds_enabled": self.sounds_enabled,
                "sound_volume": self.sound_volume,
                "vertex_ai": existing_vertex_config,
                "channels": {}
            }
            for ch_id, channel in self.ptt_channels.items():
                settings["channels"][str(ch_id)] = {
                    "mic_index": channel.mic_details.get("index"),
                    "mic_name": channel.mic_details.get("name", ""),
                    "language": channel.current_lang_code,
                    "ptt_key": channel.ptt_activation_key
                }
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self.update_status(f"Ustawienia zapisane do {SETTINGS_FILE}")
            log_message(f"Zapisano ustawienia: {settings}")
        except Exception as e:
            self.update_status(f"Blad zapisu ustawien: {e}")
            log_message(f"Blad zapisu ustawien: {e}")
    
    def _load_settings(self):
        """Wczytuje ustawienia z pliku."""
        if not os.path.exists(SETTINGS_FILE):
            log_message("Brak pliku ustawien - uzycie domyslnych")
            return
        
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            log_message(f"Wczytano ustawienia: {settings}")
            
            # Model
            saved_model = settings.get("model", "google")
            if saved_model == "gemini" and VERTEX_AVAILABLE:
                self.gemini_radio.setChecked(True)
                self.selected_model = "gemini"
            elif saved_model == "vosk":
                self.vosk_radio.setChecked(True)
                self.selected_model = "vosk"
            else:
                self.google_radio.setChecked(True)
                self.selected_model = "google"
            
            # Custom rules
            if "custom_rules" in settings:
                self.rules_text.setPlainText(settings["custom_rules"])
            
            # Dzwieki
            if "sounds_enabled" in settings:
                self.sounds_enabled = settings["sounds_enabled"]
                self.sounds_checkbox.setChecked(self.sounds_enabled)
            
            if "sound_volume" in settings:
                self.sound_volume = settings["sound_volume"]
                self.volume_slider.setValue(self.sound_volume)
                self._update_volume()
            
            # Kanaly
            for ch_id_str, ch_settings in settings.get("channels", {}).items():
                ch_id = int(ch_id_str)
                if ch_id not in self.ptt_channels:
                    continue
                channel = self.ptt_channels[ch_id]
                
                # Mikrofon - szukaj po nazwie
                saved_mic_name = ch_settings.get("mic_name", "")
                if saved_mic_name and channel.mic_combo is not None:
                    for i in range(channel.mic_combo.count()):
                        if saved_mic_name in channel.mic_combo.itemText(i):
                            channel.mic_combo.setCurrentIndex(i)
                            break
                
                # Jezyk
                saved_lang = ch_settings.get("language", "pl-PL")
                if channel.lang_combo is not None:
                    for i in range(channel.lang_combo.count()):
                        if channel.lang_combo.itemData(i) == saved_lang:
                            channel.lang_combo.setCurrentIndex(i)
                            break
                channel.current_lang_code = saved_lang
                
                # Klawisz PTT
                saved_ptt = ch_settings.get("ptt_key", "")
                if saved_ptt:
                    channel.ptt_activation_key = saved_ptt
                    if channel.ptt_button is not None:
                        channel.ptt_button.setText(saved_ptt.upper())
            
            self.update_status("Wczytano zapisane ustawienia")
        except Exception as e:
            log_message(f"Blad wczytywania ustawien: {e}")
    
    def _update_custom_rules(self):
        """Aktualizuje globalna zmienna custom rules."""
        global custom_rules_text
        custom_rules_text = self.rules_text.toPlainText()
    
    def update_status(self, message):
        """Aktualizuje status (thread-safe)."""
        self.signal_bridge.status_changed.emit(message)
        log_message(f"STATUS: {message}")
    
    def _handle_signal(self, message):
        """Obsluguje sygnaly z innych watkow."""
        if message == "SHOW_WINDOW":
            self.show_from_tray()
        elif message == "QUIT_APP":
            self.quit_application()
        else:
            self.status_label.setText(message)
    
    def set_ready_status(self, delay_ms=1500):
        """Ustawia status gotowosci po opoznieniu (thread-safe)."""
        # Wyslij sygnal do glownego watku - QTimer tam obsluzony
        self.signal_bridge.ready_signal.emit()
    
    def _set_ready(self):
        """Slot ustawiajacy status gotowosci."""
        self._update_ptt_instruction_text()
        self.status_label.setText("Gotowy.")
    
    def _update_ptt_instruction_text(self):
        """Aktualizuje tekst instrukcji PTT."""
        ch1 = self.ptt_channels["1"]
        ch2 = self.ptt_channels["2"]
        text = (f"Kanal 1 ('{ch1.ptt_activation_key.upper()}'): Nagrywaj | "
                f"Kanal 2 ('{ch2.ptt_activation_key.upper()}'): Nagrywaj\n"
                f"Pusc by przetworzyc. ESC by schowac do zasobnika.")
        self.ptt_instruction_label.setText(text)
    
    def initialize_audio(self):
        """Inicjalizuje PyAudio i wykrywa mikrofony."""
        global pyaudio_instance
        log_message("Inicjalizacja PyAudio...")
        try:
            pyaudio_instance = pyaudio.PyAudio()
            num_devices = pyaudio_instance.get_device_count()
            for i in range(num_devices):
                dev_info = pyaudio_instance.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels') > 0:
                    self.all_input_mics_details.append({
                        'index': i, 
                        'name': dev_info.get('name', f"Urzadzenie {i}"),
                        'sample_rate': int(dev_info.get('defaultSampleRate', 16000)),
                        'channels': int(dev_info.get('maxInputChannels', 1)),
                        'sample_width': pyaudio_instance.get_sample_size(AUDIO_FORMAT)
                    })
            log_message(f"Znaleziono {len(self.all_input_mics_details)} urzadzen wejsciowych.")
            return True
        except Exception as e:
            log_message(f"KRYTYCZNY BLAD: Nie mozna zainicjalizowac PyAudio: {e}")
            QMessageBox.critical(self, "Blad PyAudio", 
                                f"Nie mozna zainicjalizowac PyAudio: {e}\nProgram nie moze dzialac.")
            return False
    
    def populate_mic_comboboxes(self):
        """Wypelnia combobox'y mikrofonow."""
        log_message(f"populate_mic_comboboxes: {len(self.all_input_mics_details)} mikrofonow")
        available_mics = [f"{mic['name']} (Indeks: {mic['index']})" for mic in self.all_input_mics_details]
        used_indices = set()

        for ch_id, channel in self.ptt_channels.items():
            log_message(f"CH {ch_id}: mic_combo = {channel.mic_combo}")
            if channel.mic_combo is None: 
                log_message(f"CH {ch_id}: BRAK mic_combo!")
                continue

            channel.mic_combo.clear()
            channel.mic_combo.addItems(available_mics)
            log_message(f"CH {ch_id}: Dodano {len(available_mics)} mikrofonow do combobox")

            selected_mic = None
            if channel.target_mic_name_start:
                log_message(f"CH {ch_id}: Szukam mikrofonu zaczynajacego sie od: {channel.target_mic_name_start}")
                for mic in self.all_input_mics_details:
                    if mic['name'].lower().startswith(channel.target_mic_name_start) and mic['index'] not in used_indices:
                        selected_mic = mic
                        log_message(f"CH {ch_id}: Znaleziono preferowany: {mic['name']}")
                        break

            if not selected_mic:
                for mic in self.all_input_mics_details:
                    if mic['index'] not in used_indices:
                        selected_mic = mic
                        log_message(f"CH {ch_id}: Uzyje pierwszego wolnego: {mic['name']}")
                        break

            if not selected_mic and self.all_input_mics_details:
                selected_mic = self.all_input_mics_details[0]
                log_message(f"CH {ch_id}: Fallback do pierwszego: {selected_mic['name']}")

            if selected_mic:
                channel.mic_details = selected_mic.copy()  # Pelna kopia
                used_indices.add(selected_mic['index'])
                idx = self.all_input_mics_details.index(selected_mic)
                channel.mic_combo.setCurrentIndex(idx)
                log_message(f"CH {ch_id}: Mikrofon ustawiony: {selected_mic['name']} (index={selected_mic['index']})")
            else:
                log_message(f"CH {ch_id}: BLAD - nie znaleziono mikrofonu!")
    
    def keyboard_listener_thread_func(self):
        """Watek nasluchujacy klawiatury."""
        def key_event_handler(event: keyboard.KeyboardEvent):
            if stop_program_event.is_set(): 
                return

            if self.channel_being_configured and event.event_type == keyboard.KEY_DOWN:
                channel_to_configure = self.channel_being_configured
                self.channel_being_configured = None
                if event.name != 'esc':
                    channel_to_configure.ptt_activation_key = event.name
                    channel_to_configure.ptt_key_is_keypad = hasattr(event, 'is_keypad') and event.is_keypad
                    if channel_to_configure.ptt_button:
                        channel_to_configure.ptt_button.setText(event.name.upper())
                    self._update_ptt_instruction_text()
                else:
                    self.update_status(f"Anulowano zmiane klawisza dla kanalu {channel_to_configure.id}.")
                return

            for channel in self.ptt_channels.values():
                is_target_key = (event.name == channel.ptt_activation_key)
                if is_target_key:
                    if event.event_type == keyboard.KEY_DOWN:
                        # Ignoruj key repeat - tylko pierwszy KEY_DOWN
                        if not channel.ptt_key_pressed:
                            channel.start_recording()
                    elif event.event_type == keyboard.KEY_UP:
                        channel.stop_recording_and_process()
                    return

            if event.name == 'esc' and event.event_type == keyboard.KEY_DOWN:
                self.signal_bridge.status_changed.emit("HIDE_WINDOW")

        keyboard.hook(key_event_handler)
        stop_program_event.wait()
        keyboard.unhook_all()
        log_message("Listener klawiatury zatrzymany.")
    
    def hide_to_tray(self):
        """Chowa okno do zasobnika."""
        self.hide()
        log_message("Okno schowane do zasobnika.")
    
    def show_from_tray(self):
        """Pokazuje okno z zasobnika."""
        self.show()
        self.activateWindow()
        self.raise_()
        log_message("Okno przywrocone z zasobnika.")
    
    def closeEvent(self, event):
        """Obsluga zamkniecia okna - chowa do zasobnika."""
        event.ignore()
        self.hide_to_tray()
    
    def quit_application(self):
        """Zamyka aplikacje."""
        global pyaudio_instance, tray_icon
        log_message("Rozpoczeto zamykanie aplikacji.")
        stop_program_event.set()

        if tray_icon:
            tray_icon.stop()
            tray_icon = None

        if pyaudio_instance:
            pyaudio_instance.terminate()
            pyaudio_instance = None

        for ch in self.ptt_channels.values():
            ch.is_recording_active = False
            if ch.pyaudio_stream:
                try:
                    ch.pyaudio_stream.close()
                except:
                    pass

        QApplication.quit()
        log_message("Aplikacja zakonczona.")
    
    def setup_tray_icon(self):
        """Konfiguruje ikone w zasobniku systemowym."""
        global tray_icon
        
        def run_tray():
            try:
                image = Image.open(resource_path("_internal/tray_icon.png"))
            except Exception:
                log_message("Nie mozna zaladowac ikony zasobnika, uzycie zastepczej.")
                image = Image.new('RGB', (64, 64), 'black')

            menu = (
                pystray.MenuItem('Pokaz', lambda: self.signal_bridge.status_changed.emit("SHOW_WINDOW")),
                pystray.MenuItem('Wyjdz', lambda: self.signal_bridge.status_changed.emit("QUIT_APP"))
            )

            global tray_icon
            tray_icon = pystray.Icon("YapperTray", image, "Yapper", menu)
            log_message("Uruchamianie ikony w zasobniku systemowym.")
            tray_icon.run()
            log_message("Ikona zasobnika zatrzymana.")
        
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()
    
    def run(self):
        """Uruchamia aplikacje."""
        if os.path.exists(LOG_FILE_NAME):
            try:
                os.remove(LOG_FILE_NAME)
            except Exception:
                pass
        
        log_message("Uruchamianie Yapper v4 (PyQt6)")
        
        # Inicjalizacja Vertex AI
        if USE_GEMINI:
            log_message("Inicjalizacja Vertex AI (Gemini)...")
            if setup_vertex_ai():
                log_message("Vertex AI gotowy do uzycia.")
                self.gemini_radio.setEnabled(True)
                self.gemini_radio.setChecked(True)  # Przelacz na Gemini
                self.selected_model = "gemini"
            else:
                log_message("Vertex AI niedostepny - uzyje Google Speech API jako fallback.")
                self.gemini_radio.setEnabled(False)
        
        # Aktualizuj status modeli
        status_parts = []
        status_parts.append(f"Gemini: {'OK' if VERTEX_AVAILABLE else 'niedostepny'}")
        status_parts.append("Google: OK")
        status_parts.append(f"Vosk: {'OK' if VOSK_AVAILABLE else 'niedostepny'}")
        self.model_status_label.setText(" | ".join(status_parts))
        
        # Inicjalizacja audio
        if self.initialize_audio():
            self.populate_mic_comboboxes()
            self._load_settings()  # Wczytaj zapisane ustawienia
            self._update_ptt_instruction_text()
            self.status_label.setText("Gotowy.")
        else:
            self.status_label.setText("Blad inicjalizacji audio. Zamykanie...")
            QTimer.singleShot(3000, self.quit_application)
            return
        
        # Uruchom listener klawiatury
        kbd_thread = threading.Thread(target=self.keyboard_listener_thread_func, daemon=True)
        kbd_thread.start()
        
        # Uruchom ikone zasobnika
        self.setup_tray_icon()
        
        # Pokaz okno
        self.show()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Nowoczesny ciemny motyw z akcentami
    app.setStyleSheet("""
        /* Glowne tlo */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a1a2e, stop:1 #16213e);
        }
        QWidget {
            background-color: transparent;
            color: #e8e8e8;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        
        /* GroupBox - karty z cieniem */
        QGroupBox {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #252542, stop:1 #1f1f38);
            font-weight: bold;
            font-size: 13px;
            border: 1px solid #3a3a5c;
            border-radius: 8px;
            margin-top: 10px;
            padding: 6px 8px 6px 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #a8b4ff;
        }
        
        /* ComboBox */
        QComboBox {
            background: #3a3a5c;
            border: 1px solid #4a4a6c;
            border-radius: 4px;
            padding: 4px 8px;
            padding-right: 20px;
            color: #ffffff;
            min-height: 14px;
        }
        QComboBox:hover {
            border: 1px solid #6c6cff;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 18px;
            border: none;
            background: transparent;
        }
        QComboBox::down-arrow {
            width: 0;
            height: 0;
            border: none;
            background: transparent;
            image: none;
        }
        QComboBox QAbstractItemView {
            background: #2d2d4a;
            border: 1px solid #4a4a6c;
            border-radius: 4px;
            selection-background-color: #4a4a7c;
            outline: 0;
            padding: 0px;
            margin: 0px;
        }
        QComboBox QAbstractItemView::item {
            padding: 4px 8px;
            color: #e0e0e0;
        }
        QComboBox QAbstractItemView::item:selected {
            background: #4a4a7c;
        }
        /* Całkowite ukrycie scrollbara w ComboBox */
        QComboBox QAbstractItemView QScrollBar:vertical {
            width: 0px;
            height: 0px;
            min-width: 0px;
            max-width: 0px;
            border: none;
            background: transparent;
        }
        QComboBox QAbstractItemView QScrollBar::handle:vertical,
        QComboBox QAbstractItemView QScrollBar::add-line:vertical,
        QComboBox QAbstractItemView QScrollBar::sub-line:vertical,
        QComboBox QAbstractItemView QScrollBar::up-arrow:vertical,
        QComboBox QAbstractItemView QScrollBar::down-arrow:vertical,
        QComboBox QAbstractItemView QScrollBar::add-page:vertical,
        QComboBox QAbstractItemView QScrollBar::sub-page:vertical {
            width: 0px;
            height: 0px;
            min-width: 0px;
            max-width: 0px;
            border: none;
            background: transparent;
        }
        QComboBox QAbstractScrollArea::corner {
            border: none;
            background: transparent;
        }
        
        /* TextEdit - obszar tekstowy */
        QTextEdit {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2a2a45, stop:1 #232340);
            border: 1px solid #3a3a5c;
            border-radius: 8px;
            padding: 8px;
            color: #e0e0e0;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            selection-background-color: #5050a0;
        }
        QTextEdit:focus {
            border: 2px solid #6c6cff;
        }
        
        /* Przyciski - nowoczesne z gradientem */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5050a0, stop:1 #404080);
            border: none;
            border-radius: 6px;
            padding: 6px 12px;
            color: #ffffff;
            font-weight: 600;
            min-height: 14px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #6060b0, stop:1 #5050a0);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #404080, stop:1 #353570);
        }
        QPushButton:disabled {
            background: #3a3a4a;
            color: #666666;
        }
        
        /* Radio buttony - nowoczesne */
        QRadioButton {
            spacing: 10px;
            color: #e0e0e0;
            font-size: 13px;
        }
        QRadioButton::indicator {
            width: 20px;
            height: 20px;
            border-radius: 10px;
            border: 2px solid #5a5a8a;
            background: #2a2a45;
        }
        QRadioButton::indicator:hover {
            border: 2px solid #7c7cff;
        }
        QRadioButton::indicator:checked {
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.4,
                fx:0.5, fy:0.5, stop:0 #8080ff, stop:1 #6060c0);
            border: 2px solid #8080ff;
        }
        QRadioButton:disabled {
            color: #555555;
        }
        QRadioButton::indicator:disabled {
            border: 2px solid #404050;
            background: #2a2a35;
        }
        
        /* Label */
        QLabel {
            color: #d0d0d0;
            background: transparent;
        }
        
        /* ScrollBar - minimalistyczny */
        QScrollBar:vertical {
            background: #1a1a2e;
            width: 10px;
            border-radius: 5px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #4a4a6c;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #5a5a8c;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background: #1a1a2e;
            height: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #4a4a6c;
            border-radius: 5px;
            min-width: 30px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #5a5a8c;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
        
        /* Tooltip */
        QToolTip {
            background-color: #2d2d4a;
            color: #ffffff;
            border: 1px solid #5050a0;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
        }
    """)
    
    window = SpeechToClipboardApp()
    window.run()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
