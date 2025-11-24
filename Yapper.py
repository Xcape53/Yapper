import sys
import os
import time
import wave
import json
import threading
import subprocess
import numpy as np
import pyaudio
import keyboard
import pyperclip
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
vertex_model = None
user_credentials = None

def load_vertex_config():
    """Wczytuje konfiguracje Vertex AI z settings.json."""
    config = {
        "project_id": "",
        "location": "us-central1",
        "client_secret_file": ""
    }
    
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
                vertex_config = settings.get("vertex_ai", {})
                config["project_id"] = vertex_config.get("project_id", "")
                config["location"] = vertex_config.get("location", "us-central1")
                config["client_secret_file"] = vertex_config.get("client_secret_file", "")
        except Exception:
            pass
    
    # Fallback do zmiennych srodowiskowych
    if not config["project_id"]:
        config["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not config["client_secret_file"]:
        config["client_secret_file"] = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "")
    
    return config

def setup_vertex_ai():
    """Konfiguruje Vertex AI z OAuth2 credentials."""
    global VERTEX_AVAILABLE, vertex_model, user_credentials
    
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
        
        if not PROJECT_ID or not CLIENT_SECRET_FILE:
            print("UWAGA: Brak konfiguracji Vertex AI w settings.json (project_id, client_secret_file)")
            return False
        
        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        TOKEN_FILE = "vertex_token.pickle"
        
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                user_credentials = pickle.load(token)
        
        if not user_credentials or not user_credentials.valid:
            if user_credentials and user_credentials.expired and user_credentials.refresh_token:
                user_credentials.refresh(Request())
            else:
                client_secret_path = CLIENT_SECRET_FILE
                if not os.path.exists(client_secret_path):
                    client_secret_path = resource_path(CLIENT_SECRET_FILE)
                
                if not os.path.exists(client_secret_path):
                    print(f"UWAGA: Brak pliku {CLIENT_SECRET_FILE}")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                user_credentials = flow.run_local_server(port=0)
            
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(user_credentials, token)
        
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=user_credentials)
        vertex_model = GenerativeModel("gemini-2.0-flash-001")
        
        VERTEX_AVAILABLE = True
        print(f"Vertex AI skonfigurowany (projekt: {PROJECT_ID})")
        return True
        
    except Exception as e:
        print(f"UWAGA: Nie mozna skonfigurowac Vertex AI: {e}")
        VERTEX_AVAILABLE = False
        return False

# --- Vosk Import (opcjonalny) ---
vosk_model = None
VOSK_AVAILABLE = False
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    print("UWAGA: Vosk nie jest zainstalowany. Tryb offline niedostepny.")

# --- Konfiguracja ---
LOG_FILE_NAME = "yapper_log.txt"
SETTINGS_FILE = "settings.json"  # Plik z zapisanymi ustawieniami
SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
DELAY_AFTER_KEY_RELEASE_MS = 500
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16

USE_ENSEMBLE_FOR_POLISH = True
VOSK_MODEL_PATH = None

USE_GEMINI = True
GEMINI_MODEL = "gemini-2.0-flash-001"
BENCHMARK_MODE = False

DEFAULT_CUSTOM_RULES = """# Przykladowe reguly (usun # zeby aktywowac):
# - Usun wszystkie "yyy", "eee", "hmm"
# - Zamien "nowa linia" na znak nowej linii
# - Zamien "kropka" na "."
# - Zamien "przecinek" na ","
# - Formatuj jako lista punktowana"""

custom_rules_text = ""

# --- Zasoby Globalne ---
pyaudio_instance = None
recognizer = sr.Recognizer()
stop_program_event = threading.Event()
tray_icon = None


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
        return {'text': '', 'confidence': 0, 'system': 'Vosk', 'error': 'Vosk niedostepny'}
    
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


def transcribe_with_gemini(audio_bytes, sample_rate, lang_code="pl-PL"):
    """Transkrypcja za pomoca Gemini przez Vertex AI."""
    global vertex_model, VERTEX_AVAILABLE, custom_rules_text
    
    if not VERTEX_AVAILABLE or vertex_model is None:
        return {'text': '', 'confidence': 0, 'system': 'Gemini', 'error': 'Vertex AI niedostepny'}
    
    try:
        import io
        from vertexai.generative_models import Part
        start_time = time.time()
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        wav_buffer.seek(0)
        wav_data = wav_buffer.read()
        
        lang_name = "polskim" if lang_code.startswith("pl") else "angielskim"
        
        prompt = f"""Jestes precyzyjnym systemem transkrypcji mowy. 
Przetranśkrybuj dokladnie to co slyszysz w nagraniu audio. 
Jezyk: {lang_name}.
WAZNE: Zwroc TYLKO tekst transkrypcji, bez zadnych dodatkowych komentarzy, wyjasnien ani formatowania.
Jesli nie slyszysz zadnej mowy, zwroc pusty tekst."""

        active_rules = [line.strip() for line in custom_rules_text.split('\n') 
                       if line.strip() and not line.strip().startswith('#')]
        
        if active_rules:
            rules_text = '\n'.join(f"- {rule}" for rule in active_rules)
            prompt += f"""

DODATKOWE REGULY PRZETWARZANIA (zastosuj je do transkrypcji):
{rules_text}"""

        audio_part = Part.from_data(wav_data, mime_type="audio/wav")
        response = vertex_model.generate_content([audio_part, prompt])
        
        elapsed_time = time.time() - start_time
        
        text = response.text.strip() if response.text else ''
        
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        return {
            'text': text,
            'confidence': 0.95,
            'system': 'Gemini',
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
