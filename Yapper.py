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


class SignalBridge(QObject):
    """Most do komunikacji miedzy watkami a GUI Qt."""
    status_changed = pyqtSignal(str)
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
