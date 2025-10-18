import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
import pyperclip
import pyaudio
import keyboard
import threading
import time
import wave
import os
import subprocess
import sys
from PIL import Image
import pystray
import numpy as np  # Dodana biblioteka do obsługi audio

# --- Konfiguracja ---
LOG_FILE_NAME = "speech_to_clipboard_log.txt"
SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
DELAY_AFTER_KEY_RELEASE_MS = 500
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16

# --- Zasoby Globalne ---
pyaudio_instance = None
recognizer = sr.Recognizer()
stop_program_event = threading.Event()
tray_icon = None


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
        print(f"BŁĄD ZAPISU LOGA: {e}")


class PttChannel:
    """Zarządza jednym kanałem Push-to-Talk (mikrofon, klawisz, stan)."""

    def __init__(self, channel_id, app, initial_ptt_key, initial_lang, target_mic_name_start=""):
        self.id = channel_id
        self.app = app
        self.ptt_activation_key = initial_ptt_key
        self.ptt_key_is_keypad = True if initial_ptt_key.isdigit() else False
        self.current_lang_code = initial_lang

        self.mic_details = {"index": None, "name": "Nie wybrano", "sample_rate": 16000, "channels": 1,
                            "sample_width": 2}
        self.target_mic_name_start = target_mic_name_start.lower()

        # Stan dynamiczny
        self.is_recording_active = False
        self.ptt_key_pressed = False
        self.pyaudio_stream = None
        self.audio_frames = []
        self.current_recording_actual_channels = 1

        # Elementy GUI
        self.lang_var = None
        self.mic_combobox = None
        self.ptt_key_var = None

    def update_status(self, message):
        self.app.update_status(f"[Kanał {self.id}] {message}")

    def start_recording(self):
        if stop_program_event.is_set() or self.ptt_key_pressed or self.is_recording_active or self.app.is_any_recording_active:
            return

        self.ptt_key_pressed = True
        self.app.is_any_recording_active = True

        if self.mic_details.get("index") is None:
            self.update_status("BŁĄD: Wybierz mikrofon!")
            self.ptt_key_pressed = False
            self.app.is_any_recording_active = False
            return

        self.is_recording_active = True
        self.update_status(">>> Nagrywanie...")
        threading.Thread(target=self.record_audio_loop, daemon=True).start()

    def stop_recording_and_process(self):
        if not self.ptt_key_pressed or stop_program_event.is_set() or not self.is_recording_active:
            if not self.is_recording_active:
                self.ptt_key_pressed = False
            return

        if self.app.root and self.app.root.winfo_exists():
            self.app.root.after(DELAY_AFTER_KEY_RELEASE_MS, self._actual_stop_and_process)
        else:
            time.sleep(DELAY_AFTER_KEY_RELEASE_MS / 1000.0)
            self._actual_stop_and_process()

    def _actual_stop_and_process(self):
        if not self.is_recording_active:
            self.ptt_key_pressed = False
            return

        self.is_recording_active = False
        self.ptt_key_pressed = False
        self.app.is_any_recording_active = False

        time.sleep(0.1 + (DELAY_AFTER_KEY_RELEASE_MS / 2000.0))

        self.update_status(">>> Przetwarzanie...")

        if not self.audio_frames:
            self.update_status("Nie nagrano dźwięku.")
            self.app.set_ready_status()
            return

        recorded_audio_data = b''.join(self.audio_frames)
        self.audio_frames = []

        # ### DODANE LOGOWANIE ###
        log_message(
            f"CH {self.id}: Zakończono nagrywanie. Rozmiar danych: {len(recorded_audio_data)} bajtów. Kanały nagrania: {self.current_recording_actual_channels}.")

        # Konwersja do mono, jeśli nagranie jest wielokanałowe
        if self.current_recording_actual_channels > 1:
            log_message(
                f"CH {self.id}: Wykryto {self.current_recording_actual_channels} kanałów. Konwertowanie do mono.")
            try:
                # Użyj numpy do łatwej i szybkiej konwersji
                audio_array = np.frombuffer(recorded_audio_data, dtype=np.int16)
                audio_mono = audio_array[::self.current_recording_actual_channels]  # Weź próbki z pierwszego kanału
                recorded_audio_data = audio_mono.tobytes()
                log_message(
                    f"CH {self.id}: Konwersja do mono zakończona. Nowy rozmiar danych: {len(recorded_audio_data)} bajtów.")
            except Exception as e:
                log_message(f"CH {self.id}: Błąd podczas konwersji do mono: {e}")
                self.update_status("Błąd konwersji audio.")
                self.app.set_ready_status()
                return

        recording_filename = f"last_recording_ch{self.id}.wav"
        if SAVE_LAST_RECORDING or PLAY_LAST_RECORDING:
            self.save_and_play_audio(recorded_audio_data, recording_filename)

        if len(recorded_audio_data) < 400:  # Zwiększony próg dla pewności
            self.update_status("Nagrano zbyt mało danych.")
            self.app.set_ready_status()
            return

        try:
            # ### DODANE LOGOWANIE ###
            log_message(
                f"CH {self.id}: Próba rozpoznania mowy z parametrami: sample_rate={self.mic_details['sample_rate']}, sample_width={self.mic_details['sample_width']}")
            audio_data_obj = sr.AudioData(recorded_audio_data, self.mic_details['sample_rate'],
                                          self.mic_details['sample_width'])
            text = recognizer.recognize_google(audio_data_obj, language=self.current_lang_code)
            self.update_status(f"Rozpoznano: {text}")
            if self.app.root and self.app.root.winfo_exists():
                pyperclip.copy(text)
        except sr.UnknownValueError:
            self.update_status("Nie udało się rozpoznać mowy.")
            # ### DODANE LOGOWANIE ###
            log_message(f"CH {self.id}: Błąd rozpoznawania - sr.UnknownValueError (API nie znalazło mowy).")
        except sr.RequestError as e:
            self.update_status(f"Błąd API Google: {e}")
            log_message(f"CH {self.id}: Błąd zapytania do API - sr.RequestError: {e}")
        except Exception as e:
            self.update_status(f"Błąd przetwarzania: {e}")
            log_message(f"CH {self.id}: Inny błąd podczas przetwarzania: {e}")

        self.app.set_ready_status()

    def record_audio_loop(self):
        global pyaudio_instance

        if self.mic_details.get("index") is None:
            self.update_status("BŁĄD: Mikrofon nie jest skonfigurowany.")
            self.is_recording_active = False
            return

        stream_opened_successfully = False
        # ### GŁÓWNA POPRAWKA ###
        # Zmieniono kolejność, aby priorytetem było otwieranie strumienia w MONO (1 kanał).
        for target_ch in [1, 2, self.mic_details['channels']]:
            if stream_opened_successfully: break
            # Unikaj ponownej próby, jeśli liczba kanałów jest taka sama
            if target_ch == self.mic_details['channels'] and target_ch in [1, 2]:
                continue

            try:
                log_message(f"CH {self.id}: Próba otwarcia strumienia z {target_ch} kanałem/ami...")
                self.pyaudio_stream = pyaudio_instance.open(
                    format=AUDIO_FORMAT, channels=target_ch,
                    rate=self.mic_details['sample_rate'], input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    input_device_index=self.mic_details['index'])
                self.current_recording_actual_channels = target_ch
                stream_opened_successfully = True
                log_message(f"CH {self.id}: SUKCES. Strumień otwarty z {target_ch} kanałem/ami.")
            except Exception as e:
                log_message(f"CH {self.id}: BŁĄD. Nie udało się otworzyć strumienia z {target_ch} kanałem/ami: {e}")

        if not stream_opened_successfully:
            self.update_status(f"BŁĄD: Nie można otworzyć strumienia audio.")
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
        log_message(f"CH {self.id}: Pętla nagrywania zakończona.")

    def save_and_play_audio(self, audio_data_bytes, filename):
        # ### ZMIANA W ZAPISIE ### Zawsze zapisujemy jako mono, bo tak przetwarzamy
        channels_to_save = 1
        filepath = os.path.join(os.getcwd(), filename)
        log_message(f"CH {self.id}: Zapisywanie audio (jako mono) do: {filepath}")
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
            log_message(f"CH {self.id}: Błąd podczas zapisu/odtwarzania pliku: {e}")


# Pozostała część kodu bez zmian

class SpeechToClipboardApp:
    def __init__(self):
        self.root = None
        self.status_var = None
        self.ptt_instruction_var = None
        self.channel_being_configured = None
        self.is_any_recording_active = False
        self.all_input_mics_details = []

        self.ptt_channels = {
            "1": PttChannel(channel_id="1", app=self, initial_ptt_key="5", initial_lang="pl-PL",
                            target_mic_name_start="Voicemeeter Out B1"),
            "2": PttChannel(channel_id="2", app=self, initial_ptt_key="6", initial_lang="pl-PL",
                            target_mic_name_start="CABLE Output")
        }

    def update_status(self, message):
        if self.status_var and self.root and self.root.winfo_exists():
            try:
                self.status_var.set(message)
            except tk.TclError:
                pass
        log_message(f"STATUS_LOG: {message}")

    def set_ready_status(self, delay_ms=1500):
        if not stop_program_event.is_set() and self.root and self.root.winfo_exists():
            self.root.after(delay_ms, self.update_ptt_instruction_text)

    def initialize_audio(self):
        global pyaudio_instance
        log_message("Inicjalizacja PyAudio...")
        try:
            pyaudio_instance = pyaudio.PyAudio()
            num_devices = pyaudio_instance.get_device_count()
            for i in range(num_devices):
                dev_info = pyaudio_instance.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels') > 0:
                    self.all_input_mics_details.append({
                        'index': i, 'name': dev_info.get('name', f"Urządzenie {i}"),
                        'sample_rate': int(dev_info.get('defaultSampleRate', 16000)),
                        'channels': int(dev_info.get('maxInputChannels', 1)),
                        'sample_width': pyaudio_instance.get_sample_size(AUDIO_FORMAT)
                    })
            log_message(f"Znaleziono {len(self.all_input_mics_details)} urządzeń wejściowych.")
            return True
        except Exception as e:
            log_message(f"KRYTYCZNY BŁĄD: Nie można zainicjalizować PyAudio: {e}")
            messagebox.showerror("Błąd PyAudio", f"Nie można zainicjalizować PyAudio: {e}\nProgram nie może działać.")
            return False

    def populate_mic_comboboxes(self):
        available_mics_for_gui = [f"{mic['name']} (Indeks: {mic['index']})" for mic in self.all_input_mics_details]

        used_indices = set()

        for ch_id, channel in self.ptt_channels.items():
            if not channel.mic_combobox: continue

            channel.mic_combobox['values'] = available_mics_for_gui

            selected_mic = None
            if channel.target_mic_name_start:
                for mic in self.all_input_mics_details:
                    if mic['name'].lower().startswith(channel.target_mic_name_start) and mic[
                        'index'] not in used_indices:
                        selected_mic = mic
                        break

            if not selected_mic:
                for mic in self.all_input_mics_details:
                    if mic['index'] not in used_indices:
                        selected_mic = mic
                        break

            if not selected_mic and self.all_input_mics_details:
                selected_mic = self.all_input_mics_details[0]

            if selected_mic:
                channel.mic_details.update(selected_mic)
                used_indices.add(selected_mic['index'])
                gui_name = f"{selected_mic['name']} (Indeks: {selected_mic['index']})"
                channel.mic_combobox.set(gui_name)
                self.update_status(f"CH {ch_id}: Używany mikrofon: {selected_mic['name']}")
            elif available_mics_for_gui:
                channel.mic_combobox.set(available_mics_for_gui[0])
                self.on_mic_select(None, ch_id)
            else:
                self.update_status(f"CH {ch_id}: BŁĄD: Brak mikrofonów!")

    def on_mic_select(self, event, channel_id):
        channel = self.ptt_channels[channel_id]
        selected_gui_name = channel.mic_combobox.get()

        for mic in self.all_input_mics_details:
            gui_name_iter = f"{mic['name']} (Indeks: {mic['index']})"
            if selected_gui_name == gui_name_iter:
                channel.mic_details.update(mic)
                self.update_status(f"CH {channel.id}: Zmieniono mikrofon na {mic['name']}")
                return

    def on_language_change(self, channel_id):
        channel = self.ptt_channels[channel_id]
        channel.current_lang_code = channel.lang_var.get()
        self.update_status(f"CH {channel.id}: Język zmieniony na {channel.current_lang_code}")

    def activate_ptt_key_setting_mode(self, channel_id):
        self.channel_being_configured = self.ptt_channels[channel_id]
        self.update_status(f"Dla kanału {channel_id} wciśnij nowy klawisz PTT (ESC by anulować)...")

    def create_channel_ui(self, parent, channel_id):
        channel = self.ptt_channels[channel_id]

        frame = ttk.LabelFrame(parent, text=f"Kanał {channel_id}")
        frame.pack(padx=10, pady=5, fill="x", expand=True)

        lang_frame = ttk.Frame(frame)
        lang_frame.pack(fill='x', padx=5, pady=2)
        channel.lang_var = tk.StringVar(value=channel.current_lang_code)
        ttk.Radiobutton(lang_frame, text="Polski", variable=channel.lang_var, value="pl-PL",
                        command=lambda: self.on_language_change(channel_id)).pack(side="left")
        ttk.Radiobutton(lang_frame, text="Angielski", variable=channel.lang_var, value="en-US",
                        command=lambda: self.on_language_change(channel_id)).pack(side="left", padx=5)

        mic_frame = ttk.Frame(frame)
        mic_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(mic_frame, text="Mikrofon:").pack(side="left")
        channel.mic_combobox = ttk.Combobox(mic_frame, state="readonly", width=50)
        channel.mic_combobox.pack(side="left", fill='x', expand=True, padx=5)
        channel.mic_combobox.bind("<<ComboboxSelected>>",
                                  lambda event, c_id=channel_id: self.on_mic_select(event, c_id))

        ptt_frame = ttk.Frame(frame)
        ptt_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(ptt_frame, text="Klawisz PTT:").pack(side="left")
        channel.ptt_key_var = tk.StringVar(value=channel.ptt_activation_key.upper())
        set_ptt_key_button = ttk.Button(ptt_frame, textvariable=channel.ptt_key_var,
                                        command=lambda c_id=channel_id: self.activate_ptt_key_setting_mode(c_id))
        set_ptt_key_button.pack(side="left", padx=5)

    def create_gui(self):
        self.root = tk.Tk()
        try:
            icon_path = resource_path("_internal/wafflin.ico")
            self.root.iconbitmap(icon_path)
        except Exception:
            pass
        self.root.title("Speech-to-Clipboard")

        for ch_id in self.ptt_channels:
            self.create_channel_ui(self.root, ch_id)

        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(padx=10, pady=5, fill="x", expand=True)
        self.status_var = tk.StringVar(value="Inicjalizacja...")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=480).pack(pady=5, fill="x")

        self.ptt_instruction_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.ptt_instruction_var, justify=tk.CENTER).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window_to_tray)

        if self.initialize_audio():
            self.populate_mic_comboboxes()
            self.update_ptt_instruction_text()
        else:
            self.update_status("Błąd inicjalizacji audio. Zamykanie...")
            self.root.after(3000, self.quit_application)

        self.setup_tray_icon()
        self.root.mainloop()

    def update_ptt_instruction_text(self):
        if self.ptt_instruction_var and self.root and self.root.winfo_exists():
            ch1 = self.ptt_channels["1"]
            ch2 = self.ptt_channels["2"]
            new_text = (f"Kanał 1 ('{ch1.ptt_activation_key.upper()}'): Nagrywaj | "
                        f"Kanał 2 ('{ch2.ptt_activation_key.upper()}'): Nagrywaj\n"
                        f"Puść by przetworzyć. ESC by schować do zasobnika.")
            self.ptt_instruction_var.set(new_text)
            self.status_var.set("Gotowy.")

    def keyboard_listener_thread_func(self):
        def key_event_handler(event: keyboard.KeyboardEvent):
            if stop_program_event.is_set(): return

            if self.channel_being_configured and event.event_type == keyboard.KEY_DOWN:
                channel_to_configure = self.channel_being_configured
                self.channel_being_configured = None
                if event.name != 'esc':
                    channel_to_configure.ptt_activation_key = event.name
                    channel_to_configure.ptt_key_is_keypad = hasattr(event, 'is_keypad') and event.is_keypad
                    if channel_to_configure.ptt_key_var:
                        channel_to_configure.ptt_key_var.set(f"{event.name.upper()}")
                    self.update_ptt_instruction_text()
                else:
                    self.update_status(f"Anulowano zmianę klawisza dla kanału {channel_to_configure.id}.")
                return

            for channel in self.ptt_channels.values():
                is_target_key = (event.name == channel.ptt_activation_key)
                if is_target_key:
                    if event.event_type == keyboard.KEY_DOWN:
                        channel.start_recording()
                    elif event.event_type == keyboard.KEY_UP:
                        channel.stop_recording_and_process()
                    return

            if event.name == 'esc' and event.event_type == keyboard.KEY_DOWN:
                self.on_close_window_to_tray()

        keyboard.hook(key_event_handler)
        stop_program_event.wait()
        keyboard.unhook_all()
        log_message("Listener klawiatury zatrzymany.")

    def show_window_action(self, icon=None, item=None):
        if self.root:
            self.root.after(0, self.root.deiconify)
            self.root.after(50, self.root.lift)
            self.root.after(100, self.root.focus_force)
            log_message("Okno przywrócone z zasobnika.")

    def on_close_window_to_tray(self):
        if self.root:
            self.root.withdraw()
            log_message("Okno schowane do zasobnika.")

    def quit_action(self, icon=None, item=None):
        self.quit_application()

    def quit_application(self):
        global pyaudio_instance, tray_icon
        log_message("Rozpoczęto zamykanie aplikacji.")
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

        if self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass  # Ignoruj błąd, jeśli okno jest już niszczone
            self.root = None

        log_message("Aplikacja zakończona.")

    def run_tray_icon(self):
        global tray_icon
        try:
            image = Image.open(resource_path("_internal/tray_icon.png"))
        except Exception:
            log_message("Nie można załadować ikony zasobnika, używanie zastępczej.")
            image = Image.new('RGB', (64, 64), 'black')

        menu = (pystray.MenuItem('Pokaż', self.show_window_action, default=True),
                pystray.MenuItem('Wyjdź', self.quit_action))

        tray_icon = pystray.Icon("SpeechToClipboardTray", image, "Speech-to-Clipboard", menu)
        log_message("Uruchamianie ikony w zasobniku systemowym.")
        tray_icon.run()
        log_message("Ikona zasobnika zatrzymana.")

    def setup_tray_icon(self):
        tray_thread = threading.Thread(target=self.run_tray_icon, daemon=True)
        tray_thread.start()

    def run(self):
        if os.path.exists(LOG_FILE_NAME):
            try:
                os.remove(LOG_FILE_NAME)
            except Exception:
                pass
        log_message("Uruchamianie Speech-to-Clipboard v3 (z debugowaniem kanałów)")

        kbd_thread = threading.Thread(target=self.keyboard_listener_thread_func, daemon=True)
        kbd_thread.start()

        self.create_gui()

        log_message("Pętla GUI zakończona. Finalizowanie...")
        if not stop_program_event.is_set():
            self.quit_application()


if __name__ == "__main__":
    app = SpeechToClipboardApp()
    app.run()