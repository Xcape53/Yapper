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

CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
TARGET_MIC_NAME_START = "voicemeeter out b1 (vb-audio vo".lower()
DEFAULT_LANG = "pl-PL"
LOG_FILE_NAME = "speech_to_clipboard_log.txt"

SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
RECORDING_FILENAME = "last_recording.wav"

recognizer = sr.Recognizer()
pyaudio_instance = None
pyaudio_stream = None
audio_frames = []
is_recording_active = False
ptt_key_pressed = False
stop_program_event = threading.Event()

current_lang_code = DEFAULT_LANG
selected_mic_details = {
    "index": None, "name": "Nie wybrano", "sample_rate": 16000,
    "channels": 1, "sample_width": 2
}
current_recording_actual_channels = 1

root = None
lang_var = None
status_var = None
mic_combobox = None
ptt_activation_key = "5"
ptt_key_is_keypad = True
DELAY_AFTER_KEY_RELEASE_MS = 500
ptt_key_setting_active = False
ptt_key_var = None
ptt_instruction_var = None

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


def update_status(message):
    if status_var and root and root.winfo_exists():
        try:
            status_var.set(message)
        except tk.TclError:
            pass
    log_message(f"STATUS_LOG: {message}")


def save_and_play_audio(audio_data_bytes, filename=RECORDING_FILENAME, channels_override=None):
    if not SAVE_LAST_RECORDING and not PLAY_LAST_RECORDING: return
    actual_filename = filename
    if channels_override == 1 and selected_mic_details.get('channels', 0) > 1:
        actual_filename = filename.replace(".wav", "_mono.wav")
    elif channels_override is not None and channels_override != selected_mic_details.get('channels', 0):
        actual_filename = filename.replace(".wav", f"_{channels_override}ch.wav")
    filepath = os.path.join(os.getcwd(), actual_filename)
    log_message(f"Zapisywanie audio do: {filepath}")
    actual_channels_to_save = channels_override if channels_override is not None else current_recording_actual_channels
    try:
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(actual_channels_to_save)
            wf.setsampwidth(selected_mic_details['sample_width'])
            wf.setframerate(selected_mic_details['sample_rate'])
            wf.writeframes(audio_data_bytes)
        log_message(
            f"Audio zapisane: {filepath}, Rozmiar: {len(audio_data_bytes)} bajtów, Kanały: {actual_channels_to_save}, Rate: {selected_mic_details['sample_rate']}, Width: {selected_mic_details['sample_width']}")
        if PLAY_LAST_RECORDING:
            log_message(f"Próba odtworzenia: {filepath}")
            try:
                if os.name == 'nt':
                    os.startfile(filepath)
                elif os.name == 'posix':
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, filepath])
                log_message(f"Polecenie odtwarzania wysłane dla: {filepath}")
            except Exception as e:
                log_message(f"Błąd podczas próby odtworzenia pliku audio: {e}")
                if root and root.winfo_exists(): messagebox.showwarning("Odtwarzanie Audio",
                                                                        f"Nie udało się automatycznie odtworzyć pliku {actual_filename}.\nSpróbuj ręcznie.\nBłąd: {e}")
    except Exception as e:
        log_message(f"Błąd podczas zapisywania pliku audio: {e}")
        if root and root.winfo_exists(): messagebox.showerror("Błąd Zapisu Audio",
                                                              f"Nie udało się zapisać pliku audio {actual_filename}.\nBłąd: {e}")


def gui_initialize_microphone():
    global pyaudio_instance, selected_mic_details, mic_combobox
    log_message("Inicjalizacja PyAudio...")
    if not pyaudio_instance:
        try:
            pyaudio_instance = pyaudio.PyAudio()
            log_message("PyAudio zainicjalizowane pomyślnie.")
        except Exception as e:
            log_message(f"KRYTYCZNY BŁĄD: Nie można zainicjalizować PyAudio: {e}")
            if root and root.winfo_exists(): messagebox.showerror("Błąd PyAudio",
                                                                  f"Nie można zainicjalizować PyAudio: {e}\nProgram nie może działać.")
            if root and root.winfo_exists(): root.quit()
            return False
    selected_mic_details["sample_width"] = pyaudio_instance.get_sample_size(AUDIO_FORMAT)
    update_status("Wyszukiwanie mikrofonów...")
    available_mics_for_gui, mic_map_for_gui, target_mics_found_details, all_input_mics_details = [], {}, [], []
    try:
        num_devices = pyaudio_instance.get_device_count()
        for i in range(num_devices):
            dev_info = pyaudio_instance.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:
                mic_name = dev_info.get('name', f"Urządzenie {i}")
                mic_detail = {'index': i, 'name': mic_name,
                              'sample_rate': 16000,
                              'channels': int(dev_info.get('maxInputChannels', 1))}
                if mic_detail['channels'] == 0: mic_detail['channels'] = 1
                all_input_mics_details.append(mic_detail)
                if mic_name.lower().startswith(TARGET_MIC_NAME_START): target_mics_found_details.append(mic_detail)
        chosen_mic_detail = None
        if len(target_mics_found_details) == 1:
            chosen_mic_detail = target_mics_found_details[0]
        elif len(target_mics_found_details) > 1:
            chosen_mic_detail = target_mics_found_details[0]
        elif all_input_mics_details:
            chosen_mic_detail = all_input_mics_details[0]
        if chosen_mic_detail:
            selected_mic_details.update(chosen_mic_detail)
        else:
            log_message("BŁĄD: Nie znaleziono żadnych mikrofonów wejściowych!")
            if root and root.winfo_exists(): messagebox.showerror("Błąd mikrofonu",
                                                                  "Nie znaleziono żadnych mikrofonów wejściowych.")
            return False
        for mic_d in all_input_mics_details:
            gui_name = f"{mic_d['name']} (Indeks: {mic_d['index']})"
            available_mics_for_gui.append(gui_name)
            mic_map_for_gui[gui_name] = mic_d
        if mic_combobox:
            mic_combobox['values'] = available_mics_for_gui
            current_selection_gui_name = f"{selected_mic_details['name']} (Indeks: {selected_mic_details['index']})"
            if current_selection_gui_name in available_mics_for_gui:
                mic_combobox.set(current_selection_gui_name)
            elif available_mics_for_gui:
                mic_combobox.set(available_mics_for_gui[0])
                on_mic_select(None)
        update_status(
            f"Używany mikrofon: {selected_mic_details['name']} (R: {selected_mic_details['sample_rate']}, C: {selected_mic_details['channels']})")
        return True
    except Exception as e:
        log_message(f"KRYTYCZNY BŁĄD podczas inicjalizacji mikrofonu: {e}")
        if root and root.winfo_exists(): messagebox.showerror("Błąd mikrofonu", f"Wystąpił błąd: {e}")
        return False


def record_audio_loop():
    global pyaudio_stream, audio_frames, is_recording_active, current_recording_actual_channels
    if selected_mic_details.get("index") is None:
        update_status("BŁĄD: Mikrofon nie jest skonfigurowany.")
        is_recording_active = False;
        return
    stream_opened_successfully = False
    target_ch_attempts = [1, 2, selected_mic_details['channels']]
    for target_ch in target_ch_attempts:
        if stream_opened_successfully: break
        try:
            pyaudio_stream = pyaudio_instance.open(format=AUDIO_FORMAT, channels=target_ch,
                                                   rate=selected_mic_details['sample_rate'], input=True,
                                                   frames_per_buffer=CHUNK_SIZE,
                                                   input_device_index=selected_mic_details['index'])
            current_recording_actual_channels = target_ch;
            stream_opened_successfully = True
            log_message(f"Strumień PyAudio otwarty z {target_ch} kanałem/kanałami.")
        except Exception as e_open:
            log_message(f"Nie udało się otworzyć strumienia z {target_ch} kanałem/kanałami: {e_open}")
            if target_ch == selected_mic_details['channels'] and not stream_opened_successfully:
                update_status(f"BŁĄD: Nie można otworzyć strumienia: {e_open}")
                is_recording_active = False;
                return
    if not stream_opened_successfully: is_recording_active = False; return
    audio_frames = []
    log_message(f"Pętla nagrywania audio AKTYWNA (kanały: {current_recording_actual_channels}).")
    while is_recording_active and not stop_program_event.is_set():
        try:
            data = pyaudio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_frames.append(data)
        except IOError:
            pass
        except Exception:
            is_recording_active = False; break
    if pyaudio_stream:
        try:
            if pyaudio_stream.is_active(): pyaudio_stream.stop_stream()
            pyaudio_stream.close()
        except Exception:
            pass
    pyaudio_stream = None


def start_recording_ptt():
    global is_recording_active, ptt_key_pressed
    if stop_program_event.is_set() or ptt_key_pressed or is_recording_active: return
    ptt_key_pressed = True
    if selected_mic_details.get("index") is None:
        update_status("BŁĄD: Wybierz mikrofon!");
        ptt_key_pressed = False;
        return
    is_recording_active = True
    update_status(">>> Nagrywanie...")
    threading.Thread(target=record_audio_loop, daemon=True).start()


def stop_recording_and_process_ptt():
    global is_recording_active, ptt_key_pressed
    if not ptt_key_pressed or stop_program_event.is_set() or not is_recording_active:
        if not is_recording_active: ptt_key_pressed = False
        return
    if root and root.winfo_exists():
        root.after(DELAY_AFTER_KEY_RELEASE_MS, _actual_stop_and_process)
    else:
        time.sleep(DELAY_AFTER_KEY_RELEASE_MS / 1000.0); _actual_stop_and_process()


def _actual_stop_and_process():
    global is_recording_active, audio_frames, ptt_key_pressed, current_lang_code, current_recording_actual_channels
    if not is_recording_active: ptt_key_pressed = False; return
    is_recording_active = False;
    ptt_key_pressed = False
    time.sleep(0.1 + (DELAY_AFTER_KEY_RELEASE_MS / 2000.0))
    update_status(">>> Przetwarzanie...")
    if not audio_frames:
        update_status("Nie nagrano dźwięku.")
        if not stop_program_event.is_set() and root and root.winfo_exists(): root.after(1000, lambda: update_status(
            f"Gotowy. Naciśnij '{ptt_activation_key.upper()}'."))
        return
    recorded_audio_data = b''.join(audio_frames);
    audio_frames = []
    if SAVE_LAST_RECORDING or PLAY_LAST_RECORDING: save_and_play_audio(recorded_audio_data,
                                                                       channels_override=current_recording_actual_channels)
    if len(recorded_audio_data) < 200:
        update_status("Nagrano zbyt mało danych.")
        if not stop_program_event.is_set() and root and root.winfo_exists(): root.after(1000, lambda: update_status(
            f"Gotowy. Naciśnij '{ptt_activation_key.upper()}'."))
        return
    try:
        audio_data_obj = sr.AudioData(recorded_audio_data, selected_mic_details['sample_rate'],
                                      selected_mic_details['sample_width'])
        text = recognizer.recognize_google(audio_data_obj, language=current_lang_code)
        update_status(f"Rozpoznano: {text}")
        if root and root.winfo_exists():
            try:
                pyperclip.copy(text)
            except pyperclip.PyperclipException as e:
                update_status(f"Błąd schowka: {e}"); messagebox.showwarning("Błąd schowka", f"Nie skopiowano: {e}")
    except sr.UnknownValueError:
        update_status("Nie udało się rozpoznać mowy.")
    except sr.RequestError as e:
        update_status(f"Błąd API Google: {e}")
    except Exception as e:
        update_status(f"Błąd przetwarzania: {e}")
    if not stop_program_event.is_set() and root and root.winfo_exists(): root.after(1500, lambda: update_status(
        f"Gotowy. Naciśnij '{ptt_activation_key.upper()}'."))


def keyboard_listener_thread_func():
    global stop_program_event, ptt_key_setting_active, ptt_activation_key, ptt_key_is_keypad, ptt_key_var

    def key_event_handler(event: keyboard.KeyboardEvent):
        global ptt_key_setting_active, ptt_activation_key, ptt_key_is_keypad, ptt_key_var, ptt_instruction_var
        if stop_program_event.is_set(): return False
        if ptt_key_setting_active and event.event_type == keyboard.KEY_DOWN:
            if event.name != 'esc':
                ptt_activation_key = event.name
                ptt_key_is_keypad = hasattr(event, 'is_keypad') and event.is_keypad
                if ptt_key_var: ptt_key_var.set(f"{ptt_activation_key.upper()}")
                if ptt_instruction_var and root and root.winfo_exists(): root.after(10, update_ptt_instruction_text)
                update_status(f"Klawisz PTT: '{ptt_activation_key.upper()}'.")
            ptt_key_setting_active = False;
            return False
        if event.name == 'esc' and event.event_type == keyboard.KEY_DOWN:
            on_close_window_to_tray();
            return False
        if event.name is None: return True
        is_target_key = (event.name == ptt_activation_key)
        if ptt_key_is_keypad: is_target_key = is_target_key and hasattr(event, 'is_keypad') and event.is_keypad
        if is_target_key:
            if event.event_type == keyboard.KEY_DOWN:
                start_recording_ptt()
            elif event.event_type == keyboard.KEY_UP:
                stop_recording_and_process_ptt()
        return True

    hooks = None
    try:
        hooks = keyboard.hook(key_event_handler)
        stop_program_event.wait()
    except Exception:
        pass
    finally:
        if hooks:
            try:
                keyboard.unhook(hooks)
            except Exception:
                pass


def on_language_change():
    global current_lang_code
    if lang_var: current_lang_code = lang_var.get(); update_status(f"Język: {current_lang_code}")


def on_mic_select(event):
    global selected_mic_details, pyaudio_instance
    if not mic_combobox: return
    selected_gui_name = mic_combobox.get()
    mic_map = {}
    pa_instance_for_map = pyaudio_instance if pyaudio_instance else pyaudio.PyAudio()
    temp_instance_created = not bool(pyaudio_instance)
    try:
        for i in range(pa_instance_for_map.get_device_count()):
            dev_info = pa_instance_for_map.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:
                mic_name_raw = dev_info.get('name', f"Urządzenie {i}")
                gui_name_iter = f"{mic_name_raw} (Indeks: {i})"
                mic_map[gui_name_iter] = {'index': i, 'name': mic_name_raw,
                                          'sample_rate': int(dev_info.get('defaultSampleRate', 16000)),
                                          'channels': int(dev_info.get('maxInputChannels', 1))}
                if mic_map[gui_name_iter]['channels'] == 0: mic_map[gui_name_iter]['channels'] = 1
        if temp_instance_created: pa_instance_for_map.terminate()
    except Exception:
        if temp_instance_created and pa_instance_for_map: pa_instance_for_map.terminate()
        return
    if selected_gui_name in mic_map:
        selected_mic_details.update(mic_map[selected_gui_name])
        if pyaudio_instance: selected_mic_details["sample_width"] = pyaudio_instance.get_sample_size(AUDIO_FORMAT)
        update_status(
            f"Mikrofon: {selected_mic_details['name']} (R: {selected_mic_details['sample_rate']}, C: {selected_mic_details['channels']})")
    else:
        update_status(f"Błąd wyboru mikrofonu {selected_gui_name}")


def update_ptt_instruction_text():
    global ptt_instruction_var, ptt_activation_key, root
    if ptt_instruction_var and root and root.winfo_exists():
        new_text = f"Naciśnij '{ptt_activation_key.upper()}' aby nagrywać.\nPuść by przetworzyć. ESC by schować do zasobnika."
        ptt_instruction_var.set(new_text)
        root.title(f"Speech-to-Clipboard")


def create_gui():
    global root, lang_var, status_var, mic_combobox, ptt_key_var, ptt_activation_key, ptt_instruction_var
    root = tk.Tk()
    try:
        icon_path = resource_path("_internal/wafflin.ico")
        root.iconbitmap(icon_path)
    except Exception:
        pass
    root.title(f"Speech-to-Clipboard")

    lang_frame = ttk.LabelFrame(root, text="Język")
    lang_frame.pack(padx=10, pady=5, fill="x")
    lang_var = tk.StringVar(value=DEFAULT_LANG)
    ttk.Radiobutton(lang_frame, text="Polski", variable=lang_var, value="pl-PL", command=on_language_change).pack(
        side="left", padx=5)
    ttk.Radiobutton(lang_frame, text="Angielski", variable=lang_var, value="en-US", command=on_language_change).pack(
        side="left", padx=5)

    mic_frame = ttk.LabelFrame(root, text="Mikrofon")
    mic_frame.pack(padx=10, pady=5, fill="x")
    mic_combobox = ttk.Combobox(mic_frame, state="readonly", width=60)
    mic_combobox.pack(pady=5);
    mic_combobox.bind("<<ComboboxSelected>>", on_mic_select)

    ptt_key_frame = ttk.LabelFrame(root, text="Push-To-Talk")
    ptt_key_frame.pack(padx=10, pady=5, fill="x")
    ttk.Label(ptt_key_frame, text="Klawisz PTT: ").pack(side="left", padx=5)
    ptt_key_var = tk.StringVar(value=f"{ptt_activation_key.upper()}")
    set_ptt_key_button = ttk.Button(ptt_key_frame, textvariable=ptt_key_var, command=activate_ptt_key_setting_mode)
    set_ptt_key_button.pack(side="left", padx=5)

    status_frame = ttk.LabelFrame(root, text="Status")
    status_frame.pack(padx=10, pady=5, fill="x", expand=True)
    status_var = tk.StringVar(value="Inicjalizacja...")
    ttk.Label(status_frame, textvariable=status_var, wraplength=480).pack(pady=5, fill="x")

    ptt_instruction_var = tk.StringVar()
    ttk.Label(root, textvariable=ptt_instruction_var, justify=tk.CENTER).pack(pady=10)
    update_ptt_instruction_text()

    root.protocol("WM_DELETE_WINDOW", on_close_window_to_tray)

    if not gui_initialize_microphone():
        update_status("Błąd inicjalizacji mikrofonu. Zamykanie.")
        if root and root.winfo_exists(): root.after(3000, lambda: quit_application(force_quit=True))
    else:
        update_status(f"Gotowy. Naciśnij '{ptt_activation_key.upper()}'.")

    setup_tray_icon()
    root.mainloop()


def activate_ptt_key_setting_mode():
    global ptt_key_setting_active
    ptt_key_setting_active = True
    update_status("Wciśnij nowy klawisz PTT (ESC by anulować)...")


def show_window_action(icon_ref=None, item_ref=None):
    global root
    log_message("Akcja: Pokaż okno z menu zasobnika.")
    if root and root.winfo_exists():
        root.after(0, root.deiconify)
        root.after(50, root.lift)
        root.after(100, root.focus_force)
        log_message("Okno powinno być widoczne.")
    else:
        log_message("Główne okno (root) nie istnieje lub zostało zniszczone. Nie można pokazać.")


def on_close_window_to_tray():
    global root, tray_icon
    log_message("Polecenie: Schowaj okno do zasobnika")
    if root and root.winfo_exists():
        root.withdraw()
        log_message("Okno schowane.")
        if not tray_icon or not tray_icon.visible:
            log_message("Ikona zasobnika nie jest aktywna, próba ponownego ustawienia.")
            setup_tray_icon_thread_if_needed()


def setup_tray_icon_thread_if_needed():
    global tray_icon

    tray_already_running = False
    if tray_icon:
        try:
            if hasattr(tray_icon,
                       '_thread') and tray_icon._thread is not None and tray_icon._thread.is_alive():
                tray_already_running = True
                log_message("Ikona zasobnika (wątek) już działa.")
                return
        except AttributeError:
            pass

    if tray_icon and not tray_already_running:
        log_message("Istnieje instancja tray_icon, ale wątek nie działa. Próba zatrzymania dla pewności.")
        try:
            tray_icon.stop()
        except Exception:
            pass
        tray_icon = None

    if not tray_icon:
        log_message("Uruchamianie wątku ikony zasobnika...")
        tray_thread = threading.Thread(target=run_tray_icon, daemon=True)
        tray_thread.start()


def run_tray_icon():
    global tray_icon
    try:
        image = Image.open(resource_path("_internal/tray_icon.png"))
    except Exception as e:
        log_message(f"Nie można załadować ikony zasobnika: {e}")
        return

    menu = (pystray.MenuItem('Pokaż', show_window_action, default=True),
            pystray.MenuItem('Wyjdź', quit_action))

    tray_icon = pystray.Icon("SpeechToClipboardTray", image, "Speech-to-Clipboard", menu)
    log_message("Ikona zasobnika utworzona, uruchamiam pystray.Icon.run()...")
    try:
        tray_icon.run()
    except Exception as e:
        log_message(f"Błąd podczas działania pystray.Icon.run(): {e}")
    finally:
        log_message("pystray.Icon.run() zakończone.")


def show_window_action(icon_ref, item_ref):
    global root
    log_message("Akcja: Pokaż okno z menu zasobnika.")
    if root:
        root.after(0, root.deiconify)
        root.after(100, lambda: root.state(
            'zoomed') if root.wm_state() == 'iconic' else root.lift())
        root.after(100, root.focus_force)
        log_message("Okno powinno być widoczne.")
    else:
        log_message("Okno (root) nie istnieje, nie można pokazać.")


def quit_action(icon_ref=None, item_ref=None):
    log_message("Akcja: Wyjdź z menu zasobnika.")
    quit_application(force_quit=True)


def quit_application(force_quit=False):
    global stop_program_event, root, pyaudio_instance, is_recording_active, pyaudio_stream, tray_icon

    user_confirmed_quit = force_quit
    if not force_quit and root and root.winfo_exists():
        user_confirmed_quit = messagebox.askokcancel("Wyjście", "Czy na pewno chcesz zakończyć?")

    if user_confirmed_quit:
        log_message("Potwierdzono zamknięcie aplikacji.")
        stop_program_event.set()
        is_recording_active = False

        if tray_icon:
            log_message("Zatrzymywanie ikony zasobnika...")
            try:
                tray_icon.stop()
            except Exception as e_tray_stop:
                log_message(f"Błąd przy zatrzymywaniu ikony zasobnika: {e_tray_stop}")
            tray_icon = None
            time.sleep(0.2)

        if pyaudio_stream and hasattr(pyaudio_stream, 'is_active') and pyaudio_stream.is_active():
            try:
                pyaudio_stream.stop_stream(); pyaudio_stream.close()
            except Exception:
                pass
        pyaudio_stream = None
        if pyaudio_instance:
            try:
                pyaudio_instance.terminate()
            except Exception:
                pass
        pyaudio_instance = None

        if root and root.winfo_exists():
            try:
                root.destroy()
            except tk.TclError:
                pass
        root = None
        log_message("Aplikacja zakończona.")
    else:
        log_message("Anulowano zamknięcie aplikacji.")


def setup_tray_icon():
    log_message("setup_tray_icon: Uruchamianie wątku dla ikony zasobnika.")
    tray_thread = threading.Thread(target=run_tray_icon, daemon=True)
    tray_thread.start()


if __name__ == "__main__":
    if os.path.exists(LOG_FILE_NAME):
        try:
            os.remove(LOG_FILE_NAME)
        except Exception:
            pass
    log_message("Uruchamianie Speech-to-Clipboard")

    kbd_thread = threading.Thread(target=keyboard_listener_thread_func,
                                  daemon=True)
    kbd_thread.start()

    create_gui()

    log_message("Pętla GUI zakończona.")
    if not stop_program_event.is_set(): stop_program_event.set()

    if kbd_thread.is_alive(): kbd_thread.join(timeout=0.5)

    log_message("Program zakończony.")