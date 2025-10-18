# Speech-to-Clipboard (PTT, 2 kanały)

Prosta aplikacja desktopowa (Tkinter) do nagrywania mowy z dwóch wybranych mikrofonów z klawiszem Push‑to‑Talk i automatycznego wklejania rozpoznanego tekstu do schowka. Działa w tle z ikoną w zasobniku systemowym. Obsługuje polski i angielski, zapis ostatniego nagrania i szczegółowe logowanie.

Uwaga: Rozpoznawanie odbywa się przez API Google Web Speech (wymaga internetu). Fragmenty audio są wysyłane do zewnętrznej usługi.

## Funkcje

- 2 niezależne kanały nagrywania (oddzielny mikrofon, klawisz PTT i język dla każdego)
- Push‑to‑Talk: przytrzymaj klawisz, aby nagrywać; puść, aby rozpoznać i skopiować do schowka
- Wybór mikrofonu z listy urządzeń wejściowych
- Przełączanie języka rozpoznawania: pl-PL lub en-US (per kanał)
- Ikona w zasobniku: Pokaż/Ukryj okno, Wyjdź
- Zapis ostatniego nagrania WAV (opcjonalnie) i automatyczne odtwarzanie (opcjonalnie)
- Konwersja wielokanałowego audio do mono (numpy) dla stabilniejszego rozpoznawania
- Rozbudowane logi działania do pliku

## Domyślne skróty

- Kanał 1: klawisz „5”
- Kanał 2: klawisz „6”
- ESC: schowaj okno do zasobnika

W GUI możesz zmienić klawisze PTT. Przytrzymaj, aby nagrywać; po puszczeniu nastąpi przetwarzanie.

## Wymagania

- Python 3.8+
- System: Windows (zalecane). Linux/macOS możliwe, ale:
  - Linux: biblioteka keyboard może wymagać uprawnień root/uinput; PyAudio i pystray zależą od środowiska graficznego.
  - macOS: uprawnienia do mikrofonu oraz dostępności (dla globalnych skrótów); pystray wymaga dodatkowych zależności (pyobjc).

## Instalacja

1) Zainstaluj zależności (najlepiej w wirtualnym środowisku):
```
pip install -r requirements.txt
```

Jeśli nie masz pliku requirements.txt, użyj:
```
pip install SpeechRecognition pyperclip pyaudio keyboard pillow pystray numpy
```

Uwaga dot. PyAudio:
- Windows: polecam gotowe koła .whl z https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio, lub:
  ```
  pip install pipwin
  pipwin install pyaudio
  ```
- Linux/macOS: może wymagać PortAudio:
  - Debian/Ubuntu: sudo apt-get install portaudio19-dev && pip install pyaudio
  - macOS: brew install portaudio && pip install pyaudio

2) Uruchom aplikację:
```
python app.py
```

## Użytkowanie

- Wybierz mikrofon dla każdego kanału w rozwijanej liście.
- Ustaw język (Polski/Angielski) dla każdego kanału.
- Kliknij przycisk z klawiszem PTT, aby ustawić własny skrót (naciśnij wybrany klawisz; ESC anuluje).
- Przytrzymaj PTT, mów do wskazanego mikrofonu, puść – tekst trafi do schowka.
- ESC chowa okno do zasobnika; zarządzaj z ikony (Pokaż/Wyjdź).

Domyślnie aplikacja próbuje automatycznie przypisać urządzenia:
- Kanał 1: nazwa mikrofonu zaczynająca się od „Voicemeeter Out B1”
- Kanał 2: nazwa mikrofonu zaczynająca się od „CABLE Output”
Jeśli ich nie ma, wybierze pierwsze dostępne różne wejścia.

## Pliki i logi

- Log działania: speech_to_clipboard_log.txt
- Ostatnie nagrania: last_recording_ch1.wav, last_recording_ch2.wav (jeśli włączone)
- Ikony: _internal/wafflin.ico (okno), _internal/tray_icon.png (zasobnik)

## Konfiguracja (stałe na początku pliku)

W pliku źródłowym możesz zmienić:
```
LOG_FILE_NAME = "speech_to_clipboard_log.txt"
SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
DELAY_AFTER_KEY_RELEASE_MS = 500
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
```

- SAVE_LAST_RECORDING: zapisuje ostatnie nagranie WAV per kanał
- PLAY_LAST_RECORDING: automatycznie odtwarza zapis po nagraniu
- DELAY_AFTER_KEY_RELEASE_MS: opóźnienie po puszczeniu klawisza przed przetwarzaniem (ms)

## Architektura (skrót)

- PttChannel: klasa obsługująca pojedynczy kanał (urządzenie audio, PTT, język, nagrywanie, zapis, rozpoznawanie)
  - Otwiera strumień PyAudio (preferencyjnie 1 kanał; fallback do 2)
  - Zbiera ramki audio; po zakończeniu scala i konwertuje do mono (numpy)
  - Tworzy sr.AudioData i wywołuje recognizer.recognize_google(language=...)
  - Kopiuje wynik do schowka
- SpeechToClipboardApp: GUI (Tkinter), lista urządzeń, obsługa skrótów (keyboard), ikona zasobnika (pystray), logika aplikacji

## Budowanie EXE (Windows, PyInstaller)

Z uwagi na resource_path i folder _internal, dodaj pliki zasobów:
```
pyinstaller -F -w app.py ^
  --name "Speech-to-Clipboard" ^
  --add-data "_internal/tray_icon.png;_internal" ^
  --add-data "_internal/wafflin.ico;_internal"
```

Na macOS/Linux użyj dwukropka zamiast średnika:
```
--add-data "_internal/tray_icon.png:_internal" --add-data "_internal/wafflin.ico:_internal"
```

## Rozwiązywanie problemów

- Brak dźwięku / błąd PyAudio:
  - Upewnij się, że mikrofon jest wybrany i dostępny (Panel sterowania / Ustawienia)
  - Zmień mikrofon w GUI i spróbuj ponownie
  - Zainstaluj poprawnie PyAudio/PortAudio

- Nic się nie wpisuje do schowka:
  - Sprawdź log: speech_to_clipboard_log.txt
  - Sprawdź uprawnienia do mikrofonu i połączenie z internetem
  - Zwiększ DELAY_AFTER_KEY_RELEASE_MS (czas na domknięcie bufora)

- Skróty nie działają:
  - Linux: uruchom z sudo lub skonfiguruj uinput
  - macOS: nadaj uprawnienia Dostępność dla terminala/aplikacji Pythona

- Błąd „Google API”:
  - Problemy sieciowe lub ograniczenia API. Spróbuj ponownie później.

## Bezpieczeństwo i prywatność

- Rozpoznawanie wykorzystuje Google Web Speech – wysyła dźwięk do usługi zewnętrznej.
- Nie wysyłaj poufnych danych głosowych, jeśli to nieakceptowalne.

## Licencja

Wstaw tu wybraną licencję (np. MIT).

## Podziękowania

- speech_recognition (Google Web Speech)
- PyAudio/PortAudio
- keyboard, pystray, Pillow, numpy, Tkinter

## Minimalny przykład uruchomienia

```
git clone <repo>
cd <repo>
pip install -r requirements.txt
python app.py
```

Gotowe. Przytrzymuj PTT, mów, puść — tekst trafi do schowka.