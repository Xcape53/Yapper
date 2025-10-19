# Speech-to-Clipboard (PTT, 2 Channels)

A simple desktop application (Tkinter) for recording speech from two selected microphones with Push-to-Talk keys and automatic copying of recognized text to clipboard. Runs in the background with a system tray icon. Supports Polish and English, saves the last recording, and provides detailed logging.

Note: Recognition is performed via Google Web Speech API (requires internet). Audio fragments are sent to an external service.

## Features

- 2 independent recording channels (separate microphone, PTT key, and language for each)
- Push-to-Talk: hold key to record; release to recognize and copy to clipboard
- Microphone selection from input device list
- Language switching: pl-PL or en-US (per channel)
- System tray icon: Show/Hide window, Exit
- Optional WAV recording save (last recording) and optional automatic playback
- Multi-channel audio conversion to mono (numpy) for more stable recognition
- Detailed operation logs to file

## Default Shortcuts

- Channel 1: "5" key
- Channel 2: "6" key
- ESC: hide window to tray

You can change PTT keys in the GUI. Hold to record; processing occurs after release.

## Requirements

- Python 3.8+
- System: Windows (recommended). Linux/macOS possible, but:
  - Linux: keyboard library may require root/uinput permissions; PyAudio and pystray depend on graphical environment.
  - macOS: microphone and accessibility permissions required (for global shortcuts); pystray requires additional dependencies (pyobjc).

## Installation

1) Install dependencies (preferably in a virtual environment):
```
pip install -r requirements.txt
```

If you don't have a requirements.txt file, use:
```
pip install SpeechRecognition pyperclip pyaudio keyboard pillow pystray numpy
```

Note about PyAudio:
- Windows: I recommend pre-built wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio, or:
  ```
  pip install pipwin
  pipwin install pyaudio
  ```
- Linux/macOS: may require PortAudio:
  - Debian/Ubuntu: sudo apt-get install portaudio19-dev && pip install pyaudio
  - macOS: brew install portaudio && pip install pyaudio

2) Run the application:
```
python Speech-to-Clipboard.py
```

## Usage

- Select a microphone for each channel from the dropdown list.
- Set the language (Polish/English) for each channel.
- Click the PTT key button to set a custom shortcut (press the chosen key; ESC cancels).
- Hold PTT, speak into the indicated microphone, release – text goes to clipboard.
- ESC hides the window to tray; manage from icon (Show/Exit).

By default, the application tries to automatically assign devices:
- Channel 1: microphone name starting with "Voicemeeter Out B1"
- Channel 2: microphone name starting with "CABLE Output"
If not found, it will select the first available different inputs.

## Files and Logs

- Operation log: speech_to_clipboard_log.txt
- Last recordings: last_recording_ch1.wav, last_recording_ch2.wav (if enabled)
- Icons: _internal/wafflin.ico (window), _internal/tray_icon.png (tray)

## Configuration (constants at the beginning of file)

In the source file you can change:
```
LOG_FILE_NAME = "speech_to_clipboard_log.txt"
SAVE_LAST_RECORDING = True
PLAY_LAST_RECORDING = False
DELAY_AFTER_KEY_RELEASE_MS = 500
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
```

- SAVE_LAST_RECORDING: saves last WAV recording per channel
- PLAY_LAST_RECORDING: automatically plays recording after capture
- DELAY_AFTER_KEY_RELEASE_MS: delay after key release before processing (ms)

## Architecture (Overview)

- PttChannel: class handling a single channel (audio device, PTT, language, recording, saving, recognition)
  - Opens PyAudio stream (preferably 1 channel; fallback to 2)
  - Collects audio frames; after completion, merges and converts to mono (numpy)
  - Creates sr.AudioData and calls recognizer.recognize_google(language=...)
  - Copies result to clipboard
- SpeechToClipboardApp: GUI (Tkinter), device list, shortcut handling (keyboard), tray icon (pystray), application logic

## Building EXE (Windows, PyInstaller)

Due to resource_path and _internal folder, add resource files:
```
pyinstaller -F -w Speech-to-Clipboard.py ^
  --name "Speech-to-Clipboard" ^
  --add-data "_internal/tray_icon.png;_internal" ^
  --add-data "_internal/wafflin.ico;_internal"
```

On macOS/Linux use colon instead of semicolon:
```
--add-data "_internal/tray_icon.png:_internal" --add-data "_internal/wafflin.ico:_internal"
```

## Troubleshooting

- No sound / PyAudio error:
  - Make sure the microphone is selected and available (Control Panel / Settings)
  - Change microphone in GUI and try again
  - Install PyAudio/PortAudio correctly

- Nothing is copied to clipboard:
  - Check log: speech_to_clipboard_log.txt
  - Check microphone permissions and internet connection
  - Increase DELAY_AFTER_KEY_RELEASE_MS (time to close buffer)

- Shortcuts not working:
  - Linux: run with sudo or configure uinput

- "Google API" error:
  - Network issues or API limitations. Try again later.

## Security and Privacy

- Recognition uses Google Web Speech – sends audio to an external service.

## Acknowledgments

- speech_recognition (Google Web Speech)
- PyAudio/PortAudio
- keyboard, pystray, Pillow, numpy, Tkinter

## Minimal Quick Start

```
git clone <repo>
cd <repo>
pip install SpeechRecognition pyperclip pyaudio keyboard pillow pystray numpy
python Speech-to-Clipboard.py
```

Done. Hold PTT, speak, release - text goes to clipboard.
