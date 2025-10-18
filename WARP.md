# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Yapper** (Speech-to-Clipboard) is a Polish desktop application built with Python/Tkinter that provides dual-channel Push-to-Talk speech recognition. It captures audio from two separate microphones, processes speech through Google Web Speech API, and automatically copies recognized text to clipboard. The app runs in system tray with global hotkeys.

## Commands

### Running the Application
```bash
python Speech-to-Clipboard.py
```

### Building Executable
```bash
pyinstaller --name "SpeechToClipboard" --onefile --windowed --icon="_internal/wafflin.ico" --add-data "_internal;_internal" Speech-to-Clipboard.py
```

### Installing Dependencies
```bash
pip install SpeechRecognition pyperclip pyaudio keyboard pillow pystray numpy
```

**Note**: On Windows, PyAudio may require pre-built wheels:
```bash
pip install pipwin
pipwin install pyaudio
```

## Architecture

### Core Components

**PttChannel Class**: Manages individual Push-to-Talk channels
- Handles audio device configuration (mono/stereo fallback)
- Controls recording lifecycle with PyAudio streams  
- Processes audio data with numpy for mono conversion
- Integrates with Google Speech Recognition API
- Manages file saving and playback of recordings

**SpeechToClipboardApp Class**: Main application controller
- Tkinter GUI with dual-channel configuration interface
- Global keyboard hook management using `keyboard` library
- System tray integration with `pystray` 
- Audio device enumeration and automatic assignment
- Logging system with detailed debug information

### Audio Processing Flow
1. PyAudio stream opens with fallback (1 channel preferred, then 2, then device default)
2. Audio frames collected during PTT key press
3. Multi-channel audio converted to mono using numpy array slicing
4. Raw audio wrapped in SpeechRecognition AudioData object
5. Google Web Speech API call with language parameter
6. Result copied to clipboard via pyperclip

### Configuration System
- Channel-specific microphone, language, and PTT key assignment
- Automatic device matching by name prefixes ("Voicemeeter Out B1", "CABLE Output")
- Constants at file top for audio parameters and behavior flags
- Asset management through `resource_path()` function for PyInstaller compatibility

## Key Files

- `Speech-to-Clipboard.py` - Main application (single file architecture)
- `speech_to_clipboard_log.txt` - Runtime logging with Polish text
- `last_recording_ch*.wav` - Optional saved recordings per channel
- `_internal/` - Assets directory (icons for window and system tray)
- `save/build-comenda.txt` - PyInstaller build command reference

## Development Notes

### Language Support
- Primary interface in Polish with Polish log messages
- Speech recognition supports pl-PL and en-US per channel
- Google Web Speech API requires internet connection

### Platform Considerations  
- Primary target: Windows (keyboard library, PyAudio, pystray)
- Linux/macOS possible but may require root/accessibility permissions
- PyInstaller builds use --windowed flag for GUI deployment

### Audio Device Logic
- Prioritizes mono (1-channel) recording for stability
- Multi-channel recordings automatically converted to mono
- Device selection with fallback logic for missing preferred devices
- Detailed logging of audio stream parameters and errors

### Threading Model
- GUI thread (Tkinter mainloop)
- Keyboard listener thread (global hotkey monitoring)  
- System tray thread (pystray icon)
- Recording threads (per-channel, spawned on demand)