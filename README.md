# Yapper

Yapper is a Windows speech-to-text application with two independent push-to-talk channels. Each channel can use its own shortcut, language, and recognition engine, making the app suitable for bilingual dictation and workflows that need separate input modes.

## Features

- Two independently configured push-to-talk channels
- Online transcription with Vertex AI Gemini or Google Speech Recognition
- Offline transcription with Vosk
- Automatic MP3 encoding and chunking for longer recordings
- Clipboard output only when transcription succeeds
- Separate Polish and English interface language
- System tray controls, sound cues, and persistent settings
- Automatic fallback to a lighter Gemini model when quota limits are reached

## Requirements

- Windows 10 or Windows 11
- Python 3.11 or newer
- A working microphone
- Credentials for the selected online engine, unless Vosk is used

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item settings.example.json settings.json
```

Edit `settings.json` to select shortcuts, languages, and engines. Do not commit personal credentials or a populated `.env` file.

## Recognition engines

### Vertex AI Gemini

Copy `.env.example` to `.env` and provide the Google Cloud project and authentication values required by your environment. The application sends recorded audio to the configured Vertex AI model.

### Google Speech Recognition

Select the Google engine in `settings.json`. This mode requires an internet connection.

### Vosk

Install a compatible Vosk language model locally and point the channel configuration to it. Audio stays on the device in this mode.

## Running the application

```powershell
python Yapper.py
```

Hold the shortcut assigned to a channel while speaking, then release it to transcribe. Clips shorter than one second are ignored. Successful text is copied to the clipboard.

## Building a Windows executable

```powershell
pip install pyinstaller
pyinstaller --noconfirm Yapper.spec
```

The packaged application is written to `dist/`.

## Privacy

Gemini and Google modes transmit audio to external services. Vosk processes audio locally. Review the selected provider's data policy before using online transcription with sensitive material.

## License

No license has been granted for this repository unless a license file states otherwise.
