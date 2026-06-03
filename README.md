# 🗣️ Yapper

**Yapper** to aplikacja do transkrypcji mowy na tekst z obsługą wielu kanałów PTT (Push-to-Talk). Rozpoznany tekst automatycznie kopiowany jest do schowka.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Funkcje

- 🎙️ **2 niezależne kanały PTT** - każdy z własnym mikrofonem, językiem i klawiszem aktywacji
- 🤖 **3 silniki rozpoznawania mowy:**
  - **Google Speech API** - szybkie i dokładne (wymaga internetu)
  - **Gemini (Vertex AI)** - zaawansowane przetwarzanie z custom rules (wymaga konfiguracji GCP)
  - **Vosk** - w pełni offline
- 📋 **Automatyczne kopiowanie** - tekst trafia bezpośrednio do schowka
- 🔊 **Dźwięki PTT** - konfigurowalna głośność
- ⚙️ **Custom Rules** - własne reguły formatowania dla Gemini
- 🖥️ **System tray** - minimalizacja do zasobnika systemowego
- 💾 **Zapis konfiguracji** - wszystkie ustawienia zapisywane do pliku

## 📦 Instalacja

### Wymagania
- Python 3.11+
- Windows 10/11

### Instalacja zależności

```bash
pip install -r requirements.txt
```

Lub ręcznie:
```bash
pip install PyQt6 pyaudio keyboard pyperclip speech_recognition numpy lameenc pillow pystray python-dotenv
```

Opcjonalnie (dla dodatkowych funkcji):
```bash
pip install vosk                    # Offline speech recognition
pip install google-cloud-aiplatform # Gemini/Vertex AI
pip install google-auth-oauthlib    # OAuth dla Vertex AI
```

## ⚙️ Konfiguracja

### Podstawowa konfiguracja

1. Skopiuj `settings.example.json` do `settings.json`
2. Dostosuj ustawienia według potrzeb

### Konfiguracja Vertex AI (Gemini)

Aby korzystać z modelu Gemini, potrzebujesz:

1. **Projekt Google Cloud** z włączonym Vertex AI API
2. **OAuth Client Secret** - pobierz z Google Cloud Console
3. Skonfiguruj `settings.json`:

```json
{
  "vertex_ai": {
    "project_id": "twoj-projekt-id",
    "location": "global",
    "client_secret_file": "client_secret.json",
    "model_id": "gemini-3.1-flash-lite"
  }
}
```

Lub użyj zmiennych środowiskowych (`.env`):
```env
GOOGLE_CLOUD_PROJECT=twoj-projekt-id
GOOGLE_CLIENT_SECRET_FILE=client_secret.json
GOOGLE_VERTEX_MODEL=gemini-3.1-flash-lite
```

## 🚀 Uruchomienie

```bash
python Yapper.py
```

## 🎮 Użycie

1. **Wybierz model** transkrypcji (Google/Gemini/Vosk)
2. **Skonfiguruj kanały:**
   - Wybierz mikrofon dla każdego kanału
   - Ustaw język (Polski/Angielski)
   - Przypisz klawisz PTT (domyślnie: `5` i `6`)
3. **Przytrzymaj klawisz PTT** aby nagrywać
4. **Puść klawisz** - tekst zostanie rozpoznany i skopiowany do schowka
5. **Wklej** tekst gdzie potrzebujesz (Ctrl+V)

## 📁 Struktura projektu

```
Yapper/
├── Yapper.py              # Główny plik aplikacji
├── settings.json          # Konfiguracja (nie commitować!)
├── settings.example.json  # Przykładowa konfiguracja
├── sounds/                # Dźwięki PTT
│   ├── press.wav
│   └── release.wav
├── _internal/             # Zasoby (ikony)
│   ├── wafflin.ico
│   └── tray_icon.png
└── .env                   # Zmienne środowiskowe (nie commitować!)
```

## 🔨 Kompilacja do EXE

```bash
pip install pyinstaller
pyinstaller --noconfirm --name "Yapper" --onefile --windowed --icon="_internal/wafflin.ico" --add-data "_internal;_internal" --collect-all vosk --collect-all lameenc Yapper.py
```

Skompilowany plik znajdziesz w folderze `dist/`. Pamiętaj o skopiowaniu:
- `sounds/` - folder z dźwiękami
- `settings.json` - konfiguracja (lub `client_secret.json` + `vertex_token.pickle` dla Gemini)

## 📝 License

MIT License - możesz używać, modyfikować i dystrybuować według uznania.

## 🤝 Contributing

Pull requesty są mile widziane! Dla większych zmian, najpierw otwórz issue.

