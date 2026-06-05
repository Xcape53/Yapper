# Yapper

Yapper to aplikacja Windows/PyQt6 do transkrypcji mowy na tekst z obsługą dwóch kanałów PTT. Rozpoznany tekst jest kopiowany do schowka tylko wtedy, gdy wybrany silnik zwróci niepustą odpowiedź.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Funkcje

- 2 niezależne kanały PTT, każdy z własnym mikrofonem, językiem transkrypcji i klawiszem aktywacji.
- Silniki transkrypcji: Gemini przez Vertex AI, Google Speech API oraz Vosk offline.
- Gemini wysyła audio jako MP3, dzieli dłuższe nagrania na segmenty 30 s i przy błędzie quota/429 ponawia segment przez `gemini-2.5-flash-lite`.
- Minimalna długość nagrania to 1 s; krótsze nagrania nie są wysyłane do API i nie dotykają schowka.
- Schowek jest nadpisywany wyłącznie realnym, niepustym wynikiem transkrypcji.
- Osobny język GUI (`Polski` / `English`) oraz osobny język transkrypcji dla kanału 1 i kanału 2.
- Stały rozmiar okna, status ostatniej wiadomości i nazwa modelu, który ją przetworzył.
- Dźwięki PTT, minimalizacja do zasobnika systemowego i zapis konfiguracji.

## Instalacja ze źródeł

Wymagania:

- Windows 10/11
- Python 3.11+

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Opcjonalne komponenty:

```bash
pip install vosk                    # offline speech recognition
pip install google-cloud-aiplatform # Gemini / Vertex AI
pip install google-auth-oauthlib    # OAuth dla Vertex AI
```

## Konfiguracja

1. Skopiuj `settings.example.json` do `settings.json`.
2. Ustaw mikrofony, klawisze PTT i języki w aplikacji.
3. Jeśli używasz Gemini przez Vertex AI, uzupełnij sekcję `vertex_ai`.

Przykład konfiguracji Vertex AI:

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

Możesz też użyć zmiennych środowiskowych:

```env
GOOGLE_CLOUD_PROJECT=twoj-projekt-id
GOOGLE_CLIENT_SECRET_FILE=client_secret.json
GOOGLE_VERTEX_MODEL=gemini-3.1-flash-lite
```

Pliki lokalne z sekretami i tokenami są ignorowane przez Git:

- `.env`
- `settings.json`
- `vertex_token.pickle`
- `client_secret_*.json`
- `dist/`

## Uruchomienie

```bash
python Yapper.py
```

## Użycie

1. Wybierz model transkrypcji.
2. Ustaw język GUI w prawym górnym boksie, jeśli chcesz zmienić napisy interfejsu.
3. Dla kanału 1 i kanału 2 ustaw osobno język transkrypcji, mikrofon i klawisz PTT.
4. Przytrzymaj klawisz PTT, mów, a potem puść klawisz.
5. Jeśli transkrypcja zwróci tekst, zostanie on skopiowany do schowka.

## Build EXE

Rekomendowany build korzysta z `Yapper.spec`:

```bash
pip install pyinstaller
pyinstaller --noconfirm Yapper.spec
```

Wynikowy plik znajduje się w `dist/Yapper.exe`.

Do paczki release dołączaj tylko bezpieczne pliki:

- `dist/Yapper.exe`
- `sounds/`
- `settings.example.json`
- `README.md`

Nie pakuj lokalnych plików `settings.json`, `vertex_token.pickle`, `client_secret_*.json`, logów ani nagrań.

## Struktura projektu

```text
Yapper/
├── Yapper.py              # główna aplikacja
├── Yapper.spec            # konfiguracja PyInstaller
├── requirements.txt
├── settings.example.json
├── sounds/
│   ├── press.wav
│   └── release.wav
└── _internal/
    ├── wafflin.ico
    └── tray_icon.png
```

## Release 1.2.0

Najważniejsze zmiany:

- MP3 i chunkowanie audio dla Gemini.
- Fallback z `gemini-3.1-flash-lite` na `gemini-2.5-flash-lite` przy 429/quota.
- Ochrona schowka przed pustymi wynikami.
- Minimalne nagranie 1 s.
- Rozdzielenie języka GUI od języków transkrypcji kanałów.
- Poprawione polskie znaki w UI.

## License

MIT License.
