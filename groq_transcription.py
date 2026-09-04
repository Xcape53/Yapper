"""Groq speech API adapter for mono 16-bit PCM captured by Yapper."""
import io
import time
import wave

import requests

API_URL = 'https://api.groq.com/openai/v1'
MODELS = ('whisper-large-v3', 'whisper-large-v3-turbo')
MAX_WAV_BYTES = 24_000_000  # Leave room below the free plan's 25 MB limit.


class GroqError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _check_response(response):
    if response.status_code >= 400:
        code = {401: 'auth', 403: 'permission', 413: 'size', 429: 'rate_limit'}.get(
            response.status_code, 'server' if response.status_code >= 500 else 'request')
        # Never propagate an API response body or an exception containing credentials.
        raise GroqError(code)


def check_api_key(api_key):
    if not api_key.strip():
        raise GroqError('missing_key')
    try:
        response = requests.get(API_URL + '/models',
                                headers={'Authorization': 'Bearer ' + api_key.strip()},
                                timeout=(10, 20), allow_redirects=False)
        _check_response(response)
        data = response.json()
        if not any(item.get('id') in MODELS for item in data.get('data', [])):
            raise GroqError('permission')
    except requests.Timeout:
        raise GroqError('timeout') from None
    except requests.RequestException:
        raise GroqError('network') from None
    except (ValueError, AttributeError, TypeError):
        raise GroqError('response') from None


def _wav(pcm, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return buffer.getvalue()


def _merge(previous, current):
    """Remove an exact word overlap from the one-second shared audio boundary."""
    left, right = previous.split(), current.split()
    for count in range(min(40, len(left), len(right)), 0, -1):
        if [w.casefold().strip('.,!?;:') for w in left[-count:]] == [
                w.casefold().strip('.,!?;:') for w in right[:count]]:
            return previous + (' ' + ' '.join(right[count:]) if right[count:] else '')
    return (previous + ' ' + current).strip()


def transcribe_pcm(pcm, sample_rate, *, api_key, language='pl-PL',
                   model='whisper-large-v3', prompt=''):
    if not api_key.strip():
        raise GroqError('missing_key')
    if model not in MODELS or sample_rate <= 0 or len(pcm) % 2:
        raise GroqError('request')
    if len(pcm) < sample_rate * 2:
        raise GroqError('short_audio')
    data = {'model': model, 'response_format': 'json', 'temperature': '0'}
    if language and language != 'auto':
        data['language'] = language.split('-')[0].lower()
    if prompt.strip():
        data['prompt'] = prompt.strip()
    capacity = ((MAX_WAV_BYTES - 44) // 2) * 2
    overlap = min(sample_rate * 2, capacity // 2)
    start, chunks, text = 0, 0, ''
    started = time.monotonic()
    while start < len(pcm):
        end = min(start + capacity, len(pcm))
        if chunks:
            time.sleep(3.1)  # Avoid sending chunks in a burst against the free RPM limit.
        try:
            response = requests.post(
                API_URL + '/audio/transcriptions',
                headers={'Authorization': 'Bearer ' + api_key.strip()},
                files={'file': ('recording.wav', _wav(pcm[start:end], sample_rate), 'audio/wav')},
                data=data, timeout=(10, 180), allow_redirects=False)
            _check_response(response)
            chunk_text = response.json().get('text')
            if not isinstance(chunk_text, str):
                raise GroqError('response')
            chunk_text = chunk_text.strip()
            if chunk_text:
                text = _merge(text, chunk_text) if chunks else chunk_text
        except requests.Timeout:
            raise GroqError('timeout') from None
        except requests.RequestException:
            raise GroqError('network') from None
        except (ValueError, AttributeError, TypeError):
            raise GroqError('response') from None
        chunks += 1
        if end == len(pcm):
            break
        start = end - overlap
    return {'text': text, 'system': 'Groq', 'model_id': model, 'chunks': chunks,
            'latency': time.monotonic() - started, 'duration': len(pcm) / (sample_rate * 2)}
