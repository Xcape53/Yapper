import io
import unittest
import wave
from unittest.mock import patch, Mock

from groq_transcription import transcribe_pcm, GroqError


class GroqTranscriptionTests(unittest.TestCase):
    def response(self, payload, status=200):
        response = Mock(status_code=status, headers={})
        response.json.return_value = payload
        return response

    @patch('groq_transcription.requests.post')
    def test_pcm_sent_as_valid_wav_with_polish_language(self, post):
        post.return_value = self.response({'text': ' Zażółć gęślą. '})
        pcm = b'\x01\x00' * 16000
        result = transcribe_pcm(pcm, 16000, api_key='test-key', language='pl-PL')
        self.assertEqual(result['text'], 'Zażółć gęślą.')
        args, kwargs = post.call_args
        self.assertEqual(args[0], 'https://api.groq.com/openai/v1/audio/transcriptions')
        self.assertEqual(kwargs['data']['language'], 'pl')
        self.assertEqual(kwargs['data']['model'], 'whisper-large-v3')
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')
        with wave.open(io.BytesIO(kwargs['files']['file'][1])) as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.readframes(16000), pcm)

    @patch('groq_transcription.requests.post')
    def test_auto_omits_language_and_english_normalizes(self, post):
        post.return_value = self.response({'text': 'Hello, cześć.'})
        transcribe_pcm(b'\0\0' * 16000, 16000, api_key='test-key', language='auto')
        self.assertNotIn('language', post.call_args.kwargs['data'])
        transcribe_pcm(b'\0\0' * 16000, 16000, api_key='test-key', language='en-US')
        self.assertEqual(post.call_args.kwargs['data']['language'], 'en')

    @patch('groq_transcription.requests.post')
    def test_missing_key_and_short_audio_never_send(self, post):
        with self.assertRaises(GroqError) as caught:
            transcribe_pcm(b'\0\0' * 16000, 16000, api_key='')
        self.assertEqual(caught.exception.code, 'missing_key')
        with self.assertRaises(GroqError) as caught:
            transcribe_pcm(b'\0\0' * 100, 16000, api_key='test-key')
        self.assertEqual(caught.exception.code, 'short_audio')
        post.assert_not_called()

    @patch('groq_transcription.requests.post')
    def test_rate_limit_does_not_retry_or_leak_provider_body(self, post):
        post.return_value = self.response({'error': {'message': 'test-key secret'}}, 429)
        with self.assertRaises(GroqError) as caught:
            transcribe_pcm(b'\0\0' * 16000, 16000, api_key='test-key')
        self.assertEqual(caught.exception.code, 'rate_limit')
        self.assertNotIn('test-key', str(caught.exception))
        self.assertEqual(post.call_count, 1)

    @patch('groq_transcription.requests.post')
    def test_malformed_response_never_becomes_clipboard_text(self, post):
        post.return_value = self.response({'text': ['not', 'a', 'transcript']})
        with self.assertRaises(GroqError) as caught:
            transcribe_pcm(b'\0\0' * 16000, 16000, api_key='test-key')
        self.assertEqual(caught.exception.code, 'response')

    @patch('groq_transcription.requests.post')
    def test_failed_later_chunk_does_not_return_partial_transcript(self, post):
        post.side_effect = [self.response({'text': 'First part.'}), self.response({}, 503)]
        with patch('groq_transcription.MAX_WAV_BYTES', 64044), patch('groq_transcription.time.sleep'):
            with self.assertRaises(GroqError):
                transcribe_pcm(b'\0\0' * 64000, 16000, api_key='test-key')
        self.assertEqual(post.call_count, 2)
        self.assertTrue(all(len(c.kwargs['files']['file'][1]) <= 64044 for c in post.call_args_list))


if __name__ == '__main__':
    unittest.main()
