import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import Yapper
from PyQt6.QtWidgets import QApplication


class GroqUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.temp.name)
        self.environment = patch.dict(os.environ, {'GROQ_API_KEY': 'test-secret-not-a-real-key'})
        self.environment.start()
        with patch.object(Yapper.SpeechToClipboardApp, '_init_sounds'):
            self.window = Yapper.SpeechToClipboardApp()

    def tearDown(self):
        self.window.hide()
        self.window.deleteLater()
        self.qt.processEvents()
        self.environment.stop()
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_save_reload_preserves_microphones_and_excludes_key_from_json_and_log(self):
        Path('settings.json').write_text(json.dumps({'gemini_chunk_seconds': 123,
                                                    'groq': {'model_id': 'whisper-large-v3-turbo',
                                                             'prompt': 'Yapper, Beczek'},
                                                    'vertex_ai': {'project_id': 'old-project'}}))
        original_env = b'# manual configuration\nGROQ_API_KEY=test-secret-not-a-real-key\nGEMINI_API_KEY=preserve-me\n'
        Path('.env').write_bytes(original_env)
        self.window._load_settings()
        self.assertEqual(self.window.groq_options['model'], 'whisper-large-v3-turbo')
        self.assertEqual(self.window.groq_options['prompt'], 'Yapper, Beczek')
        self.assertFalse(hasattr(self.window, 'groq_prompt_input'))
        self.window.ptt_channels['2'].current_lang_code = 'auto'
        self.window._save_settings()
        settings = json.loads(Path('settings.json').read_text(encoding='utf-8'))
        self.assertEqual(settings['model'], 'groq')
        self.assertEqual(settings['gemini_chunk_seconds'], 123)
        self.assertEqual(settings['vertex_ai']['project_id'], 'old-project')
        self.assertEqual(settings['groq']['prompt'], 'Yapper, Beczek')
        self.assertNotIn('test-secret', Path('settings.json').read_text(encoding='utf-8'))
        self.assertNotIn('test-secret', Path('yapper_log.txt').read_text(encoding='utf-8'))
        self.assertEqual(Path('.env').read_bytes(), original_env)
        self.window._load_settings()
        self.assertEqual(self.window.groq_options['model'], 'whisper-large-v3-turbo')
        self.assertEqual(self.window.ptt_channels['2'].lang_combo.currentData(), 'auto')

    def test_gui_languages_do_not_change_transcription_language(self):
        self.window.ptt_channels['1'].lang_combo.setCurrentIndex(2)
        self.window._on_gui_language_change('en')
        self.assertEqual(self.window.ptt_channels['1'].current_lang_code, 'auto')
        self.assertTrue(self.window.rules_group.isHidden())
        self.assertFalse(self.window.groq_group.isHidden())
        self.window._on_gui_language_change('pl')

    @patch('Yapper.recognizer.recognize_google')
    @patch('groq_transcription.requests.post')
    def test_groq_error_keeps_clipboard_and_never_falls_back(self, post, google):
        self.window.fallback_to_google = True
        post.return_value = Mock(status_code=429)
        self.qt.clipboard().setText('keep this')
        self.window.ptt_channels['1']._process_audio(b'\0\0' * 16000)
        self.qt.processEvents()
        self.assertEqual(self.qt.clipboard().text(), 'keep this')
        self.assertIn('Groq', self.window.status_label.text())
        google.assert_not_called()

    @patch('groq_transcription.requests.post')
    def test_success_copies_transcript(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {'text': 'Cześć, hello world.'}
        post.return_value = response
        self.window.ptt_channels['1']._process_audio(b'\0\0' * 16000)
        self.qt.processEvents()
        self.assertEqual(self.qt.clipboard().text(), 'Cześć, hello world.')

    def test_environment_key_used_without_gui_or_creating_env_file(self):
        self.assertEqual(self.window.groq_options['api_key'], 'test-secret-not-a-real-key')
        self.assertFalse(hasattr(self.window, 'groq_key_input'))
        self.window._save_settings()
        self.assertFalse(Path('.env').exists())

    def test_typing_in_text_field_still_releases_active_ptt(self):
        channel = self.window.ptt_channels['1']
        channel.ptt_key_pressed = True
        self.window.editing_text = True
        with patch('Yapper.keyboard.hook') as hook, patch('Yapper.keyboard.unhook_all'), \
                patch.object(Yapper.stop_program_event, 'wait'):
            self.window.keyboard_listener_thread_func()
        handler = hook.call_args.args[0]
        handler(SimpleNamespace(event_type=Yapper.keyboard.KEY_UP, name=channel.ptt_activation_key))
        self.assertFalse(channel.ptt_key_pressed)

    @patch('Yapper.setup_vertex_ai', return_value=False)
    def test_google_auth_only_after_explicit_gemini_selection(self, setup):
        self.qt.processEvents()
        setup.assert_not_called()
        self.assertTrue(self.window.gemini_radio.isEnabled())
        self.window.gemini_radio.setChecked(True)
        self.qt.processEvents()
        setup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
