import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from scripts.smoke_runtime import wait_for_health


class SmokeReadinessTests(unittest.TestCase):
    def test_startup_connection_resets_are_retried(self):
        request = Mock(side_effect=[ConnectionResetError('starting'),
                                    URLError('not listening'), {'status': 'ok'}])
        with patch('scripts.smoke_runtime.time.sleep'):
            self.assertEqual(wait_for_health(request, 10), {'status': 'ok'})
        self.assertEqual(request.call_count, 3)

    def test_broken_server_still_times_out(self):
        request = Mock(side_effect=ConnectionResetError('not ready'))
        with patch('scripts.smoke_runtime.time.monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(RuntimeError, 'before timeout'):
                wait_for_health(request, 10)
