from __future__ import annotations

import ssl
import unittest
from unittest.mock import Mock, call
from urllib.error import HTTPError, URLError

from src.utils.http import urlopen_with_retry


def _http_error(code: int) -> HTTPError:
    return HTTPError(
        url="https://example.invalid/pricing",
        code=code,
        msg="test error",
        hdrs=None,
        fp=None,
    )


class HttpRetryTests(unittest.TestCase):
    def test_first_attempt_success_does_not_sleep(self) -> None:
        response = object()
        opener = Mock(return_value=response)
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        opener.assert_called_once_with("request", timeout=30)
        sleeper.assert_not_called()

    def test_timeout_then_success_retries_once(self) -> None:
        response = object()
        opener = Mock(side_effect=[TimeoutError("timed out"), response])
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        self.assertEqual(2, opener.call_count)
        self.assertEqual([call(1.0)], sleeper.call_args_list)

    def test_two_failures_then_success_uses_both_backoffs(self) -> None:
        response = object()
        opener = Mock(
            side_effect=[
                TimeoutError("timed out"),
                ConnectionResetError("reset"),
                response,
            ]
        )
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        self.assertEqual(3, opener.call_count)
        self.assertEqual([call(1.0), call(2.0)], sleeper.call_args_list)

    def test_three_ssl_eof_failures_raise_final_error(self) -> None:
        errors = [
            ssl.SSLEOFError(8, "unexpected EOF"),
            ssl.SSLEOFError(8, "unexpected EOF"),
            ssl.SSLEOFError(8, "unexpected EOF"),
        ]
        opener = Mock(side_effect=errors)
        sleeper = Mock()

        with self.assertRaises(ssl.SSLEOFError) as caught:
            urlopen_with_retry(
                "request", timeout=30, opener=opener, sleeper=sleeper
            )

        self.assertIs(errors[-1], caught.exception)
        self.assertEqual(3, opener.call_count)
        self.assertEqual([call(1.0), call(2.0)], sleeper.call_args_list)

    def test_http_429_is_retried(self) -> None:
        response = object()
        error = _http_error(429)
        self.addCleanup(error.close)
        opener = Mock(side_effect=[error, response])
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        self.assertEqual(2, opener.call_count)
        self.assertEqual([call(1.0)], sleeper.call_args_list)

    def test_http_503_is_retried(self) -> None:
        response = object()
        error = _http_error(503)
        self.addCleanup(error.close)
        opener = Mock(side_effect=[error, response])
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        self.assertEqual(2, opener.call_count)
        self.assertEqual([call(1.0)], sleeper.call_args_list)

    def test_http_404_is_not_retried(self) -> None:
        error = _http_error(404)
        self.addCleanup(error.close)
        opener = Mock(side_effect=error)
        sleeper = Mock()

        with self.assertRaises(HTTPError) as caught:
            urlopen_with_retry(
                "request", timeout=30, opener=opener, sleeper=sleeper
            )

        self.assertIs(error, caught.exception)
        self.assertEqual(1, opener.call_count)
        sleeper.assert_not_called()

    def test_http_403_is_not_retried(self) -> None:
        error = _http_error(403)
        self.addCleanup(error.close)
        opener = Mock(side_effect=error)
        sleeper = Mock()

        with self.assertRaises(HTTPError) as caught:
            urlopen_with_retry(
                "request", timeout=30, opener=opener, sleeper=sleeper
            )

        self.assertIs(error, caught.exception)
        self.assertEqual(1, opener.call_count)
        sleeper.assert_not_called()

    def test_certificate_verification_error_is_not_retried(self) -> None:
        error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        opener = Mock(side_effect=error)
        sleeper = Mock()

        with self.assertRaises(ssl.SSLCertVerificationError) as caught:
            urlopen_with_retry(
                "request", timeout=30, opener=opener, sleeper=sleeper
            )

        self.assertIs(error, caught.exception)
        self.assertEqual(1, opener.call_count)
        sleeper.assert_not_called()

    def test_url_error_wrapping_certificate_error_is_not_retried(self) -> None:
        certificate_error = ssl.SSLCertVerificationError(
            1, "certificate verify failed"
        )
        error = URLError(certificate_error)
        opener = Mock(side_effect=error)
        sleeper = Mock()

        with self.assertRaises(URLError) as caught:
            urlopen_with_retry(
                "request", timeout=30, opener=opener, sleeper=sleeper
            )

        self.assertIs(error, caught.exception)
        self.assertEqual(1, opener.call_count)
        sleeper.assert_not_called()

    def test_plain_url_error_is_retried(self) -> None:
        response = object()
        opener = Mock(side_effect=[URLError("temporary failure"), response])
        sleeper = Mock()

        result = urlopen_with_retry(
            "request", timeout=30, opener=opener, sleeper=sleeper
        )

        self.assertIs(response, result)
        self.assertEqual(2, opener.call_count)
        self.assertEqual([call(1.0)], sleeper.call_args_list)


if __name__ == "__main__":
    unittest.main()
