"""TTS retry logging must keep the synthesized text out of the log body.

A log record's body cannot be redacted — the telemetry pipeline only strips
attributes carrying a ``pii`` segment (see ``telemetry/traces.py``). The TTS retry
path interpolates the exception into the message, and that exception quotes the text
being synthesized (``APIError("no audio frames were pushed for text: ...")``), so the
text used to reach every log sink — including the Cloud log export when the project
enables redaction.
"""

from __future__ import annotations

import logging

import pytest

from livekit.agents import APIConnectOptions, APIError
from livekit.agents.telemetry import pii

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit

SYNTHESIZED_TEXT = "your card number is 4111 1111 1111 1111"


async def _synthesize_until_error(tts_v: FakeTTS, *, max_retry: int) -> None:
    stream = tts_v.synthesize(
        SYNTHESIZED_TEXT,
        conn_options=APIConnectOptions(max_retry=max_retry, timeout=0.5, retry_interval=0.0),
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass


async def test_retry_warning_keeps_the_text_out_of_the_log_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A retried synthesis failure must not put the text in the message body."""
    fake_tts = FakeTTS(fake_audio_duration=0.0)  # pushes no audio -> retryable APIError
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            await _synthesize_until_error(fake_tts, max_retry=1)
    finally:
        await fake_tts.aclose()

    retries = [r for r in caplog.records if "failed to synthesize speech" in r.getMessage()]
    assert retries, "expected the retry warning to be logged"

    for record in retries:
        assert SYNTHESIZED_TEXT not in record.getMessage(), (
            "the synthesized text must not be interpolated into the log body, "
            "which no redaction can reach"
        )
        # it is still available to the operator, under a key the PII filter can strip
        assert SYNTHESIZED_TEXT in record.__dict__["lk.pii.error"]
        assert pii.filter_attributes(record.__dict__) == {} or "lk.pii.error" not in (
            pii.filter_attributes(record.__dict__)
        )
