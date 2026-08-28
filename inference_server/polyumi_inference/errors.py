"""
Exceptions raised across the inference protocol.

Two failure modes, deliberately distinct because they have different fixes. A
:class:`WireFormatError` means the *frame* is wrong -- malformed, truncated, or missing a channel
the policy needs -- and is always the requester's fault, so every server turns it into a 422. A
:class:`TransportError` means the request never completed or came back an error: the link, the
process, or the far end's own verdict.
"""

from __future__ import annotations


class PolyumiInferenceError(Exception):
    """Base for every error this library raises, so a caller can catch one thing."""


class WireFormatError(PolyumiInferenceError, ValueError):
    """A frame that cannot be decoded, or that omits a channel the policy requires."""


class TransportError(PolyumiInferenceError):
    """
    A request that did not come back with a usable answer.

    Carries the server's own words. On a 4xx the response body IS the diagnostic -- it names the
    missing channel, or the byte the frame was cut at -- so ``str(self)`` includes it rather than
    leaving the caller with a bare status code.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Build the error. ``detail`` is the server's own words, kept for the log line."""
        self.url = url
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)
        self._message = message

    def __str__(self) -> str:
        parts = [self._message]
        if self.status_code is not None:
            parts.append(f'HTTP {self.status_code}')
        text = ': '.join(parts)
        return text if not self.detail else f'{text} - {self.detail}'
