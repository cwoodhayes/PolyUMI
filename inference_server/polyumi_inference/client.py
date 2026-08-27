"""
Client half of the inference protocol.

:class:`PolicyClient` is what the ROS-side ``policy_client_node`` holds. It owns the three
endpoints, the URL arithmetic between them, and the persistent connection; it does not own logging
or retry policy, which belong to the caller (the node logs and carries on; a script might want to
die).

Every failure is a :class:`TransportError` rather than a ``None`` return, so a caller cannot
accidentally treat "the server refused this frame" as "no actions this tick" -- and so the server's
own explanation survives into the caller's log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from polyumi_inference.errors import TransportError, WireFormatError
from polyumi_inference.types import ActionChunk, Observation

#: How much of an error response body to keep. Enough for the longest message the contract can
#: produce, short enough not to dump a stack trace page into a ROS log line.
MAX_DETAIL_CHARS = 500


@dataclass(frozen=True)
class Reply:
    """One HTTP response, reduced to what this protocol needs."""

    status_code: int
    text: str

    def json(self) -> dict:
        """Parse the body as a JSON object, or raise :class:`ValueError`."""
        doc = json.loads(self.text)
        if not isinstance(doc, dict):
            raise ValueError(f'expected a JSON object, got {type(doc).__name__}')
        return doc


class Transport:
    """
    How a :class:`PolicyClient` actually speaks HTTP.

    One method, not one per body type. The client makes three different requests -- a binary frame
    to ``/predict_cartesian/``, a JSON document to ``/reset``, and a bodiless ``GET /health`` -- but
    the thing that varies between implementations is the HTTP client, not the encoding. There are
    two implementations: :class:`RequestsTransport` in production, and the in-process one in
    :mod:`polyumi_inference.testing` that lets a real client drive a real app with no socket. That
    second one is the point of this seam; it is not a plugin surface.

    An implementation raises :class:`TransportError` when the request does not complete at all, and
    returns a :class:`Reply` for anything the server answered -- including a 4xx. Deciding what a
    status code means is :class:`PolicyClient`'s job, so both transports agree on it for free.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[bytes] = None,
        json: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> Reply:
        """Issue one request and return the response."""
        raise NotImplementedError


class RequestsTransport(Transport):
    """
    Production transport, over a persistent ``requests.Session``.

    The session, rather than ``urllib.request.urlopen``, because urlopen opens a fresh connection
    per call: on a several-hundred-kilobyte body that costs a TCP handshake plus a slow-start ramp
    every single inference, with the window reopening from the initial ~14 kB before the request can
    even finish uploading. One warm connection for the life of the client instead.
    """

    def __init__(self, session: Any = None) -> None:
        """Build a transport, reusing ``session`` if one is given."""
        # Imported here, not at module scope, so the server half of this library carries no
        # dependency on an HTTP client it never uses (the policy container installs neither).
        import requests

        self._requests = requests
        self._session = requests.Session() if session is None else session

    def request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[bytes] = None,
        json: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> Reply:
        """Issue one request through the session."""
        kwargs: dict = {}
        if content is not None:
            kwargs['data'] = content
            kwargs['headers'] = {'Content-Type': 'application/octet-stream'}
        if json is not None:
            kwargs['json'] = json
        try:
            resp = self._session.request(method, url, timeout=timeout_s, **kwargs)
        except self._requests.RequestException as e:
            raise TransportError(f'{method} {url} failed: {e}', url=url) from e
        return Reply(status_code=resp.status_code, text=resp.text)

    def close(self) -> None:
        """Close the underlying session."""
        self._session.close()


class PolicyClient:
    """
    Talks to a PolyUMI inference server.

    :param predict_url: full URL of ``/predict_cartesian/``. The other two endpoints are derived
        from its base, so one parameter configures all three -- which is how the ROS node is
        parameterized and why the derivation lives here rather than in the node.
    :param timeout_s: per-request timeout.
    :param transport: defaults to :class:`RequestsTransport`.
    """

    def __init__(
        self,
        predict_url: str,
        *,
        timeout_s: float = 5.0,
        transport: Optional[Transport] = None,
    ) -> None:
        """Build a client for one server, deriving its other two endpoints."""
        self.predict_url = predict_url
        base = predict_url.split('/predict_cartesian')[0]
        self.reset_url = base + '/reset'
        self.health_url = base + '/health'
        self.timeout_s = timeout_s
        self._transport = transport if transport is not None else RequestsTransport()

    def predict(self, obs: Observation) -> ActionChunk:
        """
        Run one observation through the policy and return its action chunk.

        :raises TransportError: on a failed request or any non-2xx, carrying the server's reason.
        :raises WireFormatError: if the reply is not a well-formed action chunk.
        """
        reply = self._transport.request('POST', self.predict_url, content=obs.to_frame(), timeout_s=self.timeout_s)
        return ActionChunk.from_json(self._body(reply, 'POST', self.predict_url))

    def reset(self, agent_pos: Sequence[float]) -> dict:
        """
        Tell the server where this episode started.

        The policy consumes ``robot0_eef_rot_axis_angle_wrt_start`` -- orientation relative to the
        episode's first pose -- and the observation frame only ever carries the *current* pose, so
        the start pose has to be sent once per rollout and cached server-side.

        :param agent_pos: a single ``[8]`` pose ``[x, y, z, qx, qy, qz, qw, gripper]``.
        """
        payload = {'agent_pos': np.asarray(agent_pos, dtype=np.float64).tolist()}
        reply = self._transport.request('POST', self.reset_url, json=payload, timeout_s=self.timeout_s)
        return self._body(reply, 'POST', self.reset_url)

    def health(self) -> dict:
        """Ask the server whether it is ready to serve."""
        reply = self._transport.request('GET', self.health_url, timeout_s=self.timeout_s)
        return self._body(reply, 'GET', self.health_url)

    def close(self) -> None:
        """Release the transport's connection, if it holds one."""
        close = getattr(self._transport, 'close', None)
        if close is not None:
            close()

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _body(self, reply: Reply, method: str, url: str) -> dict:
        """
        Turn a reply into a parsed body, or raise with the server's own words.

        A 4xx body carries the reason -- a malformed frame, a missing channel -- and that reason is
        the whole diagnostic; without it this is just "422".
        """
        if reply.status_code >= 400:
            raise TransportError(
                f'{method} {url} failed',
                url=url,
                status_code=reply.status_code,
                detail=_detail(reply.text),
            )
        try:
            return reply.json()
        except ValueError as e:
            raise TransportError(
                f'{method} {url} returned an unreadable response: {e}',
                url=url,
                status_code=reply.status_code,
                detail=_detail(reply.text),
            ) from e


def _detail(text: str) -> str:
    """
    Pull the useful part out of an error body.

    FastAPI wraps every ``HTTPException`` message in ``{"detail": ...}``; unwrapping it here means a
    log line reads as the sentence the server wrote rather than as a fragment of JSON.
    """
    try:
        doc = json.loads(text)
    except ValueError:
        return text[:MAX_DETAIL_CHARS]
    if isinstance(doc, dict) and 'detail' in doc:
        return str(doc['detail'])[:MAX_DETAIL_CHARS]
    return text[:MAX_DETAIL_CHARS]


__all__ = ['PolicyClient', 'Reply', 'RequestsTransport', 'Transport', 'TransportError', 'WireFormatError']
