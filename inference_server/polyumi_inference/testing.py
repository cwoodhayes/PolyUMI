"""
Drive a real :class:`~polyumi_inference.client.PolicyClient` against a real app, in process.

This is why :class:`~polyumi_inference.client.Transport` is a seam. With it, a test can send the
exact bytes the ROS node sends and have them decoded by the exact code the server runs, with no
socket, no port, and no second process -- which is the one pairing that was previously untestable,
and the one that used to be kept honest by comparing files.

Test-only: it imports Starlette's ``TestClient``, which the production install does not have.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from polyumi_inference.client import PolicyClient, Reply, Transport

#: TestClient resolves relative URLs against this host, so it is the base a client must be given.
TEST_BASE_URL = 'http://testserver'


class TestClientTransport(Transport):
    """Routes requests into a Starlette/FastAPI ``TestClient`` instead of over the network."""

    def __init__(self, test_client: Any) -> None:
        """Wrap an already-constructed ``TestClient``."""
        self._client = test_client

    def request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[bytes] = None,
        json: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> Reply:
        """Issue one request through the test client. ``timeout_s`` is meaningless in process."""
        kwargs: dict = {}
        if content is not None:
            kwargs['content'] = content
            kwargs['headers'] = {'Content-Type': 'application/octet-stream'}
        if json is not None:
            kwargs['json'] = json
        resp = self._client.request(method, url, **kwargs)
        return Reply(status_code=resp.status_code, text=resp.text)


@contextmanager
def in_process_client(app: Any, *, base_url: str = TEST_BASE_URL) -> Iterator[PolicyClient]:
    """
    Yield a :class:`PolicyClient` wired to ``app`` with no socket in between.

    Enters the app's lifespan, so the backend is constructed exactly as it would be at startup --
    including failing there if it is misconfigured.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield PolicyClient(f'{base_url}/predict_cartesian/', transport=TestClientTransport(test_client))
