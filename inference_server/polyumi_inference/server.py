"""
Server half of the inference protocol: the app every PolyUMI inference server is.

:func:`create_app` owns the HTTP surface -- the three routes, decoding the frame, enforcing the
contract, turning a bad frame into a 422, truncating the chunk, and the request timing. A server is
then only a :class:`PolicyBackend`: a sine oscillator for bringup, a diffusion checkpoint in
production. Neither writes a route.

That split is what makes "the dummy must refuse exactly what a checkpoint refuses" structural
rather than a convention two files have to keep matching by hand.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Callable, Iterable, Optional, Protocol, Union, runtime_checkable

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel

from polyumi_inference.contract import AGENT_POS_DIM, REQUIRED_CHANNELS, validate_observation
from polyumi_inference.errors import WireFormatError
from polyumi_inference.types import ActionChunk, Observation

# A child of uvicorn's own logger, NOT a bare getLogger(__name__). Uvicorn configures handlers for
# the 'uvicorn*' loggers only and leaves the root logger bare, so a top-level logger here propagates
# to a root with no handler and every INFO line is silently discarded. Hanging off 'uvicorn.error'
# inherits uvicorn's handler and format, so these lines interleave with the access log instead of
# needing a --log-config of their own.
logger = logging.getLogger('uvicorn.error').getChild('polyumi_inference')


@runtime_checkable
class PolicyBackend(Protocol):
    """What a server must be able to do. Everything else is :func:`create_app`'s."""

    def predict(self, obs: Observation) -> ActionChunk:
        """
        Produce actions for one observation window.

        The observation has already been decoded and checked against the contract, so a backend can
        index its channels directly. Set ``model_ms`` on the returned chunk: the backend is the only
        place the forward pass can be timed honestly (a GPU backend must time through its own
        synchronization point, not just the call).
        """

    def reset(self, agent_pos: np.ndarray) -> None:
        """Cache the episode-start pose. Called once per rollout, may be a no-op."""

    def describe(self) -> dict:
        """Describe readiness and configuration; becomes the body of ``GET /health``."""


class PredictResponse(BaseModel):
    """Response body for ``/predict_cartesian/``."""

    actions: list
    n_action_steps: int
    #: Wall time this process spent on the request, in ms -- the same number the log line prints.
    #: Nullable so a client can distinguish "the server did not say" from "it was zero".
    server_total_ms: Optional[float] = None
    #: The forward pass alone, in ms. The client plots the round trip split against this.
    model_ms: Optional[float] = None


class ResetRequest(BaseModel):
    """Body for ``/reset`` -- one wire pose captured at the start of the rollout."""

    agent_pos: list  # a single [8] pose [x,y,z,qx,qy,qz,qw,gripper]


def create_app(
    backend: Union[PolicyBackend, Callable[[], PolicyBackend]],
    *,
    title: str,
    required_channels: Iterable[str] = REQUIRED_CHANNELS,
) -> FastAPI:
    """
    Build the FastAPI app for a policy backend.

    :param backend: a backend, or a zero-argument callable returning one. Prefer the callable: it
        is invoked at startup, so a misconfigured server (no checkpoint, a nonsense ``HOME_POSE``)
        fails when it starts rather than passing a health check it cannot honour.
    :param title: OpenAPI title.
    :param required_channels: channels a request must carry. Defaults to the policy's.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend() if callable(backend) else backend
        logger.info('serving %s', title)
        yield

    app = FastAPI(title=title, lifespan=lifespan)

    @app.middleware('http')
    async def _log_request_time(request: Request, call_next):
        """
        Log how long each request took to serve, split into total and model time.

        The client already measures its own round trip, but that number bundles network,
        serialization and compute together, so a slow tick is indistinguishable from a slow link.
        Uvicorn's access log cannot close the gap -- its message format is fixed and carries no
        duration -- so time it here.

        Two numbers because they fail for different reasons and have different fixes. ``total``
        minus ``model`` is decode + FastAPI overhead, which scales with the observation payload;
        ``model`` is compute, which on a shared box scales with whoever else is running.

        The clock starts before the body has finished arriving, so ``total`` absorbs upload time on
        a slow link. That is why the client splits its round trip on ``model_ms`` and not on this.
        """
        t0 = time.perf_counter()
        # Also handed to the endpoint, which puts it in the response body: the log is for a human
        # reading this box, the wire field is for the client plotting the split live.
        request.state.t_request_start = t0
        response = await call_next(request)
        total_ms = (time.perf_counter() - t0) * 1e3
        model_ms = getattr(request.state, 'model_ms', None)
        model = f', model {model_ms:.0f}ms' if model_ms is not None else ''
        logger.info('%s %s -> %d in %.0f ms%s', request.method, request.url.path, response.status_code, total_ms, model)
        return response

    @app.get('/health')
    def health() -> dict:
        """Liveness/readiness check. The backend describes itself; the app only reports reachability."""
        backend_obj = getattr(app.state, 'backend', None)
        if backend_obj is None:
            return {'status': 'loading'}
        return backend_obj.describe()

    @app.post('/reset')
    def reset(req: ResetRequest) -> dict:
        """Cache the episode-start EEF pose. Call once at the start of each rollout."""
        if len(req.agent_pos) != AGENT_POS_DIM:
            raise HTTPException(status_code=422, detail=f'agent_pos must have length {AGENT_POS_DIM}')
        # A malformed pose (e.g. a zero-norm quaternion) makes the backend's rotation maths raise;
        # that is a bad request, not a server fault, so surface it as 422 rather than letting it 500.
        try:
            app.state.backend.reset(np.asarray(req.agent_pos, dtype=np.float64))
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=f'Invalid agent_pos: {e}') from e
        return {'status': 'ok', 'episode_start_set': True}

    @app.post('/predict_cartesian/', response_model=PredictResponse)
    def predict_cartesian(
        request: Request,
        body: Annotated[bytes, Body(media_type='application/octet-stream')],
    ) -> dict:
        """Run the backend on one observation window and return an absolute EEF action chunk."""
        # A sync def, deliberately: FastAPI runs it in the threadpool, so a multi-hundred-millisecond
        # forward pass does not block the event loop and stall every health check behind it.
        try:
            obs = Observation.from_frame(body)
            validate_observation(obs, required_channels)
        except WireFormatError as e:
            # A frame this server cannot read or cannot serve is a bad request, not a fault here.
            raise HTTPException(status_code=422, detail=str(e)) from e

        chunk = app.state.backend.predict(obs)
        # Return at most what was asked for; further truncation is the client's job (UMI's policy
        # emits the full horizon with no offset).
        chunk = chunk.truncate(obs.n_action_steps)

        if chunk.model_ms is not None:
            request.state.model_ms = chunk.model_ms
        t_start = getattr(request.state, 't_request_start', None)
        return {
            **chunk.to_json(),
            'server_total_ms': None if t_start is None else (time.perf_counter() - t_start) * 1e3,
        }

    return app


def serve(app_path: str, *, host: str = '0.0.0.0', port: int = 8000) -> None:
    """Run an app under uvicorn. Convenience for a package's ``main()`` entry point."""
    import uvicorn

    uvicorn.run(app_path, host=host, port=port, reload=False)
