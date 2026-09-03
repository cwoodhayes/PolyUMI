"""
Temporal ensembling of overlapping action chunks.

The policy re-predicts its whole receding horizon every ``steps_per_inference`` ticks, so several
chunks predict the same future instant from different observations. Executing only the newest one
means the commanded target jumps by however much the model changed its mind, every replan. This
blends the overlapping predictions instead, weighted towards the most recent, which is ACT's
temporal ensembling (Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware*) adapted to an irregular replan cadence.

Two things are deliberately different from the reference formulation:

- **Weights decay in wall-clock time, not in chunk count.** Our replan interval is not fixed —
  measured inference latency varies from ~100 ms to ~400 ms — so "two chunks ago" is not a fixed
  amount of staleness, while ``exp(-age_s / tau_s)`` is.
- **The output keeps the newest chunk's own time grid.** Action ``i`` still targets
  ``t_obs + i * action_dt``, so every downstream consumer — the stale-drop, ``first_index``, the
  anchor stamp, the interpolator on the NUC — is unaffected. Only the values change.

What this does NOT fix: it smooths disagreement between successive predictions, it does not make
any single prediction better. Blending is a trade of reactivity for smoothness — a larger
``tau_s`` averages in staler predictions, so the arm responds more slowly to genuinely new
information. For contact-critical work, keep it on the order of one replan interval.

The gripper width is blended on the same weights as the pose. That does soften a crisp close
across a replan boundary; it is uniform with the rest of the action vector rather than a special
case, and ``tau_s = 0`` turns the whole thing off if that proves to matter.
"""

from __future__ import annotations

import collections

import numpy as np

#: Column layout of one action: [x, y, z, qx, qy, qz, qw, gripper_width].
POS = slice(0, 3)
QUAT = slice(3, 7)
GRIP = 7


def _canonicalise_quaternions(actions: np.ndarray) -> np.ndarray:
    """
    Flip quaternion signs so consecutive waypoints lie in the same hemisphere.

    ``q`` and ``-q`` are the same rotation, so a chunk may contain a sign flip between adjacent
    waypoints. Interpolating the four components independently across such a flip sweeps through
    zero — the long way round the sphere — so the signs have to be made consistent first.
    """
    out = actions.copy()
    quats = out[:, QUAT]
    # Anchor the first waypoint in the w >= 0 hemisphere so two chunks that agree on the rotation
    # also agree on the sign before they are ever compared with each other.
    if quats[0, 3] < 0.0:
        quats[0] *= -1.0
    for i in range(1, len(quats)):
        if float(np.dot(quats[i], quats[i - 1])) < 0.0:
            quats[i] *= -1.0
    return out


class TemporalEnsembler:
    """
    Blends each new action chunk with the recent chunks that overlap it in time.

    Single-threaded by construction: ``policy_client_node`` runs every inference on one worker
    thread, so ``blend`` and ``reset`` are never concurrent and the buffer needs no lock.
    """

    def __init__(self, tau_s: float, action_dt: float, max_chunks: int = 8) -> None:
        """
        Build an ensembler over a fixed waypoint spacing.

        :param tau_s: recency decay constant, seconds. <= 0 disables blending entirely.
        :param action_dt: spacing between waypoints within a chunk, seconds.
        :param max_chunks: how many past chunks to keep. Anything whose weight has decayed to
            irrelevance is dead weight, so this only needs to cover a few multiples of tau_s.
        """
        self._tau_s = tau_s
        self._action_dt = action_dt
        self._buffer: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=max_chunks)

    @property
    def enabled(self) -> bool:
        """Whether blending is on; a disabled ensembler returns every chunk untouched."""
        return self._tau_s > 0.0

    def reset(self) -> None:
        """
        Drop every buffered chunk.

        Called at episode boundaries. The arm jumps back to a start pose between episodes, so a
        prediction from the previous one describes a different situation entirely and blending it
        into the first chunk of the new episode would drag the target backwards.
        """
        self._buffer.clear()

    def blend(self, t_obs: float, actions: np.ndarray) -> np.ndarray:
        """
        Add a chunk to the buffer and return it blended with the overlapping older ones.

        :param t_obs: absolute instant the chunk's observation was captured, seconds. Action ``i``
            of ``actions`` targets ``t_obs + i * action_dt``.
        :param actions: ``(n, 8)`` chunk as returned by the policy.
        :return: ``(n, 8)`` blended chunk on the same time grid. The newest chunk is returned
            unchanged when it is the only one that covers an instant, which includes every
            instant past the end of the previous chunks and the whole first chunk of an episode.
        """
        if not self.enabled or len(actions) == 0:
            return actions

        actions = _canonicalise_quaternions(np.asarray(actions, dtype=float))
        previous = list(self._buffer)
        self._buffer.append((t_obs, actions))
        if not previous:
            return actions

        query_t = t_obs + np.arange(len(actions)) * self._action_dt
        # The newest chunk seeds the accumulator at full weight, and doubles as the reference the
        # older chunks' quaternions are aligned against.
        weight = np.ones(len(actions))
        acc = actions.copy()

        for t_k, actions_k in previous:
            age_s = max(0.0, t_obs - t_k)
            w = float(np.exp(-age_s / self._tau_s))
            if w <= 0.0:
                continue
            sample_t = t_k + np.arange(len(actions_k)) * self._action_dt
            # Never extrapolate: past its last waypoint a chunk has no opinion, and np.interp
            # would silently hold the endpoint as if it did. The tolerance is because both grids
            # are built by accumulating action_dt, so a query that lands exactly on the old
            # chunk's last waypoint differs from it in the last bits — without it, whether that
            # waypoint gets blended is decided by float rounding rather than by anything real.
            edge = 1e-6 * self._action_dt
            covered = (query_t >= sample_t[0] - edge) & (query_t <= sample_t[-1] + edge)
            if not covered.any():
                continue

            contribution = np.empty((int(covered.sum()), actions.shape[1]))
            for col in range(actions.shape[1]):
                contribution[:, col] = np.interp(query_t[covered], sample_t, actions_k[:, col])

            # Align to the newest chunk's hemisphere before summing, or two quaternions naming the
            # same rotation with opposite signs would cancel instead of reinforcing.
            dots = np.einsum('ij,ij->i', contribution[:, QUAT], actions[covered][:, QUAT])
            contribution[dots < 0.0, QUAT] *= -1.0

            acc[covered] += w * contribution
            weight[covered] += w

        blended = acc / weight[:, None]
        # nlerp: the averaged quaternion is no longer unit-length. Valid here because successive
        # predictions differ by small rotations; a degenerate norm falls back to the newest chunk
        # rather than emitting a quaternion that is not a rotation.
        norms = np.linalg.norm(blended[:, QUAT], axis=1)
        degenerate = norms < 1e-9
        blended[~degenerate, QUAT] /= norms[~degenerate, None]
        blended[degenerate, QUAT] = actions[degenerate][:, QUAT]
        return blended
