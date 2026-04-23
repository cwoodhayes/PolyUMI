"""
Thin async context-manager wrapper around WirelessGoPro.

Provides the three operations needed by PolyUMI:
  - set_timestamp  (defaults to current system time)
  - start_recording
  - stop_recording

Usage::

    async with GoProWrapper('7444') as gopro:
        await gopro.set_timestamp()
        await gopro.start_recording()
        ...
        await gopro.stop_recording()
"""

import os
from datetime import datetime
from types import TracebackType

# open_gopro's WiFi driver reads LANG at import time; BLE-only mode still
# triggers the import, so set it before the first import of the package.
os.environ.setdefault('LANG', 'en_US.UTF-8')

from open_gopro import WirelessGoPro
from open_gopro.models import constants, proto


class GoProWrapper:
    def __init__(self, identifier: str) -> None:
        self._identifier = identifier
        self._gopro: WirelessGoPro | None = None

    async def __aenter__(self) -> 'GoProWrapper':
        self._gopro = WirelessGoPro(
            self._identifier,
            interfaces={WirelessGoPro.Interface.BLE},
            # Suppresses a spurious sudo prompt from the WiFi adapter __init__,
            # which runs even in BLE-only mode. Any non-empty string works on
            # systems with passwordless sudo (as configured by cloud-init on the Pi).
            host_sudo_password='unused',
        )
        await self._gopro.__aenter__()
        await self._gopro.is_ready
        await self._gopro.ble_command.set_camera_control(
            camera_control_status=proto.EnumCameraControlStatus.CAMERA_EXTERNAL_CONTROL
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._gopro is not None:
            await self._gopro.__aexit__(exc_type, exc_val, exc_tb)
            self._gopro = None

    def _require_connected(self) -> WirelessGoPro:
        if self._gopro is None:
            raise RuntimeError('GoProWrapper must be used as an async context manager.')
        return self._gopro

    async def set_timestamp(self, dt: datetime | None = None) -> None:
        """Sync the GoPro clock. Uses the current system time when *dt* is None."""
        gopro = self._require_connected()
        dt = dt or datetime.now()
        tz_offset = dt.astimezone().utcoffset()
        int_offset = int(tz_offset.total_seconds() // 3600) if tz_offset is not None else 0
        is_dst = bool(dt.astimezone().dst())
        await gopro.ble_command.set_date_time_tz_dst(
            date_time=dt,
            tz_offset=int_offset,
            is_dst=is_dst,
        )

    async def start_recording(self) -> None:
        """Start GoPro video recording."""
        gopro = self._require_connected()
        await gopro.ble_command.set_shutter(shutter=constants.Toggle.ENABLE)

    async def stop_recording(self) -> None:
        """Stop GoPro video recording."""
        gopro = self._require_connected()
        await gopro.ble_command.set_shutter(shutter=constants.Toggle.DISABLE)
