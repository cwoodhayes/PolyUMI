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

from datetime import datetime
from types import TracebackType
from typing import Any


class GoProWrapper:
    """Async context manager for BLE-only GoPro control."""

    def __init__(self, identifier: str) -> None:
        """
        Initialize GoProWrapper.

        Args:
            identifier: Last four digits of the GoPro's serial number.

        """
        from open_gopro import WirelessGoPro
        from open_gopro.models import constants, proto

        self._WirelessGoPro = WirelessGoPro
        self._constants = constants
        self._proto = proto
        self._identifier = identifier
        self._gopro: Any = None

    async def __aenter__(self) -> 'GoProWrapper':
        import os

        # open_gopro's WiFi adapter calls ensure_us_english() during __init__,
        # even in BLE-only mode, and raises if LANG != en_US.*. Temporarily
        # override just for the duration of WirelessGoPro construction so the
        # rest of the process keeps its original locale.
        _original_lang = os.environ.get('LANG')
        try:
            os.environ['LANG'] = 'en_US.UTF-8'
            self._gopro = self._WirelessGoPro(
                self._identifier,
                interfaces={self._WirelessGoPro.Interface.BLE},
                # Suppresses a spurious sudo prompt from the WiFi adapter __init__,
                # which runs even in BLE-only mode. Any non-empty string works on
                # systems with passwordless sudo (as configured by cloud-init on the Pi).
                host_sudo_password='unused',
            )
        finally:
            if _original_lang is None:
                os.environ.pop('LANG', None)
            else:
                os.environ['LANG'] = _original_lang
        await self._gopro.__aenter__()
        await self._gopro.is_ready
        await self._gopro.ble_command.set_camera_control(
            camera_control_status=self._proto.EnumCameraControlStatus.CAMERA_EXTERNAL_CONTROL
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

    def _require_connected(self) -> Any:
        if self._gopro is None:
            raise RuntimeError('GoProWrapper must be used as an async context manager.')
        return self._gopro

    async def set_timestamp(self, dt: datetime | None = None) -> datetime:
        """
        Sync the GoPro clock. Uses the current system time when *dt* is None.

        Returns:
            The datetime that was sent to the GoPro.

        """
        gopro = self._require_connected()
        dt = (dt or datetime.now()).astimezone()
        tz_offset = dt.utcoffset()
        int_offset = int(tz_offset.total_seconds() // 60) if tz_offset is not None else 0
        is_dst = bool(dt.dst())
        await gopro.ble_command.set_date_time_tz_dst(
            date_time=dt,
            tz_offset=int_offset,
            is_dst=is_dst,
        )
        return dt

    async def start_recording(self) -> None:
        """Start GoPro video recording."""
        gopro = self._require_connected()
        await gopro.ble_command.set_shutter(shutter=self._constants.Toggle.ENABLE)

    async def stop_recording(self) -> None:
        """Stop GoPro video recording."""
        gopro = self._require_connected()
        await gopro.ble_command.set_shutter(shutter=self._constants.Toggle.DISABLE)
