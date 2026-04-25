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

Pass *mac_address* to skip BLE scanning and connect directly by MAC address,
which is significantly faster when the target device is already known::

    async with GoProWrapper('7444', mac_address='XX:XX:XX:XX:XX:XX') as gopro:
        ...
"""

from datetime import datetime
from types import TracebackType
from typing import Any

# Lazily populated the first time GoProWrapper is opened with a mac_address.
_fast_ble_controller_cls: type | None = None


def _get_fast_ble_controller() -> type:
    """Return (lazily) a BleakWrapperController subclass whose scan() uses find_device_by_address()."""
    global _fast_ble_controller_cls
    if _fast_ble_controller_cls is not None:
        return _fast_ble_controller_cls

    from open_gopro.network.ble import FailedToFindDevice
    from open_gopro.network.ble.adapters.bleak_wrapper import BleakWrapperController

    class _FastBleController(BleakWrapperController):
        # Set to a MAC address string before __aenter__ to skip scanning.
        _target_mac: str | None = None

        async def scan(self, token: Any, timeout: int = 5, service_uuids: Any = None) -> Any:
            if type(self)._target_mac is not None:
                import bleak

                device = await bleak.BleakScanner.find_device_by_address(
                    type(self)._target_mac, timeout=timeout  # type: ignore
                )
                if device is not None:
                    return device
                raise FailedToFindDevice
            return await super().scan(token, timeout, service_uuids)

    _fast_ble_controller_cls = _FastBleController
    return _fast_ble_controller_cls


class GoProWrapper:
    """Async context manager for BLE-only GoPro control."""

    def __init__(self, identifier: str, mac_address: str | None = None) -> None:
        """
        Initialize GoProWrapper.

        Args:
            identifier: Last four digits of the GoPro's serial number.
            mac_address: BLE MAC address. When provided, BLE scanning is skipped
                and the device is located directly, saving several seconds.

        """
        from open_gopro import WirelessGoPro
        from open_gopro.models import constants, proto

        self._WirelessGoPro = WirelessGoPro
        self._constants = constants
        self._proto = proto
        self._identifier = identifier
        self._mac_address = mac_address
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

            kwargs: dict[str, Any] = {
                'interfaces': {self._WirelessGoPro.Interface.BLE},
                # Suppresses a spurious sudo prompt from the WiFi adapter __init__,
                # which runs even in BLE-only mode. Any non-empty string works on
                # systems with passwordless sudo (as configured by cloud-init on the Pi).
                'host_sudo_password': 'unused',
            }

            if self._mac_address:
                fast_cls = _get_fast_ble_controller()
                fast_cls._target_mac = self._mac_address  # type: ignore[attr-defined]
                kwargs['ble_adapter'] = fast_cls
            elif _fast_ble_controller_cls is not None:
                # Clear any stale MAC left by a previous connection so a later
                # mac_address=None call doesn't silently take the fast path.
                _fast_ble_controller_cls._target_mac = None  # type: ignore[attr-defined]

            self._gopro = self._WirelessGoPro(self._identifier, **kwargs)
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
        if _fast_ble_controller_cls is not None:
            _fast_ble_controller_cls._target_mac = None  # type: ignore[attr-defined]
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
