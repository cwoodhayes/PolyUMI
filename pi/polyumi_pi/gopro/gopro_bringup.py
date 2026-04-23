"""
GoPro bring-up script.

Connects to a GoPro over BLE, syncs its clock, and fires a short recording.
Originally written to exercise the GoPro integration during initial bring-up;
kept around as a standalone debug tool for diagnosing connection issues without
the rest of the PolyUMI stack running.

The opengopro is a bit finicky & light on documentation, so this is a useful reference.

Usage::

    python gopro_bringup.py

Set ``identifier`` to the last four digits of the target camera's serial number.
"""

import asyncio
import os
from datetime import datetime

# Set locale to en_US for open_gopro WiFi driver compatibility
os.environ['LANG'] = 'en_US.UTF-8'

from open_gopro import WirelessGoPro
from open_gopro.models import constants, proto
from open_gopro.util.logger import setup_logging
from rich.console import Console

console = Console()


async def main() -> None:
    """Bring-up function."""
    logger = setup_logging(__name__)
    gopro: WirelessGoPro | None = None

    # arm gopro
    # identifier = '1112'

    # gripper gopro
    identifier = '7444'

    try:
        logger.info(f'Connecting to {identifier}...')
        async with WirelessGoPro(
            identifier,
            interfaces={
                WirelessGoPro.Interface.BLE,
            },
            # Suppresses a sudo prompt from the wifi adapter's __init__, which runs
            # even in BLE-only mode. Any non-empty string works on systems with
            # passwordless sudo (like our pi, as configured by cloud-init).
            # the library's "validation" succeeds regardless.
            host_sudo_password='unused',
        ) as gopro:
            await gopro.is_ready
            logger.info(f'Connected to GoPro: {gopro.identifier}')

            await gopro.ble_command.set_camera_control(
                camera_control_status=proto.EnumCameraControlStatus.CAMERA_EXTERNAL_CONTROL
            )

            now = datetime.now()
            tz_offset = now.astimezone().utcoffset()
            if tz_offset is not None:
                int_offset = tz_offset.seconds // 3600
            else:
                int_offset = 0
            dst = bool(now.astimezone().dst())
            logger.info(f'Setting GoPro date/time to {now} with tz offset {tz_offset} ({int_offset}) and dst {dst}')
            await gopro.ble_command.set_date_time_tz_dst(date_time=now, tz_offset=int_offset, is_dst=bool(dst))

            await gopro.ble_command.set_shutter(shutter=constants.Toggle.ENABLE)
            await asyncio.sleep(2)
            await gopro.ble_command.set_shutter(shutter=constants.Toggle.DISABLE)

    except Exception as e:  # pylint: disable = broad-except
        logger.exception(e)

    if gopro:
        await gopro.close()
    console.print('Exiting...')


if __name__ == '__main__':
    asyncio.run(main())
