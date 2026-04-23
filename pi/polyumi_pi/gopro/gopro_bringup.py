# video.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# This copyright was auto-generated on Wed, Sep  1, 2021  5:05:46 PM

"""Entrypoint for taking a video demo."""

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
    # identifier = 'GoPro 1112'

    # gripper gopro
    identifier = 'GoPro 7444'

    try:
        logger.info(f'Connecting to {identifier}...')
        async with WirelessGoPro(
            identifier,
            interfaces={
                WirelessGoPro.Interface.BLE,
            },
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
