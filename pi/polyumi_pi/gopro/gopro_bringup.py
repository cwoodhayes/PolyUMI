# video.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# This copyright was auto-generated on Wed, Sep  1, 2021  5:05:46 PM

"""Entrypoint for taking a video demo."""

import argparse
import asyncio
from pathlib import Path

from open_gopro import WirelessGoPro
from open_gopro.models import constants, proto
from open_gopro.util import add_cli_args_and_parse
from open_gopro.util.logger import setup_logging
from rich.console import Console

console = Console()


async def main() -> None:
    logger = setup_logging(__name__)
    gopro: WirelessGoPro | None = None

    identifier = 'GoPro 1112'
    record_time = 3

    try:
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

            await gopro.ble_command.set_shutter(shutter=constants.Toggle.ENABLE)
            await asyncio.sleep(2)
            await gopro.ble_command.set_shutter(shutter=constants.Toggle.DISABLE)

            video = await gopro.ble_command.get_last_captured_media()
            logger.info(f'Video captured: {video}')

    except Exception as e:  # pylint: disable = broad-except
        logger.error(repr(e))

    if gopro:
        await gopro.close()
    console.print('Exiting...')


if __name__ == '__main__':
    asyncio.run(main())
