"""
Module for managing integration with the optitrack system.

This is optionally used as a secondary localization technique in addition to SLAM.

The optitrack system has an e-sync box, which is connected to the cameras.
In motive, you can set this box to latch the output of the box high when recording,
and low when not recording.
In order to use the functionality of this module, enable that setting, then:

1. POWER OFF THE POLYUMI
2. Connect esync's ground to the pi's ground (pin 39), and esync's signal to pin 37 (using a pigtail bnc cable)
3. power on the polyumi
4. use the start-scene command with the --optitrack flag enabled
5. When the polyumi indicates ready (with a double beep + blinking light), start recording on the esync
6. Polyumi should output a single beep, and the LED should become solid.
7. ORDER IS VERY IMPORTANT: unplug pin 37 (optitrack signal), and ONLY THEN unplug optitrack ground (39).
Can also just unplug bnc side first, since the connector geometry enforces this ordering.

You are now ready to start recording sessions, starting with a mapping session.
"""

import asyncio
import logging
from datetime import datetime, timezone

from polyumi_pi import sync_chirp
from polyumi_pi.constants import AUDIO_DEVICE, AUDIO_OUTPUT_SAMPLE_RATE, ESYNC_PIN
from polyumi_pi.files.scene import SceneFiles
from polyumi_pi.raspi_driver import IndicatorState, RaspiDriver

log = logging.getLogger('pi_optitrack')


async def await_optitrack_esync(
    scene: SceneFiles, hat: RaspiDriver
) -> None:
    """
    Wait for the esync start recording signal to arrive.

    Control the indicator light to make the user aware of the status, and write
    the esync start time to files when it does arrive.
    """
    from gpiozero import DigitalInputDevice

    esync = DigitalInputDevice(ESYNC_PIN, pull_up=False)
    try:
        log.info(f'Awaiting OptiTrack e-sync on GPIO {ESYNC_PIN}...')
        hat.set_indicator(IndicatorState.AWAITING_ESYNC)
        sync_chirp.beep(2, AUDIO_OUTPUT_SAMPLE_RATE, device=AUDIO_DEVICE)

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        esync.when_activated = lambda: loop.call_soon_threadsafe(event.set)
        try:
            await event.wait()
        finally:
            esync.when_activated = None

        esync_time = datetime.now(timezone.utc)
        hat.set_indicator(IndicatorState.READY)
        sync_chirp.beep(1, AUDIO_OUTPUT_SAMPLE_RATE, device=AUDIO_DEVICE)
        scene.optitrack_start_time = esync_time
        log.info(f'OptiTrack e-sync received at {esync_time.isoformat()}')
    finally:
        esync.close()
