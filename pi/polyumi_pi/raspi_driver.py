"""Driver for the RaspiAudio ULTRA++ audio HAT and its GPIO peripherals."""

import asyncio
import logging

log = logging.getLogger('raspi_driver')


class RaspiDriver:
    """Manages the RaspiAudio ULTRA++ HAT and its exposed GPIO peripherals."""

    BUTTON_PIN = 23

    def __init__(self, bounce_time_ms: int = 50) -> None:
        """Initialize GPIO button on pin 23."""
        from gpiozero import Button

        self._button = Button(self.BUTTON_PIN, bounce_time=bounce_time_ms / 1000)

    async def wait_for_press(self) -> None:
        """Wait asynchronously for a single button press."""
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        self._button.when_pressed = lambda: loop.call_soon_threadsafe(event.set)
        try:
            await event.wait()
        finally:
            self._button.when_pressed = None

    def close(self) -> None:
        """Release GPIO resources."""
        self._button.close()
