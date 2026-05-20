"""Driver for the RaspiAudio ULTRA++ audio HAT and its GPIO peripherals."""

import asyncio
import enum
import logging

from polyumi_pi.constants import BUTTON_PIN, INDICATOR_PIN

log = logging.getLogger('raspi_driver')


class IndicatorState(enum.Enum):
    """States for the GPIO indicator LED."""

    INACTIVE = enum.auto()
    READY = enum.auto()
    AWAITING_ESYNC = enum.auto()
    RECORDING = enum.auto()


class RaspiDriver:
    """Manages the RaspiAudio ULTRA++ HAT and its exposed GPIO peripherals."""

    def __init__(self, bounce_time_ms: int = 50) -> None:
        """Initialize GPIO button on pin 23 and indicator LED on pin 25."""
        from gpiozero import PWMLED, Button

        self._button = Button(BUTTON_PIN, bounce_time=bounce_time_ms / 1000)
        self._indicator = PWMLED(INDICATOR_PIN)

    async def wait_for_press(self) -> None:
        """Wait asynchronously for a single button press."""
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        self._button.when_pressed = lambda: loop.call_soon_threadsafe(event.set)
        try:
            await event.wait()
        finally:
            self._button.when_pressed = None

    def set_indicator(self, state: IndicatorState) -> None:
        """Set the indicator LED state."""
        match state:
            case IndicatorState.INACTIVE:
                self._indicator.off()
            case IndicatorState.READY:
                self._indicator.on()
            case IndicatorState.AWAITING_ESYNC:
                self._indicator.blink(on_time=0.2, off_time=0.2)
            case IndicatorState.RECORDING:
                self._indicator.pulse()
            case _:
                raise ValueError(f'Invalid IndicatorState: {state}')

    def close(self) -> None:
        """Release GPIO resources."""
        self._button.close()
        self._indicator.close()
