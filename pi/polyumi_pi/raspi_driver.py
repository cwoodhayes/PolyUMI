"""Driver for the RaspiAudio ULTRA++ audio HAT and its GPIO peripherals."""

import asyncio
import enum
import logging
import time

from polyumi_pi.constants import BUTTON_PIN, ESYNC_PIN, INDICATOR_PIN

log = logging.getLogger('raspi_driver')

_GPIOCHIP = 0


class IndicatorState(enum.Enum):
    """States for the GPIO indicator LED."""

    INACTIVE = enum.auto()
    READY = enum.auto()
    AWAITING_ESYNC = enum.auto()
    RECORDING = enum.auto()


class RaspiDriver:
    """Manages the RaspiAudio ULTRA++ HAT and its exposed GPIO peripherals."""

    def __init__(self) -> None:
        """Initialize GPIO pins and indicator LED."""
        # using lgpio here instead of gpiozero
        # because gpiozero wasn't playing nice with async/await
        # and multiple buttons/inputs.

        import lgpio as _lgpio  # type: ignore
        from gpiozero import PWMLED

        self._lgpio = _lgpio
        self._handle = _lgpio.gpiochip_open(_GPIOCHIP)
        _lgpio.gpio_claim_input(self._handle, BUTTON_PIN, _lgpio.SET_PULL_UP)
        _lgpio.gpio_claim_input(self._handle, ESYNC_PIN, _lgpio.SET_PULL_NONE)
        self._indicator = PWMLED(INDICATOR_PIN)

    def get_lgpio_handle(self) -> int:
        """Return the lgpio chip handle."""
        return self._handle

    async def wait_for_press(self) -> None:
        """Wait asynchronously for a single button press."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._poll_button)

    def _poll_button(self) -> None:
        last = self._lgpio.gpio_read(self._handle, BUTTON_PIN)
        while True:
            val = self._lgpio.gpio_read(self._handle, BUTTON_PIN)
            if val == 0 and last == 1:
                return
            last = val
            time.sleep(0.001)

    def wait_for_esync(self) -> None:
        """Block until a rising edge is detected on the esync pin."""
        last = self._lgpio.gpio_read(self._handle, ESYNC_PIN)
        while True:
            val = self._lgpio.gpio_read(self._handle, ESYNC_PIN)
            if val == 1 and last == 0:
                return
            last = val
            # faster poll due to desired accuracy on esync.
            time.sleep(0.0001)

    def set_indicator(self, state: IndicatorState) -> None:
        """Set the indicator LED state."""
        # Stop any running blink/pulse thread before switching state.
        self._indicator.off()
        match state:
            case IndicatorState.INACTIVE:
                pass
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
        self._lgpio.gpiochip_close(self._handle)
        self._indicator.close()
