"""Manages the LED strip that lights the sensor surface."""

import rpi_hardware_pwm


class LEDManager:
    """Manages the LED strip that lights the sensor surface."""

    # transistor controlling led strip is connected here
    # pin 12 on the actual header, which is PWM channel 0 on the BCM2710
    GPIO_PIN = 18
    PWM_CHANNEL = 0

    def __init__(self) -> None:
        """Initialize the LED manager."""
        self.pwm = rpi_hardware_pwm.HardwarePWM(
            self.PWM_CHANNEL, hz=1000, chip=0
        )
        self.pwm.start(0)

    def set_brightness(self, brightness: float) -> None:
        """
        Set the brightness of the LED strip.

        Args:
            brightness: Brightness in [0.0, 1.0].

        """
        duty_cycle = int(brightness * 100)
        self.pwm.change_duty_cycle(duty_cycle)

    def __del__(self) -> None:
        """Clean up the PWM on deletion."""
        # is ok if this doesn't happen, which is why I'm not using
        # a context manager
        self.pwm.stop()
