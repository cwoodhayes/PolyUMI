"""Hardware constants for the PolyUMI Pi."""

# GPIO pin numbers (BCM numbering)
BUTTON_PIN = 23
INDICATOR_PIN = 25
ESYNC_PIN = 26  # rpi header pin 37. connect gnd to pin 39 right next door.

# Audio hardware
AUDIO_DEVICE = 'wm8960-soundcard'
AUDIO_OUTPUT_SAMPLE_RATE = 44100
