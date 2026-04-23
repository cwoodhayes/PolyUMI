# Raspberry Pi Provisioning

This project uses [cloud-init](https://cloudinit.readthedocs.io/) to automate first-boot Pi setup. The config lives at [`infra/cloud-init/user-data`](../infra/cloud-init/user-data) and is version-controlled alongside the code. Flashing a new Pi takes about 10 minutes of hands-off time.

## What cloud-init handles automatically

- Hostname, user account, SSH authorized keys, passwordless sudo
- Sets up password-based login + shell access using monitor and keyboard (but disables password login over ssh) -- password = "polyumi!
- `apt` packages & upgrades
- Hardware PWM setup
- Audio HAT DKMS driver setup (Waveshare installer)
- `uv` installed for the `pi` user
- Other miscellaneous changes -- see `infra/cloud-init/user-data` for details

## What you do manually afterwards

- Run `./deploy.sh` from repo root to push app code
- Run the following to install Python deps (can't happen before code is deployed)

```bash
    cd ~/pi && uv venv --system-site-packages && uv sync --no-dev
    uv pip install -e ~/polyumi_pi_msgs
    python polyumi_pi/main.py stream
```

See [README.md](/README.md) for more on next steps.

## Step-by-step

### Prerequisites

- SD card (≥16 GB recommended)
- Your SSH public key (`cat ~/.ssh/id_ed25519.pub`)
- WiFi credentials for the network the Pi will join

### 1. Fill in your SSH key

Edit [`infra/cloud-init/user-data`](../infra/cloud-init/user-data) and replace the placeholder:

```yaml
ssh_authorized_keys:
  - ssh-ed25519 AAAA...  # ← replace this line
```

This is the only file you should need to edit before flashing.

### 2. Flash Raspberry Pi OS

Download [RPi Imager](https://www.raspberrypi.com/software/), connect your SD card to your PC, run the imager, and navigate through the menus to apply the following settings:
- Device: Raspberry Pi Zero 2W
- OS: Raspberry Pi OS (other) -> Raspberry Pi OS (Legacy, 64-bit) Lite (Debian Bookworm port)
- Then on next section ("Customization" -- the first page at time of writing is "Enter your hostname"), hit "SKIP CUSTOMIZATION" in the bottom left corner. The cloud-init workflow handles all OS-level configuration.

### 3. Copy cloud-init files to the boot partition

After flashing, the SD card's `bootfs` partition auto-mounts. On Linux it's typically at `/media/$USER/bootfs`; on macOS it's `/Volumes/bootfs`. (If it doesn't show up, mount the "bootfs" drive in Nautilus or Finder or the command line.)

Then copy the following files:

```bash
# from the repo root:
cp infra/cloud-init/user-data /media/$USER/bootfs/
touch /media/$USER/bootfs/meta-data        # required by cloud-init, can be empty

# WiFi credentials (gitignored — do not commit with real values):
cp infra/cloud-init/network-config.example /media/$USER/bootfs/network-config
# edit /media/$USER/bootfs/network-config and fill in your SSID and password
```

Safely eject the SD card.

### 4. Boot and wait

Insert the SD card, power on the Pi, and wait for cloud-init to finish. This takes **5–10 minutes** on first boot (package upgrades + WM8960 DKMS build add time).

Once the Pi is reachable over SSH, you can monitor progress:

```bash
ssh pi@polyumi-pi.local
cloud-init status --wait    # blocks until done, exits 0 on success
```

Full logs are at `/var/log/cloud-init-output.log` if anything goes wrong.

### 5. Validate audio and PWM

```bash
# Audio HAT — expect a 5-second recording with no errors:
arecord -D hw:wm8960soundcard -r 48000 -f S16_LE -c 2 -d 5 test.wav

# Hardware PWM — expect pwm_bcm2835 in the output:
lsmod | grep pwm
```

### 6. Deploy app code and install Python deps

From your dev machine:

```bash
./deploy.sh polyumi-pi.local
```

Then on the Pi:

```bash
cd ~/pi
uv venv --system-site-packages    # picks up system python3-picamera2
uv sync --no-dev
uv pip install -e ~/polyumi_pi_msgs
```

The Pi is ready. Run `python polyumi_pi/main.py stream` to verify.

## Reference

- [Raspberry Pi OS configuration docs](https://www.raspberrypi.com/documentation/computers/configuration.html) (search "cloud-init")
- [cloud-init user-data examples](https://cloudinit.readthedocs.io/en/latest/reference/examples.html)
- [network-config format (Netplan)](https://netplan.readthedocs.io/en/stable/reference/)
