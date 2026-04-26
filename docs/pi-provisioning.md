# Raspberry Pi Provisioning

This project uses [cloud-init](https://cloudinit.readthedocs.io/) to automate first-boot Pi setup. The version-controlled template lives at [`infra/cloud-init/user-data.example`](../infra/cloud-init/user-data.example); you create your local [`infra/cloud-init/user-data`](../infra/cloud-init/user-data) from it before flashing. Flashing a new Pi takes about 5-10 minutes of mostly hands-off time.

## What cloud-init handles automatically

- Hostname, user account, SSH authorized keys, passwordless sudo
- Sets up password-based login + shell access using monitor and keyboard (but disables password login over ssh) -- `password = "polyumi!"`
- `apt` packages & upgrades
- Hardware PWM setup
- Audio HAT DKMS driver setup (Waveshare installer)
- `uv` installed for the `pi` user
- Other miscellaneous changes -- see `infra/cloud-init/user-data` for details

## What you do manually afterwards

- Run `./deploy.sh` from repo root to push your local working-tree code (cloud-init bootstraps from `main`; `deploy.sh` overlays uncommitted changes)
- Pair the GoPro with `polyumi_pi.main scan-gopro`
- `sudo systemctl enable polyumi-pi` and reboot to start the autostart service

See [README.md](/README.md) for more on next steps.

## Step-by-step

### Prerequisites

- SD card (≥16 GB recommended)
- Your SSH public key (`cat ~/.ssh/id_ed25519.pub`)
- WiFi credentials for the network the Pi will join

### 1. Create your local config files

Both `user-data` and `network-config` are gitignored (they contain personal details). Create them from the committed examples:

```bash
cp infra/cloud-init/user-data.example infra/cloud-init/user-data
cp infra/cloud-init/network-config.example infra/cloud-init/network-config
```

Then edit both files in your IDE before copying to the SD card:

- **`user-data`**: replace the SSH key placeholder with your public key (`cat ~/.ssh/id_ed25519.pub`)
- **`network-config`**: fill in your WiFi SSID, password, and `regulatory-domain` country code

### 2. Flash Raspberry Pi OS

Download [RPi Imager](https://www.raspberrypi.com/software/), connect your SD card to your PC, run the imager, and navigate through the menus to apply the following settings:
- Device: Raspberry Pi Zero 2W
- OS: Raspberry Pi OS (other) -> Raspberry Pi OS Lite (**Debian Trixie port, 2025**)
- Then on next section ("Customization" -- the first page at time of writing is "Enter your hostname"), hit "SKIP CUSTOMIZATION" in the bottom left corner. The cloud-init workflow handles all OS-level configuration.

### 3. Copy cloud-init files to the boot partition

After flashing, the SD card's `bootfs` partition auto-mounts. On Linux it's typically at `/media/$USER/bootfs`; on macOS it's `/Volumes/bootfs`. (If it doesn't show up, unplug and plug back in the SD, then mount the "bootfs" drive in Nautilus or Finder or the command line.)

Copy your locally-configured files to the SD card:

```bash
# from the repo root:
cp infra/cloud-init/user-data /media/$USER/bootfs/
cp infra/cloud-init/network-config /media/$USER/bootfs/
touch /media/$USER/bootfs/meta-data        # required by cloud-init, can be empty
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

### 7. (GRIPPER ONLY) Pair the GoPro and enable the autostart service

**DO NOT PERFORM THIS STEP IF THIS PI IS FOR THE END-EFFECTOR.** We don't need a startup service there since we don't record, we only stream.

Cloud-init installs `polyumi-pi.service` but leaves it disabled, because `start-scene` needs a saved GoPro pairing to launch. 

First, turn on the GoPro attached to the UMI, and then run the pairing command from the pi:

```bash
ssh pi@polyumi-pi.local
cd ~/PolyUMI/pi
.venv/bin/python -m polyumi_pi.main scan-gopro   # follow prompts to pick your GoPro

sudo systemctl enable polyumi-pi
sudo reboot
```

After the reboot the service comes up automatically. **Confirm it's ready to record by checking that the red LED on the audio HAT is lit solid** — that's the indicator wired to GPIO25, and it means `start-scene` is running and waiting for a button press. If the LED is off, check `journalctl -u polyumi-pi` or `/var/log/polyumi-pi.log`.

## Reference

- [Raspberry Pi OS configuration docs](https://www.raspberrypi.com/documentation/computers/configuration.html) (search "cloud-init")
- [cloud-init user-data examples](https://cloudinit.readthedocs.io/en/latest/reference/examples.html)
- [network-config format (Netplan)](https://netplan.readthedocs.io/en/stable/reference/)
