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

- Run `./deploy.sh <hostname>` from repo root — deploys code, creates the Pi venv, and applies the ALSA preset
- Point the Pi's clock at your ROS host with chrony (step 6) — lab-specific, and the stream timestamps are meaningless without it
- Pair the GoPro with `polyumi_pi scan-gopro`
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

### 5. Deploy code and ALSA preset

From your PC, run deploy.sh once the Pi is reachable over SSH:

```bash
./deploy.sh pi@polyumi-pi.local
```

This rsyncs the Pi code, sets up the `.venv`, installs Python packages, and applies the ALSA mixer preset for the contact mic.

### 6. Clock sync — point the Pi at your ROS host

**This step is specific to your lab's network, and nothing in this repo can do it for you.**

Everything the Pi streams — camera frames and audio chunks alike — is stamped in epoch
nanoseconds at the capture instant (the `timestamp_ns` contract in `camera_frame.proto` and
`audio_chunk.proto`). Those stamps become ROS message headers verbatim on the receiving host, so
the Pi's clock and that host's clock have to agree. If they drift, nothing errors: the frames
keep arriving and every latency computed from them is quietly wrong by the drift.

Sync the Pi **to the machine running the ROS nodes**, not to a public pool. That host is the one
whose clock the timestamps are compared against, so agreeing with it matters more than either
machine agreeing with UTC. In this lab that host is `lamb`; yours will be something else.
Substitute your own hostnames throughout.

The stock image does not do this for you. Raspberry Pi OS ships `systemd-timesyncd`, which syncs to a public pool over SNTP and is not configurable in the same way chrony is.
chrony against the ROS host replaces it and gets this to sub-millisecond:

```bash
# On the Pi — chrony is NOT installed by default; installing it displaces systemd-timesyncd:
sudo apt install -y chrony

# Sync to the ROS host, preferring it over anything else configured:
echo 'server lamb iburst prefer' | sudo tee /etc/chrony/conf.d/ros-host-time.conf
sudo systemctl restart chrony

# On the ROS host — serve time to the Pi's subnet:
echo 'allow 10.106.0.0/16' | sudo tee /etc/chrony/conf.d/allow-pi.conf
sudo systemctl restart chrony
```

If the ROS host is not itself internet-synced, add `local stratum 8` to its drop-in so it will
serve its own clock rather than refusing to answer.

Verify from your PC — the `^*` marks the source actually selected, and the offset should be
sub-millisecond:

```bash
ssh polyumi-pi chronyc sources    # expect: ^* lamb
ssh polyumi-pi chronyc tracking   # expect: Leap status : Normal
```

`./deploy.sh` runs that `chronyc tracking` check on every deploy and warns if the Pi has no
synchronised source. It only ever warns — the server address above is yours to choose.

If the Pi comes up while the ROS host is unreachable, chrony falls back to the Pi's own clock and
may take a long time to converge once the link returns. `ssh polyumi-pi 'sudo chronyc makestep'`
re-steps it immediately.

**Recommended: connect pi<->PC over USB gadget wired connection at inference rather than WiFi** for better timesync performance.

### 7. Validate audio and PWM

```bash
# Audio HAT — expect a 5-second recording with no errors:
arecord -D hw:wm8960soundcard -r 48000 -f S16_LE -c 2 -d 5 test.wav

# Hardware PWM — expect pwm_bcm2835 in the output:
lsmod | grep pwm
```

### 8. (GRIPPER ONLY) Pair the GoPro and enable the autostart service

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

After the reboot the service comes up automatically. **Confirm it's ready to record by checking that the red LED on the audio HAT is lit solid** — that's the indicator wired to GPIO25, and it means `start-scene` is running and waiting for a button press. If the LED is off:

- `tail -f /var/log/polyumi-pi.log` — application output (Python logging goes here, since the unit redirects stdout/stderr to this file)
- `journalctl -u polyumi-pi` — systemd lifecycle events only (start, crash, restart counts)

## Reference

- [Raspberry Pi OS configuration docs](https://www.raspberrypi.com/documentation/computers/configuration.html) (search "cloud-init")
- [cloud-init user-data examples](https://cloudinit.readthedocs.io/en/latest/reference/examples.html)
- [network-config format (Netplan)](https://netplan.readthedocs.io/en/stable/reference/)
