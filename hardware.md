# Hardware on the Intel server — instructions for agents

This document tells an agent which hardware is connected to the team Intel server and how to work with it.

> **Read this document before you install drivers or software for the hardware, or before you configure, calibrate, or control the hardware.** For SSH access to the server, read [ssh.md](ssh.md) first.

## Rules

- The `intel` user can use `sudo` without a password. Use `sudo -n` for steps that need root (for example, to install `udev` rules or `apt` packages). See [ssh.md](ssh.md#sudo).
- The network is not reliable. Start long installs in `tmux` or with `nohup`, and write a log file (see [ssh.md](ssh.md)).

## The server

| Item | Value |
|---|---|
| Computer | Dell XPS 16 (DA16260), hostname `intel-XPS-16-DA16260` |
| CPU / GPU / NPU | Intel Panther Lake, Intel Arc B390 integrated GPU, Intel NPU |
| OS | Ubuntu, kernel 7.0 |
| Network | Wi-Fi only. The laptop has no Ethernet port. |

## Connected hardware

Last check: 2026-11-13.

| Device | USB ID | Where | Linux device | Notes |
|---|---|---|---|---|
| Intel RealSense D415 depth camera | `8086:0ad3` | Hub port 3, USB 3 (5 Gbit/s) | `/dev/video32`–`/dev/video37`. Stable names are in `/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_415_*` | Serial number `318123021715`. Driver: `uvcvideo`. |
| USB-C hub: VIA Labs VL822 and Realtek RTS5411 | `2109:0822`, `2109:2822`, `0bda:0411`, `0bda:5411` | Laptop USB-C port 2 | — | 7 free ports for devices. If a device does not show, make sure that the hub has power. |

### Built-in devices (ignore these for the robot work)

| Device | Notes |
|---|---|
| `Intel IPU7 ISYS Capture` (`/dev/video0`–`/dev/video31`) | Built-in laptop camera pipeline. No sensor driver is attached, so these devices give no image. Do not use them as robot cameras. |
| Synaptics SVP7500 (`06cb:0701`) | Intel USBIO bridge for the built-in camera. It is not a robot device. |

## Arm API service

The service **armd** controls the two SO-101 arms through an HTTP API on the server (`127.0.0.1:8765`). To use, run, or deploy it, read [arm-api.md](arm-api.md). Only one process can open a serial port. Release the arms in armd, or stop armd, before you start LeRobot.

## Check what is connected

Use these read-only commands:

```bash
ssh -o BatchMode=yes intel 'lsusb; lsusb -t'
ssh -o BatchMode=yes intel 'ls -l /dev/serial/by-id /dev/v4l/by-id'
ssh -o BatchMode=yes intel 'journalctl -k --no-pager --since "10 min ago" | grep -E "usb [0-9.-]+:"'
```

To find which device is which, ask the user to connect one device at a time, and read the kernel log after each one.

## Software setup

### Python and LeRobot

- Use the conda environment **`lerobot`** (Python 3.12) in `~/miniforge3`. Run `conda activate lerobot` before you use `pip`.
- **Do not install into the `base` environment.** It has Python 3.14. Packages such as numpy 2.2 have no wheels for Python 3.14, so `pip` tries to compile them. The server has no C compiler, so the install fails with `Unknown compiler(s): [['cc'], ['gcc'], ...]`.
- Install PyTorch from the Intel XPU index, so that it uses the Arc GPU and does not download the large NVIDIA CUDA packages:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
  pip install lerobot
  ```

- An install script is at `~/install_lerobot.sh`, and its log is at `~/install_lerobot.log`.
- To make sure that PyTorch can use the GPU: `python -c "import torch; print(torch.xpu.is_available())"`.

### RealSense camera

- The `uvcvideo` kernel driver gives the basic color and depth video streams through `/dev/video*`.
- For depth data and camera settings, use the `pyrealsense2` Python package (`pip install pyrealsense2` in the `lerobot` environment). LeRobot uses it for RealSense cameras.
- The RealSense tools (`rs-enumerate-devices`, `realsense-viewer`) and RealSense `udev` rules are not installed. To install them, use `sudo -n`.
- Use the serial number `318123021715` to select this camera in LeRobot camera settings.
