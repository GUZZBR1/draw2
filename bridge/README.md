# Windows capture bridge (fallback route)

This folder is the camera's plan B. The main route remains the
webcam attached to WSL via usbipd, with
[`scripts/setup_camera_wsl.sh`](../scripts/setup_camera_wsl.sh). Use the
bridge only if that route fails on your machine.

## How it works

A single `ffmpeg.exe` file runs on Windows, reads the webcam via
DirectShow and serves the video as an MJPEG stream at
`http://0.0.0.0:8080/video`. Draw keeps running on WSL and consumes
this stream via the `--source` option, which it already accepts. Nothing else goes
to Windows: no Python, no libraries, just the standalone `ffmpeg.exe`
next to `capture_windows.bat`.

## Before you start

- The camera **must not be attached to WSL** while the bridge is running.
  If you attached it with usbipd, release it first, in PowerShell:
  `usbipd detach --busid <BUSID>` (the same BUSID from `usbipd list`).
  A camera attached to WSL disappears from Windows, and ffmpeg cannot find it.
- Windows audio services must be running normally
  on this route. If the usbipd route stopped them, restart them with
  `Start-Service Audiosrv` in PowerShell as administrator.

## Step 1: download ffmpeg

Download a Windows build of ffmpeg and extract only `ffmpeg.exe`
into this `bridge` folder, on the Windows side of the disk:

- gyan.dev: https://www.gyan.dev/ffmpeg/builds/ ("release
  full" package; `ffmpeg.exe` is in the `bin` folder of the extracted archive), or
- BtbN builds on GitHub: package `ffmpeg-master-latest-win64-gpl`
  (`ffmpeg.exe` is also in the `bin` folder).

`capture_windows.bat` expects `ffmpeg.exe` in the same folder as
itself. Without it, it stops and shows this same warning.

## Step 2: check the camera name

In a prompt inside this folder, list the video devices:

```bat
ffmpeg.exe -list_devices true -f dshow -i dummy
```

The camera name appears in the list. The `.bat` default is
`1200W Camera`. If the list shows another name, open
`capture_windows.bat` in a text editor and change the value of the
`CAM` variable, in the block marked CONFIGURACAO, right at the top.

## Step 3: run the bridge

Run `capture_windows.bat` with a double-click or via the prompt. On
first run, Windows shows the Firewall warning asking
permission for ffmpeg on the network: allow on **private networks**.
The window stays open serving the video. Closing the window stops the bridge.

## In WSL: connect Draw

Draw reaches Windows via the IP that appears as `nameserver` in
WSL's `/etc/resolv.conf`:

```bash
grep nameserver /etc/resolv.conf
```

With that IP, run Draw pointing to the bridge stream:

```bash
python3 main.py --source http://<IP-DO-WINDOWS>:8080/video
```

Done: capture happens on Windows and all processing, on
WSL, as in the rest of the project.
