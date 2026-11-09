# Camera setup on Windows with WSL

Universal guide: it applies to anyone with Windows and WSL2 who
wants to run Draw with their own webcam. No step depends on
a specific machine, camera, or manufacturer.

## Requirements

- Windows 10 or Windows 11.
- WSL2 installed and updated (`wsl --update` in PowerShell).
- A USB webcam or the notebook's built-in camera.
- The Draw repository cloned inside WSL.

## One-command installation

From inside WSL, at the project root:

```bash
bash scripts/setup_camera_wsl.sh
```

That is all. The script asks for Windows administrator
confirmations in some steps (that is UAC opening). Click **Yes**. At
the end, it runs Draw's own acceptance test and prints the verdict.

Useful options:

- `bash scripts/setup_camera_wsl.sh --busid 1-4` points to a
  specific camera by its BUSID in `usbipd list`.
- `bash scripts/setup_camera_wsl.sh --dry-run` only shows what the
  script would do, without executing anything.

After restarting WSL, run the script again. Only the attach
needs to be redone.

## What the script does, and why

1. **Checks WSL.** The `vhci_hcd` (virtual USB) and
   `uvcvideo` (camera) modules need to exist in the kernel. Without
   them, nothing works. It installs `usbip` and `lsusb` if missing.
2. **Checks Windows.** `usbipd-win` needs to be installed.
   If it is not, the script installs it via `winget` and prints the
   version found.
3. **Finds the camera.** It reads `usbipd list`. If the name is
   ambiguous, it confirms via the Windows Camera and Image classes.
   With a single camera, it chooses on its own. With several, it asks
   by number. With none, it stops with a clear message.
4. **Shares and attaches.** It does the camera's `bind` and `attach`
   into WSL and clears old attach records.
5. **Checks the result.** It waits for the video device to appear
   in WSL and fixes the permission: whoever runs Draw needs to be in
   the `video` group.
6. **Runs the acceptance test.** `python3 main.py --check` says whether
   Draw really sees and opens the camera.

### Why Windows sometimes holds the camera (the audio veto)

Many webcams are composite devices: on a single cable they have the
video function and the microphone function. Windows audio services
keep the microphone function open all the time. When
usbipd tries to take the whole device to WSL, Windows
refuses the removal because of that open function. The error appears
as **Device busy**.

When that happens, the script reads usbipd-win's own
event log. The log names the exact function that vetoed the removal.
Then:

- If the function is **audio**, the script stops the Windows audio
  services, attaches the camera, tries to disable only that
  microphone function (the video one stays intact) and turns the audio
  services back on, checking whether the camera survived with audio
  back on. With the function disabled, later runs do not even need
  to stop the audio.
- If the function is **video**, the script stops the
  FrameServer service, which is the one that usually holds cameras on
  Windows, and tries the attach again. A video function is never
  disabled.
- If the veto names an **incompatible USB filter** (network capture
  drivers, for example), the script names the filter and asks
  for the uninstallation of the product that owns it. There is no safe
  fix by script in that case.

## Known problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No camera found | Camera disconnected, turned off, or in exclusive use by another application | Connect the camera, close applications using it, and run the script again. Or find the BUSID with `usbipd list` in PowerShell and run with `--busid <BUSID>`. |
| Device busy that survives the script | A component the script does not know is holding the camera | The script prints a diagnostic block at the end of execution. Copy the whole block and send it to whoever will help you. It names the vetoing component. |
| Permission denied when opening the camera in Draw | Your WSL user is not yet in the `video` group in this session | The script already adds the user to the group. In the current session, run `sg video -c "python3 main.py"`. In later sessions, the permission already applies. |
| The camera opens, but delivers no frames | Video format too heavy for virtual USB | Update the repository (`git pull`). Current Draw already requests MJPEG at 1280x720 for the local camera, which is the format that works well over usbipd. |
| Windows left without sound after setup | Audio services were stopped during attach and did not return | In PowerShell as administrator: `Start-Service Audiosrv`. The script turns audio back on by itself when it was the one that stopped it. If turning audio back on drops the camera, run the script again. |
| `vhci_hcd` or `uvcvideo` module error missing | Old WSL kernel or one without camera support | Run `wsl --update` in PowerShell, restart WSL, and try again. If it persists, save the script's diagnostic block for support. |

## No bundled capture fallback

The untested ffmpeg capture fallback was removed; usbipd is the supported
Windows/WSL route. [`bridge/`](../bridge/) contains an experimental
timestamped helper protocol, not a working alternative capture setup.
Draw still accepts an existing MJPEG stream with `--source URL`, but the
repository does not provide a Windows stream server.
