# Windows helper protocol

Experimental protocol between a Windows-side capture helper and Draw
running in WSL, for frame capture with trustworthy timestamps — ground
work for the future cursor bridge (VISION.md, layer 3). The app does
not use it yet; nothing in the default path depends on this folder.

- `PROTOCOL.md` — the wire contract: a stream of records (28-byte
  binary header + JPEG payload) over TCP or in a recorded file; the
  consumer cannot tell the two sources apart.
- `windows_helper.py` — an untested Windows-side sender sketch;
  capture and JPEG encoding are stubs, not a working capture fallback.
- `tools/helper_receiver.py` — the WSL-side receiver and recorder
  (`--selfcheck`, `--record`, `--dump`).

If the helper is not running, nothing changes: Draw captures from the
webcam attached to WSL as usual (see
[`scripts/setup_camera_wsl.sh`](../scripts/setup_camera_wsl.sh)).
