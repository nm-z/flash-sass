# ops: files that live on archy outside the repo

Copies of the host-side files the engine's host relies on. Nothing here is built; install by hand.

| File | Installed at | What it does |
|---|---|---|
| `gpu-application-clocks.service` | `/etc/systemd/system/` | persistence mode, application clocks at the maximum (2505/1177 MHz on dies 0-5, 875 on 6-7) and auto boost off on every die; `-lgc` locked clocks are not supported on the M60 |
| `thermalwd.sh` | `~/thermalwd.sh` (run it in the background) | stops `flashd` and `flashcore` if any of dies 0-5 reaches 92 C; log `~/flash-watchdog.log` |
| `systemd-user/persuader-lfm.cpu-only.conf` | `~/.config/systemd/user/persuader-lfm.service.d/cpu-only.conf` | LFM2 on the CPU (4 threads), off die 6 (the MTP model goes there) |
| `systemd-user/jev-files-embed.cpu-only.conf` | `~/.config/systemd/user/jev-files-embed@.service.d/cpu-only.conf` | Qwen3-Embedding on the CPU (4 threads), off die 7 |
| `systemd-user/sol-stop-hook-embed-m60.no-autostart.conf` | `~/.config/systemd/user/sol-stop-hook-embed-m60.service.d/00-no-autostart.conf` | the GPU embedding copy starts only if `~/.config/embed-m60.allow` exists |

The slot-4 gen2 pin is `kern/pcie-slot4-gen2.sh` and `kern/pcie-slot4-gen2.service`; install those under `/usr/local/sbin` and `/etc/systemd/system`.

After copying a drop-in: `systemctl --user daemon-reload`, then restart the service. To undo, delete the drop-in and do the same.
