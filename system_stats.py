"""Live Jetson telemetry from `tegrastats`.

Spawns `tegrastats --interval N` once, parses each line, exposes latest as a dict.
"""
import re
import subprocess
import threading
from typing import Optional


_RX_RAM   = re.compile(r"RAM (\d+)/(\d+)MB")
_RX_SWAP  = re.compile(r"SWAP (\d+)/(\d+)MB")
_RX_CPU   = re.compile(r"CPU \[([^\]]+)\]")
_RX_CORE  = re.compile(r"(\d+)%@(\d+)")
_RX_GPU   = re.compile(r"GR3D_FREQ (\d+)%")
_RX_CTEMP = re.compile(r"cpu@([\d.]+)C")
_RX_GTEMP = re.compile(r"gpu@([\d.]+)C")
_RX_TJ    = re.compile(r"tj@([\d.]+)C")
_RX_POWER = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")


def parse_tegrastats(line: str) -> Optional[dict]:
    out: dict = {}
    m = _RX_RAM.search(line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))
    m = _RX_SWAP.search(line)
    if m:
        out["swap_used_mb"] = int(m.group(1))
        out["swap_total_mb"] = int(m.group(2))
    m = _RX_CPU.search(line)
    if m:
        cores = []
        for piece in m.group(1).split(","):
            mc = _RX_CORE.match(piece.strip())
            if mc:
                cores.append({"util": int(mc.group(1)), "freq_mhz": int(mc.group(2))})
            elif "off" in piece.lower():
                cores.append({"util": 0, "freq_mhz": 0, "off": True})
        out["cpu_cores"] = cores
        active = [c for c in cores if not c.get("off")]
        if active:
            out["cpu_avg_util"] = sum(c["util"] for c in active) // len(active)
    m = _RX_GPU.search(line)
    if m: out["gpu_util"] = int(m.group(1))
    m = _RX_CTEMP.search(line)
    if m: out["cpu_temp_c"] = float(m.group(1))
    m = _RX_GTEMP.search(line)
    if m: out["gpu_temp_c"] = float(m.group(1))
    m = _RX_TJ.search(line)
    if m: out["tj_temp_c"] = float(m.group(1))
    m = _RX_POWER.search(line)
    if m:
        out["power_now_mw"] = int(m.group(1))
        out["power_avg_mw"] = int(m.group(2))
    return out if out else None


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 1500):
        self.latest: dict = {}
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            for line in self._proc.stdout:                # type: ignore[union-attr]
                if self._stop.is_set():
                    break
                parsed = parse_tegrastats(line)
                if parsed:
                    with self._lock:
                        self.latest = parsed
        except FileNotFoundError:
            print("[stats] tegrastats not found — telemetry disabled")
        except Exception as e:
            print(f"[stats] error: {e}")

    def get(self) -> dict:
        with self._lock:
            return dict(self.latest)

    def stop(self):
        self._stop.set()
        if self._proc:
            try: self._proc.terminate()
            except Exception: pass
