"""
Process physical footprint (macOS), the number that includes MPS/Metal buffers.

RSS and `ru_maxrss` do not count memory held by the MPS caching allocator: ReDimNet2 on
MPS once reached a 74-85 GB physical footprint while RSS showed 3-8 GB. This module reads
`ri_phys_footprint` / `ri_lifetime_max_phys_footprint` from `proc_pid_rusage` (the same
values as `footprint -p` and `/usr/bin/time -l` "peak memory footprint"), and provides a
guard that long loops call between batches.

Stdlib only. On other platforms every reading is None and the guard is a no-op.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

_RUSAGE_INFO_V4 = 4


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
        "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
        "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
        "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
        "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


_LIB = None


def _lib():
    global _LIB
    if _LIB is None and sys.platform == "darwin":
        _LIB = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    return _LIB


def _info(pid: int | None = None) -> _RUsageInfoV4 | None:
    lib = _lib()
    if lib is None:
        return None
    buf = _RUsageInfoV4()
    if lib.proc_pid_rusage(int(pid or os.getpid()), _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        return None
    return buf


def phys_footprint(pid: int | None = None) -> int | None:
    """Current physical footprint in bytes (None where unsupported)."""
    i = _info(pid)
    return int(i.ri_phys_footprint) if i else None


def peak_footprint(pid: int | None = None) -> int | None:
    """Lifetime peak physical footprint in bytes (None where unsupported)."""
    i = _info(pid)
    return int(i.ri_lifetime_max_phys_footprint) if i else None


def gb(x: int | None) -> float | None:
    return None if x is None else round(x / 1e9, 2)


class FootprintExceeded(MemoryError):
    pass


class FootprintGuard:
    """`check()` raises FootprintExceeded once the footprint passes `limit_gb`
    (default VOICEPRINT_FOOTPRINT_LIMIT_GB, else 24). Cheap: one syscall."""

    def __init__(self, limit_gb: float | None = None):
        self.limit_gb = float(limit_gb if limit_gb is not None
                              else os.environ.get("VOICEPRINT_FOOTPRINT_LIMIT_GB", 24))
        self.peak = 0

    def check(self, where: str = "") -> None:
        cur = phys_footprint()
        if cur is None:
            return
        self.peak = max(self.peak, cur)
        if cur > self.limit_gb * 1e9:
            raise FootprintExceeded(f"physical footprint {cur / 1e9:.1f} GB > limit "
                                    f"{self.limit_gb:g} GB{' at ' + where if where else ''}")
