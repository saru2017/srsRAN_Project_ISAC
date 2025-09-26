#!/usr/bin/env python3
# mp4_sc_zmq_dat.py — make an MP4 across all records:
#   top: unwrapped phase (deg), bottom: power (dB)
# Source format: ver=2 .dat, Parts 1/3/5/7 are complex64 subcarriers per antenna.
#
# Usage:
#   python mp4_sc_zmq_dat.py /path/to/file.dat
#
# Output:
#   ./<path_with_slashes_replaced_by_underscores>.mp4
#
import argparse
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Iterator, Tuple, List, Optional

MAGIC = b"ZMQD"
FIXED_HDR_FMT_V2 = "<4sHHQH"  # magic, ver, reserved, recv_time_ns, num_parts
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"  # legacy tolerate/skip
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

PART_TO_ANT_IDX = {1: 0, 3: 1, 5: 2, 7: 3}  # ANT1..ANT4

def iter_records(path: str) -> Iterator[Tuple[int, int, List[bytes]]]:
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                return
            rest = f.read(4)  # ver, reserved
            if len(rest) < 4:
                return
            magic = head
            ver, reserved = struct.unpack("<HH", rest)
            if magic != MAGIC:
                raise ValueError("Bad magic")
            if ver == 2:
                rest2 = f.read(FIXED_HDR_SIZE_V2 - 8)
                if len(rest2) < (FIXED_HDR_SIZE_V2 - 8):
                    return
                recv_time_ns, num_parts = struct.unpack("<QH", rest2)
                parts = []
                for _ in range(num_parts):
                    sz_b = f.read(4)
                    if len(sz_b) < 4:
                        return
                    sz = struct.unpack("<I", sz_b)[0]
                    blob = f.read(sz)
                    if len(blob) < sz:
                        return
                    parts.append(blob)
                yield (2, recv_time_ns, parts)
            elif ver == 1:
                # tolerate: skip legacy frame
                rest1 = f.read(FIXED_HDR_SIZE_V1 - 8)
                if len(rest1) < (FIXED_HDR_SIZE_V1 - 8):
                    return
                header_len, payload_len, recv_time_ns = struct.unpack("<IIQ", rest1)
                f.seek(header_len + payload_len, os.SEEK_CUR)
                yield (1, recv_time_ns, [])
            else:
                raise ValueError(f"Unknown version {ver}")

def blob_to_complex64(blob: bytes) -> np.ndarray:
    if len(blob) % 8 != 0:
        raise ValueError(f"Part size {len(blob)} is not multiple of 8 (complex64)")
    return np.frombuffer(blob, dtype=np.complex64)

def collect_all(path: str):
    """Read all records and produce per-record dict with ANT arrays and recv_time_ns."""
    recs = []
    for ver, recv_ns, parts in iter_records(path):
        if ver != 2:
            # skip legacy in movie
            continue
        ants = [None, None, None, None]  # ANT1..4
        for pidx in (1, 3, 5, 7):
            if pidx < len(parts) and parts[pidx]:
                try:
                    arr = blob_to_complex64(parts[pidx])
                except Exception:
                    arr = None
            else:
                arr = None
            ants[PART_TO_ANT_IDX[pidx]] = arr
        recs.append({"recv_ns": recv_ns, "ants": ants})
    return recs

def compute_phase_power(arr: Optional[np.ndarray]):
    if arr is None:
        return None, None
    eps = 1e-12
    phase_deg = np.unwrap(np.angle(arr)) * 180.0 / np.pi
    power_db = 20.0 * np.log10(np.abs(arr) + eps)
    return phase_deg, power_db

def main():
    ap = argparse.ArgumentParser(description="Make MP4 from ver=2 .dat (Parts 1/3/5/7 -> ANT1..ANT4).")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second for MP4 (default: 10)")
    ap.add_argument("--dpi", type=int, default=110, help="Figure DPI (default: 110)")
    args = ap.parse_args()

    in_path = args.dat
    # Output filename: replace '/' with '_' and append '.mp4'
    out_name = in_path.replace('/', '_') + ".mp4"
    # Collect all records
    recs = collect_all(in_path)
    if not recs:
        raise SystemExit("No ver=2 records found.")

    # Normalize recv_time to start at 0 ms
    t0_ns = recs[0]["recv_ns"]
    for r in recs:
        r["t_ms"] = (r["recv_ns"] - t0_ns) / 1e6

    # Precompute phase & power and global y-limits
    # We'll also determine the max N (subcarriers) to keep x-axis consistent across frames.
    global_phase_min = +np.inf
    global_phase_max = -np.inf
    global_pow_min = +np.inf
    global_pow_max = -np.inf
    max_len = 0

    for r in recs:
        r["pp"] = []  # list of (phase_deg, power_db) for ANT1..4 (or (None,None))
        for a in r["ants"]:
            phase_deg, power_db = compute_phase_power(a)
            r["pp"].append((phase_deg, power_db))
            if phase_deg is not None:
                max_len = max(max_len, len(phase_deg))
                global_phase_min = min(global_phase_min, float(np.min(phase_deg)))
                global_phase_max = max(global_phase_max, float(np.max(phase_deg)))
            if power_db is not None:
                global_pow_min = min(global_pow_min, float(np.min(power_db)))
                global_pow_max = max(global_pow_max, float(np.max(power_db)))

    if not np.isfinite([global_phase_min, global_phase_max, global_pow_min, global_pow_max]).all():
        raise SystemExit("Failed to compute global y-limits (no valid antenna data).")

    # Add small margins
    phase_margin = 0.05 * (global_phase_max - global_phase_min + 1e-9)
    pow_margin = 0.05 * (global_pow_max - global_pow_min + 1e-9)
    #y_phase = (global_phase_min - phase_margin, global_phase_max + phase_margin)
    y_phase = (-20000, 0)  # fixed for phase
    #y_pow = (global_pow_min - pow_margin, global_pow_max + pow_margin)
    y_pow = (-80, 0)      # fixed for power

    # Prepare figure & artists
    fig, (ax_phase, ax_pow) = plt.subplots(2, 1, figsize=(12, 9), dpi=args.dpi)
    # X-axis as subcarrier index [0..max_len-1]
    x = np.arange(max_len, dtype=int)

    # Initialize 4 lines per axis (some may be empty if that ANT missing)
    lines_phase = []
    lines_pow = []
    labels = ["ANT1", "ANT2", "ANT3", "ANT4"]
    for i in range(4):
        lp, = ax_phase.plot([], [], label=labels[i])
        lw, = ax_pow.plot([], [], label=labels[i])
        lines_phase.append(lp)
        lines_pow.append(lw)

    # Static axis setup
    ax_phase.set_xlim(0, max_len - 1 if max_len > 0 else 1)
    ax_phase.set_ylim(*y_phase)
    ax_phase.set_xlabel("Subcarrier Index")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, alpha=0.35)
    ax_phase.legend(title="Antenna")

    ax_pow.set_xlim(0, max_len - 1 if max_len > 0 else 1)
    ax_pow.set_ylim(*y_pow)
    ax_pow.set_xlabel("Subcarrier Index")
    ax_pow.set_ylabel("Power (dB)")
    ax_pow.grid(True, alpha=0.35)
    ax_pow.legend(title="Antenna")

    # Title: include the exact path string and t_ms with 3 decimals
    def update_title(t_ms: float):
        # Avoid repr() which inserts escapes; use the raw string directly.
        fig.suptitle(f'{in_path} — t={t_ms:.3f} ms')

    # Render/encode with FFMpegWriter
    writer = FFMpegWriter(fps=args.fps, metadata={'title': f"SC Movie: {os.path.basename(in_path)}"})
    total = len(recs)
    print(f"Encoding {total} frames to {out_name} ...")
    with writer.saving(fig, out_name, dpi=args.dpi):
        for idx, r in enumerate(recs):
            # For each ANT, set data (pad/truncate to max_len if needed)
            for ant_idx, (phase_deg, power_db) in enumerate(r["pp"]):
                if phase_deg is None or power_db is None:
                    lines_phase[ant_idx].set_data([], [])
                    lines_pow[ant_idx].set_data([], [])
                else:
                    n = len(phase_deg)
                    if n < max_len:
                        xp = x[:n]
                        lines_phase[ant_idx].set_data(xp, phase_deg)
                        lines_pow[ant_idx].set_data(xp, power_db)
                    else:
                        lines_phase[ant_idx].set_data(x, phase_deg[:max_len])
                        lines_pow[ant_idx].set_data(x, power_db[:max_len])
            update_title(r["t_ms"])
            writer.grab_frame()
            # Progress
            print(f"[{idx+1}/{total}] {idx+1:>6} frames encoded ({(idx+1)*100/total:.1f}%)")

    print(f"Done. Saved: {out_name}")

if __name__ == "__main__":
    main()
