
#!/usr/bin/env python3
"""
srs_plot_wb_phase.py
--------------------
Plot, over time, the PHASE (angle in degrees, **no unwrap**) for each antenna
(RX0..RX3) from WB (Part 9) records in a ZMQ .dat capture (v1/v2).

Definition used here:
- WB assumed to be an n_rx x n_tx complex64 matrix per record.
- We compute one complex value per RX by taking the complex mean across TX.
- Then we take the phase in degrees: angle = degrees(angle(complex_mean)).
- Time axis is relative recv_time in milliseconds from the first WB record.

Usage:
  python3 srs_plot_wb_phase.py your_dump.dat
  python3 srs_plot_wb_phase.py your_dump.dat --save phase.png
  python3 srs_plot_wb_phase.py your_dump.dat --show

Options:
  --save PATH   Save the figure to PATH (e.g., out.png). If omitted, no file is saved.
  --show        Display the plot window (useful locally).

Notes:
  - If fewer than 4 RX ports exist, missing series are omitted from the plot.
  - If more than 4 exist, only RX0..RX3 are plotted.
  - No unwrap is applied.
"""
import argparse
import math
import struct
from typing import Iterator, Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

MAGIC = b"ZMQD"
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

# ---- Part8: AUX header v1 (with timestamp_ns, total 64 bytes) ----
AUX_HDR_FMT = "<I H H Q I H H H B B f f f f f H H H H H H I"
AUX_HDR_SIZE = struct.calcsize(AUX_HDR_FMT)
AUX_HDR_FIELDS = [
    "magic", "version", "size_bytes",
    "timestamp_ns",
    "sfn", "slot", "numerology",
    "rnti", "normalized_iq_requested", "positioning_requested",
    "epre_dB", "rsrp_dB", "noise_variance",
    "frob_norm", "frob_norm_sq",
    "n_rx", "n_tx",
    "lse_len", "comb", "k0", "scs_khz",
    "crc32_hdr"
]

def iter_records(path: str) -> Iterator[Tuple[int, int, List[bytes]]]:
    """Yield (ver, recv_time_ns, parts[]) for each record in the file."""
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
                raise ValueError("Bad magic in file")
            if ver == 1:
                tail = f.read(FIXED_HDR_SIZE_V1 - 8)
                if len(tail) < (FIXED_HDR_SIZE_V1 - 8):
                    return
                header_len, payload_len, recv_time_ns = struct.unpack("<IIQ", tail)
                header = f.read(header_len)
                payload = f.read(payload_len)
                if len(header) < header_len or len(payload) < payload_len:
                    return
                yield (1, recv_time_ns, [header, payload])
            elif ver == 2:
                tail = f.read(FIXED_HDR_SIZE_V2 - 8)
                if len(tail) < (FIXED_HDR_SIZE_V2 - 8):
                    return
                recv_time_ns, num_parts = struct.unpack("<QH", tail)
                parts: List[bytes] = []
                for _ in range(num_parts):
                    sz_bytes = f.read(4)
                    if len(sz_bytes) < 4:
                        return
                    sz = struct.unpack("<I", sz_bytes)[0]
                    blob = f.read(sz)
                    if len(blob) < sz:
                        return
                    parts.append(blob)
                yield (2, recv_time_ns, parts)
            else:
                raise ValueError(f"Unknown version: {ver}")

def parse_aux(blob: bytes) -> Dict[str, int]:
    if len(blob) != AUX_HDR_SIZE:
        return {}
    vals = struct.unpack(AUX_HDR_FMT, blob)
    return dict(zip(AUX_HDR_FIELDS, vals))

def compute_rx_complex_means(wb_blob: bytes, aux_info: Dict[str, int]) -> List[complex]:
    """Return up to 4 complex means (one per RX) across TX for WB matrix."""
    if len(wb_blob) % 8 != 0:
        return []
    arr = np.frombuffer(wb_blob, dtype=np.complex64)
    n = len(arr)
    n_rx = int(aux_info.get("n_rx") or 0)
    n_tx = int(aux_info.get("n_tx") or 0)

    if n_rx > 0 and n_tx > 0 and n_rx * n_tx == n:
        mat = arr.reshape((n_rx, n_tx))  # rx-major
        vals = [complex(np.mean(mat[rx, :])) for rx in range(min(4, n_rx))]
    else:
        # Fallback: flat vector, take first up to 4 entries
        vals = [complex(x) for x in arr[:4]]
    return vals

def main():
    ap = argparse.ArgumentParser(description="Plot time series of RX phases (deg) without unwrap from WB.")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--save", help="Path to save the figure (e.g., out.png)")
    ap.add_argument("--show", action="store_true", help="Show the plot window")
    args = ap.parse_args()

    t_list: List[float] = []  # ms
    phase_series: List[List[float]] = [[], [], [], []]  # per RX
    t0_ns: Optional[int] = None

    for ver, recv_time_ns, parts in iter_records(args.dat):
        aux_info: Dict[str, int] = {}
        wb_blob: Optional[bytes] = None
        for idx, blob in enumerate(parts):
            if idx == 8 and len(blob) == AUX_HDR_SIZE:
                aux_info = parse_aux(blob)
            elif idx == 9:
                wb_blob = blob
        if wb_blob is None:
            continue

        rx_vals = compute_rx_complex_means(wb_blob, aux_info)
        if not rx_vals:
            continue

        if t0_ns is None:
            t0_ns = recv_time_ns
        t_ms = (recv_time_ns - t0_ns) / 1e6
        t_list.append(t_ms)

        # Fill each series; if fewer than 4 RX, remaining lists are left unchanged for this sample
        for i in range(3):
            if i < len(rx_vals):
#                ang_deg = float(np.degrees(np.angle(rx_vals[i])))
                ang_deg = np.angle(rx_vals[i + 1] - rx_vals[i], deg=True)  # direct in degrees
                phase_series[i].append(ang_deg)
            else:
                # To keep all series aligned in length, append NaN when RX missing
                phase_series[i].append(float("nan"))

    # Plot
    fig, ax = plt.subplots()
    labels = ["rx0","rx1","rx2","rx3"]
    for i in range(4):
        if any(not math.isnan(v) for v in phase_series[i]):
            ax.plot(t_list, phase_series[i], label=labels[i])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title("WB RX Phase vs Time (no unwrap)")
    ax.legend()
    ax.grid(True)
    plt.show()

    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=150)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
