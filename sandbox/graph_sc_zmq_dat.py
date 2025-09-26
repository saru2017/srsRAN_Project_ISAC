#!/usr/bin/env python3
# graph_sc_zmq_dat.py — plot per-antenna subcarrier phase (unwrapped) and power(dB)
# Compatible with .dat files where version=2 and Parts 1,3,5,7 contain complex64 subcarriers.
#
# Usage:
#   python graph_sc_zmq_dat.py <file.dat> --record 0 --out phase_power.png
#
# Notes:
# - This script reads the specified record (0-based) and expects that for that record,
#   Parts 1,3,5,7 contain per-antenna subcarrier arrays (complex64). If a part is
#   missing, it is skipped.
# - Power is computed as 20*log10(|c| + eps). Phase is np.unwrap(np.angle(c)) in degrees.
# - Subcarrier x-axis is simple index [0..N-1] for each antenna trace; if you need
#   physical k mapping, extend here with k0/step when available in your stream.
#
import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, List

MAGIC = b"ZMQD"
# ver=2 fixed header: "<4sHHQH" -> magic, ver, reserved, recv_time_ns, num_parts
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)

# ver=1 legacy header (not used for plotting here, but tolerated if encountered)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

def iter_records(path: str) -> Iterator[Tuple[int, int, List[bytes]]]:
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                return
            rest = f.read(2 + 2)
            if len(rest) < 4:
                return
            magic = head
            ver, reserved = struct.unpack("<HH", rest)
            if magic != MAGIC:
                raise ValueError("Bad magic in file")
            if ver == 2:
                # read remainder of ver=2 fixed header
                rest2 = f.read(FIXED_HDR_SIZE_V2 - 8)
                if len(rest2) < (FIXED_HDR_SIZE_V2 - 8):
                    return
                recv_time_ns, num_parts = struct.unpack("<QH", rest2)
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
            elif ver == 1:
                # consume ver=1 record so we can skip/ignore
                rest1 = f.read(FIXED_HDR_SIZE_V1 - 8)
                if len(rest1) < (FIXED_HDR_SIZE_V1 - 8):
                    return
                header_len, payload_len, recv_time_ns = struct.unpack("<IIQ", rest1)
                # skip the header+payload
                hdr = f.read(header_len)
                pay = f.read(payload_len)
                if len(hdr) < header_len or len(pay) < payload_len:
                    return
                yield (1, recv_time_ns, [hdr, pay])
            else:
                # Unknown version — stop.
                raise ValueError(f"Unknown version: {ver}")

def blob_to_complex64(blob: bytes) -> np.ndarray:
    if len(blob) % 8 != 0:
        raise ValueError(f"Part has size {len(blob)} not multiple of 8 (complex64)")
    return np.frombuffer(blob, dtype=np.complex64)

def main():
    ap = argparse.ArgumentParser(description="Plot unwrapped phase and power(dB) for Parts 1,3,5,7 (per-antenna subcarriers)")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--record", type=int, default=0, help="0-based record index to plot (default: 0)")
    ap.add_argument("--out", default="phase_power.png", help="Output figure filename (PNG)")
    args = ap.parse_args()

    # Find the target record
    target = None
    for rec_idx, tup in enumerate(iter_records(args.dat)):
        if rec_idx == args.record:
            target = tup
            break
    if target is None:
        raise SystemExit(f"Record {args.record} not found in {args.dat}")

    ver, recv_time_ns, parts = target
    if ver != 2:
        print(f"Warning: record ver={ver} — expected ver=2 with Parts 1/3/5/7. Proceeding anyway.")

    # Collect antenna arrays from Parts 1,3,5,7 if present
    antenna_parts = []
    for j in (1, 3, 5, 7):
        if j < len(parts) and parts[j]:
            try:
                arr = blob_to_complex64(parts[j])
                antenna_parts.append((j, arr))
            except Exception as e:
                print(f"Skip Part{j}: {e}")
        else:
            print(f"Skip Part{j}: not present in this record.")

    if not antenna_parts:
        raise SystemExit("No antenna subcarrier parts (1,3,5,7) found in this record.")

    # Prepare plots
    fig, (ax_phase, ax_pow) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(f"Per-antenna subcarriers — Record {args.record}, recv_time_ns={recv_time_ns}")

    ax_phase.set_title("Unwrapped Phase per Subcarrier")
    ax_phase.set_xlabel("Subcarrier Index")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, alpha=0.35)

    ax_pow.set_title("Power per Subcarrier")
    ax_pow.set_xlabel("Subcarrier Index")
    ax_pow.set_ylabel("Power (dB)")
    ax_pow.grid(True, alpha=0.35)

    eps = 1e-12
    index = 0
    for j, arr in antenna_parts:
        n = arr.size
        x = np.arange(n)
        phase_deg = np.unwrap(np.angle(arr)) * 180.0 / np.pi
        power_db = 20.0 * np.log10(np.abs(arr) + eps)
        ax_phase.plot(x, phase_deg, label=f"ANT{index}")
        ax_pow.plot(x, power_db, label=f"ANT{index}")
        index += 1

    ax_phase.legend();
    ax_pow.legend();

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
#    fig.savefig(args.out, bbox_inches="tight")
    #print(f"Saved figure to {args.out}")

if __name__ == "__main__":
    main()
