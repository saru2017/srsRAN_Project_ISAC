
#!/usr/bin/env python3
"""
srs_check_wb.py
---------------
For each record that contains WB (Part 9), print a CSV line:
  rx0,rx1,rx2,rx3,ang_rx1_minus_rx0_deg,ang_rx2_minus_rx1_deg,ang_rx3_minus_rx2_deg

- rx0..rx3 are printed as COMPLEX numbers (e.g., "0.12-0.34j").
- Differences are angles (degrees) of (rx1-rx0), (rx2-rx1), (rx3-rx2).
- If --include-time is given, a leading column t_ms (relative) is added.

Notes
-----
- When WB is an n_rx x n_tx complex64 matrix, we compute a single complex per RX
  by taking the mean across TX (complex mean). This yields 1 complex per RX.
- If a record has fewer than 4 RX ports, missing values are filled with 'nan'.
- If a record has more than 4 RX ports, only the first four are used.
"""
import argparse
import math
import struct
from typing import Iterator, Tuple, List, Dict, Optional

import numpy as np

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

def fmt_complex(c: complex) -> str:
    if isinstance(c, float) and math.isnan(c):
        return "nan"
    # format without parentheses, with sign on imag part
    real = f"{c.real:.9g}"
    imag = f"{c.imag:.9g}"
    sign = "+" if c.imag >= 0 else ""
    return f"{real}{sign}{imag}j"

def compute_rx_complex(wb_blob: bytes, aux_info: Dict[str, int]) -> List[complex]:
    """Return up to 4 complex values [rx0..rx3] as mean across TX per RX."""
    if len(wb_blob) % 8 != 0:
        return [float("nan")] * 4
    arr = np.frombuffer(wb_blob, dtype=np.complex64)
    n = len(arr)
    n_rx = int(aux_info.get("n_rx") or 0)
    n_tx = int(aux_info.get("n_tx") or 0)

    if n_rx > 0 and n_tx > 0 and n_rx * n_tx == n:
        mat = arr.reshape((n_rx, n_tx))  # rx-major rows
        vals = [complex(np.mean(mat[rx, :])) for rx in range(min(4, n_rx))]
    else:
        # Fallback: flat vector, take first four entries
        vals = [complex(x) for x in arr[:4]]

    while len(vals) < 4:
        vals.append(float("nan"))
    return vals[:4]

def diffs_angle_deg(vals: List[complex]) -> List[float]:
    out = []
    pairs = [(1,0),(2,1),(3,2)]
    for i,j in pairs:
        a = vals[i]
        b = vals[j]
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            out.append(float("nan"))
        else:
            d = a - b
            ang = float(np.degrees(np.angle(d)))
            out.append(ang)
    return out

def main():
    ap = argparse.ArgumentParser(description="Print WB rx0..rx3 as complex and angle diffs in degrees.")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--include-time", dest="include_time", action="store_true",
                    help="Include leading column with relative recv_time_ms")
    args = ap.parse_args()

    t0_ns: Optional[int] = None
    wrote_header = False

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

        rx_vals = compute_rx_complex(wb_blob, aux_info)
#        print(rx_vals[0], rx_vals[1], rx_vals[2], rx_vals[3])

        rx_vals[0] = rx_vals[0] - 0.3358704149723053+0.5291157364845276j
        rx_vals[1] = rx_vals[1] - 0.15472480654716492+0.45775243639945984j
        rx_vals[2] = rx_vals[2] - -0.34757092595100403-0.3040795624256134j
        rx_vals[3] = rx_vals[3] - 0.5237588882446289+0.49044445157051086j


#        (0.3358704149723053+0.5291157364845276j) (0.15472480654716492+0.45775243639945984j) (-0.34757092595100403-0.3040795624256134j) (0.5237588882446289+0.49044445157051086j)

        for val in rx_vals:
            print(float(np.degrees(np.angle(val))), end=",")

        print("", end="\n")

        continue
#        exit(0)
        angs = diffs_angle_deg(rx_vals)

        if not wrote_header:
            cols = ["rx0","rx1","rx2","rx3",
                    "ang_rx1_minus_rx0_deg","ang_rx2_minus_rx1_deg","ang_rx3_minus_rx2_deg"]
            if args.include_time:
                cols = ["t_ms"] + cols
            print(",".join(cols))
            wrote_header = True

        fields: List[str] = [fmt_complex(c) for c in rx_vals]
        fields += [f"{a:.9g}" if not math.isnan(a) else "nan" for a in angs]

        if args.include_time:
            if t0_ns is None:
                t0_ns = recv_time_ns
            t_ms = (recv_time_ns - t0_ns) / 1e6
            line = f"{t_ms:.6f}," + ",".join(fields)
        else:
            line = ",".join(fields)
        print(line)

if __name__ == "__main__":
    main()
