#!/usr/bin/env python3
# read_zmq_dat.py — parser for .dat files saved by save_zmq_to_dat.py (V1/V2)
import argparse
import struct
import sys
from typing import Iterator, Tuple, List

MAGIC = b"ZMQD"
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

def iter_records(path: str) -> Iterator[Tuple[int, int, list]]:
    """
    Yields (version, recv_time_ns, parts:list[bytes])
    """
    with open(path, "rb") as f:
        while True:
            head = f.read(4)  # magic
            if not head:
                return
            rest = f.read(2+2)  # ver,reserved
            if len(rest) < 4:
                return
            magic = head
            ver, reserved = struct.unpack("<HH", rest)
            if magic != MAGIC:
                raise ValueError("Bad magic at offset")

            if ver == 1:
                rest1 = f.read(FIXED_HDR_SIZE_V1 - 8)  # remaining of V1
                if len(rest1) < (FIXED_HDR_SIZE_V1 - 8):
                    return
                header_len, payload_len, recv_time_ns = struct.unpack("<IIQ", rest1)
                header = f.read(header_len)
                payload = f.read(payload_len)
                if len(header) < header_len or len(payload) < payload_len:
                    return
                yield (1, recv_time_ns, [header, payload] if header or payload else [])
            elif ver == 2:
                rest2 = f.read(FIXED_HDR_SIZE_V2 - 8)  # remaining of V2
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
            else:
                raise ValueError(f"Unknown version: {ver}")

def main():
    ap = argparse.ArgumentParser(description="Read .dat created by save_zmq_to_dat.py")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--peek", type=int, default=1, help="How many records to print")
    ap.add_argument("--dump-idx", type=int, default=None, help="Record index (0-based) to dump frame sizes")
    ap.add_argument("--extract", nargs=2, metavar=("REC_IDX", "OUT_BASENAME"),
                    help="Extract all frames of record REC_IDX to OUT_BASENAME_partN.bin")
    args = ap.parse_args()

    for i, (ver, t_ns, parts) in enumerate(iter_records(args.dat)):
        if i < args.peek:
            sizes = [len(p) for p in parts]
            print(f"[{i}] ver={ver} t={t_ns} ns parts={len(parts)} sizes={sizes}")
        if args.dump_idx is not None and i == args.dump_idx:
            sizes = [len(p) for p in parts]
            print(f"Record {i}: version={ver}, recv_time_ns={t_ns}, parts={len(parts)}, sizes={sizes}")
        if args.extract is not None:
            idx = int(args.extract[0])
            base = args.extract[1]
            if i == idx:
                for j, blob in enumerate(parts):
                    with open(f"{base}_part{j}.bin", "wb") as g:
                        g.write(blob)
                print(f"Extracted {len(parts)} parts to {base}_part*.bin")
                break

if __name__ == "__main__":
    main()
