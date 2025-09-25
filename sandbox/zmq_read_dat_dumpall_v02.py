#!/usr/bin/env python3
# zmq_read_dat_dumpall_v02.py — structured dump following user's C++ headers
import argparse
import struct
import numpy as np
from typing import Iterator, Tuple, List

MAGIC = b"ZMQD"
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

# ---- Part0: per-slice LSE header (20 bytes) ----
# struct sc_hdr {
#   uint32_t sfn;
#   uint16_t slot;
#   uint16_t tx_port;
#   uint16_t rx_port_idx;
#   uint16_t n_rx;
#   uint16_t n_tx;
#   uint16_t k0;
#   uint16_t step;
#   uint16_t len;
# } __attribute__((packed));
SLICE_HDR_FMT = "<I8H"
SLICE_HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)
SLICE_HDR_FIELDS = [
    "sfn", "slot", "tx_port", "rx_port_idx",
    "n_rx", "n_tx", "k0", "step", "len"
]

# ---- Part8: AUX header v1 (56 bytes) ----
# struct saru_srs_aux_hdr_v1 { ... } per user spec
AUX_HDR_FMT = "<I H H I H H H B B f f f f f H H H H H H I"
AUX_HDR_SIZE = struct.calcsize(AUX_HDR_FMT)
AUX_HDR_FIELDS = [
    "magic", "version", "size_bytes",
    "sfn", "slot", "numerology",
    "rnti", "normalized_iq_requested", "positioning_requested",
    "epre_dB", "rsrp_dB", "noise_variance",
    "frob_norm", "frob_norm_sq",
    "n_rx", "n_tx",
    "lse_len", "comb", "k0", "scs_khz",
    "crc32_hdr"
]

def iter_records(path: str) -> Iterator[Tuple[int, int, List[bytes]]]:
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                return
            rest = f.read(2+2)
            if len(rest) < 4:
                return
            magic = head
            ver, reserved = struct.unpack("<HH", rest)
            if magic != MAGIC:
                raise ValueError("Bad magic in file")
            if ver == 1:
                rest1 = f.read(FIXED_HDR_SIZE_V1 - 8)
                if len(rest1) < (FIXED_HDR_SIZE_V1 - 8):
                    return
                header_len, payload_len, recv_time_ns = struct.unpack("<IIQ", rest1)
                header = f.read(header_len)
                payload = f.read(payload_len)
                if len(header) < header_len or len(payload) < payload_len:
                    return
                yield (1, recv_time_ns, [header, payload])
            elif ver == 2:
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
            else:
                raise ValueError(f"Unknown version: {ver}")

def dump_slice_hdr_part0(blob: bytes) -> None:
    if len(blob) != SLICE_HDR_SIZE:
        print(f"  Part0: unexpected header size {len(blob)} (expected {SLICE_HDR_SIZE})")
        print(f"  raw-hex: {blob.hex()}")
        return
    vals = struct.unpack(SLICE_HDR_FMT, blob)
    print(f"  Part0: sc_hdr ({SLICE_HDR_SIZE} bytes)")
    for name, v in zip(SLICE_HDR_FIELDS, vals):
        print(f"    {name} = {v}")

def dump_subcarriers(blob: bytes, label: str, preview_count: int = 5) -> None:
    if len(blob) % 8 != 0:
        print(f"  {label}: size {len(blob)} not multiple of 8 (complex64)")
        print(f"  raw-hex: {blob[:64].hex()}...")
        return
    arr = np.frombuffer(blob, dtype=np.complex64)
    n = len(arr)
    print(f"  {label}: subcarriers total={n}")
    for i in range(min(preview_count, n)):
        c = arr[i]
        print(f"    sc[{i}] = {c.real:.6g}+{c.imag:.6g}j")

def dump_aux_part8(blob: bytes) -> dict:
    if len(blob) != AUX_HDR_SIZE:
        print(f"  Part8: unexpected AUX header size {len(blob)} (expected {AUX_HDR_SIZE})")
        print(f"  raw-hex: {blob.hex()}")
        return {}
    vals = struct.unpack(AUX_HDR_FMT, blob)
    d = dict(zip(AUX_HDR_FIELDS, vals))
    # Pretty print
    print(f"  Part8: saru_srs_aux_hdr_v1 ({AUX_HDR_SIZE} bytes)")
    # magic: show hex and ascii
    magic = d["magic"]
    ascii_try = struct.pack("<I", magic)
    try:
        magic_ascii = ascii_try.decode("ascii")
    except Exception:
        magic_ascii = repr(ascii_try)
    print(f"    magic = 0x{magic:08X} ('{magic_ascii}')")
    for k in AUX_HDR_FIELDS[1:]:
        if k == "magic":
            continue
        print(f"    {k} = {d[k]}")
    return d

def dump_wb_part9(blob: bytes, aux_info: dict) -> None:
    if len(blob) % 8 != 0:
        print(f"  Part9: size {len(blob)} not multiple of 8 (complex64)")
        print(f"  raw-hex: {blob[:64].hex()}...")
        return
    arr = np.frombuffer(blob, dtype=np.complex64)
    n = len(arr)
    n_rx = aux_info.get("n_rx")
    n_tx = aux_info.get("n_tx")
    if n_rx and n_tx and n_rx*n_tx == n:
        mat = arr.reshape(n_rx, n_tx)
        print(f"  Part9: WB matrix [{n_rx} x {n_tx}] (rx-major, then tx)")
        for rx in range(n_rx):
            row = " ".join(f"{mat[rx,tx].real:.6g}+{mat[rx,tx].imag:.6g}j" for tx in range(n_tx))
            print(f"    rx{rx}: {row}")
    else:
        print(f"  Part9: WB flat {n} coeffs (n_rx={n_rx}, n_tx={n_tx})")
        # print as single row to avoid huge output
        preview = " ".join(f"{c.real:.6g}+{c.imag:.6g}j" for c in arr[:min(8,n)])
        print(f"    preview: {preview}{' ...' if n>8 else ''}")

def main():
    ap = argparse.ArgumentParser(description="Dump .dat with structured parts per user's layout")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--record", type=int, default=0, help="Record index to dump (0-based)")
    args = ap.parse_args()

    for rec_idx, (ver, t_ns, parts) in enumerate(iter_records(args.dat)):
        if rec_idx != args.record:
            continue
        print(f"Record {rec_idx}: ver={ver}, recv_time_ns={t_ns}, parts={len(parts)}")
        aux_info = {}
        for j, blob in enumerate(parts):
            if j == 0:
                dump_slice_hdr_part0(blob)
            elif j in (2,4,6):
                print(f"  Part{j}: {len(blob)} bytes (not decoded per spec)")
            elif j in (1,3,5,7):
                dump_subcarriers(blob, f"Part{j}")
            elif j == 8:
                aux_info = dump_aux_part8(blob) or {}
            elif j == 9:
                dump_wb_part9(blob, aux_info)
            else:
                print(f"  Part{j}: {len(blob)} bytes (no decoder assigned)")
        break

if __name__ == "__main__":
    main()
