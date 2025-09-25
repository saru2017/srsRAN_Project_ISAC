#!/usr/bin/env python3
# zmq_read_dat_dumpall_v02.py — structured dump per user-defined headers
import argparse
import struct
import numpy as np
from typing import Iterator, Tuple, List

MAGIC = b"ZMQD"
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

# ---- Per-slice LSE header: 20 bytes ----
# C++:
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
#
# => little-endian: <I8H
SLICE_HDR_FMT = "<I8H"
SLICE_HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)
SLICE_HDR_FIELDS = [
    "sfn",
    "slot",
    "tx_port",
    "rx_port_idx",
    "n_rx",
    "n_tx",
    "k0",
    "step",
    "len",
]

# ---- AUX header v1: 56 bytes ----
# struct (little-endian) assumed from prior discussion:
# magic 'AUX1'(u32), version(u16), size_bytes(u16),
# sfn(u32), slot(u16), numerology(u16),
# rnti(u16), normalized_iq_requested(u8), positioning_requested(u8),
# epre_dB(f32), rsrp_dB(f32), noise_variance(f32),
# frob_norm(f32), frob_norm_sq(f32),
# n_rx(u16), n_tx(u16),
# lse_len(u16), comb(u16), k0(u16), scs_khz(u16),
# crc32_hdr(u32)
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
    "crc32_hdr",
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

def dump_slice_header(blob: bytes, label: str) -> None:
    if len(blob) != SLICE_HDR_SIZE:
        print(f"  {label}: unexpected header size {len(blob)} (expected {SLICE_HDR_SIZE})")
        print(f"  raw-hex: {blob.hex()}")
        return
    vals = struct.unpack(SLICE_HDR_FMT, blob)
    print(f"  {label}: per-slice LSE header ({SLICE_HDR_SIZE} bytes)")
    for name, v in zip(SLICE_HDR_FIELDS, vals):
        print(f"    {name} = {v}")

def dump_aux_header(blob: bytes, label: str) -> None:
    if len(blob) != AUX_HDR_SIZE:
        print(f"  {label}: unexpected AUX header size {len(blob)} (expected {AUX_HDR_SIZE})")
        print(f"  raw-hex: {blob.hex()}")
        return
    vals = struct.unpack(AUX_HDR_FMT, blob)
    print(f"  {label}: AUX header v1 ({AUX_HDR_SIZE} bytes)")
    for name, v in zip(AUX_HDR_FIELDS, vals):
        if name == "magic":
            # show both hex and ascii attempt
            ascii_try = bytes(struct.pack('<I', v))
            try:
                s = ascii_try.decode('ascii')
            except Exception:
                s = repr(ascii_try)
            print(f"    {name} = 0x{v:08X} ('{s}')")
        else:
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

def main():
    ap = argparse.ArgumentParser(description="Dump .dat with structured parts: headers vs subcarriers")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--record", type=int, default=0, help="Record index to dump (0-based)")
    args = ap.parse_args()

    for rec_idx, (ver, t_ns, parts) in enumerate(iter_records(args.dat)):
        if rec_idx != args.record:
            continue
        print(f"Record {rec_idx}: ver={ver}, recv_time_ns={t_ns}, parts={len(parts)}")
        for j, blob in enumerate(parts):
            if j in (1,3,5,7):
                dump_subcarriers(blob, f"Part{j}")
            elif j in (0,2,4,6):
                dump_slice_header(blob, f"Part{j}")
            elif j == 8:
                dump_aux_header(blob, f"Part{j}")
            else:
                print(f"  Part{j}: {len(blob)} bytes (no decoder assigned)")
        break

if __name__ == "__main__":
    main()
