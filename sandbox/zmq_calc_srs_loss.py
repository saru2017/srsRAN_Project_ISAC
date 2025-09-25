#!/usr/bin/env python3
# zmq_calc_srs_loss.py
# Compute SRS loss count and loss rate by scanning a ZMQ .dat file (V2 multipart format)
# New features:
#   - You can specify --interval-ms (recommended). The tool converts to slots using inferred mu.
#   - Prints diagnostics: inferred mu, slots per frame, median observed slot gap.
#   - Optional --assume-mu for cases where AUX (Part8) is missing.
# Backward compatible:
#   python zmq_calc_srs_loss.py <interval_slots> <file.dat>
#
import argparse
import struct
from typing import Iterator, List, Tuple, Optional
import statistics

MAGIC = b"ZMQD"

FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)

SLICE_HDR_FMT = "<I8H"
SLICE_HDR_SIZE = struct.calcsize(SLICE_HDR_FMT)

AUX_HDR_FMT = "<I H H Q I H H H B B f f f f f H H H H H H I"
AUX_HDR_SIZE = struct.calcsize(AUX_HDR_FMT)

def iter_records_v2(path: str) -> Iterator[List[bytes]]:
    # Yield the list of parts (as bytes) for each V2 record.
    with open(path, "rb") as f:
        while True:
            hdr = f.read(FIXED_HDR_SIZE_V2)
            if not hdr or len(hdr) < FIXED_HDR_SIZE_V2:
                return
            magic, ver, _reserved, _recv_time_ns, num_parts = struct.unpack(FIXED_HDR_FMT_V2, hdr)
            if magic != MAGIC or ver != 2:
                return
            parts: List[bytes] = []
            for _ in range(num_parts):
                szb = f.read(4)
                if len(szb) < 4:
                    return
                sz = struct.unpack("<I", szb)[0]
                blob = f.read(sz)
                if len(blob) < sz:
                    return
                parts.append(blob)
            yield parts

def parse_slice_hdr(blob: bytes) -> Optional[Tuple[int, int]]:
    if len(blob) != SLICE_HDR_SIZE:
        return None
    sfn, slot, *_ = struct.unpack(SLICE_HDR_FMT, blob)
    return sfn, slot

def parse_aux_mu(blob: bytes) -> Optional[int]:
    if len(blob) != AUX_HDR_SIZE:
        return None
    fields = struct.unpack(AUX_HDR_FMT, blob)
    return int(fields[6])

def slots_per_frame_from_mu(mu: int) -> int:
    # slots per 10 ms frame = 10 * 2**mu
    return 10 * (1 << mu)

def slot_ms_from_mu(mu: int) -> float:
    # slot duration in ms = 1.0 ms / 2**mu
    return 1.0 / (1 << mu)

def advance_absolute_slot(prev_abs: Optional[int], sfn: int, slot: int, spf: int) -> int:
    # Convert (sfn,slot) into monotonically increasing absolute slot index with wrap handling.
    period = 1024 * spf
    cur_mod = (sfn % 1024) * spf + (slot % spf)
    if prev_abs is None:
        return cur_mod
    prev_mod = prev_abs % period
    forward = (cur_mod - prev_mod) % period
    return prev_abs + forward

def scan(path: str):
    mu: Optional[int] = None
    spf: Optional[int] = None
    prev_abs: Optional[int] = None
    deltas: List[int] = []
    seq: List[Tuple[int,int,int]] = []  # (sfn, slot, abs_slot)

    count = 0
    for parts in iter_records_v2(path):
        if not parts:
            continue
        hdr = parse_slice_hdr(parts[0])
        if hdr is None:
            continue
        sfn, slot = hdr

        if len(parts) > 8:
            mu_try = parse_aux_mu(parts[8])
            if mu_try is not None:
                mu = mu_try

        if spf is None and mu is not None:
            spf = slots_per_frame_from_mu(mu)

        # If mu still unknown, assume mu=1 tentatively for stepping; we will only use deltas
        spf_eff = spf if spf is not None else slots_per_frame_from_mu(1)
        abs_slot = advance_absolute_slot(prev_abs, sfn, slot, spf_eff)
        if prev_abs is not None:
            d = abs_slot - prev_abs
            if d > 0:
                deltas.append(d)
        prev_abs = abs_slot
        seq.append((sfn, slot, abs_slot))
        count += 1
    return mu, (spf if spf is not None else slots_per_frame_from_mu(1)), seq, deltas

def compute_loss(interval_slots: int, seq: List[Tuple[int,int,int]]) -> Tuple[int,int,int,float]:
    received = len(seq)
    if received == 0:
        return 0, 0, 0, 0.0
    lost = 0
    next_expected: Optional[int] = None
    for _, _, abs_slot in seq:
        if next_expected is None:
            next_expected = abs_slot + interval_slots
            continue
        while next_expected < abs_slot:
            lost += 1
            next_expected += interval_slots
        if next_expected <= abs_slot:
            next_expected = abs_slot + interval_slots
    total_expected = received + lost
    rate = (lost / total_expected * 100.0) if total_expected > 0 else 0.0
    return received, total_expected, lost, rate

def main():
    ap = argparse.ArgumentParser(description="Scan ZMQ .dat (V2) and compute SRS loss statistics.")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--interval-ms", type=float, help="Expected SRS transmit interval in milliseconds (recommended)")
    ap.add_argument("interval_slots", nargs="?", type=int, help="Expected SRS interval in slots (legacy positional)")
    ap.add_argument("dat", help="Path to .dat file")
    ap.add_argument("--assume-mu", type=int, choices=[0,1,2,3,4], help="Assume numerology if AUX Part8 missing")
    ap.add_argument("--debug", action="store_true", help="Print diagnostics")
    args = ap.parse_args()

    mu, spf, seq, deltas = scan(args.dat)

    if mu is None:
        if args.assume-mu is not None:
            mu = args.assume-mu
            spf = slots_per_frame_from_mu(mu)
        else:
            # Default to mu=1 if still unknown
            mu = 1
            spf = slots_per_frame_from_mu(mu)

    slot_ms = 1.0 / (1 << mu)

    # Determine interval_slots from args
    if args.interval_ms is not None:
        interval_slots = max(1, int(round(args.interval_ms / slot_ms)))
        src = f"{args.interval_ms} ms"
    elif args.interval_slots is not None:
        interval_slots = args.interval_slots
        src = f"{args.interval_slots} slots"
    else:
        raise SystemExit("Specify either --interval-ms or <interval_slots>.")

    # Diagnostics
    median_delta = statistics.median(deltas) if deltas else 0
    recv, expected, lost, rate = compute_loss(interval_slots, seq)

    print("=== SRS Loss Report ===")
    print(f"File                : {args.dat}")
    print(f"Inferred mu         : {mu}")
    print(f"Slot duration (ms)  : {slot_ms:.6f}")
    print(f"Slots per frame     : {spf}")
    print(f"Observed median Δslot: {median_delta}")
    print(f"Interval (input)    : {src}")
    print(f"Interval (slots)    : {interval_slots}")
    print(f"Received            : {recv}")
    print(f"Expected            : {expected}")
    print(f"Lost                : {lost}")
    print(f"Loss Rate           : {rate:.3f}%")

    if args.debug and seq:
        print("\n[Debug] First 5 (sfn, slot -> abs) rows:")
        for s in seq[:5]:
            print(f"  sfn={s[0]:4d}, slot={s[1]:3d} -> abs={s[2]}")

if __name__ == "__main__":
    main()
