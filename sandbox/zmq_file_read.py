#!/usr/bin/env python3
# read_channel_chunks_hdr.py
# 各レコード: [固定ヘッダ(24B)] + [可変ヘッダ(sc_hdr 20B想定)] + [payload(complex64 * len)]
import argparse
import struct
import sys
from typing import BinaryIO
import numpy as np

MAGIC = b"ZMQD"
FIXED_HDR_FMT = "<4sHHIIQ"   # magic, ver, reserved, header_len, payload_len, recv_time_ns
FIXED_HDR_SIZE = struct.calcsize(FIXED_HDR_FMT)

# C struct (packed):
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
SC_HDR_FMT = "<I8H"          # 4 + 2*8 = 20 bytes
SC_HDR_SIZE = struct.calcsize(SC_HDR_FMT)

def read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if b is None or len(b) != n:
        raise EOFError
    return b

def center_indices(n: int, k: int) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    mid = n // 2
    half = k // 2
    start = max(0, min(n - k, mid - half))
    return np.arange(start, start + k)

def format_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Read .dat, decode sc_hdr, and print per-record subcarrier samples")
    ap.add_argument("dat_file", help=".dat created by save_zmq_to_dat.py")
    ap.add_argument("--print-sc", type=int, default=10, help="How many subcarriers to print per record (centered)")
    ap.add_argument("--dump-raw-header", action="store_true", help="Also dump raw header bytes in hex")
    args = ap.parse_args()

    rec = 0
    with open(args.dat_file, "rb") as f:
        while True:
            try:
                fixed = read_exact(f, FIXED_HDR_SIZE)
            except EOFError:
                break

            magic, ver, _res, header_len, payload_len, recv_time_ns = struct.unpack(FIXED_HDR_FMT, fixed)
            if magic != MAGIC or ver != 1:
                print(f"[WARN] Bad record at #{rec}: magic/ver mismatch (magic={magic}, ver={ver})", file=sys.stderr)
                break

            try:
                header = read_exact(f, header_len) if header_len else b""
                payload = read_exact(f, payload_len) if payload_len else b""
            except EOFError:
                print("[WARN] Truncated record at EOF", file=sys.stderr)
                break

            rec += 1

            # --- ヘッダ解釈 ---
            if header_len < SC_HDR_SIZE:
                print(f"\n=== Record #{rec} ===")
                print(f"[WARN] header_len={header_len} < SC_HDR_SIZE={SC_HDR_SIZE}（sc_hdr未満）")
                if args.dump_raw_header and header:
                    print(f"raw_header({header_len}B) = {header.hex()}")
                # 以降は payload だけ軽く確認
                print(f"recv_time_ns = {recv_time_ns}")
                print(f"payload_len  = {payload_len} bytes")
                continue

            # sc_hdr 取り出し（20バイトだけ読む。余りは将来拡張用として raw で保持）
            sc_hdr_bytes = header[:SC_HDR_SIZE]
            extra_hdr = header[SC_HDR_SIZE:] if header_len > SC_HDR_SIZE else b""

            (sfn, slot, tx_port, rx_port_idx, n_rx, n_tx, k0, step, length_sc) = struct.unpack(SC_HDR_FMT, sc_hdr_bytes)

            # --- ペイロード確認（complex64 × len想定） ---
            if payload_len % 8 != 0:
                print(f"\n=== Record #{rec} ===")
                print(f"[WARN] payload_len={payload_len} is not multiple of 8 (complex64)")
                print(f"recv_time_ns = {recv_time_ns}")
                continue

            data = np.frombuffer(payload, dtype=np.complex64)
            if data.size != length_sc:
                print(f"\n=== Record #{rec} ===")
                print(f"[WARN] payload samples={data.size} but header.len={length_sc}")
                # 続行はする（ベストエフォート）
            n = data.size

            # --- 出力 ---
            print(f"\n=== Record #{rec} ===")
            print(f"recv_time_ns = {recv_time_ns}")
            print(f"header_len   = {header_len} bytes (sc_hdr={SC_HDR_SIZE}B, extra={len(extra_hdr)}B)")
            if args.dump_raw_header and header:
                print(f"raw_header({header_len}B) = {header.hex()}")

            # sc_hdr 全項目表示
            print("sc_hdr:")
            print(f"  sfn         = {sfn}")
            print(f"  slot        = {slot}")
            print(f"  tx_port     = {tx_port}")
            print(f"  rx_port_idx = {rx_port_idx}")
            print(f"  n_rx        = {n_rx}")
            print(f"  n_tx        = {n_tx}")
            print(f"  k0          = {k0}")
            print(f"  step        = {step}")
            print(f"  len         = {length_sc}  (payload={n} complex64)")

            # 中央寄せで --print-sc 本を抜く
            idxs = center_indices(n, args.print_sc)

            # 物理SC番号（k0 + i*step）も併記
            rows = []
            for i in idxs:
                c = data[i]
                amp = float(np.abs(c))
                phs = float(np.angle(c))
                k_phys = int(k0 + i * step)
                rows.append([
                    rx_port_idx, i, k_phys,
                    f"{c.real:.6f}", f"{c.imag:.6f}", f"{amp:.6f}", f"{phs:.6f}"
                ])

            print(format_table(
                rows,
                headers=["rx_port", "i(idx)", "k_phys", "Real", "Imag", "Amp", "Phase(rad)"]
            ))

if __name__ == "__main__":
    main()
