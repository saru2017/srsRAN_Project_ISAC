#!/usr/bin/env python3
# save_zmq_to_dat.py  (multipart-aware, Ctrl-C safe)
import argparse
import datetime as dt
import signal
import struct
import sys
import time
from typing import List

import zmq  # pip install pyzmq

MAGIC = b"ZMQD"
# V2 header: magic, ver(2), reserved(2), recv_time_ns(u8), num_parts(u16)
FIXED_HDR_FMT_V2 = "<4sHHQH"
FIXED_HDR_SIZE_V2 = struct.calcsize(FIXED_HDR_FMT_V2)

# V1 header (legacy/compat): magic, ver(2), reserved(2), header_len(u32), payload_len(u32), recv_time_ns(u64)
FIXED_HDR_FMT_V1 = "<4sHHIIQ"
FIXED_HDR_SIZE_V1 = struct.calcsize(FIXED_HDR_FMT_V1)

running = True
def _stop(_sig=None, _frm=None):
    global running
    running = False

def main():
    parser = argparse.ArgumentParser(description="Subscribe ZMQ and append into a timestamped .dat (supports multipart).")
    parser.add_argument("--connect", default="tcp://127.0.0.1:5556", help="ZMQ endpoint to connect (SUB)")
    parser.add_argument("--topic", default=b"", type=lambda s: s.encode() if isinstance(s, str) else s,
                        help="Topic to subscribe (default: empty for all)")
    parser.add_argument("--outfile", default=None,
                        help="Optional output filename; if not set, use timestamp like 20250924_011954.dat")
    parser.add_argument("--rcv-timeout-ms", type=int, default=1000, help="ZMQ RCVTIMEO (ms)")
    parser.add_argument("--linger-ms", type=int, default=0, help="ZMQ LINGER on close")
    parser.add_argument("--force-v2", action="store_true",
                        help="Always save as V2 multipart format even if parts<=2 (recommended)")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = args.outfile or f"{ts}.dat"

    # Catch signals for clean exit
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    try:
        sock.setsockopt(zmq.SUBSCRIBE, args.topic)
        sock.setsockopt(zmq.RCVTIMEO, args.rcv_timeout_ms)  # timeout to allow graceful exit
        sock.setsockopt(zmq.LINGER, args.linger_ms)
        sock.connect(args.connect)

        if not args.quiet:
            print(f"[INFO] SUB connected to {args.connect}, writing to {outname} (Ctrl-C to stop)")

        n = 0
        with open(outname, "ab", buffering=0) as f:
            while running:
                try:
                    parts: List[bytes] = sock.recv_multipart(flags=0)
                except zmq.Again:
                    continue
                except KeyboardInterrupt:
                    break
                except zmq.error.ZMQError as e:
                    if not running:
                        break
                    print(f"[WARN] recv error: {e}", file=sys.stderr)
                    continue

                recv_time_ns = time.time_ns()

                # Decide saving format
                if (not args.force_v2) and len(parts) <= 2:
                    # Legacy V1 (header+payload) format for backward compatibility
                    header = parts[0] if len(parts) >= 1 else b""
                    payload = parts[1] if len(parts) >= 2 else b""

                    fixed_hdr = struct.pack(
                        FIXED_HDR_FMT_V1, MAGIC, 1, 0,
                        len(header), len(payload),
                        recv_time_ns
                    )
                    f.write(fixed_hdr)
                    if header:
                        f.write(header)
                    if payload:
                        f.write(payload)
                else:
                    # Modern V2 format: variable number of parts
                    num_parts = len(parts)
                    fixed_hdr = struct.pack(FIXED_HDR_FMT_V2, MAGIC, 2, 0, recv_time_ns, num_parts)
                    f.write(fixed_hdr)
                    # For each part: write u32 length followed by raw bytes
                    for blob in parts:
                        f.write(struct.pack("<I", len(blob)))
                        if blob:
                            f.write(blob)

                n += 1
                if (n & 0xFF) == 0 and not args.quiet:
                    print(f"[INFO] records written: {n}")
                    try:
                        f.flush()
                    except Exception:
                        pass

        if not args.quiet:
            print(f"[INFO] Stopping... total records written: {n}")
    finally:
        try:
            sock.close(args.linger_ms)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()
