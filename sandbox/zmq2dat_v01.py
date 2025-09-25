#!/usr/bin/env python3
# save_zmq_to_dat.py  (Ctrl-C で確実に止まる版)
import argparse
import datetime as dt
import signal
import struct
import sys
import time

import zmq  # pip install pyzmq

MAGIC = b"ZMQD"
FIXED_HDR_FMT = "<4sHHIIQ"  # magic, ver, reserved, header_len, payload_len, recv_time_ns
FIXED_HDR_SIZE = struct.calcsize(FIXED_HDR_FMT)
VERSION = 1

running = True
def _stop(_sig=None, _frm=None):
    global running
    running = False

def main():
    parser = argparse.ArgumentParser(description="Subscribe ZMQ and append into a timestamped .dat")
    parser.add_argument("--connect", default="tcp://127.0.0.1:5556", help="ZMQ endpoint to connect (SUB)")
    parser.add_argument("--topic", default=b"", type=lambda s: s.encode() if isinstance(s, str) else s,
                        help="Topic to subscribe (default: empty for all)")
    parser.add_argument("--outfile", default=None,
                        help="Optional output filename; if not set, use timestamp like 20250924_011954.dat")
    parser.add_argument("--rcv-timeout-ms", type=int, default=1000, help="ZMQ RCVTIMEO (ms)")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = args.outfile or f"{ts}.dat"

    # シグナルを捕まえる
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    try:
        sock.setsockopt(zmq.SUBSCRIBE, args.topic)
        sock.setsockopt(zmq.RCVTIMEO, args.rcv_timeout_ms)  # タイムアウトで EAGAIN を投げる
        sock.setsockopt(zmq.LINGER, 0)                      # 終了時にすぐ閉じる
        sock.connect(args.connect)

        print(f"[INFO] SUB connected to {args.connect}, writing to {outname} (Ctrl-C to stop)")

        n = 0
        with open(outname, "ab", buffering=0) as f:
            while running:
                try:
                    parts = sock.recv_multipart(flags=0)  # タイムアウトあり
                except zmq.Again:
                    # タイムアウト：running を見て続行/終了を判断
                    continue
                except KeyboardInterrupt:
                    break
                except zmq.error.ZMQError as e:
                    # 終了中に出ることがあるので running と合わせて扱う
                    if not running:
                        break
                    print(f"[WARN] recv error: {e}", file=sys.stderr)
                    continue

                recv_time_ns = time.time_ns()

                if len(parts) == 1:
                    header = b""
                    payload = parts[0]
                else:
                    header = parts[0]
                    payload = parts[1] if len(parts) > 1 else b""

                fixed_hdr = struct.pack(
                    FIXED_HDR_FMT,
                    MAGIC, VERSION, 0,
                    len(header), len(payload),
                    recv_time_ns
                )
                f.write(fixed_hdr)
                if header:
                    f.write(header)
                if payload:
                    f.write(payload)
                n += 1

                # たまに flush（バッファリング0でも安全に）
                if (n & 0x0FF) == 0:
                    try:
                        print("saving...")
                        f.flush()
                    except Exception:
                        pass

        print(f"[INFO] Stopping... total records written: {n}")
    finally:
        # ソケット→コンテキストを確実にクリーンアップ
        try:
            sock.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()
