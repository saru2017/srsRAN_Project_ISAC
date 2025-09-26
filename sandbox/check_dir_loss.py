#!/usr/bin/env python3
# check_dir_loss.py
# Compute SRS loss stats for every .dat file in a directory by reusing zmq_calc_srs_loss.py logic.
#
# Usage examples:
#   python check_dir_loss.py --interval-ms 5 /path/to/dir
#   python check_dir_loss.py 10 /path/to/dir
#
# Notes:
# - Produces the SAME report block per file as zmq_calc_srs_loss.py.
# - Accepts either --interval-ms (recommended) or positional <interval_slots> like the original.
#
import argparse
import statistics
import sys
import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple

# --- Dynamically load zmq_calc_srs_loss from a sibling or known path ---
def load_zmq_calc(module_path: Path):
    spec = importlib.util.spec_from_file_location("zmq_calc_srs_loss", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load zmq_calc_srs_loss from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def process_one(mod, path: Path, interval_ms: Optional[float], interval_slots_cli: Optional[int],
                assume_mu: Optional[int], debug: bool):
    # Scan the file using original logic
    mu, spf, seq, deltas = mod.scan(str(path))

    # Handle mu fallback the same way as the original
    if mu is None:
        if assume_mu is not None:
            mu = assume_mu
            spf = mod.slots_per_frame_from_mu(mu)
        else:
            mu = 1
            spf = mod.slots_per_frame_from_mu(mu)

    slot_ms = 1.0 / (1 << mu)

    # Determine interval in slots just like the original
    if interval_ms is not None:
        interval_slots = max(1, int(round(interval_ms / slot_ms)))
        src = f"{interval_ms} ms"
    elif interval_slots_cli is not None:
        interval_slots = interval_slots_cli
        src = f"{interval_slots_cli} slots"
    else:
        raise SystemExit("Specify either --interval-ms or <interval_slots>.")

    median_delta = statistics.median(deltas) if deltas else 0
    recv, expected, lost, rate = mod.compute_loss(interval_slots, seq)

    # Emit the SAME report structure as zmq_calc_srs_loss.py
    print("=== SRS Loss Report ===")
    print(f"File                : {path}")
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

    if debug and seq:
        print("\n[Debug] First 5 (sfn, slot -> abs) rows:")
        for s in seq[:5]:
            print(f"  sfn={s[0]:4d}, slot={s[1]:3d} -> abs={s[2]}")
    print()  # blank line between reports

def main():
    ap = argparse.ArgumentParser(description="Compute SRS loss for all .dat files in a directory.")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--interval-ms", type=float, help="Expected SRS transmit interval in milliseconds (recommended)")
    ap.add_argument("interval_slots", nargs="?", type=int, help="Expected SRS interval in slots (legacy positional)")
    ap.add_argument("directory", help="Directory containing .dat files")
    ap.add_argument("--assume-mu", type=int, choices=[0,1,2,3,4], help="Assume numerology if AUX Part8 missing")
    ap.add_argument("--debug", action="store_true", help="Print diagnostics per file")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--pattern", default="*.dat", help="Filename glob pattern (default: *.dat)")

    args = ap.parse_args()

    dir_path = Path(args.directory).expanduser().resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"Not a directory: {dir_path}")

    # Try to locate zmq_calc_srs_loss.py near this script first, then alongside the given directory.
    candidates: List[Path] = []
    # 1) Same directory as this script
    this_script = Path(__file__).resolve()
    candidates.append(this_script.parent / "zmq_calc_srs_loss.py")
    # 2) The working directory
    candidates.append(Path.cwd() / "zmq_calc_srs_loss.py")
    # 3) The directory provided (useful if user keeps both together)
    candidates.append(dir_path / "zmq_calc_srs_loss.py")

    module_path: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if module_path is None:
        raise SystemExit("Could not find zmq_calc_srs_loss.py next to this script, in cwd, or in the target directory. "
                         "Please place it alongside this script or pass PYTHONPATH accordingly.")

    mod = load_zmq_calc(module_path)

    # Gather files
    files: List[Path] = []
    if args.recursive:
        files = sorted(dir_path.rglob(args.pattern))
    else:
        files = sorted(dir_path.glob(args.pattern))

    if not files:
        print(f"No files matched pattern '{args.pattern}' under {dir_path}")
        return

    for f in files:
        # Skip the helper module itself if it matches pattern
        if f.name == "zmq_calc_srs_loss.dat":  # unlikely, but avoid confusion
            continue
        process_one(mod, f, args.interval_ms, args.interval_slots, args.assume_mu, args.debug)

if __name__ == "__main__":
    main()
