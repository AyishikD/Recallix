"""
Check the tamper-evident deletion log.

    python -m backend.tools.audit verify            # recompute the whole hash chain
    python -m backend.tools.audit receipts --user alice

`verify` exits 1 if any entry was changed, inserted or removed after it was written.
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    r = sub.add_parser("receipts")
    r.add_argument("--user", required=True)
    r.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)

    from backend.brain.memory.audit import AuditLog
    log = AuditLog()
    if args.cmd == "verify":
        ok, n, bad = log.verify()
        print(f"deletion log: {n} entries — " + ("intact" if ok else f"TAMPERED at entry {bad}"))
        return 0 if ok else 1
    print(json.dumps(log.receipts(args.user, args.limit), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
