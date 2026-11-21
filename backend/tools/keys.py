"""
API key management.

  python -m backend.tools.keys create --user alice [--role user|service|admin] [--label "..."]
  python -m backend.tools.keys list
  python -m backend.tools.keys revoke <prefix>
"""
import argparse
import time

from backend.api import keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--user", required=True)
    c.add_argument("--role", default="user", choices=keys.ROLES)
    c.add_argument("--label", default="")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("prefix")
    args = ap.parse_args()

    if args.cmd == "create":
        print(keys.create(args.user, args.role, args.label))
    elif args.cmd == "list":
        for k in keys.list_keys():
            created = time.strftime("%Y-%m-%d", time.localtime(k["created_at"] or 0))
            print(f"{k['prefix']:10} {k['role']:8} {k['user_id']:20} {created}  {k['label']}")
    else:
        n = keys.revoke(args.prefix)
        print(f"revoked {n} key(s)")


if __name__ == "__main__":
    main()
