"""
One-time local setup.

  python -m backend.tools.setup_env [--admin-user admin]

* Writes RECALLIC_INTERNAL_KEY (random, 64 hex chars) to .env if it is not set.
  The API reads .env automatically; export it for the engine too:
      set -a; source .env; set +a; ./backend/infra_cpp/build/memory_service
* Creates the storage directories.
* Creates an admin API key (printed once) if no keys exist yet.
"""
import argparse
import os
import secrets

from backend import settings
from backend.api import keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-user", default="admin")
    args = ap.parse_args()

    env_path = settings.PROJECT_ROOT / ".env"
    existing = env_path.read_text() if env_path.exists() else ""
    if "RECALLIC_INTERNAL_KEY=" not in existing and not os.environ.get("RECALLIC_INTERNAL_KEY"):
        with open(env_path, "a") as f:
            f.write(("" if existing.endswith("\n") or not existing else "\n") +
                    f"RECALLIC_INTERNAL_KEY={secrets.token_hex(32)}\n")
        os.chmod(env_path, 0o600)
        print(f"wrote RECALLIC_INTERNAL_KEY to {env_path}")
    else:
        print("RECALLIC_INTERNAL_KEY already configured")

    print(f"storage root: {settings.storage_path(is_dir=True)}")

    if not keys.list_keys():
        key = keys.create(args.admin_user, role="admin", label="setup_env")
        print(f"\nadmin API key (shown once, store it safely):\n  {key}\n")
    else:
        print("API keys already exist (python -m backend.tools.keys list)")


if __name__ == "__main__":
    main()
