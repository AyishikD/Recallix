"""
Download a prebuilt vector engine instead of compiling it with cmake.

    python -m backend.tools.get_engine              # latest release
    python -m backend.tools.get_engine --tag v1.0.0 # a specific release

Picks the build for this machine (Linux x86-64 with or without AVX2, Linux ARM64, macOS Apple
Silicon), checks it against the release's SHA256SUMS, and puts it where the cmake build would:
backend/infra_cpp/build/memory_service. Other platforms (Intel Mac, Windows): use Docker or cmake.
The binaries are built and tested by .github/workflows/engine-release.yml.
"""
import argparse
import hashlib
import io
import os
import platform
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Recallic/Recallic"
DEST = Path(__file__).resolve().parents[1] / "infra_cpp" / "build" / "memory_service"


def has_avx2_fma():
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return " avx2" in flags and " fma" in flags


def target():
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "linux-x86_64" if has_avx2_fma() else "linux-x86_64-noavx2"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "linux-arm64"
    if system == "Darwin" and machine == "arm64":
        return "macos-arm64"
    return None


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "recallic-get-engine"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download the prebuilt Recallic vector engine for this machine.")
    ap.add_argument("--tag", default=None, help="release tag (default: the latest release)")
    ap.add_argument("--target", default=None, help="override the detected build, e.g. linux-x86_64-noavx2")
    ap.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # tests: a local folder of release files
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args(argv)

    tgt = args.target or target()
    if not tgt:
        sys.exit(f"No prebuilt engine for {platform.system()} {platform.machine()}. "
                 "Use Docker, or build it: cmake -S backend/infra_cpp -B backend/infra_cpp/build "
                 "-DCMAKE_BUILD_TYPE=Release && cmake --build backend/infra_cpp/build -j")
    base = args.base_url or (f"https://github.com/{REPO}/releases/download/{args.tag}" if args.tag
                             else f"https://github.com/{REPO}/releases/latest/download")
    archive = None
    for name in (f"recallic-engine-{tgt}", f"recallix-engine-{tgt}"):  # releases made before the rename use recallix-
        print(f"downloading {name} from {base}")
        try:
            archive = fetch(f"{base}/{name}.tar.gz")
            break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                sys.exit(f"download failed ({e.code})")
    if archive is None:
        sys.exit(f"download failed: is there a release{' ' + args.tag if args.tag else ''} with recallic-engine-{tgt}.tar.gz?")
    try:
        sums = fetch(f"{base}/SHA256SUMS").decode()
    except urllib.error.HTTPError as e:
        sys.exit(f"download failed ({e.code}) for SHA256SUMS")

    expected = {parts[-1].lstrip("*"): parts[0] for parts in (line.split() for line in sums.splitlines()) if len(parts) >= 2}
    got = hashlib.sha256(archive).hexdigest()
    if expected.get(f"{name}.tar.gz") != got:
        sys.exit(f"checksum mismatch for {name}.tar.gz (expected {expected.get(name + '.tar.gz')}, got {got}); not installed")

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember(f"{name}/memory_service")  # only this file is used; nothing else is extracted
        if not member.isfile():
            sys.exit("unexpected archive contents; not installed")
        binary = tar.extractfile(member).read()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    tmp.write_bytes(binary)
    os.chmod(tmp, 0o755)
    tmp.replace(dest)
    print(f"installed {dest} ({tgt}, sha256 {got[:12]}...)")
    print("start it with: set -a; source .env; set +a; ./backend/infra_cpp/build/memory_service &")


if __name__ == "__main__":
    main()
