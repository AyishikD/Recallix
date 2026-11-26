"""backend.tools.get_engine: installs a release's engine only when its checksum matches."""
import hashlib
import io
import os
import tarfile

import pytest

from backend.tools import get_engine


def fake_release(folder, target="linux-arm64", payload=b"#!engine", sums_override=None):
    name = f"recallix-engine-{target}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(f"{name}/memory_service")
        info.size, info.mode = len(payload), 0o755
        tar.addfile(info, io.BytesIO(payload))
    archive = buf.getvalue()
    (folder / f"{name}.tar.gz").write_bytes(archive)
    digest = sums_override or hashlib.sha256(archive).hexdigest()
    (folder / "SHA256SUMS").write_text(f"{digest}  {name}.tar.gz\n")
    return folder.as_uri()


def test_installs_the_binary_when_the_checksum_matches(tmp_path):
    base = fake_release(tmp_path)
    dest = tmp_path / "out" / "memory_service"
    get_engine.main(["--target", "linux-arm64", "--base-url", base, "--dest", str(dest)])
    assert dest.read_bytes() == b"#!engine"
    assert os.access(dest, os.X_OK)


def test_refuses_a_binary_whose_checksum_does_not_match(tmp_path):
    base = fake_release(tmp_path, sums_override="0" * 64)
    dest = tmp_path / "out" / "memory_service"
    with pytest.raises(SystemExit):
        get_engine.main(["--target", "linux-arm64", "--base-url", base, "--dest", str(dest)])
    assert not dest.exists()


def test_detects_a_build_for_this_machine_or_says_none():
    assert get_engine.target() in (None, "linux-x86_64", "linux-x86_64-noavx2", "linux-arm64", "macos-arm64")
