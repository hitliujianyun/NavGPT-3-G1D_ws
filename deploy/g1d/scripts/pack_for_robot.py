"""Package navgpt_g1d for the G1-D PC2: dist/navgpt_g1d-<commit>.tar.gz.

    python deploy/g1d/scripts/pack_for_robot.py [--no-sdk]
    scp dist/navgpt_g1d-<commit>.tar.gz unitree@192.168.123.164:~
    # on PC2:  tar xzf navgpt_g1d-<commit>.tar.gz && sudo navgpt_g1d/pc2/install.sh --enable

Built with ``git archive`` from the committed tree (uncommitted changes are not
included), so line endings follow .gitattributes (LF) even on a Windows
checkout; shell scripts are marked executable. The unitree_sdk2_python
submodule is bundled under third_party/ for PC2s that do not have it yet.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path


MUST_BE_LF = (".sh", ".service", ".env", ".py", ".yaml", ".xml")


def git(args, cwd: Path) -> bytes:
    # Archiving a subtree does not see the root .gitattributes, so on Windows core.autocrlf would
    # turn every file into CRLF; with conversion off the archive carries the stored (LF) blobs.
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", *args]
    return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="output directory (default: <repo>/dist)")
    parser.add_argument("--no-sdk", action="store_true", help="do not bundle unitree_sdk2_python")
    args = parser.parse_args()

    repo = Path(git(["rev-parse", "--show-toplevel"], Path(__file__).parent).decode().strip())
    commit = git(["rev-parse", "--short", "HEAD"], repo).decode().strip()
    dirty = git(["status", "--porcelain", "--", "deploy/g1d"], repo).decode().strip()
    if dirty:
        print("warning: uncommitted changes in deploy/g1d are NOT packed:\n" + dirty, file=sys.stderr)

    parts = [git(["archive", "--format=tar", "--prefix=navgpt_g1d/", "HEAD:deploy/g1d"], repo)]
    versions = ["navgpt_g1d {} ({})".format(commit, time.strftime("%Y-%m-%d %H:%M:%S"))]
    sdk = repo / "code" / "unitree_sdk2_python"
    if not args.no_sdk:
        if not (sdk / ".git").exists():
            raise SystemExit("code/unitree_sdk2_python is missing: git submodule update --init code/unitree_sdk2_python")
        parts.append(git(["archive", "--format=tar", "--prefix=navgpt_g1d/third_party/unitree_sdk2_python/", "HEAD"], sdk))
        versions.append("unitree_sdk2_python {}".format(git(["rev-parse", "--short", "HEAD"], sdk).decode().strip()))

    out_dir = Path(args.out) if args.out else repo / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "navgpt_g1d-{}.tar.gz".format(commit)
    count, crlf = 0, []
    with tarfile.open(out, "w:gz") as dst:
        for blob in parts:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as src:
                for member in src.getmembers():
                    data = src.extractfile(member).read() if member.isfile() else None
                    if data is not None and member.name.endswith(MUST_BE_LF) and b"\r\n" in data:
                        crlf.append(member.name)
                    if member.isfile() and member.name.endswith(".sh"):
                        member.mode |= 0o755
                    member.uname = member.gname = "unitree"
                    dst.addfile(member, io.BytesIO(data) if data is not None else None)
                    count += member.isfile()
        text = ("\n".join(versions) + "\n").encode()
        info = tarfile.TarInfo("navgpt_g1d/VERSION")
        info.size, info.mtime, info.mode = len(text), int(time.time()), 0o644
        dst.addfile(info, io.BytesIO(text))
    if crlf:
        out.unlink()
        raise SystemExit("CRLF line endings in {} file(s), e.g. {}; fix .gitattributes and recommit".format(len(crlf), crlf[:3]))
    print("{} ({} files, {:.1f} MB)".format(out, count, out.stat().st_size / 1e6))
    print("\n".join("  " + v for v in versions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
