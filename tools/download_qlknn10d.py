# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — QLKNN-10D Dataset Downloader
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Download the QLKNN-10D public dataset from Zenodo.

The QLKNN-10D dataset contains ~300 million QuaLiKiz v2.4.0 turbulent
transport flux calculations covering ITG/TEM/ETG regimes.  This is the
gold-standard training dataset for neural transport surrogates in
tokamak plasmas.

Citation
--------
van de Plassche, K.L. et al. (2020). "Fast modeling of turbulent
transport in fusion plasmas using neural networks."
*Phys. Plasmas* 27, 022310. doi:10.1063/1.5134126

Usage
-----
    python tools/download_qlknn10d.py                # download
    python tools/download_qlknn10d.py --check         # verify only
    python tools/download_qlknn10d.py --output-dir /d  # custom dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Zenodo metadata ──────────────────────────────────────────────────

# Primary record: QLKNN-10D dataset (v3 = latest at time of writing)
ZENODO_RECORD_ID = "3497066"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

DOI = "10.5281/zenodo.3497066"
CITATION = (
    'van de Plassche, K.L. et al. (2020). "Fast modeling of turbulent '
    'transport in fusion plasmas using neural networks." '
    "Phys. Plasmas 27, 022310. doi:10.1063/1.5134126"
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "qlknn10d"


# ── Helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_with_progress(url: str, dest: Path, expected_size: int = 0) -> None:
    """Download a URL to *dest* with a progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if partial file exists, use Range header
    start_byte = 0
    mode = "wb"
    if dest.exists():
        start_byte = dest.stat().st_size
        if expected_size and start_byte >= expected_size:
            print(f"  Already complete: {dest.name}")
            return
        mode = "ab"

    req = urllib.request.Request(url)
    if start_byte > 0:
        req.add_header("Range", f"bytes={start_byte}-")
        print(f"  Resuming {dest.name} from {_human_size(start_byte)}...")

    try:
        with urllib.request.urlopen(req) as resp, open(dest, mode) as f:
            total = expected_size or int(resp.headers.get("Content-Length", 0))
            downloaded = start_byte
            t0 = time.monotonic()
            last_print = t0

            while True:
                chunk = resp.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_print > 2.0:
                    pct = (downloaded / total * 100) if total else 0
                    speed = (downloaded - start_byte) / max(now - t0, 0.01) / 1e6
                    print(
                        f"\r  {dest.name}: {_human_size(downloaded)}"
                        f" / {_human_size(total)}"
                        f" ({pct:.1f}%) [{speed:.1f} MB/s]",
                        end="", flush=True,
                    )
                    last_print = now

            print()  # newline after progress
    except urllib.error.HTTPError as e:
        if e.code == 416 and start_byte > 0:
            print(f"  Already complete: {dest.name}")
        else:
            raise


def fetch_zenodo_files() -> list[dict]:
    """Query the Zenodo API for the file listing of this record."""
    print(f"Querying Zenodo record {ZENODO_RECORD_ID}...")
    with urllib.request.urlopen(ZENODO_API_URL) as resp:
        meta = json.loads(resp.read().decode())

    files = []
    for f in meta.get("files", []):
        files.append({
            "filename": f["key"],
            "size": f["size"],
            "checksum": f.get("checksum", ""),  # "md5:abc..."
            "url": f["links"]["self"],
        })
    print(f"  Found {len(files)} files ({_human_size(sum(f['size'] for f in files))} total)")
    return files


# ── Main operations ──────────────────────────────────────────────────

def download(output_dir: Path) -> None:
    """Download all dataset files from Zenodo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = fetch_zenodo_files()

    for finfo in files:
        dest = output_dir / finfo["filename"]
        print(f"\nDownloading {finfo['filename']} ({_human_size(finfo['size'])})...")
        _download_with_progress(finfo["url"], dest, finfo["size"])

        # Verify checksum if available (Zenodo provides md5:...)
        checksum_str = finfo.get("checksum", "")
        if checksum_str.startswith("md5:"):
            expected_md5 = checksum_str[4:]
            h = hashlib.md5()
            with open(dest, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            actual_md5 = h.hexdigest()
            if actual_md5 != expected_md5:
                print(f"  CHECKSUM MISMATCH: {dest.name}")
                print(f"    Expected: {expected_md5}")
                print(f"    Got:      {actual_md5}")
                sys.exit(1)
            else:
                print(f"  Checksum OK: {dest.name}")

    # Write provenance README
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        f"# QLKNN-10D Dataset\n\n"
        f"**DOI**: {DOI}\n"
        f"**Zenodo Record**: {ZENODO_RECORD_ID}\n"
        f"**Downloaded**: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## Citation\n\n{CITATION}\n\n"
        f"## Files\n\n"
        + "\n".join(
            f"- `{f['filename']}` ({_human_size(f['size'])})"
            for f in files
        )
        + "\n\n## Description\n\n"
        "~300 million QuaLiKiz v2.4.0 turbulent transport flux calculations\n"
        "covering ITG, TEM, and ETG regimes.  10-dimensional input space\n"
        "(Ati, Ate, An, q, smag, x, Ti_Te, Zeff, alpha, Machtor) mapped to\n"
        "turbulent fluxes (efi_GB, efe_GB, pfe_GB) in gyro-Bohm units.\n",
        encoding="utf-8",
    )
    print(f"\nProvenance written to {readme_path}")
    print("Download complete.")


def check(output_dir: Path) -> bool:
    """Verify integrity of an existing download."""
    print(f"Checking {output_dir}...")
    if not output_dir.exists():
        print("  Directory does not exist. Run without --check first.")
        return False

    files = fetch_zenodo_files()
    all_ok = True

    for finfo in files:
        dest = output_dir / finfo["filename"]
        if not dest.exists():
            print(f"  MISSING: {finfo['filename']}")
            all_ok = False
            continue

        actual_size = dest.stat().st_size
        if actual_size != finfo["size"]:
            print(
                f"  SIZE MISMATCH: {finfo['filename']} "
                f"(expected {finfo['size']}, got {actual_size})"
            )
            all_ok = False
            continue

        checksum_str = finfo.get("checksum", "")
        if checksum_str.startswith("md5:"):
            expected_md5 = checksum_str[4:]
            h = hashlib.md5()
            with open(dest, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_md5:
                print(f"  CHECKSUM MISMATCH: {finfo['filename']}")
                all_ok = False
                continue

        print(f"  OK: {finfo['filename']} ({_human_size(actual_size)})")

    if all_ok:
        print("\nAll files verified.")
    else:
        print("\nSome files are missing or corrupted. Re-run without --check.")
    return all_ok


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the QLKNN-10D dataset from Zenodo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing download without downloading.",
    )
    args = parser.parse_args()

    if args.check:
        ok = check(args.output_dir)
        sys.exit(0 if ok else 1)
    else:
        download(args.output_dir)


if __name__ == "__main__":
    main()
