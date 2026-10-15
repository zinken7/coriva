"""
Download + extract the nuScenes dataset for the SP3 Kinematic Digital Twin.

nuScenes is gated: you must register at https://www.nuscenes.org and accept the
license. The **mini** split (v1.0-mini, ~4 GB) is publicly downloadable and is
enough to build/test the SP3 ingestion adapter. The full **trainval**/**test**
splits are multi-part and require download links from YOUR logged-in account —
pass those via --url (one invocation per part).

Usage
-----
    # Mini split (default URL), extract, and verify with the devkit:
    python3 scripts/download_nuscenes.py --split mini --out-dir data/nuscenes --extract --verify

    # A gated split, using a link copied from your nuScenes account page:
    python3 scripts/download_nuscenes.py --split trainval \\
        --url "https://<your-account-download-link>/v1.0-trainval01_blobs.tgz" \\
        --out-dir data/nuscenes --extract

Install the devkit first (heavy deps):  pip install -r requirements-nuscenes.txt
"""

import argparse
import os
import sys
import tarfile
import urllib.request

# Publicly downloadable split(s). Gated splits must be passed via --url.
KNOWN_URLS = {
    "mini": "https://www.nuscenes.org/data/v1.0-mini.tgz",
}

# nuScenes devkit version string per split (used by --verify).
VERSION_BY_SPLIT = {
    "mini": "v1.0-mini",
    "trainval": "v1.0-trainval",
    "test": "v1.0-test",
}


def resolve_download(split, url_override=None):
    """Return (url, filename). Gated splits without a --url raise a clear error."""
    if url_override:
        filename = url_override.split("?")[0].rstrip("/").split("/")[-1] or f"nuscenes-{split}.tgz"
        return url_override, filename
    if split in KNOWN_URLS:
        url = KNOWN_URLS[split]
        return url, url.rstrip("/").split("/")[-1]
    raise ValueError(
        f"split {split!r} is gated: register at nuscenes.org, then pass the download "
        f"link from your account via --url (e.g. v1.0-{split}*_blobs.tgz)."
    )


def download(url, dest, chunk=1 << 20):
    """Stream a URL to dest with a simple progress indicator. Skips if dest exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest)} bytes)")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[download] {url}\n        -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "coriva-nuscenes-downloader"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (intentional remote fetch)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r        {done >> 20} / {total >> 20} MB ({pct:5.1f}%)", end="", flush=True)
    print()
    return dest


def extract(tgz_path, out_dir):
    """Extract a .tgz into out_dir (path-traversal safe)."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[extract] {tgz_path} -> {out_dir}")
    with tarfile.open(tgz_path, "r:*") as tar:
        try:
            tar.extractall(out_dir, filter="data")  # Python 3.12+: blocks unsafe members
        except TypeError:
            tar.extractall(out_dir)  # older Python: no filter kwarg
    return out_dir


def verify(dataroot, version):
    """Load the dataset with the devkit and report the scene count."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("[verify] nuscenes-devkit not installed; skip "
              "(pip install -r requirements-nuscenes.txt)")
        return None
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"[verify] OK: {len(nusc.scene)} scenes, {len(nusc.sample)} samples "
          f"(version={version}, dataroot={dataroot})")
    return len(nusc.scene)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download/extract the nuScenes dataset")
    parser.add_argument("--split", default="mini", choices=["mini", "trainval", "test"])
    parser.add_argument("--url", default=None, help="override download URL (required for gated splits)")
    parser.add_argument("--out-dir", default="data/nuscenes", help="dataset root")
    parser.add_argument("--extract", action="store_true", help="extract the tarball after download")
    parser.add_argument("--verify", action="store_true", help="verify with nuscenes-devkit after extraction")
    args = parser.parse_args(argv)

    url, filename = resolve_download(args.split, args.url)
    tgz_path = os.path.join(args.out_dir, filename)

    download(url, tgz_path)
    if args.extract:
        extract(tgz_path, args.out_dir)
    if args.verify:
        verify(args.out_dir, VERSION_BY_SPLIT[args.split])

    print("\nDone. Point the SP3 ingestion adapter at:", args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
