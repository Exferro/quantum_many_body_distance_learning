#!/usr/bin/env python3
"""Download the public Zenodo dataset archive into the repo tree.

The downloaded archive is expected to contain:
- figures/
- tfim/raw_snapshots/

These are placed into the repository as:
- data/figures/
- data/tfim/raw_snapshots/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_ARCHIVE_NAME = "data_zenodo.zip"
DEFAULT_DATA_URL = (
    "https://zenodo.org/records/18892338/files/"
    "data_zenodo.zip?download=1"
)
DEFAULT_EXPECTED_SHA256 = (
    "b827b9c8d192597c4ce12c9bb36ef73c99a2848aa200b685ecfaf2540a34178d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Zenodo dataset archive and place it into the natural "
            "repository locations expected by the notebooks."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Direct dataset archive URL. If omitted, the script first reads "
            "QMBDL_DATA_URL from the environment and otherwise falls back to the "
            "current default URL in this file."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help=(
            "Optional SHA-256 checksum for the archive. If omitted, the script first "
            "reads QMBDL_DATA_SHA256 from the environment and otherwise falls back "
            "to the current default checksum in this file."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Optional repository root override. By default, the script auto-detects "
            "the repo root from its own location."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional cache directory for the downloaded archive. By default, the "
            "archive is stored under .cache/qmbdl/ in the repo."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target directories if they already exist.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive in the cache directory after extraction.",
    )
    return parser.parse_args()


def resolve_repo_root(*, repo_root_arg: str | None) -> Path:
    if repo_root_arg is not None:
        return Path(repo_root_arg).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_required_url(*, url_arg: str | None) -> str:
    url = url_arg or None
    if url is None:
        url = str.strip(os.environ.get("QMBDL_DATA_URL", ""))
    if not url:
        url = DEFAULT_DATA_URL
    return url


def resolve_expected_sha256(*, expected_sha256_arg: str | None) -> str | None:
    if expected_sha256_arg:
        return expected_sha256_arg.lower()
    env_value = str.strip(os.environ.get("QMBDL_DATA_SHA256", ""))
    if env_value:
        return env_value.lower()
    return DEFAULT_EXPECTED_SHA256


def infer_archive_name(*, url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    return filename or DEFAULT_ARCHIVE_NAME


def compute_sha256(*, archive_path: Path) -> str:
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    *,
    url: str,
    archive_path: Path,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset archive from {url}")
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
        total_size = response.headers.get("Content-Length")
        total_size_int = int(total_size) if total_size is not None else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size_int:
                percentage = 100.0 * downloaded / total_size_int
                print(
                    f"  downloaded {downloaded / 1024**3:.2f} / "
                    f"{total_size_int / 1024**3:.2f} GiB ({percentage:.1f}%)",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"  downloaded {downloaded / 1024**3:.2f} GiB",
                    end="\r",
                    flush=True,
                )
    print()
    print(f"Saved archive to {archive_path}")


def locate_payload_root(*, extraction_root: Path) -> Path:
    candidate_roots = [
        extraction_root,
        extraction_root / "data_zenodo",
        extraction_root / "data_zenodo_no_code",
    ]
    for candidate_root in candidate_roots:
        if (
            (candidate_root / "figures").is_dir()
            and (candidate_root / "tfim" / "raw_snapshots").is_dir()
        ):
            return candidate_root
    raise RuntimeError(
        "Could not find the extracted payload root containing figures/ and "
        "tfim/raw_snapshots/."
    )


def replace_tree(
    *,
    source_dir: Path,
    target_dir: Path,
    force: bool,
) -> None:
    if target_dir.exists():
        if not force:
            raise RuntimeError(
                f"Target already exists: {target_dir}. Re-run with --force to replace it."
            )
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src=source_dir, dst=target_dir)


def extract_and_install(
    *,
    archive_path: Path,
    repo_root: Path,
    force: bool,
) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Archive is not a valid ZIP file: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="qmbdl-data-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with zipfile.ZipFile(archive_path) as archive_handle:
            archive_handle.extractall(path=temp_dir)

        payload_root = locate_payload_root(extraction_root=temp_dir)
        figures_source_dir = payload_root / "figures"
        tfim_source_dir = payload_root / "tfim" / "raw_snapshots"

        figures_target_dir = repo_root / "data" / "figures"
        tfim_target_dir = repo_root / "data" / "tfim" / "raw_snapshots"

        replace_tree(
            source_dir=figures_source_dir,
            target_dir=figures_target_dir,
            force=force,
        )
        replace_tree(
            source_dir=tfim_source_dir,
            target_dir=tfim_target_dir,
            force=force,
        )

        print(f"Installed figure data to {figures_target_dir}")
        print(f"Installed TFIM raw snapshots to {tfim_target_dir}")


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(repo_root_arg=args.repo_root)
    url = resolve_required_url(url_arg=args.url)
    expected_sha256 = resolve_expected_sha256(
        expected_sha256_arg=args.expected_sha256,
    )

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else (repo_root / ".cache" / "qmbdl").resolve()
    )
    archive_name = infer_archive_name(url=url)
    archive_path = cache_dir / archive_name

    download_archive(
        url=url,
        archive_path=archive_path,
    )

    archive_sha256 = compute_sha256(archive_path=archive_path)
    print(f"Archive SHA-256: {archive_sha256}")
    if expected_sha256 is not None and archive_sha256 != expected_sha256:
        raise RuntimeError(
            "Downloaded archive checksum does not match the expected SHA-256."
        )

    extract_and_install(
        archive_path=archive_path,
        repo_root=repo_root,
        force=args.force,
    )

    if not args.keep_archive:
        archive_path.unlink()
        print(f"Removed cached archive {archive_path}")

    print("Dataset download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
