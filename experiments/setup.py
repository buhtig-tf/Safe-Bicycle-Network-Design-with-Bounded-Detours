#!/usr/bin/env python3
"""
Prepare the BNIP reproducibility artifact.

Expected repository layout:

    repo/
    ├── zipfiles/
    │   ├── original.tar.xz
    │   └── networks.zip
    └── experiments/
        ├── setup.py
        ├── generate_networks.py
        └── data/
            └── adfc/
                └── ...

Archive layout:

    original.tar.xz
    └── original/
        ├── villages/
        └── regions/

    networks.zip
    └── networks/
        ├── villages/
        └── regions/

Default setup uses only the frozen publication artifacts:

    zipfiles/original.tar.xz
        -> experiments/data/instances/original/

    zipfiles/networks.zip
        -> experiments/data/networks/

No live OpenStreetMap query is performed by default.

Use --regenerate-networks to deliberately replace the frozen networks by
rerunning generate_networks.py against current live OSM data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent

ZIPFILES_DIR = REPO_ROOT / "zipfiles"

DEFAULT_INSTANCES_ARCHIVE = (
    ZIPFILES_DIR / "original.tar.xz"
)

DEFAULT_NETWORKS_ARCHIVE = (
    ZIPFILES_DIR / "networks.zip"
)

DATA_DIR = (
    EXPERIMENTS_DIR / "data"
)

INSTANCE_ROOT = (
    DATA_DIR / "instances"
)

ORIGINAL_INSTANCE_ROOT = (
    INSTANCE_ROOT / "original"
)

REDUCED_INSTANCE_ROOT = (
    INSTANCE_ROOT / "reduced"
)

NETWORK_ROOT = (
    DATA_DIR / "networks"
)

RESULT_ROOT = (
    DATA_DIR / "results"
)

PLOT_ROOT = (
    EXPERIMENTS_DIR / "plots"
)


def safe_extract(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    """
    Extract a ZIP archive while rejecting path-traversal entries.
    """

    destination = destination.resolve()

    for member in archive.infolist():

        target = (
            destination
            /
            member.filename
        ).resolve()

        try:

            target.relative_to(
                destination
            )

        except ValueError as exc:

            raise RuntimeError(
                "Unsafe path in ZIP archive: "
                f"{member.filename!r}"
            ) from exc

    archive.extractall(
        destination
    )


def top_level_entries(
    archive: zipfile.ZipFile,
) -> set[str]:

    return {
        Path(
            member.filename
        ).parts[0]
        for member in archive.infolist()
        if (
            member.filename
            and Path(
                member.filename
            ).parts
        )
    }


def count_pickles(
    root: Path,
) -> int:

    if not root.exists():

        return 0

    return sum(
        1
        for _ in root.rglob(
            "*.pkl"
        )
    )


def extract_instances(
    archive_path: Path,
    *,
    force: bool,
) -> None:

    if not archive_path.is_file():

        raise FileNotFoundError(
            "Instance archive not found:\n"
            f"  {archive_path}"
        )

    existing_count = count_pickles(
        ORIGINAL_INSTANCE_ROOT
    )

    if existing_count > 0:

        if not force:

            print(
                "[instances] already present:"
            )

            print(
                f"  {ORIGINAL_INSTANCE_ROOT}"
            )

            print(
                f"  {existing_count} .pkl files"
            )

            return

        shutil.rmtree(
            ORIGINAL_INSTANCE_ROOT
        )

    elif (
        force
        and ORIGINAL_INSTANCE_ROOT.exists()
    ):

        shutil.rmtree(
            ORIGINAL_INSTANCE_ROOT
        )

    INSTANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[instances] extracting:"
    )

    print(
        f"  {archive_path}"
    )

    print(
        "into:"
    )

    print(
        f"  {INSTANCE_ROOT}"
    )

    with tarfile.open(
        archive_path,
        "r:xz",
    ) as archive:

        members = archive.getmembers()

        roots = {
            Path(member.name).parts[0]
            for member in members
            if (
                member.name
                and Path(member.name).parts
            )
        }

        if roots != {"original"}:

            raise RuntimeError(
                "original.tar.xz must contain exactly one "
                "top-level directory named 'original/'. "
                f"Found: {sorted(roots)!r}"
            )

        destination = INSTANCE_ROOT.resolve()

        for member in members:

            member_path = Path(
                member.name
            )

            if member_path.is_absolute():

                raise RuntimeError(
                    "Unsafe absolute path in TAR archive: "
                    f"{member.name!r}"
                )

            target = (
                destination
                /
                member_path
            ).resolve()

            try:

                target.relative_to(
                    destination
                )

            except ValueError as exc:

                raise RuntimeError(
                    "Unsafe path in TAR archive: "
                    f"{member.name!r}"
                ) from exc

            if not (
                member.isfile()
                or member.isdir()
            ):

                raise RuntimeError(
                    "Only regular files and directories are "
                    "allowed in the instance archive: "
                    f"{member.name!r}"
                )

        archive.extractall(
            INSTANCE_ROOT
        )

    instance_count = count_pickles(
        ORIGINAL_INSTANCE_ROOT
    )

    if instance_count == 0:

        raise RuntimeError(
            "original.tar.xz was extracted, but no .pkl "
            "files were found under:\n"
            f"  {ORIGINAL_INSTANCE_ROOT}"
        )

    print(
        f"[instances] extracted {instance_count} .pkl files"
    )


def extract_networks(
    archive_path: Path,
    *,
    force: bool,
) -> None:

    if not archive_path.is_file():

        raise FileNotFoundError(
            "Network archive not found:\n"
            f"  {archive_path}"
        )

    existing_count = count_pickles(
        NETWORK_ROOT
    )

    if existing_count > 0:

        if not force:

            print(
                "[networks] already present:"
            )

            print(
                f"  {NETWORK_ROOT}"
            )

            print(
                f"  {existing_count} .pkl files"
            )

            return

        shutil.rmtree(
            NETWORK_ROOT
        )

    elif (
        force
        and NETWORK_ROOT.exists()
    ):

        shutil.rmtree(
            NETWORK_ROOT
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[networks] extracting:"
    )

    print(
        f"  {archive_path}"
    )

    print(
        "into:"
    )

    print(
        f"  {DATA_DIR}"
    )

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:

        roots = top_level_entries(
            archive
        )

        if "networks" not in roots:

            raise RuntimeError(
                "networks.zip must contain a "
                "'networks/' directory at its root."
            )

        safe_extract(
            archive,
            DATA_DIR,
        )

    network_count = count_pickles(
        NETWORK_ROOT
    )

    if network_count == 0:

        raise RuntimeError(
            "networks.zip was extracted, but no .pkl "
            "files were found under:\n"
            f"  {NETWORK_ROOT}"
        )

    print(
        f"[networks] extracted {network_count} .pkl files"
    )


def regenerate_networks() -> None:
    """
    Deliberately regenerate networks from current live OSM data.

    This does not reproduce the frozen publication networks exactly.
    """

    generator = (
        EXPERIMENTS_DIR
        /
        "generate_networks.py"
    )

    if not generator.is_file():

        raise FileNotFoundError(
            "Network generator not found:\n"
            f"  {generator}"
        )

    if NETWORK_ROOT.exists():

        shutil.rmtree(
            NETWORK_ROOT
        )

    NETWORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(
            generator
        ),
        "--safety_models",
        "A",
        "B",
        "C",
        "--radii",
        "3",
    ]

    print()
    print(
        "[networks] regenerating from current OSM data"
    )

    print(
        "[networks] command:"
    )

    print(
        "  "
        +
        " ".join(
            command
        )
    )

    subprocess.run(
        command,
        cwd=EXPERIMENTS_DIR,
        check=True,
    )

    network_count = count_pickles(
        NETWORK_ROOT
    )

    if network_count == 0:

        raise RuntimeError(
            "Network generation completed, but no "
            ".pkl network files were produced under:\n"
            f"  {NETWORK_ROOT}"
        )

    print(
        f"[networks] generated {network_count} .pkl files"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare frozen BNIP paper networks and instances."
        )
    )

    parser.add_argument(
        "--instances-archive",
        type=Path,
        default=DEFAULT_INSTANCES_ARCHIVE,
        help=(
            "Default: ../zipfiles/original.tar.xz "
            "relative to experiments/setup.py"
        ),
    )

    parser.add_argument(
        "--networks-archive",
        type=Path,
        default=DEFAULT_NETWORKS_ARCHIVE,
        help=(
            "Default: ../zipfiles/networks.zip "
            "relative to experiments/setup.py"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract existing frozen networks and "
            "original instances."
        ),
    )

    parser.add_argument(
        "--regenerate-networks",
        action="store_true",
        help=(
            "Ignore networks.zip and regenerate networks "
            "from current live OSM data using "
            "generate_networks.py --safety_models A B C --radii 3. "
            "This is NOT exact paper reproduction."
        ),
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INSTANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REDUCED_INSTANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.regenerate_networks:

        regenerate_networks()

    else:

        extract_networks(
            args.networks_archive.resolve(),
            force=args.force,
        )

    extract_instances(
        args.instances_archive.resolve(),
        force=args.force,
    )

    print()
    print(
        "Setup complete."
    )

    print(
        f"Frozen/or current networks: {NETWORK_ROOT}"
    )

    print(
        f"Original paper instances:   {ORIGINAL_INSTANCE_ROOT}"
    )

    print(
        f"Reduced instances:          {REDUCED_INSTANCE_ROOT}"
    )

    print(
        f"Experiment results:         {RESULT_ROOT}"
    )

    print(
        f"Plots:                      {PLOT_ROOT}"
    )


if __name__ == "__main__":
    main()
