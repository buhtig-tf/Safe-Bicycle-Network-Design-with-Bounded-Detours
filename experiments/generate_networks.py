"""
generate_networks.py

Generate and save benchmark networks.
"""

import argparse
import os
from pathlib import Path

from data.adfc.adfc_registry import ADFC_VILLAGES
from villageregions import VillageRegion
from bicyclenetwork import BicycleNetwork


# --------------------------------------------------------
# Defaults
# --------------------------------------------------------

DEFAULT_SAFETY_MODELS = [
    "A",
    "B",
    "C",
]

DEFAULT_RADII = [
    3,
]

VILLAGE_NETWORK_DIR = Path(
    "data/networks/villages"
)

REGION_NETWORK_DIR = Path(
    "data/networks/regions"
)


# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def village_network_path(
    village_key,
    safety_model,
):

    return (
        VILLAGE_NETWORK_DIR
        /
        f"{village_key}_{safety_model}.pkl"
    )


def region_network_path(
    village_key,
    radius_km,
    safety_model,
):

    return (
        REGION_NETWORK_DIR
        /
        f"{village_key}_r{radius_km}_{safety_model}.pkl"
    )


def should_generate_path(
    path,
    overwrite,
):

    if overwrite:

        return True

    return not os.path.exists(
        path
    )


# --------------------------------------------------------
# Village networks
# --------------------------------------------------------

def generate_village_networks(
    villages,
    safety_models,
    overwrite,
):

    print()
    print("=" * 60)
    print("GENERATING VILLAGE NETWORKS")
    print("=" * 60)

    VILLAGE_NETWORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for village in villages:

        print()
        print(f"Village: {village.name}")

        for safety_model in safety_models:

            print(
                f"  safety model = {safety_model}"
            )

            try:

                target_path = village_network_path(
                    village_key=village.key,
                    safety_model=safety_model,
                )

                if not should_generate_path(
                    target_path,
                    overwrite,
                ):

                    print(
                        f"    exists -> {target_path}"
                    )

                    continue

                net = BicycleNetwork.from_place(
                    village,
                    safety_model=safety_model,
                )

                # Force metadata to match the target filename.
                net.origin_key = village.key
                net.origin_type = "village"
                net.radius_km = None
                net.safety_model = safety_model

                save_path = net.path()

                if Path(save_path) != target_path:

                    raise RuntimeError(
                        "Village save path does not match target path: "
                        f"target={target_path}, save_path={save_path}"
                    )

                if (
                    not overwrite
                    and os.path.exists(save_path)
                ):

                    print(
                        f"    exists before save -> {save_path}"
                    )

                    continue

                saved_path = net.save()

                print(
                    f"    saved -> {saved_path}"
                )

            except Exception as e:

                print(
                    f"    FAILED: {e}"
                )


# --------------------------------------------------------
# Region networks
# --------------------------------------------------------

def generate_region_networks(
    villages,
    safety_models,
    radii,
    overwrite,
):

    print()
    print("=" * 60)
    print("GENERATING REGION NETWORKS")
    print("=" * 60)

    REGION_NETWORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for village in villages:

        for radius_km in radii:

            region = VillageRegion(
                key=village.key,
                center=village,
                radius_km=radius_km,
            )

            print()
            print(
                f"Region: {village.key} "
                f"({radius_km} km)"
            )

            for safety_model in safety_models:

                print(
                    f"  safety model = {safety_model}"
                )

                try:

                    target_path = region_network_path(
                        village_key=village.key,
                        radius_km=radius_km,
                        safety_model=safety_model,
                    )

                    # Avoid live OSM queries when the target file already exists.

                    if not should_generate_path(
                        target_path,
                        overwrite,
                    ):

                        print(
                            f"    exists -> {target_path}"
                        )

                        continue

                    net = region.network(
                        safety_model=safety_model
                    )

                    # Force metadata to match the target filename.
                    net.origin_key = village.key
                    net.origin_type = "region"
                    net.radius_km = radius_km
                    net.safety_model = safety_model

                    save_path = net.path()

                    if Path(save_path) != target_path:

                        raise RuntimeError(
                            "Region save path does not match target path: "
                            f"target={target_path}, save_path={save_path}"
                        )

                    if (
                        not overwrite
                        and os.path.exists(save_path)
                    ):

                        print(
                            f"    exists before save -> {save_path}"
                        )

                        continue

                    saved_path = net.save()

                    print(
                        f"    saved -> {saved_path}"
                    )

                except Exception as e:

                    print(
                        f"    FAILED: {e}"
                    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--safety_models",
        nargs="+",
        choices=[
            "A",
            "B",
            "C",
        ],
        default=DEFAULT_SAFETY_MODELS,
        help="Safety models.",
    )

    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=DEFAULT_RADII,
        help="Region radii in km.",
    )

    parser.add_argument(
        "--village",
        default=None,
        help="Village key.",
    )

    parser.add_argument(
        "--villages_only",
        action="store_true",
    )

    parser.add_argument(
        "--regions_only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing files.",
    )

    args = parser.parse_args()

    if (
        args.villages_only
        and args.regions_only
    ):

        parser.error(
            "--villages_only and --regions_only "
            "cannot be used together."
        )

    # ----------------------------------------------------
    # Village selection
    # ----------------------------------------------------

    if args.village is None:

        villages = list(
            ADFC_VILLAGES.values()
        )

    else:

        if args.village not in ADFC_VILLAGES:

            raise ValueError(
                f"Unknown ADFC village key: {args.village}"
            )

        villages = [
            ADFC_VILLAGES[
                args.village
            ]
        ]

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    print(
        "Villages:",
        [v.key for v in villages],
    )

    print(
        "Safety models:",
        args.safety_models,
    )

    print(
        "Radii:",
        args.radii,
    )

    print(
        "Villages only:",
        args.villages_only,
    )

    print(
        "Regions only:",
        args.regions_only,
    )

    print(
        "Overwrite:",
        args.overwrite,
    )

    print(
        "Village output dir:",
        VILLAGE_NETWORK_DIR,
    )

    print(
        "Region output dir:",
        REGION_NETWORK_DIR,
    )

    # ----------------------------------------------------
    # Generation
    # ----------------------------------------------------

    if not args.regions_only:

        generate_village_networks(
            villages=villages,
            safety_models=args.safety_models,
            overwrite=args.overwrite,
        )

    if not args.villages_only:

        generate_region_networks(
            villages=villages,
            safety_models=args.safety_models,
            radii=args.radii,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":

    main()
