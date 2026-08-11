"""
generate_instances.py

Generate benchmark instances
for all stored networks.
"""

import argparse
import os

from bicyclenetwork import BicycleNetwork
from instancegenerator import InstanceGenerator


# --------------------------------------------------------
# Generation
# --------------------------------------------------------

def generate_instances(
    networks,
    alphas,
    fractions,
    instance_ids,
    overwrite,
):

    print()
    print("=" * 60)
    print("GENERATING INSTANCES")
    print("=" * 60)

    for net in networks:

        print()
        print(
            f"Network: "
            f"{net.origin_key} "
            f"[{net.safety_model}]"
        )

        for alpha in alphas:

            for fraction in fractions:

                igen = InstanceGenerator(
                    network=net,
                    alpha=alpha,
                    target_pair_fraction=fraction,
                )

                print(
                    "  random_pairs"
                    f" | alpha={alpha}"
                    f" | frac={fraction}"
                )

                for instance_id in instance_ids:

                    try:

                        inst = igen.random_pairs(
                            instance_id=instance_id
                        )

                        path = inst.path()

                        if (
                            not overwrite
                            and os.path.exists(path)
                        ):

                            print(
                                f"    exists "
                                f"id={instance_id}"
                            )

                            continue

                        path = inst.save()

                        print(
                            f"    saved "
                            f"id={instance_id}"
                        )

                    except Exception as e:

                        print(
                            f"    FAILED "
                            f"id={instance_id}: "
                            f"{e}"
                        )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--num_instances",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit instance IDs. "
            "Overrides --num_instances."
        ),
    )

    parser.add_argument(
        "--network",
        default=None,
        help=(
            "Only networks whose "
            "origin_key contains this string."
        ),
    )

    parser.add_argument(
        "--safety_models",
        nargs="+",
        default=None,
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
    # Instance IDs
    # ----------------------------------------------------

    if args.instance_ids is None:

        instance_ids = list(
            range(args.num_instances)
        )

    else:

        instance_ids = sorted(
            set(args.instance_ids)
        )

    # ----------------------------------------------------
    # Load networks
    # ----------------------------------------------------

    networks = []

    for path in BicycleNetwork.all_paths():

        net = BicycleNetwork.load(path)

        #
        # Safety model filter
        #
        if (
            args.safety_models is not None
            and net.safety_model
            not in args.safety_models
        ):
            continue

        #
        # Network name filter
        #
        if (
            args.network is not None
            and args.network
            not in net.origin_key
        ):
            continue

        #
        # Village / region filters
        #
        is_region = net.origin_type == "region"

        if (
            args.villages_only
            and is_region
        ):
            continue

        if (
            args.regions_only
            and not is_region
        ):
            continue

        networks.append(net)

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    print(
        f"networks: {len(networks)}"
    )

    print(
        f"alphas: {args.alphas}"
    )

    print(
        f"fractions: {args.fractions}"
    )

    print(
        f"instance_ids: "
        f"{instance_ids}"
    )

    print(
        f"overwrite: "
        f"{args.overwrite}"
    )

    # ----------------------------------------------------
    # Generate
    # ----------------------------------------------------

    generate_instances(
        networks=networks,
        alphas=args.alphas,
        fractions=args.fractions,
        instance_ids=instance_ids,
        overwrite=args.overwrite,
    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    main()
