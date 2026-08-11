#!/usr/bin/env python3

import argparse
import itertools
import subprocess
import os
import json
from datetime import datetime

from bicycleinstance import BicycleInstance
from experimentresult import ExperimentResult


# ============================================================
# CONFIGURATION
# ============================================================

SOLVERS = [
    "ilp",
    "ilp_tw_dp_cuts",
    "ilp_tw_dp_cuts2",
    "ilp_tw_dp_cuts12",
]

DEFAULT_SOLVERS = [
    "ilp",
    "ilp_tw_dp_cuts",
]


PREPROCESSORS = [
    "none",
    "kernel1",
]

DEFAULT_PREPROCESSORS = [
    "none",
    "kernel1",
]


# ============================================================
# INSTANCE FILTER
# ============================================================

def match_alpha_filter(instance_path, args):
    """
    Decide whether an instance filename matches the requested alpha.

    Filenames encode alpha as tokens like:
        _a12_ for alpha=1.2
        _a13_ for alpha=1.3
        _a15_ for alpha=1.5
    """

    if args.alpha is None:

        return True

    alpha_token = f"a{int(round(10 * float(args.alpha)))}"

    stem = os.path.basename(
        str(instance_path)
    )

    if stem.endswith(".pkl"):

        stem = stem[:-4]

    tokens = stem.split(
        "_"
    )

    return alpha_token in tokens

# ============================================================
# INSTANCE FILTER
# ============================================================

def match_filter(instance, args):
    """
    Decide whether a loaded instance should be included.
    """

    # ---- village / region filter ----
    if (
        args.kind == "village_only"
        and instance.network.origin_type != "village"
    ):
        return False

    if (
        args.kind == "region_only"
        and instance.network.origin_type != "region"
    ):
        return False

    # ---- origin filter ----
    if (
        args.origin is not None
        and instance.network.origin_key != args.origin
    ):
        return False

    # ---- safety-model filter ----
    if (
        args.safety_model is not None
        and instance.network.safety_model != args.safety_model
    ):
        return False

    return True

def now_string():
    """
    Human-readable local wall-clock time for monitoring runs.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# RESULT CHECK
# ============================================================

def result_object(instance, solver, preprocessor):

    return ExperimentResult(
        instance_filename=instance.filename(),
        solver=solver,
        preprocessor=(
            ""
            if preprocessor == "none"
            else preprocessor
        ),
    )


def result_path(instance, solver, preprocessor):

    return result_object(
        instance,
        solver,
        preprocessor,
    ).path()


def load_existing_result(path):
    """
    Load an existing ExperimentResult.

    Supports JSON result files and, as fallback, pickle files.
    """

    if path.endswith(".json"):

        with open(path, "r") as f:
            return json.load(f)

    # Fallback for older pickle result files, if any.
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def result_is_optimal(path):
    """
    Return True iff the existing result has optimal == True.

    Missing/corrupt results or missing optimal flags are treated as
    non-optimal, so they are rerun when --rerun-nonoptimal is used.
    """

    try:

        result = load_existing_result(path)

    except Exception as exc:

        print(
            "[WARN] Could not load existing result:",
            path,
            exc,
        )

        return False

    if isinstance(result, dict):

        value = result.get(
            "optimal",
            False,
        )

    else:

        value = getattr(
            result,
            "optimal",
            False,
        )

    if isinstance(value, bool):

        return value

    return str(value).lower() == "true"


def should_run_job(instance, solver, preprocessor, args):
    """
    Decide whether this job should be launched.
    """

    path = result_path(
        instance,
        solver,
        preprocessor,
    )

    if not os.path.exists(path):

        return True, "missing"

    if args.rerun_nonoptimal:

        if result_is_optimal(path):

            return False, "existing optimal"

        return True, "existing non-optimal"

    return False, "existing"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kind",
        default="all",
        choices=[
            "all",
            "village_only",
            "region_only",
        ],
        help="Filter instance types",
    )

    parser.add_argument(
        "--origin",
        default=None,
        help="Filter by origin key",
    )

    parser.add_argument(
        "--safety_model",
        default=None,
        help="Filter by safety model",
    )

    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=SOLVERS,
        default=DEFAULT_SOLVERS,
        help="Solvers to run.",
    )

    parser.add_argument(
        "--preprocessors",
        nargs="+",
        choices=PREPROCESSORS,
        default=DEFAULT_PREPROCESSORS,
        help="Preprocessors to run.",
    )

    parser.add_argument(
        "--instance",
        default=None,
        help="Only solve one instance filename.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Solver timeout in seconds",
    )

    parser.add_argument(
        "--rerun-nonoptimal",
        action="store_true",
        help=(
            "If a result already exists, rerun it only when "
            "result.optimal is not True. This recomputes timeouts "
            "and other non-optimal results."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Only consider instances with this alpha value. "
            "Example: --alpha 1.3 selects filenames containing _a13_."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load instances
    # --------------------------------------------------------

    instance_paths = BicycleInstance.all_paths()

    # ---- explicit instance-name filter ----
    if args.instance is not None:

        instance_paths = [
            path
            for path in instance_paths
            if args.instance in path
        ]

    # ---- alpha filter, e.g. alpha=1.3 -> _a13_ ----
    instance_paths = [
        path
        for path in instance_paths
        if match_alpha_filter(
            path,
            args,
        )
    ]

    instances = [
        BicycleInstance.load(path)
        for path in instance_paths
    ]

    # ---- origin / safety-model filters ----
    instances = [
        inst
        for inst in instances
        if match_filter(
            inst,
            args,
        )
    ]

    print()
    print("=" * 60)
    print("SOLVE ALL INSTANCES")
    print("=" * 60)

    print(f"Filtered instances: {len(instances)}")

    print(
        "Alpha filter:",
        args.alpha,
    )

    print(
        "Solvers:",
        args.solvers,
    )

    print(
        "Preprocessors:",
        args.preprocessors,
    )

    jobs = list(
        itertools.product(
            instances,
            args.preprocessors,
            args.solvers,
        )
    )

    print(f"Total jobs: {len(jobs)}")

    skipped = 0
    launched = 0

    rerun_nonoptimal = 0

    for inst, preprocessor, solver in jobs:

        should_run, reason = should_run_job(
            inst,
            solver,
            preprocessor,
            args,
        )

        if not should_run:

            skipped += 1

            print(
                "[SKIP]",
                inst.filename(),
                preprocessor,
                solver,
                f"({reason})",
            )

            continue

        launched += 1

        if reason == "existing non-optimal":

            rerun_nonoptimal += 1

            tag = "[RERUN]"

        else:

            tag = "[RUN  ]"

        print(
            tag,
            now_string(),
            inst.filename(),
            preprocessor,
            solver,
            f"({reason})",
            flush=True,
        )

        subprocess.run(
            [
                "python3",
                "solve_instance.py",
                "--instance",
                inst.path(),
                "--solver",
                solver,
                "--preprocessor",
                preprocessor,
                "--timeout",
                str(args.timeout),
            ],
            check=True,
        )

    print()
    print("=" * 60)
    print(
        f"Done. launched={launched}, skipped={skipped}, "
        f"rerun_nonoptimal={rerun_nonoptimal}"
    )
    print("=" * 60)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
