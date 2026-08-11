#!/usr/bin/env python3

import argparse

from bicycleinstance import BicycleInstance
from experimentresult import ExperimentResult
from solutionverifier import SolutionVerifier

from ilpsolver import ILPSolver
from exhaustivereducer import ExhaustiveReducer


# ============================================================
# Registries
# ============================================================

SOLVERS = {
    "ilp": {},

    "ilp_tw_dp_cuts": {
        "treewidth_dp_cuts": True,
    },

    "ilp_tw_dp_cuts2": {
        "treewidth_dp_cuts": True,
        "treewidth_dp_cuts_setup": 2,
    },

    "ilp_tw_dp_cuts12": {
        "treewidth_dp_cuts": True,
        "treewidth_dp_cuts_setup": 3,
    },
}


PREPROCESSORS = (
    "none",
    "kernel1",
)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instance",
        required=True,
    )

    parser.add_argument(
        "--solver",
        default="ilp",
        choices=SOLVERS.keys(),
    )

    parser.add_argument(
        "--preprocessor",
        default="none",
        choices=PREPROCESSORS,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Solver timeout in seconds",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load instance
    # --------------------------------------------------------

    inst = BicycleInstance.load(
        args.instance
    )

    original_inst = inst

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    reducer = None

    if args.preprocessor != "none":

        reducer = ExhaustiveReducer(inst)

        method = getattr(
            reducer,
            f"run_{args.preprocessor}"
        )

        inst = method()

        inst.save()

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    solver_config = SOLVERS[
        args.solver
    ]

    solver = ILPSolver(
        inst,
        timeout=args.timeout,
        **solver_config,
    )

    result = solver.solve()


    # --------------------------------------------------------
    # Verify on ORIGINAL instance
    # --------------------------------------------------------

    if result.feasible:

        verifier = SolutionVerifier(
            original_inst,
            result.selected_edges,
        )

        verified = verifier.check()

        if not verified:

            raise RuntimeError(
                "Solver returned a solution that failed "
                "verification on the original instance."
            )

    else:

        print(
            "No feasible solver solution available; "
            "skipping solution verification."
        )

    # --------------------------------------------------------
    # Build ExperimentResult
    # --------------------------------------------------------

    exp = ExperimentResult(

        instance_filename=
            original_inst.filename(),

        solver=args.solver,

        preprocessor=
            (
                ""
                if args.preprocessor == "none"
                else args.preprocessor
            ),

        objective=
            result.objective,

        optimal=
            result.optimal,

        feasible=
            result.feasible,

        status=
            result.status,

        selected_edges=
            list(
                result.selected_edges
            ),

        timeout=
            args.timeout,

        termination_reason=
            result.termination_reason,
    )

    # --------------------------------------------------------
    # Solver statistics
    # --------------------------------------------------------

    exp.solver_runtime = (
        solver.runtime
    )

    exp.solver_buildtime = (
        solver.buildtime
    )

    # --------------------------------------------------------
    # Reduction statistics
    # --------------------------------------------------------

    if reducer is not None:

        exp.reduction_runtime = (
            reducer.total_runtime
        )

        exp.total_runtime = (
            reducer.total_runtime
            + solver.buildtime
            + solver.runtime
        )

        exp.budget_delta = (
            reducer.total_budget_delta
        )

        exp.safe_removed = sum(
            h["removed_safe"]
            for h in reducer.history
        )

        exp.unsafe_removed = sum(
            h["removed_unsafe"]
            for h in reducer.history
        )

        if reducer.history:

            first = reducer.history[0]
            last = reducer.history[-1]

            exp.vertices_before = (
                first["n_before"]
            )

            exp.vertices_after = (
                last["n_after"]
            )

            exp.edges_before = (
                first["m_before"]
            )

            exp.edges_after = (
                last["m_after"]
            )

    else:

        exp.total_runtime = (
            solver.buildtime
            + solver.runtime
        )

        exp.vertices_before = (
            len(original_inst.network.vertices)
        )

        exp.vertices_after = (
            len(original_inst.network.vertices)
        )

        exp.edges_before = (
            len(original_inst.network.safe_edges)
            +
            len(original_inst.network.unsafe_edges)
        )

        exp.edges_after = (
            exp.edges_before
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    exp.save()

    exp.print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
