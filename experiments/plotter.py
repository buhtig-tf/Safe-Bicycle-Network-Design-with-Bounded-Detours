#!/usr/bin/env python3

import math
import os
import argparse
import re
import matplotlib.pyplot as plt
import statistics
import numpy as np

from experimentresult import ExperimentResult

# ------------------------------------------------------------
# Display names used in plots / paper figures
# ------------------------------------------------------------

NAMEDICT = {
    # No preprocessing
    "ilp": r"$\mathsf{ILP}$",
    "ilp_tw_dp_cuts": r"$\mathsf{ILP}_{\mathsf{twc}}$",
    "ilp_tw_dp_cuts2": r"$\mathsf{ILP}_{\mathsf{twc2}}$",
    "ilp_tw_dp_cuts12": r"$\mathsf{ILP}_{\mathsf{twc1/2}}$",

    # Kernel 1 / preprocessing
    "kernel1_ilp": r"$\mathsf{ILP}^{\mathsf{pre}}$",
    "kernel1_ilp_tw_dp_cuts": r"$\mathsf{ILP}_{\mathsf{twc}}^{\mathsf{pre}}$",
    "kernel1_ilp_tw_dp_cuts2": r"$\mathsf{ILP}_{\mathsf{twc2}}^{\mathsf{pre}}$",
    "kernel1_ilp_tw_dp_cuts12": r"$\mathsf{ILP}_{\mathsf{twc1/2}}^{\mathsf{pre}}$",
}


def display_name(
    name,
):

    name = str(
        name
    )

    if name not in NAMEDICT:

        print(
            "[WARN] missing display name:",
            name,
        )

    return NAMEDICT.get(
        name,
        name,
    )

# ------------------------------------------------------------
# Matplotlib / LaTeX-style plot settings
# ------------------------------------------------------------

def use_latex_style_plots(
    usetex=False,
):
    """
    Configure Matplotlib to use LaTeX-like fonts.

    Parameters
    ----------
    usetex:
        False:
            Use Matplotlib's built-in mathtext with serif/STIX fonts.
            This does not require a LaTeX installation and is robust.

        True:
            Use an actual LaTeX installation for all text rendering.
            This requires LaTeX to be installed on the system.
    """

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            # Font family
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
            ],

            # Math font
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",

            # Optional real LaTeX rendering
            "text.usetex": usetex,

            # Sizes
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,

            # Lines / axes
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,

            # PDF/PS font embedding
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

# ============================================================
# Result loading
# ============================================================

def load_all_results(
    root="data/results",
    methods=None,
):

    results = []

    for method in os.listdir(root):

        if methods is not None:
            if method not in methods:
                continue

        method_dir = os.path.join(
            root,
            method,
        )

        if not os.path.isdir(method_dir):
            continue

        for fn in os.listdir(method_dir):

            if not fn.endswith(".json"):
                continue

            instance_filename = (
                fn[:-5]
            )

            try:

                result = (
                    ExperimentResult.load(
                        instance_filename,
                        method,
                    )
                )

                results.append(
                    result
                )

            except Exception as e:

                print(
                    "Failed:",
                    fn,
                    e,
                )

    return results

# ============================================================
# Plotting result helpers
# ============================================================

def _plot_result_is_optimal(result):

    optimal = getattr(
        result,
        "optimal",
        None,
    )

    if optimal is not None:
        return bool(optimal)

    status = getattr(
        result,
        "status",
        None,
    )

    if status is None:
        return True

    return str(status).upper() == "OPTIMAL"


def _plot_result_instance(result):

    instance = getattr(
        result,
        "instance",
        None,
    )

    if instance is not None:
        return str(instance)

    instance = getattr(
        result,
        "instance_name",
        None,
    )

    if instance is not None:
        return str(instance)

    instance = getattr(
        result,
        "instance_filename",
        None,
    )

    if instance is not None:
        return str(instance)

    return None


def _plot_result_method(result):

    method = getattr(
        result,
        "method",
        None,
    )

    if method is not None:
        return str(method)

    solver = getattr(
        result,
        "solver",
        None,
    )

    preprocessor = getattr(
        result,
        "preprocessor",
        None,
    )

    if solver is None:
        return None

    if (
        preprocessor is None
        or str(preprocessor).strip() == ""
        or str(preprocessor).lower() == "none"
    ):

        return str(solver)

    return f"{preprocessor}_{solver}"


def _plot_result_runtime(result):

    runtime = getattr(
        result,
        "total_runtime",
        None,
    )

    if runtime is not None:
        return float(runtime)

    runtime = getattr(
        result,
        "runtime",
        None,
    )

    if runtime is not None:
        return float(runtime)

    solver_runtime = getattr(
        result,
        "solver_runtime",
        None,
    )

    reduction_runtime = getattr(
        result,
        "reduction_runtime",
        0.0,
    )

    buildtime = getattr(
        result,
        "solver_buildtime",
        0.0,
    )

    if solver_runtime is not None:

        return (
            float(reduction_runtime)
            +
            float(buildtime)
            +
            float(solver_runtime)
        )

    return None


def _collect_best_runtimes():

    results = load_all_results()

    best = {}

    for result in results:

        if not _plot_result_is_optimal(result):
            continue

        instance = _plot_result_instance(
            result
        )

        if instance is None:
            continue

        method = _plot_result_method(
            result
        )

        if method is None:
            continue

        runtime = _plot_result_runtime(
            result
        )

        if runtime is None:
            continue

        if not math.isfinite(runtime):
            continue

        key = (
            method,
            instance,
        )

        if (
            key not in best
            or runtime < best[key]
        ):

            best[key] = runtime

    return best



# ============================================================
# Runtime scatter
# ============================================================

def runtime_scatter(
    solver="ilp",
    preprocessor="kernel1",
    other_solver="ilp",
    other_preprocessor=None,
    main_mode="log-log",
    inset_mode=None,
    runtime_mode="total_runtime",
    output_dir="plots/scatter",
    show_title=False,
):
    """
    Scatter plot comparing runtimes of two methods.

    The figure shows both:
        - a log-log version
        - a linear/unaltered-axis version

    One is the main plot and the other is shown as an inset in the
    lower-right corner.

    Parameters
    ----------
    main_mode:
        "log-log" or "linear"

    inset_mode:
        "log-log", "linear", or None.
        If None, use the opposite of main_mode.

    Examples
    --------
    Main plot log-log, inset linear:

        runtime_scatter(main_mode="log-log")

    Main plot linear, inset log-log:

        runtime_scatter(main_mode="linear")
    """

    import os
    import math
    import matplotlib.pyplot as plt

    if main_mode not in {
        "log-log",
        "linear",
    }:

        raise ValueError(
            "main_mode must be one of: 'log-log', 'linear'"
        )

    if inset_mode is None:

        if main_mode == "log-log":

            inset_mode = "linear"

        else:

            inset_mode = "log-log"

    if inset_mode not in {
        "log-log",
        "linear",
    }:

        raise ValueError(
            "inset_mode must be one of: 'log-log', 'linear', None"
        )

    results = load_all_results()

    # ------------------------------------------------------------
    # Helper: display names, if NAMEDICT/display_name exists.
    # ------------------------------------------------------------

    def pretty_name(
        name,
    ):

        if "display_name" in globals():

            return display_name(
                name
            )

        return str(
            name
        )

    # ------------------------------------------------------------
    # Helper: construct method name.
    # ------------------------------------------------------------

    def method_name(
        solver_name,
        preprocessor_name=None,
    ):

        if preprocessor_name is None:

            return solver_name

        preprocessor_name = str(
            preprocessor_name
        )

        if preprocessor_name.lower() in {
            "",
            "none",
            "no",
            "nopre",
        }:

            return solver_name

        return (
            f"{preprocessor_name}_{solver_name}"
        )

    # ------------------------------------------------------------
    # Backward-compatible behavior.
    # ------------------------------------------------------------

    if other_solver is None:

        if preprocessor is None:

            raise ValueError(
                "If other_solver is None, preprocessor must be given. "
                "Example: runtime_scatter(solver='ilp', preprocessor='kernel1')"
            )

        x_method = method_name(
            solver,
            None,
        )

        y_method = method_name(
            solver,
            preprocessor,
        )

    else:

        x_method = method_name(
            solver,
            preprocessor,
        )

        y_method = method_name(
            other_solver,
            other_preprocessor,
        )

    # ------------------------------------------------------------
    # Helper: robust timeout detection.
    # ------------------------------------------------------------

    def is_timeout(
        result,
    ):

        for attr in [
            "timed_out",
            "timeout",
            "time_limit_reached",
            "is_timeout",
        ]:

            if hasattr(
                result,
                attr,
            ):

                value = getattr(
                    result,
                    attr,
                )

                if isinstance(
                    value,
                    bool,
                ):

                    return value

        for attr in [
            "status",
            "solver_status",
            "gurobi_status",
        ]:

            if hasattr(
                result,
                attr,
            ):

                value = str(
                    getattr(
                        result,
                        attr,
                    )
                ).lower()

                if value in {
                    "timeout",
                    "time_limit",
                    "timelimit",
                }:

                    return True

                if (
                    "time" in value
                    and (
                        "limit" in value
                        or "out" in value
                    )
                ):

                    return True

        for attr in [
            "optimal",
        ]:

            if hasattr(
                result,
                attr,
            ):

                value = str(
                    getattr(
                        result,
                        attr,
                    )
                ).lower()

                if value in {
                    "false",
                }:

                    return True

        return False

    # ------------------------------------------------------------
    # Helper: robust runtime extraction.
    # ------------------------------------------------------------

    def get_runtime(
        result,
    ):

        if hasattr(
            result,
            runtime_mode,
        ):

            value = getattr(
                result,
                runtime_mode,
            )

            if value is not None:

                return value

        # fallback
        if hasattr(
            result,
            "runtime",
        ):

            value = getattr(
                result,
                "runtime",
            )

            if value is not None:

                return value

        return None

    # ------------------------------------------------------------
    # Helper: robust instance-name extraction.
    # ------------------------------------------------------------

    def get_instance_name(
        result,
    ):

        for attr in [
            "instance_filename",
            "instance",
            "instance_name",
        ]:

            if hasattr(
                result,
                attr,
            ):

                value = getattr(
                    result,
                    attr,
                )

                if value is not None:

                    return value

        return None

    # ------------------------------------------------------------
    # Helper: robust method-name extraction.
    # ------------------------------------------------------------

    def get_result_method(
        result,
    ):

        if hasattr(
            result,
            "method",
        ):

            return getattr(
                result,
                "method",
            )

        result_solver = getattr(
            result,
            "solver",
            None,
        )

        result_preprocessor = getattr(
            result,
            "preprocessor",
            None,
        )

        if result_solver is None:

            return None

        return method_name(
            result_solver,
            result_preprocessor,
        )

    # ------------------------------------------------------------
    # instance -> method -> result
    # ------------------------------------------------------------

    table = {}

    for r in results:

        instance = get_instance_name(
            r
        )

        if instance is None:

            continue

        result_method = get_result_method(
            r
        )

        if result_method is None:

            continue

        table.setdefault(
            instance,
            {},
        )[str(result_method)] = r

    # ------------------------------------------------------------
    # Collect points by timeout category.
    # ------------------------------------------------------------

    categories = {
        "none": {
            "x": [],
            "y": [],
            "marker": "o",
            "label": "no timeout",
        },
        "x": {
            "x": [],
            "y": [],
            "marker": "x",
            "label": f"timeout: {pretty_name(x_method)}",
        },
        "y": {
            "x": [],
            "y": [],
            "marker": "^",
            "label": f"timeout: {pretty_name(y_method)}",
        },
        "both": {
            "x": [],
            "y": [],
            "marker": "s",
            "label": "timeout both",
        },
    }

    for inst, methods in table.items():

        if (
            x_method not in methods
            or y_method not in methods
        ):

            continue

        x_result = methods[
            x_method
        ]

        y_result = methods[
            y_method
        ]

        x_runtime = get_runtime(
            x_result
        )

        y_runtime = get_runtime(
            y_result
        )

        if (
            x_runtime is None
            or y_runtime is None
        ):

            continue

        x_runtime = float(
            x_runtime
        )

        y_runtime = float(
            y_runtime
        )

        if (
            x_runtime <= 0
            or y_runtime <= 0
            or not math.isfinite(x_runtime)
            or not math.isfinite(y_runtime)
        ):

            continue

        x_timeout = is_timeout(
            x_result
        )

        y_timeout = is_timeout(
            y_result
        )

        if x_timeout and y_timeout:

            key = "both"

        elif x_timeout:

            key = "x"

        elif y_timeout:

            key = "y"

        else:

            key = "none"

        categories[key]["x"].append(
            x_runtime
        )

        categories[key]["y"].append(
            y_runtime
        )

    xs = []
    ys = []

    for cat in categories.values():

        xs.extend(
            cat["x"]
        )

        ys.extend(
            cat["y"]
        )

    if not xs:

        print(
            "No matching results found for "
            f"{x_method} vs {y_method}."
        )

        return

    # ------------------------------------------------------------
    # Helper: draw one scatter plot on a given axis.
    # ------------------------------------------------------------

    def draw_scatter_axis(
        ax,
        axis_mode,
        include_legend=False,
        small=False,
    ):

        marker_size = (
            4
            if small
            else 6
        )

        alpha = (
            0.75
            if small
            else 0.85
        )

        for key in [
            "none",
            "x",
            "y",
            "both",
        ]:

            cat = categories[key]

            if not cat["x"]:

                continue

            ax.scatter(
                cat["x"],
                cat["y"],
                alpha=alpha,
                marker=cat["marker"],
                label=cat["label"],
                s=marker_size,
            )

        mn = min(
            min(xs),
            min(ys),
        )

        mx = max(
            max(xs),
            max(ys),
        )

        ax.plot(
            [
                mn,
                mx,
            ],
            [
                mn,
                mx,
            ],
            linestyle="--",
            linewidth=1.0 if small else 1.4,
            color="black",
            alpha=0.65,
            label="equal runtime",
        )

        if axis_mode == "log-log":

            ax.set_xscale(
                "log"
            )

            ax.set_yscale(
                "log"
            )

            grid_which = "major"

        else:

            ax.set_xscale(
                "linear"
            )

            ax.set_yscale(
                "linear"
            )

            grid_which = "major"

        ax.grid(
            True,
            which=grid_which,
            linestyle="-",
            linewidth=0.25 if not small else 0.2,
            alpha=0.22 if not small else 0.18,
        )

        ax.grid(
            False,
            which="minor",
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=7 if small else 9,
        )

        if include_legend:

            ax.legend(
                fontsize=8,
                frameon=False,
            )

    # ------------------------------------------------------------
    # Plot main + inset.
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            6.0,
            6.0,
        )
    )

    draw_scatter_axis(
        ax,
        main_mode,
        include_legend=True,
        small=False,
    )

    ax.set_xlabel(
        f"{pretty_name(x_method)} runtime [s]"
    )

    ax.set_ylabel(
        f"{pretty_name(y_method)} runtime [s]"
    )

    if show_title:

        ax.set_title(
            (
                f"Runtime comparison: "
                f"{pretty_name(x_method)} vs {pretty_name(y_method)}"
            )
        )

    inset = ax.inset_axes(
        [
            0.55,
            0.035,
            0.43,
            0.41,
        ]
    )

    draw_scatter_axis(
        inset,
        inset_mode,
        include_legend=False,
        small=True,
    )

    inset.set_title(
        inset_mode,
        fontsize=8,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    suffix_parts = [
        runtime_mode,
        "scatter",
        x_method,
        "vs",
        y_method,
        "main",
        main_mode.replace(
            "-",
            "_",
        ),
        "inset",
        inset_mode.replace(
            "-",
            "_",
        ),
    ]

    filename_base = "_".join(
        suffix_parts
    )

    pdf = os.path.join(
        output_dir,
        f"{filename_base}.pdf",
    )

    png = os.path.join(
        output_dir,
        f"{filename_base}.png",
    )

    fig.tight_layout(
        pad=0.05
    )

    plt.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print()
    print(
        f"Compared {x_method} vs {y_method}"
    )

    print(
        f"Instances plotted: {len(xs)}"
    )

    print(
        f"main mode: {main_mode}"
    )

    print(
        f"inset mode: {inset_mode}"
    )

    for key in [
        "none",
        "x",
        "y",
        "both",
    ]:

        print(
            f"{categories[key]['label']}: "
            f"{len(categories[key]['x'])}"
        )

    print(
        f"Saved {pdf}"
    )

    print(
        f"Saved {png}"
    )



# ============================================================
# Speedup profile (ordered by instance difficulty)
# ============================================================

def speedup_profile():

    results = load_all_results()

    #
    # instance -> method -> result
    #
    table = {}

    for r in results:

        table.setdefault(
            r.instance_filename,
            {}
        )[r.method] = r

    rows = []

    for inst, methods in table.items():

        if (
            "ilp" not in methods
            or
            "kernel1_ilp" not in methods
        ):
            continue

        ilp = methods["ilp"]

        kernel = methods[
            "kernel1_ilp"
        ]

        ilp_runtime = (
            ilp.total_runtime
        )

        reduced_solver_runtime = (
            kernel.solver_runtime
        )

        reduced_total_runtime = (
            kernel.total_runtime
        )

        if (
            ilp_runtime <= 0
            or reduced_solver_runtime <= 0
            or reduced_total_runtime <= 0
        ):
            continue

        solver_speedup = (
            ilp_runtime
            /
            reduced_solver_runtime
        )

        total_speedup = (
            ilp_runtime
            /
            reduced_total_runtime
        )

        difficulty = max(
            ilp_runtime,
            reduced_total_runtime,
        )

        rows.append(
            (
                difficulty,
                solver_speedup,
                total_speedup,
                inst,
            )
        )

    if not rows:

        print(
            "No matching results found."
        )

        return

    #
    # sort by difficulty
    #
    rows.sort(
        key=lambda x: x[0]
    )

    x = list(
        range(
            1,
            len(rows) + 1
        )
    )

    solver_speedups = [
        row[1]
        for row in rows
    ]

    total_speedups = [
        row[2]
        for row in rows
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        solver_speedups,
        linewidth=2,
        label="Solver speedup",
    )

    plt.plot(
        x,
        total_speedups,
        linewidth=2,
        label="Total speedup",
    )

    #
    # Trend lines (fit in log-space)
    #
    solver_coef = np.polyfit(
        x,
        np.log10(solver_speedups),
        deg=1,
    )

    solver_trend = 10 ** np.polyval(
        solver_coef,
        x,
    )

    total_coef = np.polyfit(
        x,
        np.log10(total_speedups),
        deg=1,
    )

    total_trend = 10 ** np.polyval(
        total_coef,
        x,
    )

    sol_coef = 10 ** solver_coef[0]

    plt.plot(
        x,
        solver_trend,
        linestyle=":",
        linewidth=2,
        label=(
            f"Solver trend "
            f"(slope={sol_coef:.4f})"
        ),
    )

    tot_coef = 10 ** total_coef[0]

    plt.plot(
        x,
        total_trend,
        linestyle=":",
        linewidth=2,
        label=(
            f"Total trend "
            f"(slope={tot_coef:.4f})"
        ),
    )

    print()
    print("TREND LINES")

    print(
        f"solver slope: "
        f"{sol_coef:.6f}"
    )

    print(
        f"total slope: "
        f"{tot_coef:.6f}"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=2,
    )

    plt.yscale("log")

    plt.xlabel(
        "Instance rank (by difficulty)"
    )

    plt.ylabel(
        "Speedup"
    )

    plt.title(
        "Kernel speedup vs instance difficulty"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3,
    )

    plt.legend()

    os.makedirs(
        "plots",
        exist_ok=True,
    )

    pdf = (
        "plots/speedup_profile.pdf"
    )

    png = (
        "plots/speedup_profile.png"
    )

    plt.tight_layout()

    plt.savefig(pdf)

    plt.savefig(
        png,
        dpi=300,
    )

    #
    # Statistics
    #
    def median(values):

        vals = sorted(values)

        return vals[
            len(vals) // 2
        ]

    print()

    print(
        f"instances: {len(rows)}"
    )

    print()

    print(
        "SOLVER SPEEDUP"
    )

    print(
        f"  min: "
        f"{min(solver_speedups):.3f}"
    )

    print(
        f"  median: "
        f"{median(solver_speedups):.3f}"
    )

    print(
        f"  max: "
        f"{max(solver_speedups):.3f}"
    )

    print()

    print(
        "TOTAL SPEEDUP"
    )

    print(
        f"  min: "
        f"{min(total_speedups):.3f}"
    )

    print(
        f"  median: "
        f"{median(total_speedups):.3f}"
    )

    print(
        f"  max: "
        f"{max(total_speedups):.3f}"
    )

    print()

    print(
        f"Saved {pdf}"
    )

    print(
        f"Saved {png}"
    )


# ============================================================
# Solver performance comparison
# ============================================================

def solver_performance(
    output_dir="plots",
    methods=["ilp","kernel1_ilp","ilp_tw_dp_cuts","kernel1_ilp_tw_dp_cuts"],
    common_instances=True,
    logscale=True,
    runtime_metric="total_runtime",
    beta=None,
    header=False,
    small=True,
):
    """
    Plot cactus curves comparing all available solver/preprocessor
    combinations.

    Parameters
    ----------
    runtime_metric:
        "total_runtime"
            Uses result.total_runtime if available,
            otherwise result.runtime.

        "solver_runtime"
            Uses result.solver_runtime if available.
            Falls back to result.gurobi_runtime,
            then result.optimize_walltime,
            then result.runtime.

    beta:
        None or a number in (0, 1).

        If beta is not None, add a lower-right inset showing only the
        tail of the cactus curve:

            runtimes_inset = runtimes[round(beta * len(runtimes)):]

        Thus beta=0.8 shows approximately the last 20% of each curve.

    If common_instances=True, only instances solved by every method are
    used.
    """

    if runtime_metric not in {
        "total_runtime",
        "solver_runtime",
        "build_runtime",
    }:

        raise ValueError(
            "runtime_metric must be one of: "
            "'total_runtime', 'solver_runtime', 'build_runtime'"
    )

    if beta is not None:

        beta = float(
            beta
        )

        if not (
            0.0 < beta < 1.0
        ):

            raise ValueError(
                "beta must be None or a number in (0, 1)"
            )

    def get_runtime(
        result,
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            candidates = [
                "total_runtime",
                "runtime",
            ]

        elif runtime_metric == "solver_runtime":

            candidates = [
                "solver_runtime",
                "gurobi_runtime",
                "optimize_walltime",
                "runtime",
            ]

        elif runtime_metric == "build_runtime":

            candidates = [
                "solver_buildtime",
                "build_runtime",
                "buildtime",
            ]

        for field in candidates:

            value = getattr(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def runtime_label(
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            return "Total runtime [s]"

        if runtime_metric == "solver_runtime":

            return "Solver runtime [s]"

        if runtime_metric == "build_runtime":

            return "Build runtime [s]"

        raise ValueError(
            runtime_metric
        )

        raise ValueError(
            runtime_metric
        )

    results = load_all_results(methods=methods)

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Collect best runtime per method-instance pair.
    # ------------------------------------------------------------

    best = {}

    for result in results:

        status = getattr(
            result,
            "status",
            None,
        )

        optimal = getattr(
            result,
            "optimal",
            None,
        )

        if optimal is not None and not optimal:

            continue

        if (
            optimal is None
            and status is not None
            and str(status).upper() != "OPTIMAL"
        ):

            continue

        instance = getattr(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = getattr(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = getattr(
                result,
                "instance_filename",
                None,
            )

        if instance is None:

            print(
                "[WARN] result without instance name skipped"
            )

            continue

        solver = getattr(
            result,
            "solver",
            None,
        )

        preprocessor = getattr(
            result,
            "preprocessor",
            None,
        )

        if solver is None:

            method = getattr(
                result,
                "method",
                None,
            )

            if method is None:

                print(
                    "[WARN] result without solver/method skipped:",
                    instance,
                )

                continue

        else:

            if (
                preprocessor is None
                or str(preprocessor).strip() == ""
                or str(preprocessor).lower() == "none"
            ):

                method = str(
                    solver
                )

            else:

                method = (
                    f"{preprocessor}_{solver}"
                )

        runtime = get_runtime(
            result,
            runtime_metric,
        )

        if runtime is None:

            print(
                f"[WARN] result without {runtime_metric} skipped:",
                instance,
                method,
            )

            continue

        runtime = float(
            runtime
        )

        if not math.isfinite(
            runtime
        ):

            continue

        key = (
            str(method),
            str(instance),
        )

        if (
            key not in best
            or runtime < best[key]
        ):

            best[key] = runtime

    methods = sorted(
        {
            method
            for method, _ in best.keys()
        }
    )

    if not methods:

        print(
            "[WARN] No solved results found."
        )

        return

    # ------------------------------------------------------------
    # Select instances.
    # ------------------------------------------------------------

    instances_by_method = {
        method: {
            instance
            for m, instance in best.keys()
            if m == method
        }
        for method in methods
    }

    if common_instances:

        instances_to_use = None

        for method in methods:

            if instances_to_use is None:

                instances_to_use = set(
                    instances_by_method[method]
                )

            else:

                instances_to_use &= instances_by_method[method]

        if instances_to_use is None:

            instances_to_use = set()

        print(
            "common instances:",
            len(instances_to_use),
        )

    else:

        instances_to_use = {
            instance
            for _, instance in best.keys()
        }

        print(
            "all instances:",
            len(instances_to_use),
        )

    if not instances_to_use:

        print(
            "[WARN] No instances to plot."
        )

        return

    # ------------------------------------------------------------
    # Precompute sorted runtimes by method.
    # ------------------------------------------------------------

    runtimes_by_method = {}

    for method in methods:

        runtimes = [
            best[(method, instance)]
            for instance in instances_to_use
            if (method, instance) in best
        ]

        if not runtimes:

            continue

        runtimes_by_method[method] = sorted(
            runtimes
        )

    if not runtimes_by_method:

        print(
            "[WARN] No runtimes to plot."
        )

        return

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print()
    print(
        f"Solver performance summary "
        f"({runtime_metric})"
    )
    print("--------------------------")

    for method in methods:

        if method not in runtimes_by_method:

            continue

        runtimes = runtimes_by_method[
            method
        ]

        avg = sum(runtimes) / len(runtimes)

        med = runtimes[
            len(runtimes) // 2
        ]

        print(
            f"{method:35s} "
            f"n={len(runtimes):4d} "
            f"avg={avg:10.3f} "
            f"med={med:10.3f} "
            f"max={max(runtimes):10.3f}"
        )

    print()

    # ------------------------------------------------------------
    # Plot cactus curves.
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
    )

    line_by_method = {}

    for method in methods:

        if method not in runtimes_by_method:

            continue

        runtimes = runtimes_by_method[
            method
        ]

        x_values = list(
            range(
                1,
                len(runtimes) + 1,
            )
        )

        line_list = ax.plot(
            x_values,
            runtimes,
            marker=None,
            markersize=3,
            linewidth=1 if not small else 3,
            label=display_name(method),
        )

        line_by_method[method] = line_list[0]

    ax.set_xlabel(
        "Solved instances",
        fontsize=24  if not small else 28,
    )

    ax.set_ylabel(
        runtime_label(
            runtime_metric
        ),
        fontsize=24 if not small else 28,
        labelpad=-2,
    )

    if common_instances:

        if header:
            ax.set_title(
                "Solver performance on common solved instances"
            )

        suffix = "common"

    else:

        if header:
            ax.set_title(
                "Solver performance on all solved instances"
            )

        suffix = "all"

    if logscale:

        ax.set_yscale(
            "log"
        )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.25,
        alpha=0.4,
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend(
        fontsize=20  if not small else 24,
        frameon=False,
        ncol=2,
        loc="lower right" if runtime_metric == "total_runtime" else "upper left",
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=16  if not small else 20,
        top=True,
        right=True,
    )

    # ------------------------------------------------------------
    # Optional inset: tail of cactus curves.
    # ------------------------------------------------------------

    if beta is not None:

        if runtime_metric == "total_runtime":
            inset = ax.inset_axes(
                [
                    0.05,
                    0.46,
                    0.50,
                    0.52,
                ]
            )

        elif runtime_metric == "solver_runtime":
            inset = ax.inset_axes(
                [
                    0.52,
                    0.06,
                    0.47,
                    0.46,
                ]
            )

        elif runtime_metric == "build_runtime":
            inset = ax.inset_axes(
                [
                    0.52,
                    0.06,
                    0.47,
                    0.47,
                ]
            )
        else:
            inset = ax.inset_axes(
                [
                    0.52,
                    0.06,
                    0.47,
                    0.47,
                ]
            )

        for method in methods:

            if method not in runtimes_by_method:

                continue

            runtimes = runtimes_by_method[
                method
            ]

            start = int(
                round(
                    beta * len(runtimes)
                )
            )

            # Ensure the inset is nonempty.
            start = min(
                start,
                len(runtimes) - 1,
            )

            start = max(
                start,
                0,
            )

            runtimes_inset = runtimes[
                start:
            ]

            x_values_inset = list(
                range(
                    start + 1,
                    len(runtimes) + 1,
                )
            )

            color = line_by_method[
                method
            ].get_color()

            inset.plot(
                x_values_inset,
                runtimes_inset,
                marker=None,
                linewidth=0.9  if not small else 1.8,
                color=color,
            )

        if logscale:

            inset.set_yscale(
                "log"
            )

        inset.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.2,
            alpha=0.2,
        )

        inset.grid(
            False,
            which="minor",
        )

        inset.tick_params(
            axis="both",
            which="both",
            direction="in",
            labelsize=12  if not small else 14,
            top=True,
            right=True,
        )

        #inset.set_title(
            #f"tail from {100.0 * beta:.0f}\%",
            #fontsize=16,
        #)

        inset.text(
            0.5,
            0.96,
            f"tail from {100.0 * beta:.0f}\\%",
            transform=inset.transAxes,
            ha="center",
            va="top",
            fontsize=16  if not small else 20,
        )

    fig.tight_layout()

    beta_suffix = (
        ""
        if beta is None
        else f"_inset_b{int(round(100 * beta))}"
    )

    pdf_path = os.path.join(
        output_dir,
        f"solver_performance_{runtime_metric}_{suffix}{beta_suffix}.pdf",
    )

    png_path = os.path.join(
        output_dir,
        f"solver_performance_{runtime_metric}_{suffix}{beta_suffix}.png",
    )

    plt.savefig(
        pdf_path
    )

    plt.savefig(
        png_path,
        dpi=300,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )



# ============================================================
# Reducer performance
# ============================================================

def reducer_performance(
    output_dir="plots",
    source="reduced_instances",
    reduced_root="data/instances/reduced",
    kind="all",
    common_instances=True,
    logscale=True,
    beta=None,
    header=False,
    solver="ilp",
    preprocessors=["kernel1"],
    only_optimal_results=True,
):
    """
    Plot cactus curves for reducer runtimes.

    Parameters
    ----------
    source:
        "results"
            Use ExperimentResult.reduction_runtime from solved runs.
            Only results with solver == solver and non-empty preprocessing
            are considered.

        "reduced_instances"
            Load reduced BicycleInstance objects from

                data/instances/reduced/villages/
                data/instances/reduced/regions/

            and use instance.total_reduction_runtime.

    reduced_root:
        Root directory for reduced instances.

    kind:
        "all", "village_only", or "region_only".

    common_instances:
        If True, only instances for which every reducer has data are used.

    logscale:
        If True, use logarithmic y-axis.

    beta:
        None or a number in (0, 1). If given, add an inset showing the
        tail of the cactus curves.

    solver:
        Solver used when source == "results".
        Usually keep this as "ilp" to avoid counting the same reduction
        several times for different ILP variants.

    preprocessors:
        Optional list of reducer/preprocessor names to include, e.g.

            ["kernel1"]

        If None, all non-empty preprocessors/reduction pipelines are used.

    only_optimal_results:
        Only relevant for source == "results".
        If True, use only optimal ILP result files.
    """

    import os
    import math
    import pickle
    import statistics
    import matplotlib.pyplot as plt

    if source not in {
        "results",
        "reduced_instances",
    }:

        raise ValueError(
            "source must be one of: "
            "'results', 'reduced_instances'"
        )

    if kind not in {
        "all",
        "village_only",
        "region_only",
    }:

        raise ValueError(
            "kind must be one of: "
            "'all', 'village_only', 'region_only'"
        )

    if beta is not None:

        beta = float(
            beta
        )

        if not (
            0.0 < beta < 1.0
        ):

            raise ValueError(
                "beta must be None or a number in (0, 1)"
            )

    if preprocessors is not None:

        preprocessors = {
            str(preprocessor)
            for preprocessor in preprocessors
        }

    # ------------------------------------------------------------
    # Generic helpers.
    # ------------------------------------------------------------

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def normalize_preprocessor(
        preprocessor,
    ):

        if preprocessor is None:

            return ""

        preprocessor = str(
            preprocessor
        ).strip()

        if preprocessor.lower() in {
            "",
            "none",
            "no",
            "nopre",
        }:

            return ""

        return preprocessor

    def method_name(
        solver_name,
        preprocessor_name,
    ):

        preprocessor_name = normalize_preprocessor(
            preprocessor_name
        )

        if preprocessor_name == "":

            return str(
                solver_name
            )

        return (
            f"{preprocessor_name}_{solver_name}"
        )

    def label_for_method(
        method,
    ):

        if "display_name" in globals():

            return display_name(
                method
            )

        return str(
            method
        )

    def get_instance_name_from_result(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def result_is_optimal(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        status = get_field(
            result,
            "status",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return False

    def get_reduction_runtime_from_result(
        result,
    ):

        candidates = [
            "reduction_runtime",
            "total_reduction_runtime",
        ]

        for field in candidates:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        extra = get_field(
            result,
            "extra",
            {},
        )

        if isinstance(
            extra,
            dict,
        ):

            for field in candidates:

                value = extra.get(
                    field,
                    None,
                )

                if value is not None:

                    return value

        return None

    def get_instance_key_from_bicycle_instance(
        instance,
        path,
    ):
        """
        Use the original instance filename if available.

        This is important because different reduced versions of the same
        original instance should be grouped together.
        """

        if hasattr(
            instance,
            "filename",
        ):

            filename_attr = getattr(
                instance,
                "filename",
            )

            if callable(
                filename_attr
            ):

                try:

                    return filename_attr()

                except Exception:

                    pass

            else:

                return str(
                    filename_attr
                )

        return os.path.basename(
            path
        )

    def load_bicycle_instance_from_path(
        path,
    ):

        if "BicycleInstance" in globals():

            try:

                return BicycleInstance.load(
                    path
                )

            except Exception:

                pass

        with open(
            path,
            "rb",
        ) as f:

            return pickle.load(
                f
            )

    def iter_reduced_instance_paths():
        """
        Recursively iterate over reduced instance pickle files.
        """

        if kind == "all":

            subdirs = [
                "villages",
                "regions",
            ]

        elif kind == "village_only":

            subdirs = [
                "villages",
            ]

        elif kind == "region_only":

            subdirs = [
                "regions",
            ]

        else:

            raise ValueError(
                kind
            )

        for subdir in subdirs:

            root = os.path.join(
                reduced_root,
                subdir,
            )

            if not os.path.isdir(
                root
            ):

                print(
                    "[WARN] reduced-instance directory does not exist:",
                    root,
                )

                continue

            for current_root, _, filenames in os.walk(
                root
            ):

                for filename in filenames:

                    if not filename.endswith(
                        ".pkl"
                    ):

                        continue

                    yield os.path.join(
                        current_root,
                        filename,
                    )

    # ------------------------------------------------------------
    # Collect best runtime per reducer-instance pair.
    # ------------------------------------------------------------

    best = {}

    skipped_nonpositive = 0
    skipped_missing_runtime = 0
    skipped_missing_method = 0
    skipped_not_optimal = 0

    if source == "results":

        results = load_all_results()

        if not results:

            print(
                "[WARN] No results found."
            )

            return

        for result in results:

            if only_optimal_results and not result_is_optimal(
                result
            ):

                skipped_not_optimal += 1

                continue

            result_solver = get_field(
                result,
                "solver",
                None,
            )

            if str(result_solver) != str(solver):

                continue

            preprocessor = normalize_preprocessor(
                get_field(
                    result,
                    "preprocessor",
                    None,
                )
            )

            if preprocessor == "":

                continue

            if (
                preprocessors is not None
                and preprocessor not in preprocessors
            ):

                continue

            instance = get_instance_name_from_result(
                result
            )

            if instance is None:

                print(
                    "[WARN] result without instance name skipped"
                )

                continue

            runtime = get_reduction_runtime_from_result(
                result
            )

            if runtime is None:

                skipped_missing_runtime += 1

                print(
                    "[WARN] result without reduction_runtime skipped:",
                    instance,
                    preprocessor,
                )

                continue

            method = method_name(
                solver,
                preprocessor,
            )

            runtime = float(
                runtime
            )

            if not math.isfinite(
                runtime
            ):

                continue

            if runtime < 0.0:

                skipped_nonpositive += 1

                continue

            if logscale and runtime <= 0.0:

                skipped_nonpositive += 1

                continue

            key = (
                method,
                str(instance),
            )

            if (
                key not in best
                or runtime < best[key]
            ):

                best[key] = runtime

    elif source == "reduced_instances":

        paths = list(
            iter_reduced_instance_paths()
        )

        if not paths:

            print(
                "[WARN] No reduced instance files found under:",
                reduced_root,
            )

            return

        for path in paths:

            try:

                instance = load_bicycle_instance_from_path(
                    path
                )

            except Exception as exc:

                print(
                    "[WARN] could not load reduced instance:",
                    path,
                    exc,
                )

                continue

            preprocessor = normalize_preprocessor(
                getattr(
                    instance,
                    "reduction_pipeline",
                    "",
                )
            )

            if preprocessor == "":

                skipped_missing_method += 1

                continue

            if (
                preprocessors is not None
                and preprocessor not in preprocessors
            ):

                continue

            runtime = getattr(
                instance,
                "total_reduction_runtime",
                None,
            )

            if runtime is None:

                skipped_missing_runtime += 1

                continue

            runtime = float(
                runtime
            )

            if not math.isfinite(
                runtime
            ):

                continue

            if runtime < 0.0:

                skipped_nonpositive += 1

                continue

            if logscale and runtime <= 0.0:

                skipped_nonpositive += 1

                continue

            instance_key = get_instance_key_from_bicycle_instance(
                instance,
                path,
            )

            method = method_name(
                solver,
                preprocessor,
            )

            key = (
                method,
                str(instance_key),
            )

            if (
                key not in best
                or runtime < best[key]
            ):

                best[key] = runtime

    methods = sorted(
        {
            method
            for method, _ in best.keys()
        }
    )

    if not methods:

        print(
            "[WARN] No reducer runtime data found."
        )

        print(
            "source:",
            source,
        )

        print(
            "skipped_missing_runtime:",
            skipped_missing_runtime,
        )

        print(
            "skipped_missing_method:",
            skipped_missing_method,
        )

        print(
            "skipped_nonpositive:",
            skipped_nonpositive,
        )

        print(
            "skipped_not_optimal:",
            skipped_not_optimal,
        )

        return

    # ------------------------------------------------------------
    # Select instances.
    # ------------------------------------------------------------

    instances_by_method = {
        method: {
            instance
            for m, instance in best.keys()
            if m == method
        }
        for method in methods
    }

    if common_instances:

        instances_to_use = None

        for method in methods:

            if instances_to_use is None:

                instances_to_use = set(
                    instances_by_method[method]
                )

            else:

                instances_to_use &= instances_by_method[method]

        if instances_to_use is None:

            instances_to_use = set()

        suffix = "common"

    else:

        instances_to_use = {
            instance
            for _, instance in best.keys()
        }

        suffix = "all"

    if not instances_to_use:

        print(
            "[WARN] No instances to plot."
        )

        return

    # ------------------------------------------------------------
    # Precompute sorted reduction runtimes by method.
    # ------------------------------------------------------------

    runtimes_by_method = {}

    for method in methods:

        runtimes = [
            best[
                (
                    method,
                    instance,
                )
            ]
            for instance in instances_to_use
            if (
                method,
                instance,
            )
            in best
        ]

        if not runtimes:

            continue

        runtimes_by_method[method] = sorted(
            runtimes
        )

    if not runtimes_by_method:

        print(
            "[WARN] No reduction runtimes to plot."
        )

        return

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print()
    print("Reducer performance summary")
    print("---------------------------")
    print("source:", source)
    print("reduced root:", reduced_root)
    print("kind:", kind)
    print("solver label:", solver)
    print("common instances:", common_instances)
    print("instances used:", len(instances_to_use))
    print("logscale:", logscale)
    print("only optimal results:", only_optimal_results)
    print("skipped_missing_runtime:", skipped_missing_runtime)
    print("skipped_missing_method:", skipped_missing_method)
    print("skipped_nonpositive:", skipped_nonpositive)

    if source == "results":

        print("skipped_not_optimal:", skipped_not_optimal)

    print()

    for method in methods:

        if method not in runtimes_by_method:

            continue

        runtimes = runtimes_by_method[
            method
        ]

        avg = sum(runtimes) / len(runtimes)
        med = statistics.median(runtimes)
        mn = min(runtimes)
        mx = max(runtimes)

        print(
            f"{method:35s} "
            f"n={len(runtimes):4d} "
            f"avg={avg:10.3f} "
            f"med={med:10.3f} "
            f"min={mn:10.3f} "
            f"max={mx:10.3f}"
        )

    print()

    # ------------------------------------------------------------
    # Plot cactus curves.
    # ------------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
    )

    line_by_method = {}

    for method in methods:

        if method not in runtimes_by_method:

            continue

        runtimes = runtimes_by_method[
            method
        ]

        x_values = list(
            range(
                1,
                len(runtimes) + 1,
            )
        )

        line_list = ax.plot(
            x_values,
            runtimes,
            marker=None,
            markersize=3,
            linewidth=1.0,
            label=label_for_method(method),
        )

        line_by_method[method] = line_list[0]

    ax.set_xlabel(
        "Instances",
        fontsize=24,
    )

    ax.set_ylabel(
        "Reduction runtime [s]",
        fontsize=24,
    )

    if header:

        if source == "results":

            source_title = "solved results"

        else:

            source_title = "stored reduced instances"

        if common_instances:

            ax.set_title(
                f"Reducer performance on common {source_title}"
            )

        else:

            ax.set_title(
                f"Reducer performance on all {source_title}"
            )

    if logscale:

        ax.set_yscale(
            "log"
        )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.25,
        alpha=0.4,
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend(
        fontsize=20,
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=16,
        top=True,
        right=True,
    )

    # ------------------------------------------------------------
    # Optional inset: tail of cactus curves.
    # ------------------------------------------------------------

    if beta is not None:

        inset = ax.inset_axes(
            [
                0.52,
                0.06,
                0.47,
                0.47,
            ]
        )

        for method in methods:

            if method not in runtimes_by_method:

                continue

            runtimes = runtimes_by_method[
                method
            ]

            start = int(
                round(
                    beta * len(runtimes)
                )
            )

            start = min(
                start,
                len(runtimes) - 1,
            )

            start = max(
                start,
                0,
            )

            runtimes_inset = runtimes[
                start:
            ]

            x_values_inset = list(
                range(
                    start + 1,
                    len(runtimes) + 1,
                )
            )

            color = line_by_method[
                method
            ].get_color()

            inset.plot(
                x_values_inset,
                runtimes_inset,
                marker=None,
                linewidth=0.9,
                color=color,
            )

        if logscale:

            inset.set_yscale(
                "log"
            )

        inset.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.2,
            alpha=0.2,
        )

        inset.grid(
            False,
            which="minor",
        )

        inset.tick_params(
            axis="both",
            which="both",
            direction="in",
            labelsize=12,
            top=True,
            right=True,
        )

        inset.text(
            0.5,
            0.96,
            f"tail from {100.0 * beta:.0f}\\%",
            transform=inset.transAxes,
            ha="center",
            va="top",
            fontsize=16,
        )

    fig.tight_layout()

    beta_suffix = (
        ""
        if beta is None
        else f"_inset_b{int(round(100 * beta))}"
    )

    source_suffix = (
        "results"
        if source == "results"
        else "reduced_instances"
    )

    pdf_path = os.path.join(
        output_dir,
        (
            f"reducer_performance_"
            f"{source_suffix}_"
            f"{kind}_"
            f"{suffix}"
            f"{beta_suffix}.pdf"
        ),
    )

    png_path = os.path.join(
        output_dir,
        (
            f"reducer_performance_"
            f"{source_suffix}_"
            f"{kind}_"
            f"{suffix}"
            f"{beta_suffix}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )









def solver_pair_score_plot_multi(
    method_pairs,
    output_dir="plots/score",
    runtime_metric="total_runtime",
    sort_by="best",
    cumulative=True,
    runtime_epsilon=1e-9,
    normalize_axes=False,
    show_title=False,
):
    """
    Compare several method pairs in one score plot.

    Parameters
    ----------
    method_pairs:
        List of pairs, e.g.

            [
                ["ilp", "kernel1_ilp"],
                ["ilp", "ilp_tw_dp_cuts"],
            ]

        For each pair [A, B]:

            unweighted score:
                +1 if A is faster than B
                -1 if B is faster than A
                 0 if tied

            weighted score:
                runtime(B) - runtime(A)

        Thus positive values always favor A, and negative values favor B.

    runtime_metric:
        "total_runtime", "solver_runtime", or "build_runtime".

    sort_by:
        "best"      sort each pair by min(runtime(A), runtime(B))
        "a"         sort each pair by runtime(A)
        "b"         sort each pair by runtime(B)
        "diff_abs"  sort each pair by abs(runtime(B) - runtime(A))
        "instance"  sort each pair by instance name

    cumulative:
        If True, plot cumulative curves.
        If False, plot per-instance curves.

    normalize_axes:
        If False:
            left axis = raw +1/-1 score values
            right axis = raw weighted runtime score in seconds

        If True:
            left axis and right axis are each divided by their maximum
            absolute observed value, so both axes are in [-1, 1].
    """

    import os
    import math
    import statistics
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if runtime_metric not in {
        "total_runtime",
        "solver_runtime",
        "build_runtime",
    }:

        raise ValueError(
            "runtime_metric must be one of: "
            "'total_runtime', 'solver_runtime', 'build_runtime'"
        )

    if sort_by not in {
        "best",
        "a",
        "b",
        "diff_abs",
        "instance",
    }:

        raise ValueError(
            "sort_by must be one of: "
            "'best', 'a', 'b', 'diff_abs', 'instance'"
        )

    if not method_pairs:

        raise ValueError(
            "method_pairs must not be empty"
        )

    method_pairs = [
        (
            str(pair[0]),
            str(pair[1]),
        )
        for pair in method_pairs
    ]

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def get_runtime(
        result,
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            candidates = [
                "total_runtime",
                "runtime",
            ]

        elif runtime_metric == "solver_runtime":

            candidates = [
                "solver_runtime",
                "gurobi_runtime",
                "optimize_walltime",
                "runtime",
            ]

        elif runtime_metric == "build_runtime":

            candidates = [
                "solver_buildtime",
                "build_runtime",
                "buildtime",
            ]

        for field in candidates:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def is_optimal_result(
        result,
    ):

        status = get_field(
            result,
            "status",
            None,
        )

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return True

    def get_instance_name(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def get_method_name(
        result,
    ):

        solver = get_field(
            result,
            "solver",
            None,
        )

        preprocessor = get_field(
            result,
            "preprocessor",
            None,
        )

        if solver is None:

            return get_field(
                result,
                "method",
                None,
            )

        if (
            preprocessor is None
            or str(preprocessor).strip() == ""
            or str(preprocessor).lower() == "none"
        ):

            return str(
                solver
            )

        return (
            f"{preprocessor}_{solver}"
        )

    def display_method(
        method,
    ):

        if "display_name" in globals():

            return display_name(
                method
            )

        return str(
            method
        )

    def sanitize_filename_part(
        text,
    ):

        text = str(
            text
        )

        for bad in [
            "/",
            "\\",
            " ",
            ":",
            ";",
            ",",
            "(",
            ")",
            "[",
            "]",
        ]:

            text = text.replace(
                bad,
                "_",
            )

        return text

    results = load_all_results()

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Collect best runtime per method-instance pair.
    # ------------------------------------------------------------

    best = {}

    for result in results:

        if not is_optimal_result(
            result
        ):

            continue

        instance = get_instance_name(
            result
        )

        if instance is None:

            print(
                "[WARN] result without instance name skipped"
            )

            continue

        method = get_method_name(
            result
        )

        if method is None:

            print(
                "[WARN] result without method skipped:",
                instance,
            )

            continue

        runtime = get_runtime(
            result,
            runtime_metric,
        )

        if runtime is None:

            continue

        runtime = float(
            runtime
        )

        if not math.isfinite(
            runtime
        ):

            continue

        key = (
            str(method),
            str(instance),
        )

        if (
            key not in best
            or runtime < best[key]
        ):

            best[key] = runtime

    # ------------------------------------------------------------
    # Build curves for each pair.
    # ------------------------------------------------------------

    pair_curves = []

    for method_a, method_b in method_pairs:

        instances_a = {
            instance
            for method, instance in best.keys()
            if method == method_a
        }

        instances_b = {
            instance
            for method, instance in best.keys()
            if method == method_b
        }

        common_instances = sorted(
            instances_a & instances_b
        )

        if not common_instances:

            print(
                "[WARN] No common solved instances for",
                method_a,
                "and",
                method_b,
            )

            continue

        rows = []

        for instance in common_instances:

            runtime_a = best[
                (
                    method_a,
                    instance,
                )
            ]

            runtime_b = best[
                (
                    method_b,
                    instance,
                )
            ]

            # Positive weighted score means A is faster.
            weighted_score = (
                runtime_b
                -
                runtime_a
            )

            if runtime_a < runtime_b - runtime_epsilon:

                score = 1

            elif runtime_b < runtime_a - runtime_epsilon:

                score = -1

            else:

                score = 0

            rows.append(
                {
                    "instance": instance,
                    "runtime_a": runtime_a,
                    "runtime_b": runtime_b,
                    "best_runtime": min(
                        runtime_a,
                        runtime_b,
                    ),
                    "weighted_score": weighted_score,
                    "abs_diff": abs(
                        weighted_score
                    ),
                    "score": score,
                }
            )

        if sort_by == "best":

            rows.sort(
                key=lambda row: (
                    row["best_runtime"],
                    row["instance"],
                )
            )

        elif sort_by == "a":

            rows.sort(
                key=lambda row: (
                    row["runtime_a"],
                    row["instance"],
                )
            )

        elif sort_by == "b":

            rows.sort(
                key=lambda row: (
                    row["runtime_b"],
                    row["instance"],
                )
            )

        elif sort_by == "diff_abs":

            rows.sort(
                key=lambda row: (
                    row["abs_diff"],
                    row["instance"],
                )
            )

        elif sort_by == "instance":

            rows.sort(
                key=lambda row: row["instance"]
            )

        x_values = list(
            range(
                1,
                len(rows) + 1,
            )
        )

        scores = [
            row["score"]
            for row in rows
        ]

        weighted_scores = [
            row["weighted_score"]
            for row in rows
        ]

        if cumulative:

            score_values = []
            weighted_values = []

            score_sum = 0
            weighted_sum = 0.0

            for score, weighted_score in zip(
                scores,
                weighted_scores,
            ):

                score_sum += score
                weighted_sum += weighted_score

                score_values.append(
                    score_sum
                )

                weighted_values.append(
                    weighted_sum
                )

            plot_kind = "cumulative"

        else:

            score_values = scores
            weighted_values = weighted_scores

            plot_kind = "per_instance"

        wins_a = sum(
            1
            for row in rows
            if row["score"] == 1
        )

        wins_b = sum(
            1
            for row in rows
            if row["score"] == -1
        )

        ties = sum(
            1
            for row in rows
            if row["score"] == 0
        )

        total_weighted_score = sum(
            weighted_scores
        )

        pair_curves.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "rows": rows,
                "x_values": x_values,
                "score_values": score_values,
                "weighted_values": weighted_values,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "total_weighted_score": total_weighted_score,
            }
        )

    if not pair_curves:

        print(
            "[WARN] No method pair produced a curve."
        )

        return

    # ------------------------------------------------------------
    # Normalize axes if requested.
    # ------------------------------------------------------------

    score_norm = 1.0
    weighted_norm = 1.0

    if normalize_axes:

        max_abs_score = max(
            abs(value)
            for pair in pair_curves
            for value in pair["score_values"]
        )

        max_abs_weighted = max(
            abs(value)
            for pair in pair_curves
            for value in pair["weighted_values"]
        )

        if max_abs_score > 0.0:

            score_norm = max_abs_score

        if max_abs_weighted > 0.0:

            weighted_norm = max_abs_weighted

        for pair in pair_curves:

            pair["score_values"] = [
                value / score_norm
                for value in pair["score_values"]
            ]

            pair["weighted_values"] = [
                value / weighted_norm
                for value in pair["weighted_values"]
            ]

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print()
    print("Multi-pair solver score plot")
    print("----------------------------")
    print("runtime metric:", runtime_metric)
    print("sort by:", sort_by)
    print("cumulative:", cumulative)
    print("normalize axes:", normalize_axes)

    if normalize_axes:

        print("score normalization:", f"{score_norm:.6g}")
        print("weighted normalization:", f"{weighted_norm:.6g}")

    print()

    for pair in pair_curves:

        method_a = pair["method_a"]
        method_b = pair["method_b"]
        rows = pair["rows"]

        print(
            f"{method_a} vs {method_b}"
        )

        print(
            f"  common solved instances: {len(rows)}"
        )

        print(
            f"  {method_a} wins: {pair['wins_a']}"
        )

        print(
            f"  {method_b} wins: {pair['wins_b']}"
        )

        print(
            f"  ties: {pair['ties']}"
        )

        print(
            "  sum runtime(B) - runtime(A):",
            f"{pair['total_weighted_score']:.3f}",
        )

        if pair["total_weighted_score"] > 0:

            print(
                f"  weighted interpretation: {method_a} is faster in total"
            )

        elif pair["total_weighted_score"] < 0:

            print(
                f"  weighted interpretation: {method_b} is faster in total"
            )

        else:

            print(
                "  weighted interpretation: tied in total runtime"
            )

        weighted_scores = [
            row["weighted_score"]
            for row in rows
        ]

        print(
            "  weighted avg/med/min/max:",
            f"{statistics.mean(weighted_scores):.3f}",
            f"{statistics.median(weighted_scores):.3f}",
            f"{min(weighted_scores):.3f}",
            f"{max(weighted_scores):.3f}",
        )

        print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(
            8.4,
            5.2,
        )
    )

    ax2 = ax1.twinx()

    pair_color_handles = []

    for pair in pair_curves:

        method_a = pair["method_a"]
        method_b = pair["method_b"]

        label_base = (
            f"{display_method(method_a)} vs {display_method(method_b)}"
        )

        weighted_line = ax2.plot(
            pair["x_values"],
            pair["weighted_values"],
            linestyle="-",
            linewidth=2,
            label="_nolegend_",
        )[0]

        color = weighted_line.get_color()

        ax1.plot(
            pair["x_values"],
            pair["score_values"],
            linestyle="--",
            linewidth=2,
            color=color,
            label="_nolegend_",
        )

        pair_color_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2,
                label=label_base,
            )
        )

    ax1.axhline(
        0,
        linewidth=0.8,
        color="black",
        alpha=0.65,
    )

    ax2.axhline(
        0,
        linestyle=":",
        linewidth=0.8,
        color="black",
        alpha=0.45,
    )

    if cumulative:

        ax1.set_xlabel(
            "Common solved instances", #, sorted separately per pair
            fontsize=24,
        )

        if normalize_axes:

            ax1.set_ylabel(
                "Normalized cumulative $\\pm 1$ score"
            )

            ax2.set_ylabel(
                "Normalized cumulative weighted runtime score"
            )

        else:

            ax1.set_ylabel(
                "Cumulative $\\pm 1$ score",
                fontsize=24,
            )

            ax2.set_ylabel(
                "Cumulative weighted runtime score [s]",
                fontsize=24,
            )

    else:

        ax1.set_xlabel(
            "Common solved instances, sorted separately per pair"
        )

        if normalize_axes:

            ax1.set_ylabel(
                "Normalized per-instance $\\pm 1$ score"
            )

            ax2.set_ylabel(
                "Normalized per-instance weighted runtime score"
            )

        else:

            ax1.set_ylabel(
                "Per-instance $\\pm 1$ score"
            )

            ax2.set_ylabel(
                "Per-instance weighted runtime score [s]"
            )

    if normalize_axes:

        ax1.set_ylim(
            -1.05,
            1.05,
        )

        ax2.set_ylim(
            -1.05,
            1.05,
        )

    ax1.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.25,
        alpha=0.22,
    )

    ax1.grid(
        False,
        which="minor",
    )

    if show_title:

        ax1.set_title(
            "Multi-pair solver score comparison"
        )

    pattern_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.4,
            label="weighted score",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=r"$\pm 1$ score",
        ),
    ]

    pattern_legend = ax1.legend(
        handles=pattern_handles,
        fontsize=18,
        frameon=False,
        loc="upper center" if runtime_metric == "total_runtime" else "center left",
        bbox_to_anchor=(0.62, 1.0) if runtime_metric == "total_runtime" else None,
        #title="Line style",
        #title_fontsize=16,
    )

    ax1.add_artist(
        pattern_legend
    )

    ax1.legend(
        handles=pair_color_handles,
        fontsize=18,
        frameon=False,
        loc="upper left" if runtime_metric == "total_runtime" else "lower left",
        #title="Method pair",
        #title_fontsize=8,
    )

    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=False,
        labelsize=18,
    )

    ax2.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=18,
    )

    fig.tight_layout()

    pair_name_part = "__".join(
        (
            sanitize_filename_part(method_a)
            +
            "_vs_"
            +
            sanitize_filename_part(method_b)
        )
        for method_a, method_b in method_pairs
    )

    if len(pair_name_part) > 120:

        pair_name_part = (
            f"{len(method_pairs)}_pairs"
        )

    norm_part = (
        "normalized"
        if normalize_axes
        else
        "raw"
    )

    pdf_path = os.path.join(
        output_dir,
        (
            f"solver_pair_score_multi_"
            f"{pair_name_part}_"
            f"{runtime_metric}_"
            f"{plot_kind}_"
            f"sort_{sort_by}_"
            f"{norm_part}.pdf"
        ),
    )

    png_path = os.path.join(
        output_dir,
        (
            f"solver_pair_score_multi_"
            f"{pair_name_part}_"
            f"{runtime_metric}_"
            f"{plot_kind}_"
            f"sort_{sort_by}_"
            f"{norm_part}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )





# ============================================================
# Hard-instance solver comparison
# ============================================================

def hard_instance_performance(
    reference_method=None,
    top_k=20,
    output_dir="plots",
    logscale=True,
    show_instance_labels=False,
    plot_style="ladder",
    methods_to_plot=None,
    header=False,
):
    """
    Plot solver runtimes on the hardest instances.

    If reference_method is given, hardness is measured by that method.

    If reference_method is None, hardness is measured by the fastest
    available runtime for each instance. Thus the selected instances
    are hard even for the best solver.

    Parameters
    ----------
    show_instance_labels:
        If True, show shortened instance names on the x-axis.
        If False, show only instance ranks 1, ..., top_k.

    plot_style:
        "points"
            Recommended default. Methods are shown as horizontally
            shifted scatter points per instance. This is clearer because
            instances are categorical, not a continuous x-axis.

        "line"
            Old style. Connects runtimes by method. Can look chaotic and
            creates broken lines when a method is missing on some instances.

    methods_to_plot:
        Optional list of methods to include. If None, all methods are used.

    header:
        If True, add a title above the plot.
    """
    from matplotlib.lines import Line2D

    if plot_style not in {
        "points",
        "line",
        "ladder",
    }:

        raise ValueError(
            "plot_style must be one of: 'points', 'line', 'ladder'"
        )

    def method_label(
        method,
    ):

        if "display_name" in globals():

            return display_name(
                method
            )

        return str(
            method
        )

    best = _collect_best_runtimes()

    if not best:

        print(
            "[WARN] No solved results found."
        )

        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    methods = sorted(
        {
            method
            for method, _ in best.keys()
        }
    )

    if methods_to_plot is not None:

        methods_to_plot = [
            str(method)
            for method in methods_to_plot
        ]

        methods = [
            method
            for method in methods
            if method in methods_to_plot
        ]

    if not methods:

        print(
            "[WARN] No methods to plot."
        )

        return

    instances_by_method = {
        method: {
            instance
            for m, instance in best.keys()
            if m == method
        }
        for method in methods
    }

    # ------------------------------------------------------------
    # Select hardest instances.
    # ------------------------------------------------------------

    if reference_method is None:

        all_instances = sorted(
            {
                instance
                for _, instance in best.keys()
            }
        )

        instance_best_runtimes = []

        for instance in all_instances:

            runtimes = [
                runtime
                for (method, inst), runtime in best.items()
                if inst == instance
            ]

            if not runtimes:

                continue

            instance_best_runtimes.append(
                (
                    instance,
                    min(runtimes),
                )
            )

        selected = sorted(
            instance_best_runtimes,
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        reference_label = "fastest solver"

    else:

        reference_method = str(
            reference_method
        )

        all_available_methods = sorted(
            {
                method
                for method, _ in best.keys()
            }
        )

        if reference_method not in all_available_methods:

            print(
                "[WARN] Unknown reference method:",
                reference_method,
            )

            print(
                "Available methods:"
            )

            for method in all_available_methods:

                print(
                    " ",
                    method,
                )

            return

        selected = sorted(
            [
                (
                    instance,
                    best[
                        (
                            reference_method,
                            instance,
                        )
                    ],
                )
                for instance in {
                    instance
                    for method, instance in best.keys()
                    if method == reference_method
                }
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        reference_label = reference_method

    if not selected:

        print(
            "[WARN] No hard instances selected."
        )

        return

    selected_instances = [
        instance
        for instance, _ in selected
    ]

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print()
    print(
        "Hard-instance plot"
    )

    print(
        "hardness reference:",
        reference_label,
    )

    print(
        "top k:",
        len(selected_instances),
    )

    print(
        "plot style:",
        plot_style,
    )

    print(
        "show instance labels:",
        show_instance_labels,
    )

    print()

    print(
        "selected instances:"
    )

    for instance, runtime in selected:

        print(
            f"  {instance:50s} {runtime:10.3f}s"
        )

    print()

    print(
        "method coverage on selected instances:"
    )

    for method in methods:

        solved_count = sum(
            1
            for instance in selected_instances
            if (
                method,
                instance,
            )
            in best
        )

        print(
            f"  {method:35s} "
            f"{solved_count:3d}/{len(selected_instances):3d}"
        )

    print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.5, 5.8)
    )

    x_values = list(
        range(
            len(selected_instances)
        )
    )

    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get(
        "color",
        [
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
        ],
    )

    method_colors = {
        method: color_cycle[
            index % len(color_cycle)
        ]
        for index, method in enumerate(methods)
    }

    # ------------------------------------------------------------
    # Recommended style: points with horizontal offsets.
    # ------------------------------------------------------------

    if plot_style == "ladder":

        band_width = 0.72
        line_width = 1.5
        band_alpha = 0.14

        for index, instance in enumerate(selected_instances):

            solved = []

            for method in methods:

                value = best.get(
                    (
                        method,
                        instance,
                    ),
                    None,
                )

                if value is None:

                    continue

                if not math.isfinite(
                    value
                ):

                    continue

                if logscale and value <= 0.0:

                    continue

                solved.append(
                    (
                        method,
                        value,
                    )
                )

            if not solved:

                continue

            solved.sort(
                key=lambda item: item[1]
            )

            x_left = index - band_width / 2.0
            x_right = index + band_width / 2.0

            # Draw transparent runtime gaps.
            for rank, (method, value) in enumerate(solved[:-1]):

                next_value = solved[
                    rank + 1
                ][1]

                if next_value <= value:

                    continue

                color = method_colors[
                    method
                ]

                ax.fill_between(
                    [
                        x_left,
                        x_right,
                    ],
                    [
                        value,
                        value,
                    ],
                    [
                        next_value,
                        next_value,
                    ],
                    color=color,
                    alpha=band_alpha,
                    linewidth=0.0,
                    zorder=1,
                )

            # Draw actual solver runtimes as thick horizontal lines.
            for rank, (method, value) in enumerate(solved):

                color = method_colors[
                    method
                ]

                ax.hlines(
                    y=value,
                    xmin=x_left,
                    xmax=x_right,
                    color=color,
                    linewidth=line_width,
                    zorder=3,
                )

    elif plot_style == "points":

        if len(methods) == 1:

            offsets = {
                methods[0]: 0.0
            }

        else:

            max_offset = 0.35

            offsets = {}

            for i, method in enumerate(methods):

                offsets[method] = (
                    -max_offset
                    +
                    2.0 * max_offset * i / (len(methods) - 1)
                )

        for method in methods:

            xs = []
            ys = []

            for index, instance in enumerate(selected_instances):

                value = best.get(
                    (
                        method,
                        instance,
                    ),
                    None,
                )

                if value is None:

                    continue

                if not math.isfinite(
                    value
                ):

                    continue

                if logscale and value <= 0.0:

                    continue

                xs.append(
                    index + offsets[method]
                )

                ys.append(
                    value
                )

            if not xs:

                continue

            ax.scatter(
                xs,
                ys,
                s=22,
                alpha=0.85,
                color=method_colors[method],
                label=method_label(method),
            )

    # ------------------------------------------------------------
    # Old style: connected lines.
    # ------------------------------------------------------------

    elif plot_style == "line":

        for method in methods:

            y_values = []

            has_value = False

            for instance in selected_instances:

                value = best.get(
                    (
                        method,
                        instance,
                    ),
                    math.nan,
                )

                if (
                    value is not None
                    and math.isfinite(value)
                    and (
                        not logscale
                        or value > 0.0
                    )
                ):

                    has_value = True

                    y_values.append(
                        value
                    )

                else:

                    y_values.append(
                        math.nan
                    )

            if not has_value:

                continue

            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=2.2,
                linewidth=0.9,
                alpha=0.85,
                color=method_colors[method],
                label=method_label(method),
            )

    # ------------------------------------------------------------
    # X ticks.
    # ------------------------------------------------------------

    ax.set_xticks(
        x_values
    )

    if show_instance_labels:

        labels = [
            _short_instance_label(instance)
            for instance in selected_instances
        ]

        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )

    else:

        labels = [
            str(index + 1)
            for index in x_values
        ]

        ax.set_xticklabels(
            labels,
            fontsize=10,
        )

    ax.set_xlabel(
        (
            f"Top {len(selected_instances)} hardest instances "
            f"by {method_label(reference_label)}"
        )
    )

    ax.set_ylabel(
        "Total runtime [s]"
    )

    if header:

        ax.set_title(
            "Solver performance on hard instances"
        )

    if logscale:

        ax.set_yscale(
            "log"
        )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.25,
        alpha=0.22,
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=method_colors[method],
            linewidth=3.2,
            label=method_label(method),
        )
        for method in methods
    ]

    ax.legend(
        handles=method_handles,
        fontsize=8,
        frameon=False,
        loc="best",
        ncol=2,
    )

    fig.tight_layout()

    safe_reference = (
        str(reference_label)
        .replace("/", "_")
        .replace(" ", "_")
    )

    label_suffix = (
        "labels"
        if show_instance_labels
        else
        "nolabels"
    )

    pdf_path = os.path.join(
        output_dir,
        (
            f"hard_instance_performance_"
            f"{safe_reference}_"
            f"{top_k}_"
            f"{plot_style}_"
            f"{label_suffix}.pdf"
        ),
    )

    png_path = os.path.join(
        output_dir,
        (
            f"hard_instance_performance_"
            f"{safe_reference}_"
            f"{top_k}_"
            f"{plot_style}_"
            f"{label_suffix}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )


def _short_instance_label(
    instance,
    max_len=28,
):

    instance = str(
        instance
    )

    if len(instance) <= max_len:
        return instance

    return (
        instance[: max_len - 3]
        + "..."
    )



def kernel1_reduction_effect_by_safety_plot(
    kind="villages",
    instances_root="data/instances",
    reduced_label="kernel1",
    safety_model="all",
    mode="remaining",
    save=True,
    show=False,
    header=False,
):
    """
    Plot how much kernel1 reduces instances, separated by safety model.

    If kind is "villages" or "regions", the plot contains one box per
    safety model group.

    If kind is "both", each safety model group contains two adjacent
    boxes:
        left  = villages
        right = regions

    Parameters
    ----------
    kind:
        "villages", "regions", or "both". Default: "villages".

    instances_root:
        Root folder containing original/ and reduced/.

    reduced_label:
        Label used to identify reduced files if filenames differ.
        Default: "kernel1".

    safety_model:
        "all", "A", "B", or "C".
        If "all", plot A, B, C, and Overall.
        If one of "A", "B", "C", plot only that safety model.

    mode:
        "remaining" plots 100 * reduced / original.
        "removed" plots 100 * (original - reduced) / original.

    save:
        If True, save PDF and PNG to plots/.

    show:
        If True, display the plot.
    """

    import os
    import pickle
    import statistics
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if kind not in {"villages", "regions", "both"}:
        raise ValueError(
            "kind must be one of: 'villages', 'regions', 'both'"
        )

    safety_model = str(safety_model).upper()

    if safety_model not in {"ALL", "A", "B", "C"}:
        raise ValueError(
            "safety_model must be one of: 'all', 'A', 'B', 'C'"
        )

    if mode not in {"remaining", "removed"}:
        raise ValueError(
            "mode must be either 'remaining' or 'removed'"
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def load_pickle(path):

        with open(path, "rb") as f:
            return pickle.load(f)

    def get_network(obj):

        if hasattr(obj, "network"):
            return obj.network

        return obj

    def all_edges(net):

        return (
            set(net.safe_edges)
            |
            set(net.unsafe_edges)
        )

    def normalize_safety_model(value):

        if value is None:
            return None

        value = str(value).upper()

        aliases = {
            "HIER": "A",
            "SEC": "B",
            "PES": "C",
        }

        value = aliases.get(
            value,
            value,
        )

        if value in {"A", "B", "C"}:
            return value

        return None

    def infer_safety_model(obj, path):

        # First try object metadata.
        for candidate in [
            obj,
            get_network(obj),
        ]:

            if hasattr(candidate, "safety_model"):

                model = normalize_safety_model(
                    getattr(candidate, "safety_model")
                )

                if model is not None:
                    return model

        # Then try filename tokens.
        stem = path.stem

        tokens = (
            stem.replace("-", "_")
            .replace(".", "_")
            .split("_")
        )

        for token in reversed(tokens):

            model = normalize_safety_model(
                token
            )

            if model is not None:
                return model

        return None

    def normalize_stem(stem):

        s = stem

        for token in [
            reduced_label,
            "kernel1",
            "reduced",
        ]:

            s = s.replace(f"_{token}", "")
            s = s.replace(f"{token}_", "")
            s = s.replace(f"-{token}", "")
            s = s.replace(f"{token}-", "")

        return s

    def find_reduced_file(original_path, reduced_dir):

        exact = reduced_dir / original_path.name

        if exact.exists():
            return exact

        original_stem = original_path.stem
        original_norm = normalize_stem(original_stem)

        candidates = []

        for p in sorted(reduced_dir.glob("*.pkl")):

            reduced_stem = p.stem
            reduced_norm = normalize_stem(reduced_stem)

            if reduced_stem == original_stem:

                candidates.append(p)

            elif reduced_norm == original_norm:

                candidates.append(p)

            elif (
                original_stem in reduced_stem
                and reduced_label in reduced_stem
            ):

                candidates.append(p)

            elif (
                original_norm in reduced_norm
                and reduced_label in reduced_stem
            ):

                candidates.append(p)

        if not candidates:
            return None

        candidates = sorted(
            candidates,
            key=lambda p: (
                len(p.name),
                p.name,
            ),
        )

        return candidates[0]

    def instance_stats(obj):

        net = get_network(obj)

        edges = all_edges(net)

        total_length = sum(
            net.length[e]
            for e in edges
            if e in net.length
        )

        unsafe_upgrade_cost = sum(
            net.upgrade_cost[e]
            for e in net.unsafe_edges
            if e in net.upgrade_cost
        )

        terminal_pairs = None

        if hasattr(obj, "terminal_pairs"):
            terminal_pairs = len(obj.terminal_pairs)

        return {
            "Vertices": len(net.vertices),
            "Edges": len(edges),
            "Unsafe edges": len(net.unsafe_edges),
            "Unsafe upgrade cost": unsafe_upgrade_cost,
            "Total length": total_length,
            "Terminal pairs": terminal_pairs,
        }

    # ------------------------------------------------------------
    # Load and match original/reduced instances.
    # ------------------------------------------------------------

    metric_names = [
        "Vertices",
        "Edges",
        "Unsafe edges",
        "Unsafe upgrade cost",
        "Total length",
        "Terminal pairs",
    ]

    if kind == "both":

        kinds_to_load = [
            "villages",
            "regions",
        ]

    else:

        kinds_to_load = [
            kind,
        ]

    rows = []
    missing = []
    unknown_safety_model = []

    for current_kind in kinds_to_load:

        original_dir = (
            Path(instances_root)
            / "original"
            / current_kind
        )

        reduced_dir = (
            Path(instances_root)
            / "reduced"
            / current_kind
        )

        if not original_dir.exists():
            raise FileNotFoundError(
                f"Original instance folder not found: {original_dir}"
            )

        if not reduced_dir.exists():
            raise FileNotFoundError(
                f"Reduced instance folder not found: {reduced_dir}"
            )

        for original_path in sorted(original_dir.glob("*.pkl")):

            reduced_path = find_reduced_file(
                original_path,
                reduced_dir,
            )

            if reduced_path is None:
                missing.append(
                    f"{current_kind}/{original_path.name}"
                )
                continue

            original_obj = load_pickle(original_path)
            reduced_obj = load_pickle(reduced_path)

            model = infer_safety_model(
                original_obj,
                original_path,
            )

            if model is None:
                unknown_safety_model.append(
                    f"{current_kind}/{original_path.name}"
                )
                continue

            if safety_model != "ALL" and model != safety_model:
                continue

            before = instance_stats(original_obj)
            after = instance_stats(reduced_obj)

            row = {
                "instance": original_path.stem,
                "kind": current_kind,
                "safety_model": model,
                "original_file": str(original_path),
                "reduced_file": str(reduced_path),
            }

            for metric in metric_names:

                b = before[metric]
                a = after[metric]

                if b is None or a is None or b == 0:
                    row[metric] = None
                    continue

                if mode == "remaining":

                    value = (
                        100.0
                        *
                        a
                        /
                        b
                    )

                else:

                    value = (
                        100.0
                        *
                        (
                            b
                            -
                            a
                        )
                        /
                        b
                    )

                row[metric] = value

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No matched instance pairs found for kind={kind} "
            f"and safety_model={safety_model}."
        )

    # ------------------------------------------------------------
    # Prepare groups.
    # ------------------------------------------------------------

    if safety_model == "ALL":

        groups = [
            "A",
            "B",
            "C",
            "Overall",
        ]

    else:

        groups = [
            safety_model,
        ]

    def values_for(metric, group, current_kind=None):

        selected = rows

        if current_kind is not None:

            selected = [
                row
                for row in selected
                if row["kind"] == current_kind
            ]

        if group == "Overall":

            return [
                row[metric]
                for row in selected
                if row[metric] is not None
            ]

        return [
            row[metric]
            for row in selected
            if (
                row["safety_model"] == group
                and row[metric] is not None
            )
        ]

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        sharey=True,
    )

    village_color = "deepskyblue"
    region_color = "red"

    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):

        if kind != "both":

            data = []
            labels = []

            for group in groups:

                vals = values_for(
                    metric,
                    group,
                )

                if vals:

                    data.append(vals)
                    labels.append(group)

            if not data:

                ax.set_title(
                    metric
                )

                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                continue

            ax.boxplot(
                data,
                labels=labels,
                showmeans=True,
                labelsize=16,
            )

            # Median labels.
            for i, vals in enumerate(data, start=1):

                median = statistics.median(vals)

                ax.text(
                    i,
                    median,
                    f"{median:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        else:

            data = []
            positions = []
            box_labels = []

            group_centers = list(
                range(
                    1,
                    len(groups) + 1,
                )
            )

            offset = 0.18
            width = 0.32

            for center, group in zip(group_centers, groups):

                village_vals = values_for(
                    metric,
                    group,
                    current_kind="villages",
                )

                region_vals = values_for(
                    metric,
                    group,
                    current_kind="regions",
                )

                if village_vals:

                    data.append(
                        village_vals
                    )

                    positions.append(
                        center - offset
                    )

                    box_labels.append(
                        ("villages", center - offset)
                    )

                if region_vals:

                    data.append(
                        region_vals
                    )

                    positions.append(
                        center + offset
                    )

                    box_labels.append(
                        ("regions", center + offset)
                    )

            if not data:

                ax.set_title(
                    metric
                )

                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width,
                showmeans=True,
                patch_artist=True,
            )

            # Use hatching rather than custom colors, so the plot is
            # still readable in grayscale.
            for patch, (box_kind, _) in zip(
                bp["boxes"],
                box_labels,
            ):

                if box_kind == "villages":

                    #patch.set_hatch("")
                    patch.set_facecolor(
                        village_color
                    )

                    patch.set_edgecolor(
                        "black"
                    )

                    patch.set_alpha(
                        0.3
                    )

                    patch.set_hatch(
                        None
                    )

                else:

                    #patch.set_hatch("//")
                    patch.set_facecolor(
                        region_color
                    )

                    patch.set_edgecolor(
                        "black"
                    )

                    patch.set_alpha(
                        0.3
                    )

                    patch.set_hatch(
                        None
                    )

            ax.set_xticks(
                group_centers
            )

            ax.set_xticklabels(
                groups
            )

            # Median labels.
            for pos, vals in zip(positions, data):

                median = statistics.median(vals)

                ax.text(
                    pos,
                    median,
                    f"{median:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_title(
            metric,
            fontsize=16,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=10
        )

    if mode == "remaining":

        ylabel = "Remaining (\\%)"
        title_mode = "remaining"

    else:

        ylabel = "Removed by kernel1 (\\%)"
        title_mode = "removed"

    axes[0].set_ylabel(
        ylabel,
        fontsize=16,
    )

    axes[3].set_ylabel(
        ylabel,
        fontsize=16,
    )

    if header:
        if safety_model == "ALL":

            title = (
                f"Kernel1 reduction effect by safety model "
                f"({kind}, {title_mode})"
            )

        else:

            title = (
                f"Kernel1 reduction effect for safety model "
                f"{safety_model} ({kind}, {title_mode})"
            )

        fig.suptitle(
            title
        )


    if kind == "both":

        #legend_handles = [
            #Patch(
                #label="Villages",
                #hatch="",
            #),
            #Patch(
                #label="Regions",
                #hatch="//",
            #),
        #]

        #fig.legend(
            #handles=legend_handles,
            #loc="upper center",
            #ncol=2,
            #bbox_to_anchor=(0.5, 0.96),
            #frameon=False,
        #)

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=village_color,
                edgecolor="black",
                alpha=0.3,
                label="Villages",
            ),
            Patch(
                facecolor=region_color,
                edgecolor="black",
                alpha=0.3,
                label="Regions",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="center",
            ncol=2,
            bbox_to_anchor=(0.525, 0.55),
            fontsize=16
        )

    if header:
        plt.tight_layout(
            rect=(
                0,
                0,
                1,
                0.94 if kind == "both" else 0.96,
            ),
            #pad=0.05
        )
    else:
        fig.tight_layout(
            pad=0.05,
        )

    if save:

        os.makedirs(
            "plots",
            exist_ok=True,
        )

        filename_base = (
            f"kernel1_reduction_effect_by_safety_"
            f"{kind}_{safety_model.lower()}_{mode}"
        )

        pdf = f"plots/{filename_base}.pdf"
        png = f"plots/{filename_base}.png"

        plt.savefig(
            pdf,
            #pad_inches=0.01,
        )

        plt.savefig(
            png,
            dpi=300,
            #pad_inches=0.01,
        )

        print(
            f"Saved {pdf}"
        )

        print(
            f"Saved {png}"
        )

    # ------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------

    print()
    print(
        f"Kernel1 reduction effect by safety model for {kind}"
    )

    print(
        f"Matched instances: {len(rows)}"
    )

    if missing:

        print(
            f"Missing reduced counterparts: {len(missing)}"
        )

        print(
            "First missing files:",
            missing[:5],
        )

    if unknown_safety_model:

        print(
            f"Unknown safety model: {len(unknown_safety_model)}"
        )

        print(
            "First unknown files:",
            unknown_safety_model[:5],
        )

    for metric in metric_names:

        print()
        print(metric)

        if kind != "both":

            for group in groups:

                vals = values_for(
                    metric,
                    group,
                )

                if not vals:
                    continue

                print(
                    f"  {group}: "
                    f"n={len(vals)}, "
                    f"mean={statistics.mean(vals):.2f}, "
                    f"median={statistics.median(vals):.2f}, "
                    f"min={min(vals):.2f}, "
                    f"max={max(vals):.2f}"
                )

        else:

            for group in groups:

                for current_kind in [
                    "villages",
                    "regions",
                ]:

                    vals = values_for(
                        metric,
                        group,
                        current_kind=current_kind,
                    )

                    if not vals:
                        continue

                    print(
                        f"  {group} / {current_kind}: "
                        f"n={len(vals)}, "
                        f"mean={statistics.mean(vals):.2f}, "
                        f"median={statistics.median(vals):.2f}, "
                        f"min={min(vals):.2f}, "
                        f"max={max(vals):.2f}"
                    )

    if show:
        plt.show()
    else:
        plt.close()


def solved_instances_by_solver_plot(
    output_dir="plots",
    group_by="method",
    sort=True,
    show_values=True,
):
    """
    Plot the number of optimally solved instances per solver.

    Parameters
    ----------
    group_by:
        "method"
            Count by preprocessor+solver combination, e.g.
            "kernel1_ilp_tw_dp_cuts".

        "solver"
            Count only by solver name, ignoring preprocessor, e.g.
            "ilp_tw_dp_cuts".

    sort:
        If True, sort bars by number solved, descending.

    show_values:
        If True, print the count above each bar.
    """

    if group_by not in {
        "method",
        "solver",
    }:

        raise ValueError(
            "group_by must be one of: 'method', 'solver'"
        )

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def is_optimal_result(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        status = get_field(
            result,
            "status",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return False

    def get_instance_name(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def get_solver_name(
        result,
    ):

        solver = get_field(
            result,
            "solver",
            None,
        )

        if solver is None:

            solver = get_field(
                result,
                "method",
                None,
            )

        if solver is None:

            return None

        return str(
            solver
        )

    def get_method_name(
        result,
    ):

        solver = get_field(
            result,
            "solver",
            None,
        )

        preprocessor = get_field(
            result,
            "preprocessor",
            None,
        )

        if solver is None:

            method = get_field(
                result,
                "method",
                None,
            )

            if method is None:

                return None

            return str(
                method
            )

        if (
            preprocessor is None
            or str(preprocessor).strip() == ""
            or str(preprocessor).lower() == "none"
        ):

            return str(
                solver
            )

        return (
            f"{preprocessor}_{solver}"
        )

    results = load_all_results()

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    solved_instances = {}

    for result in results:

        if not is_optimal_result(
            result
        ):

            continue

        instance = get_instance_name(
            result
        )

        if instance is None:

            print(
                "[WARN] result without instance skipped"
            )

            continue

        if group_by == "method":

            name = get_method_name(
                result
            )

        else:

            name = get_solver_name(
                result
            )

        if name is None:

            print(
                "[WARN] result without solver/method skipped:",
                instance,
            )

            continue

        solved_instances.setdefault(
            name,
            set(),
        ).add(
            str(instance)
        )

    rows = [
        (
            name,
            len(instances),
        )
        for name, instances in solved_instances.items()
    ]

    if not rows:

        print(
            "[WARN] No optimal results found."
        )

        return

    if sort:

        rows.sort(
            key=lambda row: (
                -row[1],
                row[0],
            )
        )

    else:

        rows.sort(
            key=lambda row: row[0]
        )

    names = [
        row[0]
        for row in rows
    ]

    counts = [
        row[1]
        for row in rows
    ]

    print()
    print(
        "Optimally solved instances by",
        group_by,
    )
    print("--------------------------------")

    for name, count in rows:

        print(
            f"{name:35s} {count:5d}"
        )

    print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    width = max(
        7.0,
        0.35 * len(names),
    )

    plt.figure(
        figsize=(
            width,
            4.8,
        )
    )

    x_values = list(
        range(
            len(names)
        )
    )

    bars = plt.bar(
        x_values,
        counts,
        edgecolor="black",
        linewidth=0.6,
    )

    plt.xticks(
        x_values,
        names,
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Optimally solved instances"
    )

    plt.xlabel(
        "Solver" if group_by == "solver" else "Method"
    )

    plt.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    plt.grid(
        True,
        axis="y",
        linestyle="-",
        linewidth=0.2,
    )

    if show_values:

        for bar, count in zip(
            bars,
            counts,
        ):

            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(
        pad=0.05
    )

    pdf_path = os.path.join(
        output_dir,
        f"solved_instances_by_{group_by}.pdf",
    )

    png_path = os.path.join(
        output_dir,
        f"solved_instances_by_{group_by}.png",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )






def alpha_optimal_cost_plot(
    output_dir="plots",
    method=None,
    solver=None,
    preprocessor=None,
    cost_field="objective",
    alphas=None,
    sort_by="smallest_alpha_cost",
    normalize=False,
    show_relative_reduction=True,
    relative_reduction_ylim=(0, 100),
    relative_reduction_bin_size=23,
    plot_mode="cost_main",
    reduction_sort_pair=None,
    reduction_sort_descending=False,
    absolute_cost_bin_size=23,
    absolute_cost_ylim=None,
    show_title=False,
    show_instance_labels=False,
):
    """
    Plot optimal costs for congruent instances across different alpha values.

    Congruent means:
        same instance filename after removing the alpha token, e.g.

            wettringen_A_hotspots_tn50_a12_i0
            wettringen_A_hotspots_tn50_a15_i0

        become the same base key after removing a12/a15.

    Only base instances for which all selected/available alphas have an
    optimal result are plotted.

    Parameters
    ----------
    method:
        Exact method name, e.g. "ilp", "kernel1_ilp",
        "kernel1_ilp_tw_dp_cuts". If None, all optimal results are used and
        the best available objective value per base-instance/alpha is kept.

    solver, preprocessor:
        Alternative filter if method is None. For example:
            solver="ilp_tw_dp_cuts", preprocessor="kernel1"

    cost_field:
        Optional explicit result field for the objective value.
        If None, several common field names are tried.

    alphas:
        Optional list of alpha values to compare, e.g. [1.2, 1.5].
        If None, all alpha values found in the filtered results are used.

    sort_by:
        "smallest_alpha_cost"  sort by cost at smallest alpha
        "largest_alpha_cost"   sort by cost at largest alpha
        "average_cost"         sort by average cost over alphas
        "instance"             sort by base instance key

    normalize:
        If True, divide every cost curve by the cost at the smallest alpha
        for that instance. This shows relative cost reduction.

    show_instance_labels:
        If True, use base instance names as x tick labels. Usually too
        crowded for large experiments.
    """

    if sort_by not in {
        "smallest_alpha_cost",
        "largest_alpha_cost",
        "average_cost",
        "instance",
    }:

        raise ValueError(
            "sort_by must be one of: "
            "'smallest_alpha_cost', 'largest_alpha_cost', "
            "'average_cost', 'instance'"
        )


    if plot_mode not in {
        "cost_main",
        "reduction_main",
    }:

        raise ValueError(
            "plot_mode must be one of: "
            "'cost_main', 'reduction_main'"
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Robust result access helpers.
    # ------------------------------------------------------------

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def is_optimal_result(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        status = get_field(
            result,
            "status",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return False

    def get_instance_name(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def get_method_name(
        result,
    ):

        result_solver = get_field(
            result,
            "solver",
            None,
        )

        result_preprocessor = get_field(
            result,
            "preprocessor",
            None,
        )

        if result_solver is None:

            result_method = get_field(
                result,
                "method",
                None,
            )

            if result_method is None:

                return None

            return str(
                result_method
            )

        if (
            result_preprocessor is None
            or str(result_preprocessor).strip() == ""
            or str(result_preprocessor).lower() == "none"
        ):

            return str(
                result_solver
            )

        return (
            f"{result_preprocessor}_{result_solver}"
        )

    def result_matches_filter(
        result,
    ):

        result_method = get_method_name(
            result
        )

        if method is not None:

            return (
                result_method
                ==
                method
            )

        if solver is not None:

            result_solver = get_field(
                result,
                "solver",
                None,
            )

            if str(result_solver) != str(solver):

                return False

        if preprocessor is not None:

            result_preprocessor = get_field(
                result,
                "preprocessor",
                None,
            )

            if (
                preprocessor is None
                or str(preprocessor).strip() == ""
                or str(preprocessor).lower() == "none"
            ):

                expected = ""

            else:

                expected = str(
                    preprocessor
                )

            actual = (
                ""
                if result_preprocessor is None
                else str(result_preprocessor)
            )

            if actual.lower() == "none":

                actual = ""

            if actual != expected:

                return False

        return True

    def get_objective_cost(
        result,
    ):

        if cost_field is not None:

            value = get_field(
                result,
                cost_field,
                None,
            )

            if value is not None:

                return value

            return None

        candidates = [
            "objective_value",
            "objective",
            "optimal_cost",
            "solution_cost",
            "obj_val",
            "obj_value",
            "ObjVal",
            "cost",
            "value",
        ]

        for field in candidates:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def parse_alpha_and_base_key(
        instance_name,
    ):
        """
        Parse alpha token from an instance filename and return

            alpha, base_key

        where base_key is the filename without the alpha token.

        Example:
            foo_bar_a12_i0.pkl -> 1.2, foo_bar_i0
        """

        stem = os.path.basename(
            str(instance_name)
        )

        if stem.endswith(
            ".pkl"
        ):

            stem = stem[:-4]

        if stem.endswith(
            ".json"
        ):

            stem = stem[:-5]

        parts = stem.split(
            "_"
        )

        alpha = None
        base_parts = []

        for part in parts:

            match = re.fullmatch(
                r"a(\d+)",
                part,
            )

            if match is not None:

                alpha = int(
                    match.group(1)
                ) / 10.0

            else:

                base_parts.append(
                    part
                )

        if alpha is None:

            return None, None

        base_key = "_".join(
            base_parts
        )

        return alpha, base_key

    def sanitize_filename_part(
        text,
    ):

        text = str(
            text
        )

        for bad in [
            "/",
            "\\",
            " ",
            ":",
            ";",
            ",",
            "(",
            ")",
        ]:

            text = text.replace(
                bad,
                "_",
            )

        return text

    # ------------------------------------------------------------
    # Load and collect best cost per base-instance/alpha.
    # ------------------------------------------------------------

    results = load_all_results()

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    best_cost = {}

    observed_alphas = set()

    for result in results:

        if not is_optimal_result(
            result
        ):

            continue

        if not result_matches_filter(
            result
        ):

            continue

        instance_name = get_instance_name(
            result
        )

        if instance_name is None:

            print(
                "[WARN] result without instance name skipped"
            )

            continue

        alpha, base_key = parse_alpha_and_base_key(
            instance_name
        )

        if alpha is None:

            print(
                "[WARN] could not parse alpha from instance:",
                instance_name,
            )

            continue

        cost = get_objective_cost(
            result
        )

        if cost is None:

            print(
                "[WARN] result without objective cost skipped:",
                instance_name,
                get_method_name(result),
            )

            continue

        cost = float(
            cost
        )

        if not math.isfinite(
            cost
        ):

            continue

        if alphas is not None:

            alpha_allowed = any(
                abs(alpha - float(a)) <= 1e-9
                for a in alphas
            )

            if not alpha_allowed:

                continue

        observed_alphas.add(
            alpha
        )

        key = (
            base_key,
            alpha,
        )

        # If multiple methods/runs are present, keep the best optimal cost.
        # With exact optimal solves these should agree, but this is robust.
        if (
            key not in best_cost
            or cost < best_cost[key]
        ):

            best_cost[key] = cost

    alpha_values = sorted(
        observed_alphas
    )

    if alphas is not None:

        alpha_values = sorted(
            float(a)
            for a in alphas
        )

    if not alpha_values:

        print(
            "[WARN] No alpha values found."
        )

        return

    # ------------------------------------------------------------
    # Keep only complete congruence classes.
    # ------------------------------------------------------------

    base_keys = sorted(
        {
            base_key
            for base_key, _ in best_cost.keys()
        }
    )

    complete_rows = []

    for base_key in base_keys:

        if all(
            (
                base_key,
                alpha,
            )
            in best_cost
            for alpha in alpha_values
        ):

            costs = {
                alpha: best_cost[
                    (
                        base_key,
                        alpha,
                    )
                ]
                for alpha in alpha_values
            }

            complete_rows.append(
                {
                    "base_key": base_key,
                    "costs": costs,
                }
            )

    if not complete_rows:

        print(
            "[WARN] No complete instance groups found for alphas:",
            alpha_values,
        )

        return

    smallest_alpha = alpha_values[0]
    largest_alpha = alpha_values[-1]

    def relative_reduction_for_row(
        row,
        alpha_from,
        alpha_to,
    ):

        c_from = row["costs"][alpha_from]
        c_to = row["costs"][alpha_to]

        if abs(c_from) <= 1e-12:

            return 0.0

        return (
            c_from - c_to
        ) / c_from


    if reduction_sort_pair is None:

        reduction_sort_pair = (
            smallest_alpha,
            largest_alpha,
        )

    else:

        reduction_sort_pair = tuple(
            float(a)
            for a in reduction_sort_pair
        )

    if reduction_sort_pair[0] not in alpha_values:

        raise ValueError(
            f"reduction_sort_pair start alpha "
            f"{reduction_sort_pair[0]} not in alpha_values={alpha_values}"
        )

    if reduction_sort_pair[1] not in alpha_values:

        raise ValueError(
            f"reduction_sort_pair end alpha "
            f"{reduction_sort_pair[1]} not in alpha_values={alpha_values}"
        )

    if reduction_sort_pair[0] >= reduction_sort_pair[1]:

        raise ValueError(
            "reduction_sort_pair must be increasing, "
            "e.g. (1.2, 1.5)"
        )


    if plot_mode == "reduction_main":

        alpha_from, alpha_to = reduction_sort_pair

        complete_rows.sort(
            key=lambda row: (
                relative_reduction_for_row(
                    row,
                    alpha_from,
                    alpha_to,
                ),
                row["base_key"],
            ),
            reverse=reduction_sort_descending,
        )

    elif sort_by == "smallest_alpha_cost":

        complete_rows.sort(
            key=lambda row: (
                row["costs"][smallest_alpha],
                row["base_key"],
            )
        )

    elif sort_by == "largest_alpha_cost":

        complete_rows.sort(
            key=lambda row: (
                row["costs"][largest_alpha],
                row["base_key"],
            )
        )

    elif sort_by == "average_cost":

        complete_rows.sort(
            key=lambda row: (
                sum(row["costs"].values()) / len(alpha_values),
                row["base_key"],
            )
        )

    elif sort_by == "instance":

        complete_rows.sort(
            key=lambda row: row["base_key"]
        )

    # ------------------------------------------------------------
    # Statistics: cost reductions for all alpha pairs.
    # ------------------------------------------------------------

    alpha_pairs = []

    for i, alpha_from in enumerate(alpha_values):

        for alpha_to in alpha_values[i + 1:]:

            alpha_pairs.append(
                (
                    alpha_from,
                    alpha_to,
                )
            )

    reductions_by_pair = {}
    relative_reductions_by_pair = {}
    relative_reduction_curves = {}

    for alpha_from, alpha_to in alpha_pairs:

        reductions = []
        relative_reductions = []
        relative_curve = []

        for row in complete_rows:

            c_from = row["costs"][alpha_from]
            c_to = row["costs"][alpha_to]

            reduction = c_from - c_to

            reductions.append(
                reduction
            )

            if abs(c_from) <= 1e-12:

                relative = 0.0

            else:

                relative = reduction / c_from

            relative_reductions.append(
                relative
            )

            relative_curve.append(
                100.0 * relative
            )

        reductions_by_pair[
            (
                alpha_from,
                alpha_to,
            )
        ] = reductions

        relative_reductions_by_pair[
            (
                alpha_from,
                alpha_to,
            )
        ] = relative_reductions

        relative_reduction_curves[
            (
                alpha_from,
                alpha_to,
            )
        ] = relative_curve

    def latex_alpha_reduction_pairs(
        alpha_values,
        alpha_pairs,
    ):
        """
        For alphas [1.2, 1.3, 1.5], return pairs in the order

            1.2 -> 1.3, 1.3 -> 1.5, 1.2 -> 1.5

        More generally: consecutive alpha pairs first, then the
        smallest-to-largest pair.
        """

        available = set(
            alpha_pairs
        )

        pairs = []

        for i in range(
            len(alpha_values) - 1
        ):

            pair = (
                alpha_values[i],
                alpha_values[i + 1],
            )

            if pair in available:

                pairs.append(
                    pair
                )

        if len(alpha_values) >= 2:

            pair = (
                alpha_values[0],
                alpha_values[-1],
            )

            if (
                pair in available
                and pair not in pairs
            ):

                pairs.append(
                    pair
                )

        return pairs


    def latex_taban_summary(
        values,
        scale=1.0,
        decimals=2,
    ):
        """
        Format values as

            avg \\taban{med}{min}{max}

        If scale=100, relative reductions are printed as percentages.
        """

        scaled_values = [
            scale * value
            for value in values
        ]

        avg = (
            sum(scaled_values)
            /
            len(scaled_values)
        )

        med = statistics.median(
            scaled_values
        )

        mn = min(
            scaled_values
        )

        mx = max(
            scaled_values
        )

        return (
            f"${avg:.{decimals}f}$ "
            f"\\taban"
            f"{{{med:.{decimals}f}}}"
            f"{{{mn:.{decimals}f}}}"
            f"{{{mx:.{decimals}f}}}"
        )


    def print_alpha_reduction_latex_table():
        """
        Print LaTeX table for relative cost reductions.

        Entries are percentages:
            avg \\taban{med}{min}{max}
        """

        latex_pairs = latex_alpha_reduction_pairs(
            alpha_values,
            alpha_pairs,
        )

        if not latex_pairs:

            return

        column_spec = (
            "l"
            +
            "r" * len(latex_pairs)
        )

        headers = [
            rf"${alpha_from:g}\to{alpha_to:g}$"
            for alpha_from, alpha_to in latex_pairs
        ]

        values = [
            latex_taban_summary(
                relative_reductions_by_pair[
                    (
                        alpha_from,
                        alpha_to,
                    )
                ],
                scale=100.0,
                decimals=2,
            )
            for alpha_from, alpha_to in latex_pairs
        ]

        print()
        print("% LaTeX table: relative reductions in percent")
        print("% Entries are avg \\taban{med}{min}{max}")
        print(r"\begin{tabular}{" + column_spec + r"}")
        print(r"\toprule")
        print(
            " & "
            +
            " & ".join(headers)
            +
            r" \\"
        )
        print(r"\midrule")
        print(
            r"Cost reduction (in \%)"
            +
            " & "
            +
            " & ".join(values)
            +
            r" \\"
        )
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print()

    print()
    print("Alpha optimal-cost sensitivity")
    print("------------------------------")
    print("method filter:", method)
    print("solver filter:", solver)
    print("preprocessor filter:", preprocessor)
    print("alphas:", alpha_values)
    print("relative reduction bin size:", relative_reduction_bin_size)
    print("complete instance groups:", len(complete_rows))
    print()

    for alpha_from, alpha_to in alpha_pairs:

        reductions = reductions_by_pair[
            (
                alpha_from,
                alpha_to,
            )
        ]

        relative_reductions = relative_reductions_by_pair[
            (
                alpha_from,
                alpha_to,
            )
        ]

        avg_reduction = sum(reductions) / len(reductions)
        med_reduction = statistics.median(reductions)
        min_reduction = min(reductions)
        max_reduction = max(reductions)

        avg_relative_reduction = (
            sum(relative_reductions)
            /
            len(relative_reductions)
        )

        med_relative_reduction = statistics.median(
            relative_reductions
        )

        min_relative_reduction = min(
            relative_reductions
        )

        max_relative_reduction = max(
            relative_reductions
        )

        print(
            f"alpha {alpha_from:g} -> {alpha_to:g}: "
            f"abs avg={avg_reduction:.3f}, "
            f"med={med_reduction:.3f}, "
            f"min={min_reduction:.3f}, "
            f"max={max_reduction:.3f}; "
            f"rel avg={100.0 * avg_relative_reduction:.2f}%, "
            f"med={100.0 * med_relative_reduction:.2f}%, "
            f"min={100.0 * min_relative_reduction:.2f}%, "
            f"max={100.0 * max_relative_reduction:.2f}%"
        )

    print_alpha_reduction_latex_table()

    print()

    def bin_xy_values(
        x_values,
        y_values,
        bin_size,
    ):
        """
        Average x/y values in consecutive bins.

        If bin_size is None or <= 1, return the original values.
        """

        if bin_size is None or bin_size <= 1:

            return x_values, y_values

        bin_size = int(
            bin_size
        )

        binned_x = []
        binned_y = []

        for start in range(
            0,
            len(y_values),
            bin_size,
        ):

            x_chunk = x_values[
                start:start + bin_size
            ]

            y_chunk = y_values[
                start:start + bin_size
            ]

            if not y_chunk:

                continue

            binned_x.append(
                sum(x_chunk) / len(x_chunk)
            )

            binned_y.append(
                sum(y_chunk) / len(y_chunk)
            )

        return binned_x, binned_y

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    x_values = list(
        range(
            1,
            len(complete_rows) + 1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.2,
        )
    )

    cost_lines = []
    relative_lines = []

    reduction_colors = [
        "green",
        "darkorange",
        "purple",
        "brown",
        "gray",
        "olive",
    ]

    # ------------------------------------------------------------
    # Mode 1: current behavior
    # main axis = costs, second axis = binned relative reductions
    # ------------------------------------------------------------

    if plot_mode == "cost_main":

        for alpha in alpha_values:

            y_values = []

            for row in complete_rows:

                value = row["costs"][alpha]

                if normalize:

                    denominator = row["costs"][smallest_alpha]

                    if abs(denominator) <= 1e-12:

                        value = 0.0

                    else:

                        value = value / denominator

                y_values.append(
                    value
                )

            line = ax.plot(
                x_values,
                y_values,
                linewidth=1.1,
                marker=None,
                label=(
                    rf"$\alpha={alpha:g}$"
                ),
            )

            cost_lines.extend(
                line
            )

        if show_relative_reduction:

            ax2 = ax.twinx()

            for pair_index, (alpha_from, alpha_to) in enumerate(alpha_pairs):

                color = reduction_colors[
                    pair_index % len(reduction_colors)
                ]

                relative_x_values, relative_y_values = bin_xy_values(
                    x_values,
                    relative_reduction_curves[
                        (
                            alpha_from,
                            alpha_to,
                        )
                    ],
                    relative_reduction_bin_size,
                )

                line = ax2.plot(
                    relative_x_values,
                    relative_y_values,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    color=color,
                    marker=None,
                    label=(
                        rf"reduction "
                        rf"${alpha_from:g}\to{alpha_to:g}$"
                    ),
                )

                relative_lines.extend(
                    line
                )

            ax2.set_ylabel(
                "Relative cost reduction [\\%]"
            )

            if relative_reduction_ylim is not None:

                ax2.set_ylim(
                    relative_reduction_ylim
                )

            ax2.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
            )

            ax2.axhline(
                0.0,
                linestyle="--",
                linewidth=0.8,
                color="black",
                alpha=0.5,
            )

        if normalize:

            ax.set_ylabel(
                f"Optimal cost / cost at alpha {smallest_alpha:g}"
            )

        else:

            ax.set_ylabel(
                "Optimal cost"
            )

    # ------------------------------------------------------------
    # Mode 2: switched roles
    # main axis = sorted relative reductions
    # second axis = binned absolute costs
    # ------------------------------------------------------------

    elif plot_mode == "reduction_main":

        for pair_index, (alpha_from, alpha_to) in enumerate(alpha_pairs):

            color = reduction_colors[
                pair_index % len(reduction_colors)
            ]

            line = ax.plot(
                x_values,
                relative_reduction_curves[
                    (
                        alpha_from,
                        alpha_to,
                    )
                ],
                linestyle="-",
                linewidth=1.0,
                alpha=0.85,
                color=color,
                marker=None,
                label=(
                    rf"reduction "
                    rf"${alpha_from:g}\to{alpha_to:g}$"
                ),
            )

            relative_lines.extend(
                line
            )

        ax.set_ylabel(
            "Relative cost reduction [\\%]"
        )

        if relative_reduction_ylim is not None:

            ax.set_ylim(
                relative_reduction_ylim
            )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=0.8,
            color="black",
            alpha=0.5,
        )

        ax2 = ax.twinx()

        for alpha in alpha_values:

            y_values = [
                row["costs"][alpha]
                for row in complete_rows
            ]

            cost_x_values, cost_y_values = bin_xy_values(
                x_values,
                y_values,
                absolute_cost_bin_size,
            )

            line = ax2.plot(
                cost_x_values,
                cost_y_values,
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                marker=None,
                label=(
                    rf"$\alpha={alpha:g}$"
                ),
            )

            cost_lines.extend(
                line
            )

        ax2.set_ylabel(
            "Optimal cost"
        )

        if absolute_cost_ylim is not None:

            ax2.set_ylim(
                absolute_cost_ylim
            )

        ax2.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
        )

    if show_title:

        if plot_mode == "cost_main":

            ax.set_title(
                "Optimal cost by alpha"
            )

        else:

            ax.set_title(
                "Relative cost reduction by alpha"
            )

    if plot_mode == "reduction_main":

        ax.set_xlabel(
            (
                "Congruent instance groups "
                rf"sorted by reduction ${reduction_sort_pair[0]:g}\to{reduction_sort_pair[1]:g}$"
            )
        )

    else:

        ax.set_xlabel(
            "Congruent instance groups"
        )

    if show_instance_labels:

        ax.set_xticks(
            x_values
        )

        ax.set_xticklabels(
            [
                row["base_key"]
                for row in complete_rows
            ],
            rotation=90,
            fontsize=5,
        )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=10,
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.25,
        alpha=0.22,
    )

    ax.grid(
        False,
        which="minor",
    )

    if plot_mode == "cost_main" and show_relative_reduction:

        lines = (
            cost_lines
            +
            relative_lines
        )

    elif plot_mode == "reduction_main":

        lines = (
            relative_lines
            +
            cost_lines
        )

    else:

        lines = cost_lines

    labels = [
        line.get_label()
        for line in lines
    ]

    ax.legend(
        lines,
        labels,
        frameon=False,
        fontsize=10,
        ncol=2,
    )

    fig.tight_layout(
        pad=0.05
    )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    if method is not None:

        method_part = sanitize_filename_part(
            method
        )

    elif solver is not None:

        if preprocessor is None:

            method_part = sanitize_filename_part(
                solver
            )

        else:

            method_part = sanitize_filename_part(
                f"{preprocessor}_{solver}"
            )

    else:

        method_part = "all_methods_best"

    mode_part = plot_mode

    alpha_part = "_".join(
        f"a{int(round(10 * alpha))}"
        for alpha in alpha_values
    )

    norm_part = (
        "normalized"
        if normalize
        else
        "absolute"
    )

    relative_part = (
        "with_relative_reduction"
        if show_relative_reduction
        else
        "no_relative_reduction"
    )

    pdf_path = os.path.join(
        output_dir,
        (
            f"alpha_optimal_cost_"
            f"{method_part}_"
            f"{alpha_part}_"
            f"{norm_part}_{relative_part}_{mode_part}.pdf"
        ),
    )


    png_path = os.path.join(
        output_dir,
        (
            f"alpha_optimal_cost_"
            f"{method_part}_"
            f"{alpha_part}_"
            f"{norm_part}_{relative_part}_{mode_part}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )



# ============================================================
# Runtime by target fraction / target terminal-pair count
# ============================================================

def runtime_by_target_fraction_plot(
    output_dir="plots",
    runtime_metric="total_runtime",
    methods=["ilp","ilp_tw_dp_cuts","kernel1_ilp","kernel1_ilp_tw_dp_cuts"],
    method=None,
    solver=None,
    preprocessor=None,
    kind="all",
    x_metric="target_fraction",
    only_optimal=True,
    center_lines=("median", "mean"),
    show_points=True,
    logscale=True,
    show_title=False,
):
    """
    Plot how runtimes correlate with the target terminal-pair quantity.

    This function supports multiple methods in one plot and separates
    villages and regions into two panels.

    Parameters
    ----------
    methods:
        Optional list/set of exact method names, e.g.

            ["ilp", "kernel1_ilp", "ilp_tw_dp_cuts"]

        If methods is given, method/solver/preprocessor filters are ignored.

    method:
        Backward-compatible single-method filter.

    solver, preprocessor:
        Backward-compatible filter if method and methods are None.

    runtime_metric:
        "total_runtime", "solver_runtime", or "build_runtime".

    kind:
        "all", "villages", or "regions".

    x_metric:
        "target_fraction"
            Parse filename token tf45 as 45%.

        "target_num_pairs"
            Parse filename token tn50 as 50 terminal pairs, if present.

    aggregate:
        "median" or "mean".
        Determines the line value shown for each method/fraction group.

    show_points:
        If True, show individual runtimes as faint points.

    only_optimal:
        If True, only include optimal results.

    logscale:
        If True, use logarithmic y-axis.

    show_title:
        If True, add panel titles.
    """

    import os
    import re
    import math
    import statistics
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if runtime_metric not in {
        "total_runtime",
        "solver_runtime",
        "build_runtime",
    }:

        raise ValueError(
            "runtime_metric must be one of: "
            "'total_runtime', 'solver_runtime', 'build_runtime'"
        )

    if kind not in {
        "all",
        "villages",
        "regions",
    }:

        raise ValueError(
            "kind must be one of: 'all', 'villages', 'regions'"
        )

    if x_metric not in {
        "target_fraction",
        "target_num_pairs",
    }:

        raise ValueError(
            "x_metric must be one of: "
            "'target_fraction', 'target_num_pairs'"
        )

    allowed_center_lines = {
        "median",
        "mean",
    }

    if isinstance(
        center_lines,
        str,
    ):

        center_lines = (
            center_lines,
        )

    center_lines = tuple(
        center_lines
    )

    if not set(center_lines).issubset(
        allowed_center_lines
    ):

        raise ValueError(
            "center_lines must contain only: 'median', 'mean'"
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def is_optimal_result(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        status = get_field(
            result,
            "status",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return False

    def get_runtime(
        result,
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            candidates = [
                "total_runtime",
                "runtime",
            ]

        elif runtime_metric == "solver_runtime":

            candidates = [
                "solver_runtime",
                "gurobi_runtime",
                "optimize_walltime",
                "runtime",
            ]

        elif runtime_metric == "build_runtime":

            candidates = [
                "solver_buildtime",
                "build_runtime",
                "buildtime",
            ]

        for field in candidates:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def runtime_label(
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            return "Total runtime [s]"

        if runtime_metric == "solver_runtime":

            return "Solver runtime [s]"

        if runtime_metric == "build_runtime":

            return "Build runtime [s]"

        raise ValueError(
            runtime_metric
        )

    def get_instance_name(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def get_method_name(
        result,
    ):

        result_solver = get_field(
            result,
            "solver",
            None,
        )

        result_preprocessor = get_field(
            result,
            "preprocessor",
            None,
        )

        if result_solver is None:

            result_method = get_field(
                result,
                "method",
                None,
            )

            if result_method is None:

                return None

            return str(
                result_method
            )

        if (
            result_preprocessor is None
            or str(result_preprocessor).strip() == ""
            or str(result_preprocessor).lower() == "none"
        ):

            return str(
                result_solver
            )

        return (
            f"{result_preprocessor}_{result_solver}"
        )

    def display_method(
        method_name,
    ):

        if "display_name" in globals():

            return display_name(
                method_name
            )

        return str(
            method_name
        )

    def result_matches_filter(
        result,
    ):

        result_method = get_method_name(
            result
        )

        if methods is not None:

            return result_method in methods

        if method is not None:

            return (
                result_method
                ==
                method
            )

        if solver is not None:

            result_solver = get_field(
                result,
                "solver",
                None,
            )

            if str(result_solver) != str(solver):

                return False

        if preprocessor is not None:

            result_preprocessor = get_field(
                result,
                "preprocessor",
                None,
            )

            expected = (
                ""
                if (
                    preprocessor is None
                    or str(preprocessor).strip() == ""
                    or str(preprocessor).lower() == "none"
                )
                else str(preprocessor)
            )

            actual = (
                ""
                if result_preprocessor is None
                else str(result_preprocessor)
            )

            if actual.lower() == "none":

                actual = ""

            if actual != expected:

                return False

        return True

    def parse_target_value(
        instance_name,
        x_metric,
    ):

        stem = os.path.basename(
            str(instance_name)
        )

        if stem.endswith(
            ".pkl"
        ):

            stem = stem[:-4]

        if stem.endswith(
            ".json"
        ):

            stem = stem[:-5]

        tokens = stem.split(
            "_"
        )

        if x_metric == "target_fraction":

            for token in tokens:

                match = re.fullmatch(
                    r"tf(\d+)",
                    token,
                )

                if match is not None:

                    return (
                        int(match.group(1))
                        /
                        100.0
                    )

        elif x_metric == "target_num_pairs":

            for token in tokens:

                match = re.fullmatch(
                    r"tn(\d+)",
                    token,
                )

                if match is not None:

                    return int(
                        match.group(1)
                    )

        return None

    def parse_kind_from_instance(
        instance_name,
    ):

        stem = os.path.basename(
            str(instance_name)
        )

        tokens = stem.split(
            "_"
        )

        for token in tokens:

            if re.fullmatch(
                r"r\d+",
                token,
            ):

                return "regions"

        return "villages"

    def pearson_correlation(
        xs,
        ys,
    ):

        n = len(
            xs
        )

        if n < 2:

            return None

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum(
            (
                x - mean_x
            )
            *
            (
                y - mean_y
            )
            for x, y in zip(
                xs,
                ys,
            )
        )

        var_x = sum(
            (
                x - mean_x
            )
            **
            2
            for x in xs
        )

        var_y = sum(
            (
                y - mean_y
            )
            **
            2
            for y in ys
        )

        if var_x <= 0 or var_y <= 0:

            return None

        return cov / math.sqrt(
            var_x * var_y
        )

    def rankdata_average_ties(
        values,
    ):

        indexed = sorted(
            enumerate(values),
            key=lambda item: item[1],
        )

        ranks = [
            0.0
            for _ in values
        ]

        i = 0

        while i < len(indexed):

            j = i

            while (
                j + 1 < len(indexed)
                and indexed[j + 1][1] == indexed[i][1]
            ):

                j += 1

            rank = (
                i + j + 2
            ) / 2.0

            for k in range(
                i,
                j + 1,
            ):

                ranks[indexed[k][0]] = rank

            i = j + 1

        return ranks

    def spearman_correlation(
        xs,
        ys,
    ):

        if len(xs) < 2:

            return None

        rx = rankdata_average_ties(
            xs
        )

        ry = rankdata_average_ties(
            ys
        )

        return pearson_correlation(
            rx,
            ry,
        )

    def correlation_string(
        value,
    ):

        if value is None:

            return "NA"

        return f"{value:.4f}"

    def sanitize_filename_part(
        text,
    ):

        text = str(
            text
        )

        for bad in [
            "/",
            "\\",
            " ",
            ":",
            ";",
            ",",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]:

            text = text.replace(
                bad,
                "_",
            )

        return text

    # ------------------------------------------------------------
    # Normalize method list.
    # ------------------------------------------------------------

    if methods is not None:

        methods = [
            str(m)
            for m in methods
        ]

    # ------------------------------------------------------------
    # Collect rows.
    # ------------------------------------------------------------

    results = load_all_results()

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    rows = []

    for result in results:

        if only_optimal and not is_optimal_result(
            result
        ):

            continue

        if not result_matches_filter(
            result
        ):

            continue

        instance = get_instance_name(
            result
        )

        if instance is None:

            continue

        x_value = parse_target_value(
            instance,
            x_metric,
        )

        if x_value is None:

            continue

        instance_kind = parse_kind_from_instance(
            instance
        )

        if (
            kind != "all"
            and instance_kind != kind
        ):

            continue

        result_method = get_method_name(
            result
        )

        if result_method is None:

            continue

        runtime = get_runtime(
            result,
            runtime_metric,
        )

        if runtime is None:

            continue

        runtime = float(
            runtime
        )

        if not math.isfinite(
            runtime
        ):

            continue

        if logscale and runtime <= 0.0:

            continue

        rows.append(
            {
                "instance": str(instance),
                "kind": instance_kind,
                "x_value": x_value,
                "runtime": runtime,
                "method": result_method,
            }
        )

    if not rows:

        print(
            "[WARN] No matching results with target values found."
        )

        return

    if methods is None:

        methods = sorted(
            {
                row["method"]
                for row in rows
            }
        )

    else:

        # Keep user order but drop methods without data.
        available_methods = {
            row["method"]
            for row in rows
        }

        methods = [
            m
            for m in methods
            if m in available_methods
        ]

    if not methods:

        print(
            "[WARN] None of the requested methods has matching data."
        )

        return

    # ------------------------------------------------------------
    # Group data.
    # ------------------------------------------------------------

    groups = {}

    for row in rows:

        key = (
            row["kind"],
            row["method"],
            row["x_value"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row["runtime"]
        )

    kinds_to_plot = [
        "villages",
        "regions",
    ]

    if kind != "all":

        kinds_to_plot = [
            kind,
        ]

    # ------------------------------------------------------------
    # Print summary statistics.
    # ------------------------------------------------------------

    print()
    print("Runtime by target terminal-pair quantity")
    print("----------------------------------------")
    print("runtime metric:", runtime_metric)
    print("x metric:", x_metric)
    print("methods:", methods)
    print("kind:", kind)
    print("only optimal:", only_optimal)
    print("center lines:", center_lines)
    print("n results:", len(rows))
    print()

    print("Grouped runtime statistics:")
    print()

    for k in kinds_to_plot:

        print(k)
        print("-" * len(k))

        for m in methods:

            x_values_for_method = sorted(
                {
                    x_value
                    for group_kind, group_method, x_value in groups.keys()
                    if group_kind == k and group_method == m
                }
            )

            if not x_values_for_method:

                continue

            print(
                f"  {m}"
            )

            for x_value in x_values_for_method:

                values = sorted(
                    groups[
                        (
                            k,
                            m,
                            x_value,
                        )
                    ]
                )

                avg = sum(values) / len(values)
                med = statistics.median(values)
                mn = min(values)
                mx = max(values)

                if x_metric == "target_fraction":

                    x_text = (
                        f"tf={int(round(100 * x_value)):3d}"
                    )

                else:

                    x_text = (
                        f"tn={int(x_value):4d}"
                    )

                print(
                    f"    {x_text} "
                    f"n={len(values):5d} "
                    f"avg={avg:10.3f} "
                    f"med={med:10.3f} "
                    f"min={mn:10.3f} "
                    f"max={mx:10.3f}"
                )

            print()

    print("Correlation statistics:")
    print()

    for k in kinds_to_plot:

        print(k)
        print("-" * len(k))

        for m in methods:

            corr_rows = [
                row
                for row in rows
                if (
                    row["kind"] == k
                    and row["method"] == m
                )
            ]

            if len(corr_rows) < 2:

                continue

            xs = [
                row["x_value"]
                for row in corr_rows
            ]

            ys = [
                row["runtime"]
                for row in corr_rows
            ]

            pearson_raw = pearson_correlation(
                xs,
                ys,
            )

            spearman_raw = spearman_correlation(
                xs,
                ys,
            )

            positive_rows = [
                row
                for row in corr_rows
                if row["runtime"] > 0.0
            ]

            pearson_log = None
            spearman_log = None

            if len(positive_rows) >= 2:

                xs_log = [
                    row["x_value"]
                    for row in positive_rows
                ]

                ys_log = [
                    math.log10(
                        row["runtime"]
                    )
                    for row in positive_rows
                ]

                pearson_log = pearson_correlation(
                    xs_log,
                    ys_log,
                )

                spearman_log = spearman_correlation(
                    xs_log,
                    ys_log,
                )

            aggregated_correlation_parts = []

            for center_line in center_lines:

                aggregate_xs = []
                aggregate_ys = []

                x_values_for_method = sorted(
                    {
                        row["x_value"]
                        for row in corr_rows
                    }
                )

                for x_value in x_values_for_method:

                    values = [
                        row["runtime"]
                        for row in corr_rows
                        if row["x_value"] == x_value
                    ]

                    if not values:

                        continue

                    if center_line == "median":

                        aggregate_value = statistics.median(
                            values
                        )

                    elif center_line == "mean":

                        aggregate_value = (
                            sum(values)
                            /
                            len(values)
                        )

                    aggregate_xs.append(
                        x_value
                    )

                    aggregate_ys.append(
                        aggregate_value
                    )

                pearson_agg = pearson_correlation(
                    aggregate_xs,
                    aggregate_ys,
                )

                spearman_agg = spearman_correlation(
                    aggregate_xs,
                    aggregate_ys,
                )

                aggregated_correlation_parts.append(
                    f"Pearson({center_line})={correlation_string(pearson_agg)} "
                    f"Spearman({center_line})={correlation_string(spearman_agg)}"
                )

            print(
                f"  {m:35s} "
                f"n={len(corr_rows):5d} "
                f"Pearson={correlation_string(pearson_raw)} "
                f"Spearman={correlation_string(spearman_raw)} "
                f"Pearson(log10)={correlation_string(pearson_log)} "
                f"Spearman(log10)={correlation_string(spearman_log)} "
                +
                " ".join(
                    aggregated_correlation_parts
                )
            )

        print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    n_panels = len(
        kinds_to_plot
    )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            7.2 if n_panels == 1 else 10.8,
            4.8,
        ),
        sharey=True,
    )

    if n_panels == 1:

        axes = [
            axes
        ]

    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get(
        "color",
        [
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
        ],
    )

    method_colors = {
        m: color_cycle[
            index % len(color_cycle)
        ]
        for index, m in enumerate(methods)
    }

    for ax, k in zip(
        axes,
        kinds_to_plot,
    ):

        for m in methods:

            method_rows = [
                row
                for row in rows
                if (
                    row["kind"] == k
                    and row["method"] == m
                )
            ]

            if not method_rows:

                continue

            x_values_for_method = sorted(
                {
                    row["x_value"]
                    for row in method_rows
                }
            )

            line_xs = []
            center_values_by_type = {
                center_line: []
                for center_line in center_lines
            }
            lower_ys = []
            upper_ys = []

            for x_value in x_values_for_method:

                values = sorted(
                    row["runtime"]
                    for row in method_rows
                    if row["x_value"] == x_value
                )

                if not values:

                    continue

                line_xs.append(
                    x_value
                )

                for center_line in center_lines:

                    if center_line == "median":

                        center = statistics.median(
                            values
                        )

                    elif center_line == "mean":

                        center = (
                            sum(values)
                            /
                            len(values)
                        )

                    center_values_by_type[
                        center_line
                    ].append(
                        center
                    )

                lower_ys.append(
                    min(values)
                )

                upper_ys.append(
                    max(values)
                )

            if show_points:

                raw_xs = [
                    row["x_value"]
                    for row in method_rows
                ]

                raw_ys = [
                    row["runtime"]
                    for row in method_rows
                ]

                ax.scatter(
                    raw_xs,
                    raw_ys,
                    s=9,
                    alpha=0.18,
                    color=method_colors[m],
                    linewidths=0.0,
                )

            for center_line in center_lines:

                linestyle = (
                    "-"
                    if center_line == "median"
                    else "--"
                )

                label = (
                    display_method(m)
                    if center_line == center_lines[0]
                    else "_nolegend_"
                )

                ax.plot(
                    line_xs,
                    center_values_by_type[center_line],
                    marker="o",
                    markersize=3.0,
                    linewidth=1.25,
                    linestyle=linestyle,
                    color=method_colors[m],
                    label=label,
                )

            # Light min--max band.
            if len(line_xs) >= 2:

                ax.fill_between(
                    line_xs,
                    lower_ys,
                    upper_ys,
                    color=method_colors[m],
                    alpha=0.08,
                    linewidth=0.0,
                )

        if x_metric == "target_fraction":

            ax.set_xlabel(
                "Target terminal-pair fraction [%]"
            )

            all_x_values = sorted(
                {
                    row["x_value"]
                    for row in rows
                    if row["kind"] == k
                }
            )

            ax.set_xticks(
                all_x_values
            )

            ax.set_xticklabels(
                [
                    f"{int(round(100 * value))}"
                    for value in all_x_values
                ]
            )

        else:

            ax.set_xlabel(
                "Target number of terminal pairs"
            )

        if show_title:

            ax.set_title(
                "Villages" if k == "villages" else "Regions"
            )

        if logscale:

            ax.set_yscale(
                "log"
            )

        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.25,
            alpha=0.22,
        )

        ax.grid(
            False,
            which="minor",
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
        )

    axes[0].set_ylabel(
        runtime_label(
            runtime_metric
        )
    )

    # One shared method-color legend.
    handles = []
    labels = []

    for ax in axes:

        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(
            h,
            l,
        ):

            if (
                label not in labels
                and label != "_nolegend_"
            ):

                handles.append(
                    handle
                )

                labels.append(
                    label
                )

    if handles:

        method_legend = axes[-1].legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            loc="lower center",
            ncol=2,
            #title="Solver",
            #title_fontsize=8,
        )

        axes[-1].add_artist(
            method_legend
        )

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.25,
            label="median",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="mean",
        ),
    ]

    axes[-1].legend(
        handles=style_handles,
        frameon=False,
        fontsize=8,
        loc="lower right",
        #title="Line",
        #title_fontsize=8,
    )

    fig.tight_layout(
        pad=0.05
    )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    if methods is not None and len(methods) <= 4:

        method_part = "_".join(
            sanitize_filename_part(m)
            for m in methods
        )

    elif methods is not None:

        method_part = (
            f"{len(methods)}_methods"
        )

    elif method is not None:

        method_part = sanitize_filename_part(
            method
        )

    elif solver is not None:

        if preprocessor is None:

            method_part = sanitize_filename_part(
                solver
            )

        else:

            method_part = sanitize_filename_part(
                f"{preprocessor}_{solver}"
            )

    else:

        method_part = "all_methods"

    pdf_path = os.path.join(
        output_dir,
        (
            f"runtime_by_target_"
            f"{x_metric}_"
            f"{method_part}_"
            f"{runtime_metric}_"
            f"{kind}_"
            f"{'_'.join(center_lines)}.pdf"
        ),
    )

    png_path = os.path.join(
        output_dir,
        (
            f"runtime_by_target_"
            f"{x_metric}_"
            f"{method_part}_"
            f"{runtime_metric}_"
            f"{kind}_"
            f"{'_'.join(center_lines)}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )



# ============================================================
# Runtime by safety model
# ============================================================

def runtime_by_safety_model_plot(
    output_dir="plots",
    runtime_metric="total_runtime",
    methods=[
                "ilp",
                "ilp_tw_dp_cuts",
                "kernel1_ilp",
                "kernel1_ilp_tw_dp_cuts",
            ],
    method=None,
    solver=None,
    preprocessor=None,
    kind="all",
    only_optimal=True,
    logscale=True,
    show_points=False,
    show_fliers=False,
    show_title=False,
):
    """
    Plot runtime distributions by safety model.

    Safety models are parsed from instance filenames via tokens

        A, B, C

    Examples
    --------
    Village instance:

        altenholz_A_random-pairs_tf25_a12_i0.json

    Region instance:

        altenholz_r3_A_random-pairs_tf15_a12_i0.json

    Parameters
    ----------
    methods:
        Optional list/set of exact method names, e.g.

            ["ilp", "kernel1_ilp", "ilp_tw_dp_cuts"]

        If methods is given, method/solver/preprocessor filters are ignored.

    method:
        Backward-compatible single-method filter.

    solver, preprocessor:
        Backward-compatible filter if method and methods are None.

    kind:
        "all", "villages", or "regions".

    runtime_metric:
        "total_runtime", "solver_runtime", or "build_runtime".

    only_optimal:
        If True, only include optimal results.

    logscale:
        If True, use logarithmic y-axis.

    show_points:
        If True, overlay individual runtimes as faint jittered points.

    show_fliers:
        If True, show Matplotlib boxplot fliers.

    show_title:
        If True, add panel titles.
    """

    import os
    import re
    import math
    import random
    import statistics
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if runtime_metric not in {
        "total_runtime",
        "solver_runtime",
        "build_runtime",
    }:

        raise ValueError(
            "runtime_metric must be one of: "
            "'total_runtime', 'solver_runtime', 'build_runtime'"
        )

    if kind not in {
        "all",
        "villages",
        "regions",
    }:

        raise ValueError(
            "kind must be one of: 'all', 'villages', 'regions'"
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------

    def get_field(
        result,
        field,
        default=None,
    ):

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                field,
                default,
            )

        return getattr(
            result,
            field,
            default,
        )

    def is_optimal_result(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            None,
        )

        status = get_field(
            result,
            "status",
            None,
        )

        if optimal is not None:

            if isinstance(
                optimal,
                bool,
            ):

                return optimal

            optimal_string = str(
                optimal
            ).strip().lower()

            if optimal_string == "true":

                return True

            if optimal_string == "false":

                return False

        if status is not None:

            return (
                str(status).strip().upper()
                ==
                "OPTIMAL"
            )

        return False

    def get_runtime(
        result,
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            candidates = [
                "total_runtime",
                "runtime",
            ]

        elif runtime_metric == "solver_runtime":

            candidates = [
                "solver_runtime",
                "gurobi_runtime",
                "optimize_walltime",
                "runtime",
            ]

        elif runtime_metric == "build_runtime":

            candidates = [
                "solver_buildtime",
                "build_runtime",
                "buildtime",
            ]

        for field in candidates:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def runtime_label(
        runtime_metric,
    ):

        if runtime_metric == "total_runtime":

            return "Total runtime [s]"

        if runtime_metric == "solver_runtime":

            return "Solver runtime [s]"

        if runtime_metric == "build_runtime":

            return "Build runtime [s]"

        raise ValueError(
            runtime_metric
        )

    def get_instance_name(
        result,
    ):

        instance = get_field(
            result,
            "instance",
            None,
        )

        if instance is None:

            instance = get_field(
                result,
                "instance_name",
                None,
            )

        if instance is None:

            instance = get_field(
                result,
                "instance_filename",
                None,
            )

        return instance

    def get_method_name(
        result,
    ):

        result_solver = get_field(
            result,
            "solver",
            None,
        )

        result_preprocessor = get_field(
            result,
            "preprocessor",
            None,
        )

        if result_solver is None:

            result_method = get_field(
                result,
                "method",
                None,
            )

            if result_method is None:

                return None

            return str(
                result_method
            )

        if (
            result_preprocessor is None
            or str(result_preprocessor).strip() == ""
            or str(result_preprocessor).lower() == "none"
        ):

            return str(
                result_solver
            )

        return (
            f"{result_preprocessor}_{result_solver}"
        )

    def display_method(
        method_name,
    ):

        if "display_name" in globals():

            return display_name(
                method_name
            )

        return str(
            method_name
        )

    def result_matches_filter(
        result,
    ):

        result_method = get_method_name(
            result
        )

        if methods is not None:

            return result_method in methods

        if method is not None:

            return (
                result_method
                ==
                method
            )

        if solver is not None:

            result_solver = get_field(
                result,
                "solver",
                None,
            )

            if str(result_solver) != str(solver):

                return False

        if preprocessor is not None:

            result_preprocessor = get_field(
                result,
                "preprocessor",
                None,
            )

            expected = (
                ""
                if (
                    preprocessor is None
                    or str(preprocessor).strip() == ""
                    or str(preprocessor).lower() == "none"
                )
                else str(preprocessor)
            )

            actual = (
                ""
                if result_preprocessor is None
                else str(result_preprocessor)
            )

            if actual.lower() == "none":

                actual = ""

            if actual != expected:

                return False

        return True

    def parse_safety_model(
        instance_name,
    ):

        stem = os.path.basename(
            str(instance_name)
        )

        if stem.endswith(
            ".pkl"
        ):

            stem = stem[:-4]

        if stem.endswith(
            ".json"
        ):

            stem = stem[:-5]

        tokens = stem.split(
            "_"
        )

        for token in tokens:

            if token in {
                "A",
                "B",
                "C",
            }:

                return token

        return None

    def parse_kind_from_instance(
        instance_name,
    ):

        stem = os.path.basename(
            str(instance_name)
        )

        tokens = stem.split(
            "_"
        )

        for token in tokens:

            if re.fullmatch(
                r"r\d+",
                token,
            ):

                return "regions"

        return "villages"

    def sanitize_filename_part(
        text,
    ):

        text = str(
            text
        )

        for bad in [
            "/",
            "\\",
            " ",
            ":",
            ";",
            ",",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]:

            text = text.replace(
                bad,
                "_",
            )

        return text

    def print_summary_line(
        values,
        prefix,
    ):

        values = sorted(
            values
        )

        avg = sum(values) / len(values)
        med = statistics.median(
            values
        )

        q1 = statistics.quantiles(
            values,
            n=4,
            method="inclusive",
        )[0]

        q3 = statistics.quantiles(
            values,
            n=4,
            method="inclusive",
        )[2]

        mn = min(
            values
        )

        mx = max(
            values
        )

        print(
            f"{prefix} "
            f"n={len(values):5d} "
            f"avg={avg:10.3f} "
            f"med={med:10.3f} "
            f"q1={q1:10.3f} "
            f"q3={q3:10.3f} "
            f"min={mn:10.3f} "
            f"max={mx:10.3f}"
        )

    # ------------------------------------------------------------
    # Normalize method list.
    # ------------------------------------------------------------

    if methods is not None:

        methods = [
            str(m)
            for m in methods
        ]

    # ------------------------------------------------------------
    # Collect rows.
    # ------------------------------------------------------------

    results = load_all_results()

    if not results:

        print(
            "[WARN] No results found."
        )

        return

    rows = []

    skipped_missing_safety = 0

    for result in results:

        if only_optimal and not is_optimal_result(
            result
        ):

            continue

        if not result_matches_filter(
            result
        ):

            continue

        instance = get_instance_name(
            result
        )

        if instance is None:

            continue

        safety_model = parse_safety_model(
            instance
        )

        if safety_model is None:

            skipped_missing_safety += 1

            continue

        instance_kind = parse_kind_from_instance(
            instance
        )

        if (
            kind != "all"
            and instance_kind != kind
        ):

            continue

        result_method = get_method_name(
            result
        )

        if result_method is None:

            continue

        runtime = get_runtime(
            result,
            runtime_metric,
        )

        if runtime is None:

            continue

        runtime = float(
            runtime
        )

        if not math.isfinite(
            runtime
        ):

            continue

        if logscale and runtime <= 0.0:

            continue

        rows.append(
            {
                "instance": str(instance),
                "kind": instance_kind,
                "safety_model": safety_model,
                "runtime": runtime,
                "method": result_method,
            }
        )

    if not rows:

        print(
            "[WARN] No matching results with safety models found."
        )

        if skipped_missing_safety:

            print(
                "skipped missing safety model:",
                skipped_missing_safety,
            )

        return

    if methods is None:

        methods = sorted(
            {
                row["method"]
                for row in rows
            }
        )

    else:

        available_methods = {
            row["method"]
            for row in rows
        }

        methods = [
            m
            for m in methods
            if m in available_methods
        ]

    if not methods:

        print(
            "[WARN] None of the requested methods has matching data."
        )

        return

    safety_models = [
        "A",
        "B",
        "C",
    ]

    if kind == "all":

        kinds_to_plot = [
            "villages",
            "regions",
        ]

    else:

        kinds_to_plot = [
            kind,
        ]

    # ------------------------------------------------------------
    # Group data.
    # ------------------------------------------------------------

    groups = {}

    for row in rows:

        key = (
            row["kind"],
            row["method"],
            row["safety_model"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row["runtime"]
        )

    # ------------------------------------------------------------
    # Print statistics.
    # ------------------------------------------------------------

    print()
    print("Runtime by safety model")
    print("-----------------------")
    print("runtime metric:", runtime_metric)
    print("methods:", methods)
    print("kind:", kind)
    print("only optimal:", only_optimal)
    print("n results:", len(rows))
    print("skipped missing safety model:", skipped_missing_safety)
    print()

    for k in kinds_to_plot:

        print(k)
        print("-" * len(k))

        for m in methods:

            has_any = any(
                (
                    k,
                    m,
                    safety_model,
                )
                in groups
                for safety_model in safety_models
            )

            if not has_any:

                continue

            print(
                f"  {m}"
            )

            medians = {}
            means = {}

            for safety_model in safety_models:

                key = (
                    k,
                    m,
                    safety_model,
                )

                if key not in groups:

                    print(
                        f"    {safety_model} "
                        f"n={0:5d}"
                    )

                    continue

                values = groups[
                    key
                ]

                print_summary_line(
                    values,
                    prefix=f"    {safety_model}",
                )

                medians[safety_model] = statistics.median(
                    values
                )

                means[safety_model] = (
                    sum(values)
                    /
                    len(values)
                )

            if "A" in medians:

                ratio_parts = []

                for safety_model in [
                    "B",
                    "C",
                ]:

                    if safety_model in medians:

                        ratio_parts.append(
                            f"med({safety_model})/med(A)="
                            f"{medians[safety_model] / medians['A']:.3f}"
                        )

                for safety_model in [
                    "B",
                    "C",
                ]:

                    if safety_model in means:

                        ratio_parts.append(
                            f"mean({safety_model})/mean(A)="
                            f"{means[safety_model] / means['A']:.3f}"
                        )

                if ratio_parts:

                    print(
                        "    "
                        +
                        " ".join(
                            ratio_parts
                        )
                    )

            print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    n_panels = len(
        kinds_to_plot
    )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            6.4 if n_panels == 1 else 10.4,
            4.8,
        ),
        sharey=True,
    )

    if n_panels == 1:

        axes = [
            axes
        ]

    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get(
        "color",
        [
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
        ],
    )

    method_colors = {
        m: color_cycle[
            index % len(color_cycle)
        ]
        for index, m in enumerate(methods)
    }

    base_positions = {
        "A": 0.0,
        "B": 1.0,
        "C": 2.0,
    }

    if len(methods) == 1:

        box_width = 0.30

        method_offsets = {
            methods[0]: 0.0
        }

    else:

        group_width = 0.78
        box_width = min(
            0.16,
            group_width / len(methods),
        )

        method_offsets = {}

        for index, m in enumerate(methods):

            method_offsets[m] = (
                (
                    index
                    -
                    (len(methods) - 1) / 2.0
                )
                *
                box_width
                *
                1.15
            )

    rng = random.Random(
        1
    )

    for ax, k in zip(
        axes,
        kinds_to_plot,
    ):

        data = []
        positions = []
        box_methods = []

        for safety_model in safety_models:

            for m in methods:

                key = (
                    k,
                    m,
                    safety_model,
                )

                if key not in groups:

                    continue

                data.append(
                    groups[
                        key
                    ]
                )

                positions.append(
                    base_positions[safety_model]
                    +
                    method_offsets[m]
                )

                box_methods.append(
                    m
                )

        if data:

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=show_fliers,
                showmeans=True,
                meanprops={
                    "marker": "^",
                    "markersize": 3.5,
                    "markeredgecolor": "black",
                    "markerfacecolor": "black",
                },
            )

            for patch, m in zip(
                bp["boxes"],
                box_methods,
            ):

                patch.set_facecolor(
                    method_colors[m]
                )

                patch.set_alpha(
                    0.45
                )

                patch.set_edgecolor(
                    "black"
                )

                patch.set_linewidth(
                    0.8
                )

            for element in [
                "whiskers",
                "caps",
                "medians",
            ]:

                for item in bp[element]:

                    item.set_color(
                        "black"
                    )

                    item.set_linewidth(
                        0.8
                    )

            if show_points:

                for values, position, m in zip(
                    data,
                    positions,
                    box_methods,
                ):

                    jittered_xs = [
                        position
                        +
                        rng.uniform(
                            -box_width * 0.22,
                            box_width * 0.22,
                        )
                        for _ in values
                    ]

                    ax.scatter(
                        jittered_xs,
                        values,
                        s=5,
                        alpha=0.12,
                        color=method_colors[m],
                        linewidths=0.0,
                        zorder=1,
                    )

        ax.set_xticks(
            [
                base_positions[safety_model]
                for safety_model in safety_models
            ]
        )

        ax.set_xticklabels(
            safety_models
        )

        ax.set_xlabel(
            "Safety model"
        )

        if show_title:

            ax.set_title(
                "Villages" if k == "villages" else "Regions"
            )

        if logscale:

            ax.set_yscale(
                "log"
            )

        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.25,
            alpha=0.22,
        )

        ax.grid(
            False,
            which="minor",
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
        )

    axes[0].set_ylabel(
        runtime_label(
            runtime_metric
        )
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=method_colors[m],
            linewidth=5.0,
            label=display_method(m),
        )
        for m in methods
    ]

    axes[-1].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="best",
        title="Solver",
        title_fontsize=8,
    )

    fig.tight_layout(
        pad=0.05
    )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    if methods is not None and len(methods) <= 4:

        method_part = "_".join(
            sanitize_filename_part(m)
            for m in methods
        )

    elif methods is not None:

        method_part = (
            f"{len(methods)}_methods"
        )

    elif method is not None:

        method_part = sanitize_filename_part(
            method
        )

    elif solver is not None:

        if preprocessor is None:

            method_part = sanitize_filename_part(
                solver
            )

        else:

            method_part = sanitize_filename_part(
                f"{preprocessor}_{solver}"
            )

    else:

        method_part = "all_methods"

    pdf_path = os.path.join(
        output_dir,
        (
            f"runtime_by_safety_model_"
            f"{method_part}_"
            f"{runtime_metric}_"
            f"{kind}.pdf"
        ),
    )

    png_path = os.path.join(
        output_dir,
        (
            f"runtime_by_safety_model_"
            f"{method_part}_"
            f"{runtime_metric}_"
            f"{kind}.png"
        ),
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close()

    print(
        "saved",
        pdf_path,
    )

    print(
        "saved",
        png_path,
    )


# ============================================================
# Plot one solved instance geographically
# ============================================================

def plot_instance(
    instance_name,
    solver,
    output_dir="plots/instances",
    show_title=False,
    show_terminals=False,
    terminal_colormap="turbo",
    terminal_marker_diameter=8,
    show_terminal_colorbar=False,
    show_nodes=False,
    save=True,
    show=False,
):
    """
    Plot one instance together with the upgraded edges selected by a solver.

    Only results without preprocessing are considered.

    Parameters
    ----------
    instance_name:
        Instance filename, stem, or path, for example

            "altenholz_A_random-pairs_tf25_a12_i0.json"

        The extension may be omitted. The stored instance may itself use
        another extension, such as .pkl.

    solver:
        Exact solver name, for example

            "ilp"
            "ilp_tw_dp_cuts"

        Preprocessed methods are explicitly excluded.

    output_dir:
        Directory for the generated PDF and PNG.

    show_title:
        Add the instance and solver name above the plot.

    show_terminals:
        Mark all vertices occurring in terminal pairs.

    show_nodes:
        Draw all network vertices as small points.

    save:
        Save PDF and PNG files.

    show:
        Display the Matplotlib window interactively.

    Returns
    -------
    dict
        Contains the instance, selected result, figure, axes, and paths.
    """

    import os
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from bicycleinstance import BicycleInstance
    from bicyclenetwork import BicycleNetwork
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Wedge, Circle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, BoundaryNorm


    # ------------------------------------------------------------
    # Generic access helpers.
    # ------------------------------------------------------------

    def get_field(
        obj,
        field,
        default=None,
    ):

        if isinstance(
            obj,
            dict,
        ):

            return obj.get(
                field,
                default,
            )

        return getattr(
            obj,
            field,
            default,
        )

    def normalized_instance_key(
        value,
    ):

        name = os.path.basename(
            str(value)
        )

        for extension in [
            ".json",
            ".pkl",
            ".pickle",
        ]:

            if name.endswith(
                extension
            ):

                name = name[
                    :-len(extension)
                ]

                break

        return name

    def get_result_instance_name(
        result,
    ):

        for field in [
            "instance_filename",
            "instance_name",
            "instance",
        ]:

            value = get_field(
                result,
                field,
                None,
            )

            if value is not None:

                return value

        return None

    def has_no_preprocessing(
        result,
    ):

        value = get_field(
            result,
            "preprocessor",
            None,
        )

        if value is None:

            return True

        return (
            str(value).strip().lower()
            in {
                "",
                "none",
                "null",
            }
        )

    def is_optimal_result(
        result,
    ):

        optimal = get_field(
            result,
            "optimal",
            False,
        )

        if isinstance(
            optimal,
            bool,
        ):

            return optimal

        if str(optimal).strip().lower() == "true":

            return True

        status = get_field(
            result,
            "status",
            "",
        )

        return (
            str(status).strip().upper()
            ==
            "OPTIMAL"
        )

    def is_feasible_result(
        result,
    ):

        feasible = get_field(
            result,
            "feasible",
            False,
        )

        if isinstance(
            feasible,
            bool,
        ):

            return feasible

        return (
            str(feasible).strip().lower()
            ==
            "true"
        )

    def result_runtime(
        result,
    ):

        for field in [
            "total_runtime",
            "solver_runtime",
        ]:

            value = get_field(
                result,
                field,
                None,
            )

            if value is None:

                continue

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if math.isfinite(
                value
            ):

                return value

        return math.inf

    def canonical_edge(
        network,
        edge,
    ):

        if not isinstance(
            edge,
            (
                list,
                tuple,
            ),
        ):

            return None

        if len(edge) < 2:

            return None

        u = edge[0]
        v = edge[1]

        if hasattr(
            network,
            "canonical_edge",
        ):

            return network.canonical_edge(
                u,
                v,
            )

        try:

            return (
                (u, v)
                if u < v
                else (v, u)
            )

        except TypeError:

            # Fallback if node types cannot be ordered.
            return (
                u,
                v,
            )

    def get_edge_geometry(
        network,
        edge,
    ):

        geometry = network.edge_geometry.get(
            edge,
            None,
        )

        if geometry is not None:

            return geometry

        reverse_edge = (
            edge[1],
            edge[0],
        )

        return network.edge_geometry.get(
            reverse_edge,
            None,
        )

    def geometry_coordinate_sequences(
        geometry,
    ):
        """
        Extract line-coordinate sequences from common Shapely geometry
        objects and plain coordinate lists.
        """

        if geometry is None:

            return []

        geometry_type = getattr(
            geometry,
            "geom_type",
            None,
        )

        if geometry_type == "LineString":

            try:

                return [
                    [
                        (
                            float(coordinate[0]),
                            float(coordinate[1]),
                        )
                        for coordinate in geometry.coords
                    ]
                ]

            except Exception:

                return []

        if geometry_type == "MultiLineString":

            # First try to merge connected pieces into one complete line.
            try:

                from shapely.ops import linemerge

                merged = linemerge(
                    geometry
                )

                if getattr(
                    merged,
                    "geom_type",
                    None,
                ) == "LineString":

                    return geometry_coordinate_sequences(
                        merged
                    )

            except Exception:

                pass

            sequences = []

            for part in geometry.geoms:

                sequences.extend(
                    geometry_coordinate_sequences(
                        part
                    )
                )

            return sequences

        if geometry_type == "GeometryCollection":

            sequences = []

            for part in geometry.geoms:

                sequences.extend(
                    geometry_coordinate_sequences(
                        part
                    )
                )

            return sequences

        if hasattr(
            geometry,
            "coords",
        ):

            try:

                return [
                    [
                        (
                            float(coordinate[0]),
                            float(coordinate[1]),
                        )
                        for coordinate in geometry.coords
                    ]
                ]

            except Exception:

                return []

        if isinstance(
            geometry,
            (
                list,
                tuple,
            ),
        ):

            if len(geometry) < 2:

                return []

            first = geometry[0]

            if (
                isinstance(
                    first,
                    (
                        list,
                        tuple,
                    ),
                )
                and len(first) >= 2
            ):

                try:

                    return [
                        [
                            (
                                float(coordinate[0]),
                                float(coordinate[1]),
                            )
                            for coordinate in geometry
                        ]
                    ]

                except Exception:

                    return []

        return []

    def edge_coordinate_sequences(
        network,
        edge,
        diagnostics=None,
    ):
        """
        Return complete drawable coordinate sequences for an edge.

        Stored geometry is oriented and, when necessary, extended to
        the geographic coordinates of both edge endpoints.
        """

        if diagnostics is None:

            diagnostics = {}

        def increment(
            key,
        ):

            diagnostics[key] = (
                diagnostics.get(
                    key,
                    0,
                )
                +
                1
            )

        def squared_distance(
            first,
            second,
        ):

            return (
                (
                    first[0]
                    -
                    second[0]
                )
                **
                2
                +
                (
                    first[1]
                    -
                    second[1]
                )
                **
                2
            )

        u, v = edge

        if (
            u not in network.node_x
            or u not in network.node_y
            or v not in network.node_x
            or v not in network.node_y
        ):

            increment(
                "missing_endpoint_coordinates"
            )

            return []

        u_position = (
            float(network.node_x[u]),
            float(network.node_y[u]),
        )

        v_position = (
            float(network.node_x[v]),
            float(network.node_y[v]),
        )

        geometry = network.edge_geometry.get(
            edge,
            None,
        )

        sequences = geometry_coordinate_sequences(
            geometry
        )

        if not sequences:

            increment(
                "straight_fallback"
            )

            return [
                [
                    u_position,
                    v_position,
                ]
            ]

        # Use a tolerance relative to the complete network extent.
        all_x_values = list(
            network.node_x.values()
        )

        all_y_values = list(
            network.node_y.values()
        )

        if all_x_values and all_y_values:

            coordinate_span = max(
                max(all_x_values) - min(all_x_values),
                max(all_y_values) - min(all_y_values),
            )

        else:

            coordinate_span = 1.0

        tolerance = max(
            1e-12,
            coordinate_span * 1e-7,
        )

        tolerance_squared = (
            tolerance
            *
            tolerance
        )

        # ------------------------------------------------------------
        # The usual case: one complete LineString.
        # ------------------------------------------------------------

        if len(sequences) == 1:

            coordinates = list(
                sequences[0]
            )

            if len(coordinates) < 2:

                increment(
                    "invalid_geometry"
                )

                return [
                    [
                        u_position,
                        v_position,
                    ]
                ]

            forward_error = (
                squared_distance(
                    coordinates[0],
                    u_position,
                )
                +
                squared_distance(
                    coordinates[-1],
                    v_position,
                )
            )

            reverse_error = (
                squared_distance(
                    coordinates[0],
                    v_position,
                )
                +
                squared_distance(
                    coordinates[-1],
                    u_position,
                )
            )

            if reverse_error < forward_error:

                coordinates.reverse()

                increment(
                    "reversed_geometry"
                )

            if (
                squared_distance(
                    coordinates[0],
                    u_position,
                )
                >
                tolerance_squared
            ):

                coordinates.insert(
                    0,
                    u_position,
                )

                increment(
                    "extended_at_u"
                )

            else:

                # Snap the geometry endpoint exactly onto the vertex.
                coordinates[0] = u_position

            if (
                squared_distance(
                    coordinates[-1],
                    v_position,
                )
                >
                tolerance_squared
            ):

                coordinates.append(
                    v_position
                )

                increment(
                    "extended_at_v"
                )

            else:

                coordinates[-1] = v_position

            return [
                coordinates
            ]

        # ------------------------------------------------------------
        # Several disconnected or unmerged pieces.
        # ------------------------------------------------------------

        increment(
            "multipart_geometry"
        )

        cleaned_sequences = [
            list(sequence)
            for sequence in sequences
            if len(sequence) >= 2
        ]

        if not cleaned_sequences:

            increment(
                "invalid_geometry"
            )

            return [
                [
                    u_position,
                    v_position,
                ]
            ]

        # Draw every stored piece, but ensure that at least one piece
        # reaches each endpoint.
        sequence_endpoints = []

        for sequence_index, sequence in enumerate(
            cleaned_sequences
        ):

            sequence_endpoints.append(
                (
                    squared_distance(
                        sequence[0],
                        u_position,
                    ),
                    sequence_index,
                    0,
                )
            )

            sequence_endpoints.append(
                (
                    squared_distance(
                        sequence[-1],
                        u_position,
                    ),
                    sequence_index,
                    -1,
                )
            )

        _, sequence_index_u, endpoint_index_u = min(
            sequence_endpoints,
            key=lambda item: item[0],
        )

        sequence_u = cleaned_sequences[
            sequence_index_u
        ]

        if endpoint_index_u == -1:

            sequence_u.reverse()

        if (
            squared_distance(
                sequence_u[0],
                u_position,
            )
            >
            tolerance_squared
        ):

            sequence_u.insert(
                0,
                u_position,
            )

            increment(
                "extended_at_u"
            )

        else:

            sequence_u[0] = u_position

        sequence_endpoints = []

        for sequence_index, sequence in enumerate(
            cleaned_sequences
        ):

            sequence_endpoints.append(
                (
                    squared_distance(
                        sequence[0],
                        v_position,
                    ),
                    sequence_index,
                    0,
                )
            )

            sequence_endpoints.append(
                (
                    squared_distance(
                        sequence[-1],
                        v_position,
                    ),
                    sequence_index,
                    -1,
                )
            )

        _, sequence_index_v, endpoint_index_v = min(
            sequence_endpoints,
            key=lambda item: item[0],
        )

        sequence_v = cleaned_sequences[
            sequence_index_v
        ]

        if endpoint_index_v == 0:

            sequence_v.reverse()

        if (
            squared_distance(
                sequence_v[-1],
                v_position,
            )
            >
            tolerance_squared
        ):

            sequence_v.append(
                v_position
            )

            increment(
                "extended_at_v"
            )

        else:

            sequence_v[-1] = v_position

        return cleaned_sequences

    def draw_edges(
        ax,
        network,
        edges,
        color,
        linewidth,
        alpha,
        zorder,
        diagnostics=None,
    ):

        if diagnostics is None:

            diagnostics = {}

        drawn_edges = 0
        skipped_edges = []

        for edge in edges:

            sequences = edge_coordinate_sequences(
                network,
                edge,
                diagnostics=diagnostics,
            )

            edge_was_drawn = False

            for coordinates in sequences:

                if len(coordinates) < 2:

                    continue

                xs = [
                    coordinate[0]
                    for coordinate in coordinates
                ]

                ys = [
                    coordinate[1]
                    for coordinate in coordinates
                ]

                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=zorder,
                )

                edge_was_drawn = True

            if edge_was_drawn:

                drawn_edges += 1

            else:

                skipped_edges.append(
                    edge
                )

        return {
            "requested": len(edges),
            "drawn": drawn_edges,
            "skipped": skipped_edges,
        }

    def sanitize_filename_part(
        value,
    ):

        value = str(
            value
        )

        for character in [
            "/",
            "\\",
            " ",
            ":",
            ";",
            ",",
            "(",
            ")",
        ]:

            value = value.replace(
                character,
                "_",
            )

        return value

    # ------------------------------------------------------------
    # Locate and load the instance.
    # ------------------------------------------------------------

    target_key = normalized_instance_key(
        instance_name
    )

    if os.path.isfile(
        str(instance_name)
    ):

        instance_path = str(
            instance_name
        )

    else:

        instance_paths = BicycleInstance.all_paths()

        matching_paths = [
            path
            for path in instance_paths
            if normalized_instance_key(path) == target_key
        ]

        if not matching_paths:

            raise FileNotFoundError(
                "Could not find an instance matching "
                f"{instance_name!r}."
            )

        if len(matching_paths) > 1:

            print(
                "[WARN] Multiple instance files matched:"
            )

            for path in matching_paths:

                print(
                    " ",
                    path,
                )

            print(
                "[WARN] Using:",
                matching_paths[0],
            )

        instance_path = matching_paths[0]

    instance = BicycleInstance.load(
        instance_path
    )

    network = instance.network

    # ------------------------------------------------------------
    # Locate the unpreprocessed solver result.
    # ------------------------------------------------------------

    results = load_all_results()

    matching_results = []

    for result in results:

        result_instance = get_result_instance_name(
            result
        )

        if result_instance is None:

            continue

        if (
            normalized_instance_key(result_instance)
            !=
            target_key
        ):

            continue

        result_solver = get_field(
            result,
            "solver",
            None,
        )

        if str(result_solver) != str(solver):

            continue

        if not has_no_preprocessing(
            result
        ):

            continue

        matching_results.append(
            result
        )

    if not matching_results:

        available = sorted(
            {
                str(
                    get_field(
                        result,
                        "solver",
                        "",
                    )
                )
                for result in results
                if (
                    get_result_instance_name(result) is not None
                    and normalized_instance_key(
                        get_result_instance_name(result)
                    )
                    ==
                    target_key
                    and has_no_preprocessing(result)
                )
            }
        )

        raise ValueError(
            "No unpreprocessed result found for "
            f"instance={instance_name!r}, solver={solver!r}. "
            f"Available unpreprocessed solvers: {available}"
        )

    # Prefer an optimal result, then a feasible one, then the shortest
    # runtime among otherwise equivalent repeated runs.
    matching_results.sort(
        key=lambda result: (
            not is_optimal_result(result),
            not is_feasible_result(result),
            result_runtime(result),
        )
    )

    selected_result = matching_results[0]

    # ------------------------------------------------------------
    # Normalize edge sets.
    # ------------------------------------------------------------

    safe_edges = set(
        network.safe_edges
    )

    unsafe_edges = set(
        network.unsafe_edges
    )

    all_network_edges = (
        safe_edges
        |
        unsafe_edges
    )

    safe_edges.discard(
        None
    )

    unsafe_edges.discard(
        None
    )

    selected_edges_raw = get_field(
        selected_result,
        "selected_edges",
        [],
    )

    if selected_edges_raw is None:

        selected_edges_raw = []

    # JSON may convert tuples to lists, so restore only the tuple
    # container. Do not alter or canonicalize the endpoint IDs.
    upgraded_edges = {
        tuple(edge)
        for edge in selected_edges_raw
        if (
            isinstance(edge, (list, tuple))
            and len(edge) == 2
        )
    }

    missing_upgraded_edges = (
        upgraded_edges
        -
        set(network.unsafe_edges)
    )

    if missing_upgraded_edges:

        raise ValueError(
            "Selected edges are not contained in network.unsafe_edges. "
            f"Examples: {list(missing_upgraded_edges)[:10]}"
        )

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.0, 8.0)
    )

    geometry_diagnostics = {}

    safe_draw_stats = draw_edges(
        ax=ax,
        network=network,
        edges=safe_edges,
        color="black",
        linewidth=0.65,
        alpha=0.75,
        zorder=1,
        diagnostics=geometry_diagnostics,
    )

    unsafe_draw_stats = draw_edges(
        ax=ax,
        network=network,
        edges=unsafe_edges,
        color="red",
        linewidth=0.85,
        alpha=0.72,
        zorder=2,
        diagnostics=geometry_diagnostics,
    )

    upgraded_draw_stats = draw_edges(
        ax=ax,
        network=network,
        edges=upgraded_edges,
        color="green",
        linewidth=2.4,
        alpha=0.95,
        zorder=4,
        diagnostics=geometry_diagnostics,
    )

    print()
    print("Edge drawing diagnostics")
    print("------------------------")

    print(
        "safe edges:",
        safe_draw_stats["drawn"],
        "/",
        safe_draw_stats["requested"],
    )

    print(
        "unsafe edges:",
        unsafe_draw_stats["drawn"],
        "/",
        unsafe_draw_stats["requested"],
    )

    print(
        "upgraded edges:",
        upgraded_draw_stats["drawn"],
        "/",
        upgraded_draw_stats["requested"],
    )

    for key in sorted(
        geometry_diagnostics
    ):

        print(
            f"{key}:",
            geometry_diagnostics[key],
        )

    if safe_draw_stats["skipped"]:

        print(
            "example skipped safe edges:",
            safe_draw_stats["skipped"][:10],
        )

    if unsafe_draw_stats["skipped"]:

        print(
            "example skipped unsafe edges:",
            unsafe_draw_stats["skipped"][:10],
        )

    if upgraded_draw_stats["skipped"]:

        print(
            "example skipped upgraded edges:",
            upgraded_draw_stats["skipped"][:10],
        )

    print()

    if show_nodes:

        xs = [
            network.node_x[node]
            for node in network.vertices
            if node in network.node_x
        ]

        ys = [
            network.node_y[node]
            for node in network.vertices
            if node in network.node_y
        ]

        ax.scatter(
            xs,
            ys,
            s=2,
            color="black",
            alpha=0.35,
            linewidths=0.0,
            zorder=5,
        )

    pair_colors = []

    if show_terminals:

        terminal_pairs = list(
            instance.terminal_pairs
        )

        num_terminal_pairs = len(
            terminal_pairs
        )

        if num_terminal_pairs > 0:

            cmap = plt.get_cmap(
                terminal_colormap
            )

            # One evenly spaced color for every stored terminal pair.
            pair_colors = [
                cmap(
                    (
                        pair_index + 0.5
                    )
                    /
                    num_terminal_pairs
                )
                for pair_index in range(
                    num_terminal_pairs
                )
            ]

            # --------------------------------------------------------
            # Record the terminal-pair incidences of every vertex.
            #
            # No canonicalization or ID conversion is applied.
            # --------------------------------------------------------

            pair_indices_by_vertex = {}

            for pair_index, pair in enumerate(
                terminal_pairs
            ):

                if (
                    not isinstance(
                        pair,
                        (
                            list,
                            tuple,
                        ),
                    )
                    or len(pair) != 2
                ):

                    print(
                        "[WARN] Invalid terminal pair:",
                        pair,
                    )

                    continue

                source, target = pair

                for vertex in (
                    source,
                    target,
                ):

                    if vertex not in network.vertices:

                        raise ValueError(
                            "Terminal vertex is absent from the "
                            f"BicycleNetwork: {vertex!r}"
                        )

                    if (
                        vertex not in network.node_x
                        or vertex not in network.node_y
                    ):

                        raise ValueError(
                            "Terminal vertex has no geographic "
                            f"coordinates: {vertex!r}"
                        )

                    pair_indices_by_vertex.setdefault(
                        vertex,
                        [],
                    ).append(
                        pair_index
                    )

            # --------------------------------------------------------
            # Draw the cakes in display coordinates.
            #
            # AnnotationBbox anchors the center of each cake at the
            # geographic data coordinate of the terminal vertex.
            # --------------------------------------------------------

            diameter = float(
                terminal_marker_diameter
            )

            radius = (
                diameter / 2.0
                -
                0.6
            )

            center = (
                diameter / 2.0,
                diameter / 2.0,
            )

            for vertex, pair_indices in pair_indices_by_vertex.items():

                x = network.node_x[
                    vertex
                ]

                y = network.node_y[
                    vertex
                ]

                num_slices = len(
                    pair_indices
                )

                if num_slices == 0:

                    continue

                drawing_area = DrawingArea(
                    diameter,
                    diameter,
                    0,
                    0,
                    clip=False,
                )

                angle_per_slice = (
                    360.0
                    /
                    num_slices
                )

                for slice_index, pair_index in enumerate(
                    pair_indices
                ):

                    theta1 = (
                        90.0
                        +
                        slice_index
                        *
                        angle_per_slice
                    )

                    theta2 = (
                        90.0
                        +
                        (
                            slice_index + 1
                        )
                        *
                        angle_per_slice
                    )

                    wedge = Wedge(
                        center=center,
                        r=radius,
                        theta1=theta1,
                        theta2=theta2,
                        facecolor=pair_colors[
                            pair_index
                        ],
                        edgecolor="white",
                        linewidth=0.35,
                    )

                    drawing_area.add_artist(
                        wedge
                    )

                border = Circle(
                    center,
                    radius=radius,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.7,
                )

                drawing_area.add_artist(
                    border
                )

                cake = AnnotationBbox(
                    drawing_area,
                    (
                        x,
                        y,
                    ),
                    xycoords="data",
                    frameon=False,
                    box_alignment=(
                        0.5,
                        0.5,
                    ),
                    pad=0.0,
                    zorder=8,
                )

                ax.add_artist(
                    cake
                )

            print()
            print("Terminal plotting")
            print("-----------------")
            print(
                "terminal pairs:",
                num_terminal_pairs,
            )
            print(
                "distinct terminal vertices:",
                len(pair_indices_by_vertex),
            )
            print(
                "maximum pairs incident to one vertex:",
                max(
                    (
                        len(indices)
                        for indices
                        in pair_indices_by_vertex.values()
                    ),
                    default=0,
                ),
            )
            print()

            # --------------------------------------------------------
            # Optional pair-index colorbar.
            # --------------------------------------------------------

            if show_terminal_colorbar:

                discrete_cmap = ListedColormap(
                    pair_colors
                )

                boundaries = [
                    index - 0.5
                    for index in range(
                        num_terminal_pairs + 1
                    )
                ]

                norm = BoundaryNorm(
                    boundaries,
                    discrete_cmap.N,
                )

                mappable = ScalarMappable(
                    norm=norm,
                    cmap=discrete_cmap,
                )

                mappable.set_array(
                    []
                )

                colorbar = fig.colorbar(
                    mappable,
                    ax=ax,
                    fraction=0.035,
                    pad=0.015,
                )

                colorbar.set_label(
                    "Terminal-pair index"
                )

                if num_terminal_pairs <= 12:

                    ticks = list(
                        range(
                            num_terminal_pairs
                        )
                    )

                else:

                    ticks = sorted(
                        {
                            int(
                                round(
                                    index
                                    *
                                    (
                                        num_terminal_pairs - 1
                                    )
                                    /
                                    6
                                )
                            )
                            for index in range(7)
                        }
                    )

                colorbar.set_ticks(
                    ticks
                )

                colorbar.set_ticklabels(
                    [
                        str(index + 1)
                        for index in ticks
                    ]
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            label="Safe edge",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            label="Unsafe edge",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=3.0,
            label="Upgraded edge",
        ),
    ]

    if show_terminals:

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=6,
                linewidth=0.0,
                label="Terminal node",
            )
        )

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="best",
    )

    if show_title:

        if "NAMEDICT" in globals():

            solver_label = NAMEDICT.get(
                str(solver),
                str(solver),
            )

        else:

            solver_label = str(
                solver
            )

        ax.set_title(
            (
                f"{target_key}\n"
                f"{solver_label}, "
                f"{len(upgraded_edges)} upgraded edges"
            )
        )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.margins(
        0.01
    )

    ax.set_axis_off()

    fig.tight_layout(
        pad=0.02
    )

    # ------------------------------------------------------------
    # Print summary and save.
    # ------------------------------------------------------------

    print()
    print("Instance plot")
    print("-------------")
    print("instance:", instance_path)
    print("solver:", solver)
    print("preprocessor: none")
    print("optimal:", is_optimal_result(selected_result))
    print("feasible:", is_feasible_result(selected_result))
    print("runtime:", result_runtime(selected_result))
    print("safe edges:", len(safe_edges))
    print("unsafe edges:", len(unsafe_edges))
    print("upgraded edges:", len(upgraded_edges))
    print()

    pdf_path = None
    png_path = None

    if save:

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        filename_base = (
            f"instance_"
            f"{sanitize_filename_part(target_key)}_"
            f"{sanitize_filename_part(solver)}"
        )

        pdf_path = os.path.join(
            output_dir,
            filename_base + ".pdf",
        )

        png_path = os.path.join(
            output_dir,
            filename_base + ".png",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.01,
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01,
        )

        print(
            "saved",
            pdf_path,
        )

        print(
            "saved",
            png_path,
        )

    if show:

        plt.show()

    else:

        plt.close(
            fig
        )

    return {
        "instance": instance,
        "result": selected_result,
        "figure": fig,
        "axes": ax,
        "pdf_path": pdf_path,
        "png_path": png_path,
    }

# ============================================================
# Main
# ============================================================

# ============================================================
# Plot recipes and command dispatch
# ============================================================

PAPER_MAIN_METHOD_PAIRS = [
    ["ilp", "kernel1_ilp"],
    ["ilp", "ilp_tw_dp_cuts"],
    ["ilp", "kernel1_ilp_tw_dp_cuts"],
    ["kernel1_ilp", "kernel1_ilp_tw_dp_cuts"],
]


PAPER_TWC_METHODS = [
    "ilp_tw_dp_cuts",
    "ilp_tw_dp_cuts2",
    "ilp_tw_dp_cuts12",
    "kernel1_ilp_tw_dp_cuts",
    "kernel1_ilp_tw_dp_cuts2",
    "kernel1_ilp_tw_dp_cuts12",
]


PAPER_TWC_METHOD_PAIRS = [
    ["ilp_tw_dp_cuts2", "ilp_tw_dp_cuts"],
    ["ilp_tw_dp_cuts2", "ilp_tw_dp_cuts12"],
    ["kernel1_ilp_tw_dp_cuts2", "kernel1_ilp_tw_dp_cuts"],
    ["kernel1_ilp_tw_dp_cuts2", "kernel1_ilp_tw_dp_cuts12"],
]


RUNTIME_METRICS = [
    "total_runtime",
    "solver_runtime",
    "build_runtime",
]


def paper_plots():
    """
    Reproduce exactly the 15 plots used for the conference publication.

    The expected output set is:

        3 solver-performance plots with beta=0.96
        3 solver-performance plots with beta=0.97
        3 solver-performance plots with beta=0.98
        2 main four-method score plots
        3 treewidth-cut-variant score plots
        1 preprocessing-reduction plot
    """

    # Main four-method cactus plots.
    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            beta=0.96,
        )

    # Treewidth-cut variant cactus plots.
    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            methods=PAPER_TWC_METHODS,
            beta=0.97,
        )

    # Main four-method cactus plots with a smaller tail inset.
    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            beta=0.98,
        )

    # Main method-comparison score plots.
    for runtime_metric in [
        "total_runtime",
        "solver_runtime",
    ]:
        solver_pair_score_plot_multi(
            method_pairs=PAPER_MAIN_METHOD_PAIRS,
            runtime_metric=runtime_metric,
            cumulative=True,
            normalize_axes=False,
        )

    # Treewidth-cut variant score plots.
    for runtime_metric in RUNTIME_METRICS:
        solver_pair_score_plot_multi(
            method_pairs=PAPER_TWC_METHOD_PAIRS,
            runtime_metric=runtime_metric,
            cumulative=True,
            normalize_axes=False,
        )

    # Preprocessing reduction by safety model.
    kernel1_reduction_effect_by_safety_plot(
        kind="both",
    )


def run_kernel1_reduction_effect_by_safety_plot(args):
    kernel1_reduction_effect_by_safety_plot(
        kind="both",
    )


def run_runtime_by_target_fraction_plot(args):
    runtime_by_target_fraction_plot()
    runtime_by_target_fraction_plot(
        runtime_metric="solver_runtime",
    )


def run_runtime_by_safety_model_plot(args):
    runtime_by_safety_model_plot()


def run_runtime_scatter(args):
    runtime_scatter(
        solver="ilp",
        preprocessor=None,
        other_solver="ilp",
        other_preprocessor="kernel1",
        main_mode="log-log",
    )

    runtime_scatter(
        solver="ilp",
        preprocessor=None,
        other_solver="ilp_tw_dp_cuts",
        other_preprocessor=None,
        main_mode="log-log",
    )

    runtime_scatter(
        solver="ilp",
        preprocessor=None,
        other_solver="ilp_tw_dp_cuts",
        other_preprocessor="kernel1",
        main_mode="log-log",
    )

    runtime_scatter(
        solver="ilp",
        preprocessor=None,
        other_solver="ilp_tw_dp_cuts",
        other_preprocessor="kernel1",
        main_mode="log-log",
        runtime_mode="solver_runtime",
    )


def run_speedup_profile(args):
    speedup_profile()


def run_solver_performance(args):
    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            beta=0.96,
        )

    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            methods=PAPER_TWC_METHODS,
            beta=0.97,
        )

    for runtime_metric in RUNTIME_METRICS:
        solver_performance(
            runtime_metric=runtime_metric,
            beta=0.98,
        )


def run_reducer_performance(args):
    reducer_performance()


def run_hard_instance_performance(args):
    hard_instance_performance(
        reference_method=args.reference_method,
        top_k=args.top_k,
    )

    hard_instance_performance(
        reference_method=args.reference_method,
        top_k=args.top_k,
        show_instance_labels=True,
    )


def run_solver_pair_score_plot_multi(args):
    for runtime_metric in [
        "build_runtime",
        "solver_runtime",
        "total_runtime",
    ]:
        solver_pair_score_plot_multi(
            method_pairs=PAPER_TWC_METHOD_PAIRS,
            runtime_metric=runtime_metric,
            cumulative=True,
            normalize_axes=False,
        )


def run_solved_instances_by_solver_plot(args):
    solved_instances_by_solver_plot()


def run_alpha_optimal_cost_plot(args):
    alpha_optimal_cost_plot(
        method="kernel1_ilp_tw_dp_cuts",
        alphas=[1.2, 1.3, 1.5],
        show_relative_reduction=True,
    )

    alpha_optimal_cost_plot(
        method="kernel1_ilp_tw_dp_cuts",
        alphas=[1.2, 1.3, 1.5],
        plot_mode="reduction_main",
        show_relative_reduction=True,
    )


def run_plot_instance(args):
    instances = [
        "weida_C_random-pairs_tf25_a12_i0.json",
        "weida_r3_C_random-pairs_tf15_a12_i0.json",
        "zwenkau_C_random-pairs_tf25_a12_i0.json",
        "zwenkau_r3_C_random-pairs_tf15_a12_i0.json",
    ]

    for instance_name in instances:
        plot_instance(
            instance_name=instance_name,
            solver="ilp",
            show_terminals=True,
        )


def run_paper(args):
    paper_plots()


PLOT_COMMANDS = {
    "kernel1_reduction_effect_by_safety_plot": (
        run_kernel1_reduction_effect_by_safety_plot
    ),
    "runtime_by_target_fraction_plot": (
        run_runtime_by_target_fraction_plot
    ),
    "runtime_by_safety_model_plot": (
        run_runtime_by_safety_model_plot
    ),
    "runtime_scatter": run_runtime_scatter,
    "speedup_profile": run_speedup_profile,
    "solver_performance": run_solver_performance,
    "reducer_performance": run_reducer_performance,
    "hard_instance_performance": run_hard_instance_performance,
    "solver_pair_score_plot_multi": run_solver_pair_score_plot_multi,
    "solved_instances_by_solver_plot": run_solved_instances_by_solver_plot,
    "alpha_optimal_cost_plot": run_alpha_optimal_cost_plot,
    "plot_instance": run_plot_instance,
    "paper": run_paper,
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate BNIP experiment plots. "
            "Use 'paper' to reproduce the conference-publication figures."
        )
    )

    parser.add_argument(
        "plot",
        choices=sorted(PLOT_COMMANDS),
        help="Plot or plot recipe to generate.",
    )

    parser.add_argument(
        "--reference-method",
        default=None,
        help=(
            "Reference method for selecting hard instances, "
            "e.g. ilp or kernel1_ilp_tw_dp_cuts."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of hardest instances to show.",
    )

    args = parser.parse_args()

    use_latex_style_plots(
        usetex=True,
    )

    PLOT_COMMANDS[
        args.plot
    ](
        args
    )


if __name__ == "__main__":
    main()
