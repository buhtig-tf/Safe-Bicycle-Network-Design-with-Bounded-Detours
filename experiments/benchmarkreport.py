import math
import os
import statistics
from collections.abc import Iterable
from pathlib import Path

from bicyclenetwork import BicycleNetwork


class BenchmarkReport:
    """
    Publication-quality benchmark report generator
    for BicycleNetwork experiments.

    Supports:
        - Villages (Village objects or dicts)
        - Village regions
        - Arbitrary BicycleNetwork instances

    Produces LaTeX tables with:
        - graph statistics
        - feedback edge set statistics
        - treewidth upper bound
    """

    # ------------------------------------------------------------
    # Internal: ensure iterable format
    # ------------------------------------------------------------
    @staticmethod
    def _normalize(input_data):

        if input_data is None:
            return []

        if isinstance(input_data, dict):
            return list(input_data.values())

        if isinstance(input_data, Iterable) and not isinstance(input_data, (str, BicycleNetwork)):
            return list(input_data)

        return [input_data]


    @staticmethod
    def _network_stat_row(
        net,
    ):
        import networkx as nx

        G = net.to_networkx()

        n = len(
            net.vertices
        )

        m = (
            len(net.safe_edges)
            +
            len(net.unsafe_edges)
        )

        unsafe_frac = (
            len(net.unsafe_edges) / m
            if m > 0
            else 0.0
        )

        R = net.minimum_feedback_edge_set()

        fes_frac = (
            len(R) / m
            if m > 0
            else 0.0
        )

        if hasattr(
            net,
            "treewidth_upper_bound",
        ):

            tw_upper = net.treewidth_upper_bound()

        else:

            tw_upper, _ = nx.approximation.treewidth_min_fill_in(
                G
            )

        if len(G) > 0:

            max_degree = max(
                dict(
                    G.degree()
                ).values()
            )

        else:

            max_degree = 0

        is_planar, _ = nx.check_planarity(
            G
        )

        return {
            "n": n,
            "m": m,
            "unsafe_frac": unsafe_frac,
            "fes_size": len(R),
            "fes_frac": fes_frac,
            "tw_upper": tw_upper,
            "max_degree": max_degree,
            "is_planar": is_planar,
        }

    @staticmethod
    def _mean_median_min_max(
        values,
    ):
        import statistics

        values = list(
            values
        )

        if not values:

            raise ValueError(
                "Cannot summarize empty value list."
            )

        return {
            "mean": statistics.mean(
                values
            ),
            "median": statistics.median(
                values
            ),
            "min": min(
                values
            ),
            "max": max(
                values
            ),
        }


    @staticmethod
    def _fmt_int_or_float(
        x,
    ):
        if abs(
            x - round(x)
        ) <= 1e-9:

            return str(
                int(
                    round(x)
                )
            )

        return f"{x:.2f}"


    @staticmethod
    def _struct_summary_cell(
        values,
        integer_like=False,
    ):
        s = BenchmarkReport._mean_median_min_max(
            values
        )

        mean = f"{s['mean']:.2f}"

        if integer_like:

            median = BenchmarkReport._fmt_int_or_float(
                s["median"]
            )

            minimum = BenchmarkReport._fmt_int_or_float(
                s["min"]
            )

            maximum = BenchmarkReport._fmt_int_or_float(
                s["max"]
            )

        else:

            median = f"{s['median']:.2f}"
            minimum = f"{s['min']:.2f}"
            maximum = f"{s['max']:.2f}"

        return (
            f"{mean} "
            f"\\taban{{{median}}}{{{minimum}}}{{{maximum}}}"
        )


    @staticmethod
    def _struct_type_row(
        type_name,
        rows,
    ):
        planar_count = sum(
            1
            for r in rows
            if r["is_planar"]
        )

        total_count = len(
            rows
        )

        return (
            f"{type_name}\n"
            f"& {BenchmarkReport._struct_summary_cell([r['n'] for r in rows], integer_like=True)}\n"
            f"& {BenchmarkReport._struct_summary_cell([r['m'] for r in rows], integer_like=True)}\n"
            f"& {BenchmarkReport._struct_summary_cell([r['fes_frac'] for r in rows], integer_like=False)}\n"
            f"& {BenchmarkReport._struct_summary_cell([r['tw_upper'] for r in rows], integer_like=True)}\n"
            f"& {BenchmarkReport._struct_summary_cell([r['max_degree'] for r in rows], integer_like=True)}\n"
            f"& {planar_count}/{total_count}\n"
            f"\\\\"
        )

    @staticmethod
    def village_region_struct_table(
        villages=None,
        regions=None,
        safety_model="A",
        village_folder="data/networks/villages",
        region_folder="data/networks/regions",
        output_dir="tables",
        filename="village_region_struct_table.tex",
        caption=(
            "Structural statistics for our village and region networks "
            "as to the means with median, minimum, and maximum in parentheses."
        ),
        label="tab:village:struct",
    ):
        """
        Create a combined LaTeX table with structural statistics for
        village and region networks.

        If villages is None, all available village files matching
        '*_<safety_model>.pkl' are loaded from village_folder.

        If regions is None, all available region files matching
        '*_<safety_model>.pkl' are loaded from region_folder.
        """

        from pathlib import Path

        village_rows = []
        region_rows = []

        # ------------------------------------------------------------
        # Villages
        # ------------------------------------------------------------

        if villages is None:

            village_folder = Path(
                village_folder
            )

            village_files = sorted(
                village_folder.glob(
                    f"*_{safety_model}.pkl"
                )
            )

            if not village_files:

                raise FileNotFoundError(
                    f"No village network files found in {village_folder} "
                    f"matching pattern '*_{safety_model}.pkl'"
                )

            for file in village_files:

                print(
                    f"[benchmark] loading {file.name}"
                )

                net = BenchmarkReport._load_network_from_file(
                    file
                )

                row = BenchmarkReport._network_stat_row(
                    net
                )

                row["name"] = file.stem.replace(
                    f"_{safety_model}",
                    "",
                )

                village_rows.append(
                    row
                )

        else:

            villages = BenchmarkReport._normalize(
                villages
            )

            for v in villages:

                net = BicycleNetwork.from_place(
                    v,
                    safety_model=safety_model,
                )

                row = BenchmarkReport._network_stat_row(
                    net
                )

                row["name"] = getattr(
                    v,
                    "short_name",
                    None,
                ) or getattr(
                    v,
                    "name",
                    None,
                ) or getattr(
                    v,
                    "key",
                    None,
                )

                row["population"] = getattr(
                    v,
                    "population",
                    None,
                )

                village_rows.append(
                    row
                )

        # ------------------------------------------------------------
        # Regions
        # ------------------------------------------------------------

        region_folder = Path(
            region_folder
        )

        if regions is None:

            region_files = sorted(
                region_folder.glob(
                    f"*_{safety_model}.pkl"
                )
            )

            if not region_files:

                raise FileNotFoundError(
                    f"No region network files found in {region_folder} "
                    f"matching pattern '*_{safety_model}.pkl'"
                )

        else:

            regions = BenchmarkReport._normalize(
                regions
            )

            region_files = []

            for r in regions:

                key = getattr(
                    r,
                    "key",
                    None,
                ) or str(
                    r
                )

                file = region_folder / f"{key}_{safety_model}.pkl"

                if not file.exists():

                    raise FileNotFoundError(
                        f"Region network file not found: {file}"
                    )

                region_files.append(
                    file
                )

        for file in region_files:

            print(
                f"[benchmark] loading {file.name}"
            )

            net = BenchmarkReport._load_network_from_file(
                file
            )

            row = BenchmarkReport._network_stat_row(
                net
            )

            row["name"] = file.stem.replace(
                f"_{safety_model}",
                "",
            )

            region_rows.append(
                row
            )

        # ------------------------------------------------------------
        # LaTeX table
        # ------------------------------------------------------------

        lines = [
            r"\begin{table}",
            r"  \centering",
            r"  \begin{tabular}{lrrrrrr}",
            r"  \toprule",
            r"  Type & Vertices & Edges~$m$ & $\fes/m$ & $\tw_{rm fill-in}\geq \tw$ & $\Delta$ & Planar \\",
            r"  \midrule",
            "  "
            + BenchmarkReport._struct_type_row(
                "Village",
                village_rows,
            ).replace(
                "\n",
                "\n  ",
            ),
            "  "
            + BenchmarkReport._struct_type_row(
                "Region",
                region_rows,
            ).replace(
                "\n",
                "\n  ",
            ),
            r"  \bottomrule",
            r"  \end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"\end{table}",
        ]

        table = "\n".join(
            lines
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        path = Path(
            output_dir
        ) / filename

        with open(
            path,
            "w",
        ) as f:

            f.write(
                table
            )

            f.write(
                "\n"
            )

        print(
            "saved",
            path,
        )

        return str(
            path
        )

    # ------------------------------------------------------------
    # Load network from file (future-proof hook)
    # ------------------------------------------------------------
    @staticmethod
    def _load_network_from_file(
        path,
    ):

        path = Path(
            path
        )

        if path.suffix != ".pkl":

            raise ValueError(
                f"Unsupported network file: {path}"
            )

        return BicycleNetwork.load(
            path
        )

    def safety_model_summary_table(
        self,
        output_dir="tables",
        network_root="data/networks",
        filename="safety_model_summary_table.tex",
        decimals=2,
    ):
        """
        Create LaTeX table summarizing the three safety models.

        Unsafe fractions are freshly computed from saved BicycleNetwork
        files in

            data/networks/villages
            data/networks/regions

        for each safety model A, B, C.

        The output table uses tabularray X columns instead of fixed p{}
        columns. This fixes wrapping problems for cells that span several
        columns.
        """

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------

        def normalize_safety_model(
            model,
        ):

            if model is None:

                return ""

            model = str(
                model
            ).strip().upper()

            aliases = {
                "HIER": "A",
                "SEC": "B",
                "PES": "C",
            }

            return aliases.get(
                model,
                model,
            )

        def model_from_path(
            path,
        ):

            stem = Path(
                path
            ).stem

            token = stem.split(
                "_"
            )[-1]

            return normalize_safety_model(
                token
            )

        def kind_from_path(
            path,
        ):

            parts = Path(
                path
            ).parts

            if "villages" in parts:

                return "villages"

            if "regions" in parts:

                return "regions"

            return None

        def unsafe_fraction(
            net,
        ):

            safe = len(
                net.safe_edges
            )

            unsafe = len(
                net.unsafe_edges
            )

            total = safe + unsafe

            if total == 0:

                return None

            return unsafe / total

        def format_number(
            value,
        ):

            return f"{value:.{decimals}f}"

        def format_stats_cell(
            values,
        ):

            if not values:

                return "--"

            values = [
                float(v)
                for v in values
                if math.isfinite(float(v))
            ]

            if not values:

                return "--"

            avg = sum(values) / len(values)
            med = statistics.median(values)
            mn = min(values)
            mx = max(values)

            return (
                f"${format_number(avg)}$ "
                f"\\taban{{{format_number(med)}}}"
                f"{{{format_number(mn)}}}"
                f"{{{format_number(mx)}}}"
            )

        # ------------------------------------------------------------
        # Collect unsafe fractions.
        # ------------------------------------------------------------

        fractions = {
            "villages": {
                "A": [],
                "B": [],
                "C": [],
            },
            "regions": {
                "A": [],
                "B": [],
                "C": [],
            },
        }

        paths = []

        for subdir in [
            "villages",
            "regions",
        ]:

            root = Path(
                network_root
            ) / subdir

            if not root.exists():

                print(
                    "[WARN] network directory does not exist:",
                    root,
                )

                continue

            for dirpath, _, filenames in os.walk(
                root
            ):

                for fn in filenames:

                    if fn.endswith(
                        ".pkl"
                    ):

                        paths.append(
                            Path(dirpath) / fn
                        )

        for path in sorted(
            paths
        ):

            try:

                net = BicycleNetwork.load(
                    path
                )

            except Exception as e:

                print(
                    "[WARN] could not load network:",
                    path,
                    f"{type(e).__name__}: {e}",
                )

                continue

            kind = getattr(
                net,
                "origin_type",
                None,
            )

            if kind is not None:

                kind = str(
                    kind
                ).strip().lower() + "s"

            if kind not in {
                "villages",
                "regions",
            }:

                kind = kind_from_path(
                    path
                )

            if kind not in {
                "villages",
                "regions",
            }:

                print(
                    "[WARN] could not determine network kind:",
                    path,
                )

                continue

            model = normalize_safety_model(
                getattr(
                    net,
                    "safety_model",
                    None,
                )
            )

            if model not in {
                "A",
                "B",
                "C",
            }:

                model = model_from_path(
                    path
                )

            if model not in {
                "A",
                "B",
                "C",
            }:

                print(
                    "[WARN] could not determine safety model:",
                    path,
                )

                continue

            frac = unsafe_fraction(
                net
            )

            if frac is None:

                print(
                    "[WARN] network without edges skipped:",
                    path,
                )

                continue

            fractions[kind][model].append(
                frac
            )

        # ------------------------------------------------------------
        # Format computed rows.
        # ------------------------------------------------------------

        village_A = format_stats_cell(
            fractions["villages"]["A"]
        )

        village_B = format_stats_cell(
            fractions["villages"]["B"]
        )

        village_C = format_stats_cell(
            fractions["villages"]["C"]
        )

        region_A = format_stats_cell(
            fractions["regions"]["A"]
        )

        region_B = format_stats_cell(
            fractions["regions"]["B"]
        )

        region_C = format_stats_cell(
            fractions["regions"]["C"]
        )

        print()
        print("Safety-model unsafe fractions")
        print("-----------------------------")

        for kind in [
            "villages",
            "regions",
        ]:

            for model in [
                "A",
                "B",
                "C",
            ]:

                print(
                    f"{kind:8s} model {model}: "
                    f"n={len(fractions[kind][model])}"
                )

        print()

        # ------------------------------------------------------------
        # LaTeX table.
        #
        # Important for wrapping:
        #   - use tabularray X columns,
        #   - set width = \\textwidth,
        #   - use l-aligned multicolumn cells, not c-aligned cells.
        # ------------------------------------------------------------

        table = rf"""
    % Requires:
    % \usepackage{{tabularray}}
    % \UseTblrLibrary{{booktabs}}
    % \usepackage[table]{{xcolor}}
    % \newcommand{{\taban}}[3]{{\scriptsize (#1,#2,#3)}}

    \begin{{table}}[t]
    \centering
    \begin{{tblr}}{{
    width = \textwidth,
    colspec = {{X[1.10,l] X[1.00,l] X[1.00,l] X[1.55,l]}},
    cells = {{valign=m}},
    row{{1}} = {{halign=c}},
    hspan = minimal,
    }}

    \toprule
    & Model A & Model B & Model C \\
    \midrule
    Always safe
    &
    \SetCell[c=3]{{l,bg=gray!10}} Cycleways, bicycle infrastructure, living streets, traffic-calmed roads
    &
    &
    \\

    Residential roads
    &
    \SetCell[r=2]{{c,bg=gray!10}} Safe
    &
    \SetCell[r=2]{{l,bg=gray!5}} Safe if speed is missing or at most 30 km/h
    &
    Safe if speed is missing or at most 30 km/h
    \\

    Service / unclassified roads
    &
    &
    &
    \SetCell[r=1]{{l,bg=gray!10}} Safe only if at most 20 km/h or traffic-calmed
    \\

    Tertiary roads
    &
    \SetCell[c=2]{{l,bg=gray!15}} Safe only if at most 30 km/h or traffic-calmed
    &
    &
    Unsafe unless protected by bicycle infrastructure or traffic calming
    \\

    Paths / pedestrian streets
    &
    \SetCell[c=2]{{l,bg=gray!10}} Safe unless bicycles are explicitly forbidden
    &
    &
    Safe only if bicycles are explicitly allowed
    \\

    \SetCell[r=2]{{l}} Average unsafe fraction
    &
    {village_A}
    &
    {village_B}
    &
    {village_C}
    \\
    &
    {region_A}
    &
    {region_B}
    &
    {region_C}
    \\
    \bottomrule
    \end{{tblr}}
    \caption{{
    Summary of the three edge safety models.
    Last row reports average unsafe-edge fractions, with median, minimum, and maximum in parentheses, for villages (top) and regions (bottom).
    }}
    \label{{tab:safety-models}}
    \end{{table}}
    """.strip()

        path = Path(
            output_dir
        ) / filename

        with open(
            path,
            "w",
        ) as f:

            f.write(
                table
            )

            f.write(
                "\n"
            )

        print(
            "saved",
            path,
        )

        return str(
            path
        )



if __name__ == "__main__":

    report = BenchmarkReport()

    print()
    print("Village / region structural statistics")
    print("--------------------------------------")

    report.village_region_struct_table()

    print()
    print("Safety-model statistics")
    print("-----------------------")

    report.safety_model_summary_table()
