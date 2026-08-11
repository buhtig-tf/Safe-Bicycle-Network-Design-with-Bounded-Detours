import copy
import networkx as nx

from reductionrule import ReductionRule
from bicycleinstance import BicycleInstance


class MandatoryEdgeReduction(ReductionRule):
    """
    Let (s,t) be a terminal pair.

    Compute a shortest path P.

    For every unsafe edge e on P:

        remove e

        check whether there still exists an admissible path of length
        at most

            alpha * d

        where d is the original shortest-path distance.

    If not, e is mandatory and must be upgraded.

    All mandatory unsafe edges are upgraded.
    """

    name = "MandatoryEdgeReduction"

    def apply(self, instance):

        EPS = 1e-9

        net = copy.deepcopy(
            instance.network
        )

        G = net.to_networkx()

        budget = instance.budget

        mandatory_records = []

        for s, t in instance.terminal_pairs:

            # ----------------------------------------------------
            # Shortest-path distance in the full graph.
            # ----------------------------------------------------

            try:

                d = nx.shortest_path_length(
                    G,
                    s,
                    t,
                    weight="length",
                )

            except nx.NetworkXNoPath:

                continue

            threshold = (
                instance.alpha
                *
                d
            )

            # ----------------------------------------------------
            # One shortest path.
            # ----------------------------------------------------

            P = nx.shortest_path(
                G,
                s,
                t,
                weight="length",
            )

            mandatory_edges = set()
            pair_records = []

            # ----------------------------------------------------
            # Test every unsafe edge on the shortest path.
            # ----------------------------------------------------

            for i in range(len(P) - 1):

                e = net.canonical_edge(
                    P[i],
                    P[i + 1],
                )

                # Only currently unsafe edges can become mandatory.
                # If the edge was already upgraded due to an earlier
                # pair in this same reduction call, it is now safe and
                # must not be charged again.
                if e not in net.unsafe_edges:

                    continue

                u, v = e

                if not G.has_edge(
                    u,
                    v,
                ):

                    raise RuntimeError(
                        "MandatoryEdgeReduction: shortest path edge "
                        f"{e} is missing from G"
                    )

                edge_data = dict(
                    G[u][v]
                )

                G.remove_edge(
                    u,
                    v,
                )

                mandatory = False
                alt_dist = None

                try:

                    try:

                        alt_dist = nx.shortest_path_length(
                            G,
                            s,
                            t,
                            weight="length",
                        )

                        if alt_dist > threshold + EPS:

                            mandatory = True

                    except nx.NetworkXNoPath:

                        mandatory = True

                finally:

                    G.add_edge(
                        u,
                        v,
                        **edge_data,
                    )

                if mandatory:

                    mandatory_edges.add(
                        e
                    )

                    pair_records.append(
                        {
                            "edge": e,
                            "alt_dist_without_edge": alt_dist,
                            "threshold": threshold,
                        }
                    )

            if not mandatory_edges:

                continue

            # ----------------------------------------------------
            # Compute forced cost.
            # ----------------------------------------------------

            forced_cost = 0.0

            for e in mandatory_edges:

                if e not in net.upgrade_cost:

                    raise RuntimeError(
                        "MandatoryEdgeReduction: mandatory unsafe edge "
                        f"has no upgrade cost: {e}"
                    )

                cost = net.upgrade_cost[e]

                if cost < 0.0:

                    raise RuntimeError(
                        "MandatoryEdgeReduction: negative upgrade cost "
                        f"for edge {e}: {cost}"
                    )

                forced_cost += cost

            # ----------------------------------------------------
            # Budget check.
            # ----------------------------------------------------

            if forced_cost > budget + EPS:

                raise ValueError(
                    "Instance infeasible: mandatory edges exceed "
                    "remaining budget. "
                    f"pair={(s, t)}, "
                    f"forced_cost={forced_cost}, "
                    f"budget={budget}"
                )

            # ----------------------------------------------------
            # Upgrade mandatory edges.
            # ----------------------------------------------------

            for e in mandatory_edges:

                if e not in net.unsafe_edges:

                    raise RuntimeError(
                        "MandatoryEdgeReduction: edge to be upgraded "
                        f"is no longer unsafe: {e}"
                    )

                net.unsafe_edges.remove(
                    e
                )

                net.safe_edges.add(
                    e
                )

                net.upgrade_cost.pop(
                    e,
                    None,
                )

            budget -= forced_cost

            mandatory_records.append(
                {
                    "pair": (s, t),
                    "shortest_path": list(P),
                    "shortest_distance": d,
                    "threshold": threshold,
                    "mandatory_edges": sorted(mandatory_edges),
                    "forced_cost": forced_cost,
                    "edge_tests": pair_records,
                }
            )

        # --------------------------------------------------------
        # Consistency checks.
        # --------------------------------------------------------

        if net.safe_edges & net.unsafe_edges:

            overlap = (
                net.safe_edges
                &
                net.unsafe_edges
            )

            raise RuntimeError(
                "MandatoryEdgeReduction produced edges that are both "
                f"safe and unsafe: {list(overlap)[:5]}"
            )

        all_edges = (
            net.safe_edges
            |
            net.unsafe_edges
        )

        for e in all_edges:

            if e not in net.length:

                raise RuntimeError(
                    "MandatoryEdgeReduction left edge without length: "
                    f"{e}"
                )

        for e in net.unsafe_edges:

            if e not in net.upgrade_cost:

                raise RuntimeError(
                    "MandatoryEdgeReduction left unsafe edge without "
                    f"upgrade cost: {e}"
                )

        # --------------------------------------------------------
        # Construct reduced instance.
        # --------------------------------------------------------

        reduced = BicycleInstance(
            network=net,
            terminal_pairs=list(
                instance.terminal_pairs
            ),
            alpha=instance.alpha,
            budget=budget,
            info=(
                instance.info
                + " [mandatory-edge]"
            ),
        )

        if hasattr(
            reduced,
            "clone_metadata_from",
        ):

            reduced.clone_metadata_from(
                instance
            )

        reduced.mandatory_edge_records = mandatory_records

        return reduced
