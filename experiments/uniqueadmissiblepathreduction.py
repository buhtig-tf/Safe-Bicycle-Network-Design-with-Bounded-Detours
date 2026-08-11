import copy
import networkx as nx

from reductionrule import ReductionRule
from bicycleinstance import BicycleInstance


class UniqueAdmissiblePathReduction(ReductionRule):
    """
    Let (s,t) be a terminal pair.

    Let d be the shortest-path distance in G.

    A path is admissible if its length is at most

        alpha * d.

    If there is exactly one admissible path P, then every unsafe edge
    on P must be upgraded.

    Therefore:

        - make these edges safe,
        - decrease the budget,
        - delete the terminal pair.
    """

    name = "UniqueAdmissiblePathReduction"

    def apply(self, instance):

        EPS = 1e-9

        net = copy.deepcopy(
            instance.network
        )

        G = net.to_networkx()

        budget = instance.budget

        remaining_pairs = []

        forced_pair_records = []

        for s, t in instance.terminal_pairs:

            # ----------------------------------------------------
            # Shortest-path distance in full graph.
            # ----------------------------------------------------

            try:

                d = nx.shortest_path_length(
                    G,
                    s,
                    t,
                    weight="length",
                )

            except nx.NetworkXNoPath:

                remaining_pairs.append(
                    (s, t)
                )

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

            # ----------------------------------------------------
            # Test whether another admissible path exists.
            # ----------------------------------------------------

            second_exists = False

            for i in range(len(P) - 1):

                u = P[i]
                v = P[i + 1]

                if not G.has_edge(
                    u,
                    v,
                ):
                    continue

                edge_data = dict(
                    G[u][v]
                )

                G.remove_edge(
                    u,
                    v,
                )

                try:

                    try:

                        alt_dist = nx.shortest_path_length(
                            G,
                            s,
                            t,
                            weight="length",
                        )

                        if alt_dist <= threshold + EPS:

                            second_exists = True

                            break

                    except nx.NetworkXNoPath:

                        pass

                finally:

                    G.add_edge(
                        u,
                        v,
                        **edge_data,
                    )

                if second_exists:
                    break

            # ----------------------------------------------------
            # Another admissible path exists.
            # ----------------------------------------------------

            if second_exists:

                remaining_pairs.append(
                    (s, t)
                )

                continue

            # ----------------------------------------------------
            # Unique admissible path.
            # ----------------------------------------------------

            forced_edges = []
            forced_cost = 0.0

            for i in range(len(P) - 1):

                e = net.canonical_edge(
                    P[i],
                    P[i + 1],
                )

                if e in net.unsafe_edges:

                    forced_edges.append(
                        e
                    )

                    if e not in net.upgrade_cost:

                        raise RuntimeError(
                            "Unsafe edge has no upgrade cost: "
                            f"{e}"
                        )

                    forced_cost += net.upgrade_cost[e]

            # ----------------------------------------------------
            # Budget check.
            # ----------------------------------------------------

            if forced_cost > budget + EPS:

                raise ValueError(
                    "Instance infeasible: forced upgrades exceed "
                    "remaining budget. "
                    f"pair={(s, t)}, "
                    f"forced_cost={forced_cost}, "
                    f"budget={budget}"
                )

            # ----------------------------------------------------
            # Apply forced upgrades.
            # ----------------------------------------------------

            for e in forced_edges:

                if e not in net.unsafe_edges:

                    raise RuntimeError(
                        "Forced edge is not unsafe anymore: "
                        f"{e}"
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

            forced_pair_records.append(
                {
                    "pair": (s, t),
                    "path": list(P),
                    "path_length": d,
                    "threshold": threshold,
                    "forced_edges": list(forced_edges),
                    "forced_cost": forced_cost,
                }
            )

            # Pair is removed by not adding it to remaining_pairs.

        reduced = BicycleInstance(
            network=net,
            terminal_pairs=remaining_pairs,
            alpha=instance.alpha,
            budget=budget,
            info=(
                instance.info
                + " [unique-path]"
            ),
        )

        reduced.unique_path_forced_pairs = forced_pair_records

        return reduced
