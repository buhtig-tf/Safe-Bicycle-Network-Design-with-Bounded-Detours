from reductionrule import ReductionRule

import copy
import networkx as nx

from bicycleinstance import BicycleInstance


class RemoveSatisfiedPairsReduction(ReductionRule):

    name = "RemoveSatisfiedPairsReduction"

    def apply(self, instance):

        EPS = 1e-9

        net = instance.network

        G = net.to_networkx()

        S = nx.Graph()

        S.add_nodes_from(
            net.vertices
        )

        for e in net.safe_edges:

            S.add_edge(
                e[0],
                e[1],
                length=net.length[e],
            )

        remaining_pairs = []
        removed_pairs = []

        for s, t in instance.terminal_pairs:

            try:

                d_full = nx.shortest_path_length(
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

            try:

                d_safe = nx.shortest_path_length(
                    S,
                    s,
                    t,
                    weight="length",
                )

            except nx.NetworkXNoPath:

                remaining_pairs.append(
                    (s, t)
                )

                continue

            if d_safe <= instance.alpha * d_full + EPS:

                removed_pairs.append(
                    {
                        "pair": (s, t),
                        "d_full": d_full,
                        "d_safe": d_safe,
                        "ratio": d_safe / d_full if d_full > 0 else 1.0,
                    }
                )

                continue

            remaining_pairs.append(
                (s, t)
            )

        reduced = BicycleInstance(
            network=copy.deepcopy(net), # net,
            terminal_pairs=remaining_pairs,
            alpha=instance.alpha,
            budget=instance.budget,
            info=(
                instance.info
                + " [satisfied pairs removed]"
            ),
        )

        reduced.removed_satisfied_pairs = removed_pairs

        return reduced
