from reductionrule import ReductionRule

import copy
import networkx as nx

from bicycleinstance import BicycleInstance


class AlphaRelevantEdgeReduction(ReductionRule):

    name = "AlphaRelevantEdgeReduction"

    def apply(
        self,
        instance,
    ):

        EPS = 1e-9

        net = copy.deepcopy(
            instance.network
        )

        G = net.to_networkx()

        terminal_pairs = list(
            instance.terminal_pairs
        )

        all_edges = (
            set(net.safe_edges)
            |
            set(net.unsafe_edges)
        )

        # --------------------------------------------------------
        # No pairs or no edges: return a proper reduced object.
        # --------------------------------------------------------

        if not terminal_pairs or not all_edges:

            reduced = BicycleInstance(
                network=net,
                terminal_pairs=terminal_pairs,
                alpha=instance.alpha,
                budget=instance.budget,
                info=(
                    instance.info
                    + " [alpha-relevant edges]"
                ),
            )

            reduced.alpha_relevant_removed_edges = set()
            reduced.alpha_relevant_removed_vertices = set()

            return reduced

        # --------------------------------------------------------
        # Dijkstra cache.
        # --------------------------------------------------------

        distance_cache = {}

        def distances_from(source):

            if source not in distance_cache:

                distance_cache[source] = (
                    nx.single_source_dijkstra_path_length(
                        G,
                        source,
                        weight="length",
                    )
                )

            return distance_cache[source]

        # --------------------------------------------------------
        # Determine alpha-relevant edges.
        # --------------------------------------------------------

        relevant_edges = set()

        for s, t in terminal_pairs:

            if s not in G or t not in G:

                raise RuntimeError(
                    "AlphaRelevantEdgeReduction found a terminal "
                    f"not present in the graph: {(s, t)}"
                )

            dist_s = distances_from(
                s
            )

            if t not in dist_s:

                raise RuntimeError(
                    "AlphaRelevantEdgeReduction found a disconnected "
                    f"terminal pair: {(s, t)}"
                )

            dist_t = distances_from(
                t
            )

            base_dist = dist_s[t]

            threshold = (
                instance.alpha
                *
                base_dist
            )

            for e in all_edges:

                u, v = e

                if e not in net.length:

                    raise RuntimeError(
                        f"Missing length for edge {e}"
                    )

                length = net.length[e]

                relevant = False

                if (
                    u in dist_s
                    and v in dist_t
                    and dist_s[u] + length + dist_t[v]
                    <= threshold + EPS
                ):

                    relevant = True

                elif (
                    v in dist_s
                    and u in dist_t
                    and dist_s[v] + length + dist_t[u]
                    <= threshold + EPS
                ):

                    relevant = True

                if relevant:

                    relevant_edges.add(
                        e
                    )

        removed_edges = (
            all_edges
            -
            relevant_edges
        )

        # --------------------------------------------------------
        # Build reduced network.
        # --------------------------------------------------------

        new_safe_edges = {
            e
            for e in net.safe_edges
            if e in relevant_edges
        }

        new_unsafe_edges = {
            e
            for e in net.unsafe_edges
            if e in relevant_edges
        }

        new_length = {
            e: net.length[e]
            for e in relevant_edges
            if e in net.length
        }

        new_upgrade_cost = {
            e: net.upgrade_cost[e]
            for e in new_unsafe_edges
            if e in net.upgrade_cost
        }

        # --------------------------------------------------------
        # Keep all terminals and all vertices incident to relevant edges.
        # --------------------------------------------------------

        terminals = set()

        for s, t in terminal_pairs:

            terminals.add(
                s
            )

            terminals.add(
                t
            )

        incident_vertices = set()

        for u, v in relevant_edges:

            incident_vertices.add(
                u
            )

            incident_vertices.add(
                v
            )

        new_vertices = (
            incident_vertices
            |
            terminals
        )

        new_vertices = {
            v
            for v in new_vertices
            if v in net.vertices
        }

        removed_vertices = (
            set(net.vertices)
            -
            new_vertices
        )

        # --------------------------------------------------------
        # Update network.
        # --------------------------------------------------------

        net.vertices = new_vertices
        net.safe_edges = new_safe_edges
        net.unsafe_edges = new_unsafe_edges
        net.length = new_length
        net.upgrade_cost = new_upgrade_cost

        net.node_x = {
            v: net.node_x[v]
            for v in new_vertices
            if v in net.node_x
        }

        net.node_y = {
            v: net.node_y[v]
            for v in new_vertices
            if v in net.node_y
        }

        net.edge_geometry = {
            e: net.edge_geometry[e]
            for e in relevant_edges
            if e in net.edge_geometry
        }

        # --------------------------------------------------------
        # Sanity checks.
        # --------------------------------------------------------

        if net.safe_edges & net.unsafe_edges:

            raise RuntimeError(
                "AlphaRelevantEdgeReduction produced an edge that is "
                "both safe and unsafe"
            )

        for e in net.safe_edges | net.unsafe_edges:

            u, v = e

            if u not in net.vertices or v not in net.vertices:

                raise RuntimeError(
                    "AlphaRelevantEdgeReduction left an edge with "
                    f"missing endpoint: {e}"
                )

            if e not in net.length:

                raise RuntimeError(
                    f"AlphaRelevantEdgeReduction removed length for {e}"
                )

        for e in net.unsafe_edges:

            if e not in net.upgrade_cost:

                raise RuntimeError(
                    "AlphaRelevantEdgeReduction left unsafe edge "
                    f"without upgrade cost: {e}"
                )

        for s, t in terminal_pairs:

            if s not in net.vertices or t not in net.vertices:

                raise RuntimeError(
                    "AlphaRelevantEdgeReduction deleted a terminal "
                    f"of pair {(s, t)}"
                )

        # --------------------------------------------------------
        # Construct reduced instance.
        # --------------------------------------------------------

        reduced = BicycleInstance(
            network=net,
            terminal_pairs=terminal_pairs,
            alpha=instance.alpha,
            budget=instance.budget,
            info=(
                instance.info
                + " [alpha-relevant edges]"
            ),
        )

        reduced.alpha_relevant_removed_edges = removed_edges
        reduced.alpha_relevant_removed_vertices = removed_vertices

        return reduced
