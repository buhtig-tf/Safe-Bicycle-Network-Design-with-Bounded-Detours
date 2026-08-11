import copy
import networkx as nx
from reductionrule import ReductionRule
from bicycleinstance import BicycleInstance

class LeafPruningReduction(ReductionRule):
    """
    Let R be a feedback edge set.

    Consider T = G - R.

    Recursively delete leaves of T that

        - are not terminals
        - are not endpoints of edges in R

    The corresponding vertices are removed
    from the original network.
    """

    name = "LeafPruningReduction"

    def apply(
        self,
        instance,
        feedback_edges,
    ):

        net = copy.deepcopy(
            instance.network
        )

        all_edges = (
            set(net.safe_edges)
            |
            set(net.unsafe_edges)
        )

        feedback_edges = {
            net.canonical_edge(
                u,
                v,
            )
            for u, v in feedback_edges
        }

        missing_feedback_edges = (
            feedback_edges
            -
            all_edges
        )

        if missing_feedback_edges:

            raise RuntimeError(
                "LeafPruningReduction received feedback edges "
                "that are not present in the current network: "
                f"{list(missing_feedback_edges)[:5]}"
            )

        G = net.to_networkx()

        T = G.copy()

        T.remove_edges_from(
            feedback_edges
        )

        #
        # Terminal vertices.
        #
        terminals = set()

        for s, t in instance.terminal_pairs:

            terminals.add(s)
            terminals.add(t)

        #
        # Endpoints of feedback edges.
        #
        feedback_endpoints = set()

        for u, v in feedback_edges:

            feedback_endpoints.add(u)
            feedback_endpoints.add(v)

        #
        # Recursive leaf pruning.
        #
        changed = True

        deleted_vertices = set()

        while changed:

            changed = False

            leaves = [
                v
                for v in T.nodes()
                if T.degree(v) <= 1
            ]

            for v in leaves:

                if v in terminals:
                    continue

                if v in feedback_endpoints:
                    continue

                T.remove_node(v)

                deleted_vertices.add(v)

                changed = True

        #
        # Remove these vertices
        # from the original graph.
        #
        for v in deleted_vertices:

            if v not in net.vertices:
                continue

            net.vertices.remove(v)

            net.node_x.pop(v, None)
            net.node_y.pop(v, None)

        #
        # Remove incident edges.
        #
        safe_remove = set()

        for e in net.safe_edges:

            u, v = e

            if (
                u in deleted_vertices
                or
                v in deleted_vertices
            ):
                safe_remove.add(e)

        unsafe_remove = set()

        for e in net.unsafe_edges:

            u, v = e

            if (
                u in deleted_vertices
                or
                v in deleted_vertices
            ):
                unsafe_remove.add(e)

        net.safe_edges -= safe_remove
        net.unsafe_edges -= unsafe_remove

        #
        # Remove edge data.
        #
        #for e in safe_remove | unsafe_remove:

            #net.length.pop(e, None)

            #net.upgrade_cost.pop(
                #e,
                #None,
            #)

            #net.edge_geometry.pop(
                #e,
                #None,
            #)

        for e in safe_remove | unsafe_remove:

            reverse_e = (
                e[1],
                e[0],
            )

            net.length.pop(e, None)
            net.length.pop(reverse_e, None)

            net.upgrade_cost.pop(e, None)
            net.upgrade_cost.pop(reverse_e, None)

            net.edge_geometry.pop(e, None)
            net.edge_geometry.pop(reverse_e, None)

        #
        # Construct reduced instance.
        #
        reduced = BicycleInstance(
            network=net,
            terminal_pairs=list(
                instance.terminal_pairs
            ),
            alpha=instance.alpha,
            budget=instance.budget,
            info=(
                instance.info
                + " [leaf-pruned]"
            ),
        )

        return reduced
