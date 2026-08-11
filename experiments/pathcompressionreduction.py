from reductionrule import ReductionRule

import copy
import networkx as nx

from bicyclenetwork import BicycleNetwork
from bicycleinstance import BicycleInstance


class PathCompressionReduction(ReductionRule):

    name = "PathCompressionReduction"

    def apply(
        self,
        instance,
        feedback_edges,
    ):

        net = copy.deepcopy(
            instance.network
        )

        # ------------------------------------------------------------
        # Canonicalize and validate feedback edge set.
        # ------------------------------------------------------------

        feedback_edges = {
            BicycleNetwork.canonical_edge(
                u,
                v,
            )
            for u, v in feedback_edges
        }

        all_edges = (
            set(net.safe_edges)
            |
            set(net.unsafe_edges)
        )

        missing_feedback_edges = (
            feedback_edges
            -
            all_edges
        )

        if missing_feedback_edges:

            raise RuntimeError(
                "PathCompressionReduction received feedback edges "
                "that are not present in the current network: "
                f"{list(missing_feedback_edges)[:5]}"
            )

        # ------------------------------------------------------------
        # Original graph.
        # ------------------------------------------------------------

        G = net.to_networkx()

        # ------------------------------------------------------------
        # T = G - R.
        #
        # T must be a forest. If this is not true, then the feedback
        # edge set is stale, incomplete, or not computed for this graph.
        # ------------------------------------------------------------

        T = G.copy()

        T.remove_edges_from(
            feedback_edges
        )

        if not nx.is_forest(T):

            raise RuntimeError(
                "PathCompressionReduction expected T = G - R to be "
                "a forest, but T still contains a cycle. The feedback "
                "edge set may be stale or incomplete."
            )

        # ------------------------------------------------------------
        # Terminal vertices.
        # ------------------------------------------------------------

        terminals = set()

        for s, t in instance.terminal_pairs:

            terminals.add(
                s
            )

            terminals.add(
                t
            )

        # ------------------------------------------------------------
        # Feedback endpoints.
        # ------------------------------------------------------------

        feedback_endpoints = set()

        for u, v in feedback_edges:

            feedback_endpoints.add(
                u
            )

            feedback_endpoints.add(
                v
            )

        # ------------------------------------------------------------
        # Important vertices.
        #
        # Important vertices are:
        #   - terminals,
        #   - endpoints of feedback edges,
        #   - leaves of T,
        #   - branching vertices of T.
        # ------------------------------------------------------------

        important = set()

        for v in T.nodes():

            if v in terminals:

                important.add(
                    v
                )

            elif v in feedback_endpoints:

                important.add(
                    v
                )

            elif T.degree(v) == 1 or T.degree(v) >= 3:

                important.add(
                    v
                )

        # If nothing is important, nothing can be compressed in a
        # meaningful way. Return a fresh instance for clean bookkeeping.
        if not important:

            reduced = BicycleInstance(
                network=net,
                terminal_pairs=list(
                    instance.terminal_pairs
                ),
                alpha=instance.alpha,
                budget=instance.budget,
                info=(
                    instance.info
                    + " [path compressed]"
                ),
            )

            if hasattr(
                reduced,
                "clone_metadata_from",
            ):

                reduced.clone_metadata_from(
                    instance
                )

            reduced.path_compression_records = []

            return reduced

        # ------------------------------------------------------------
        # Root every connected component of T separately.
        #
        # We use deterministic roots to make the reduction reproducible.
        # ------------------------------------------------------------

        parent = {}
        depth = {}
        roots = set()

        components = sorted(
            nx.connected_components(T),
            key=lambda component: min(component),
        )

        for component in components:

            component = set(
                component
            )

            component_important = (
                component
                &
                important
            )

            if not component_important:

                continue

            root = min(
                component_important
            )

            roots.add(
                root
            )

            parent[root] = None
            depth[root] = 0

            bfs = [
                root
            ]

            for v in bfs:

                for w in sorted(T.neighbors(v)):

                    if w not in component:

                        continue

                    if w in parent:

                        continue

                    parent[w] = v

                    depth[w] = (
                        depth[v]
                        +
                        1
                    )

                    bfs.append(
                        w
                    )

        # ------------------------------------------------------------
        # New graph data.
        # ------------------------------------------------------------

        new_safe_edges = set()
        new_unsafe_edges = set()

        new_length = {}
        new_upgrade_cost = {}

        # ------------------------------------------------------------
        # Feedback edges survive unchanged, as well as edges directly
        # between important vertices.
        # ------------------------------------------------------------

        for e in sorted(net.safe_edges):

            if (
                e in feedback_edges
                or (
                    e[0] in important
                    and e[1] in important
                )
            ):

                if e not in net.length:

                    raise RuntimeError(
                        f"Missing length for safe edge {e}"
                    )

                new_safe_edges.add(
                    e
                )

                new_length[e] = net.length[e]

        for e in sorted(net.unsafe_edges):

            if (
                e in feedback_edges
                or (
                    e[0] in important
                    and e[1] in important
                )
            ):

                if e not in net.length:

                    raise RuntimeError(
                        f"Missing length for unsafe edge {e}"
                    )

                if e not in net.upgrade_cost:

                    raise RuntimeError(
                        f"Missing upgrade cost for unsafe edge {e}"
                    )

                new_unsafe_edges.add(
                    e
                )

                new_length[e] = net.length[e]

                new_upgrade_cost[e] = net.upgrade_cost[e]

        # ------------------------------------------------------------
        # Surviving vertices.
        # ------------------------------------------------------------

        new_vertices = set(
            important
        )

        if net.vertices:

            next_subdivision_vertex = (
                max(net.vertices)
                +
                1
            )

        else:

            next_subdivision_vertex = 0

        new_node_x = {}
        new_node_y = {}

        for v in sorted(important):

            if (
                hasattr(net, "node_x")
                and v in net.node_x
            ):

                new_node_x[v] = net.node_x[v]

            if (
                hasattr(net, "node_y")
                and v in net.node_y
            ):

                new_node_y[v] = net.node_y[v]

        # ------------------------------------------------------------
        # Helper for adding subdivision coordinates.
        # ------------------------------------------------------------

        def set_subdivision_coordinates(
            x,
            u,
            v,
        ):

            if (
                hasattr(net, "node_x")
                and u in net.node_x
                and v in net.node_x
            ):

                new_node_x[x] = (
                    net.node_x[u]
                    +
                    net.node_x[v]
                ) / 2.0

            if (
                hasattr(net, "node_y")
                and u in net.node_y
                and v in net.node_y
            ):

                new_node_y[x] = (
                    net.node_y[u]
                    +
                    net.node_y[v]
                ) / 2.0

        # ------------------------------------------------------------
        # Helper for adding one edge of a compressed path to the
        # accumulated length and cost.
        # ------------------------------------------------------------

        def add_original_edge_to_path(
            e,
        ):

            if e not in net.length:

                raise RuntimeError(
                    f"Compressed path uses edge without length: {e}"
                )

            edge_length = net.length[e]

            if edge_length <= 0.0:

                raise RuntimeError(
                    f"Compressed path uses non-positive length edge: "
                    f"{e}, length={edge_length}"
                )

            edge_cost = 0.0
            edge_is_unsafe = False

            if e in net.unsafe_edges:

                edge_is_unsafe = True

                if e not in net.upgrade_cost:

                    raise RuntimeError(
                        f"Compressed path uses unsafe edge without "
                        f"upgrade cost: {e}"
                    )

                edge_cost = net.upgrade_cost[e]

                if edge_cost < 0.0:

                    raise RuntimeError(
                        f"Compressed path uses negative upgrade cost: "
                        f"{e}, cost={edge_cost}"
                    )

            elif e not in net.safe_edges:

                raise RuntimeError(
                    f"Compressed path uses edge that is neither safe "
                    f"nor unsafe: {e}"
                )

            return edge_length, edge_cost, edge_is_unsafe

        # ------------------------------------------------------------
        # Audit records.
        # ------------------------------------------------------------

        compression_records = []

        # ------------------------------------------------------------
        # Pareto statistics for unsafe compressed paths parallel to an
        # existing unsafe edge.
        # ------------------------------------------------------------

        EPS = 1e-9

        pareto_unsafe_parallel_comparisons = 0
        pareto_existing_unsafe_edge_wins = 0
        pareto_compressed_unsafe_path_wins = 0

        # ------------------------------------------------------------
        # Process every important vertex except the root of its
        # connected component.
        #
        # For each important vertex v, walk upward in the BFS tree
        # until the next important ancestor is reached. The resulting
        # path is compressed.
        # ------------------------------------------------------------

        for v in sorted(important):

            if v in roots:

                continue

            if v not in parent:

                continue

            current = parent[v]

            if current is None or current in important:

                # Direct edge to an important ancestor is already kept.
                continue

            total_length = 0.0
            total_cost = 0.0
            unsafe_present = False

            path_vertices = [
                v
            ]

            path_edges = []

            prev = v

            while current not in important:

                e = BicycleNetwork.canonical_edge(
                    prev,
                    current,
                )

                edge_length, edge_cost, edge_is_unsafe = (
                    add_original_edge_to_path(
                        e
                    )
                )

                total_length += edge_length
                total_cost += edge_cost
                unsafe_present = (
                    unsafe_present
                    or edge_is_unsafe
                )

                path_edges.append(
                    e
                )

                path_vertices.append(
                    current
                )

                nxt = parent[current]

                prev = current
                current = nxt

                if current is None:

                    raise RuntimeError(
                        "PathCompressionReduction reached component "
                        "root before finding an important ancestor."
                    )

            # --------------------------------------------------------
            # Final edge to the important ancestor.
            # --------------------------------------------------------

            e = BicycleNetwork.canonical_edge(
                prev,
                current,
            )

            edge_length, edge_cost, edge_is_unsafe = (
                add_original_edge_to_path(
                    e
                )
            )

            total_length += edge_length
            total_cost += edge_cost
            unsafe_present = (
                unsafe_present
                or edge_is_unsafe
            )

            path_edges.append(
                e
            )

            path_vertices.append(
                current
            )

            if total_length <= 0.0:

                raise RuntimeError(
                    "PathCompressionReduction created a compressed "
                    f"path with non-positive length: {total_length}"
                )

            v0 = v
            v1 = current

            if v0 == v1:

                continue

            compressed_edge = BicycleNetwork.canonical_edge(
                v0,
                v1,
            )

            # Check against the already constructed graph, not only the
            # original graph. This prevents accidental overwrites.
            edge_exists_safe = (
                compressed_edge
                in new_safe_edges
            )

            edge_exists_unsafe = (
                compressed_edge
                in new_unsafe_edges
            )

            if edge_exists_safe and edge_exists_unsafe:

                raise RuntimeError(
                    "Compressed edge already exists as both safe and "
                    f"unsafe: {compressed_edge}"
                )

            case_name = None

            # --------------------------------------------------------
            # CASE 1: edge does not exist.
            # --------------------------------------------------------

            if not edge_exists_safe and not edge_exists_unsafe:

                new_length[compressed_edge] = total_length

                if unsafe_present:

                    new_unsafe_edges.add(
                        compressed_edge
                    )

                    new_upgrade_cost[compressed_edge] = total_cost

                    case_name = "new_unsafe_edge"

                else:

                    new_safe_edges.add(
                        compressed_edge
                    )

                    case_name = "new_safe_edge"

            # --------------------------------------------------------
            # CASE 2: existing safe edge.
            # --------------------------------------------------------

            elif edge_exists_safe:

                # Safe compressed path: keep the shorter safe edge.
                if not unsafe_present:

                    old_length = new_length[
                        compressed_edge
                    ]

                    new_length[compressed_edge] = min(
                        old_length,
                        total_length,
                    )

                    new_safe_edges.add(
                        compressed_edge
                    )

                    case_name = "merge_with_existing_safe_edge"

                # Unsafe compressed path parallel to an existing safe
                # edge: add a subdivision gadget.
                else:

                    x = next_subdivision_vertex
                    next_subdivision_vertex += 1

                    new_vertices.add(
                        x
                    )

                    set_subdivision_coordinates(
                        x,
                        v0,
                        v1,
                    )

                    e1 = BicycleNetwork.canonical_edge(
                        v0,
                        x,
                    )

                    e2 = BicycleNetwork.canonical_edge(
                        x,
                        v1,
                    )

                    if (
                        e1 in new_safe_edges
                        or e1 in new_unsafe_edges
                        or e2 in new_safe_edges
                        or e2 in new_unsafe_edges
                    ):

                        raise RuntimeError(
                            "Subdivision edge already exists. "
                            f"Subdivision vertex={x}"
                        )

                    new_unsafe_edges.add(
                        e1
                    )

                    new_safe_edges.add(
                        e2
                    )

                    new_length[e1] = (
                        total_length
                        /
                        2.0
                    )

                    new_length[e2] = (
                        total_length
                        /
                        2.0
                    )

                    new_upgrade_cost[e1] = total_cost

                    case_name = "parallel_unsafe_path_to_existing_safe_edge"

            # --------------------------------------------------------
            # CASE 3: existing unsafe edge.
            # --------------------------------------------------------

            else:

                old_length = new_length[
                    compressed_edge
                ]

                old_cost = new_upgrade_cost[
                    compressed_edge
                ]

                # ----------------------------------------------------
                # CASE 3a:
                # Unsafe compressed path parallel to an existing unsafe
                # edge.
                #
                # Compare the two alternatives:
                #
                #   existing unsafe edge:
                #       length = old_length
                #       cost   = old_cost
                #
                #   compressed unsafe path:
                #       length = total_length
                #       cost   = total_cost
                #
                # If one Pareto-dominates the other, keep only the
                # dominating alternative. If neither dominates, keep both
                # using the subdivision gadget.
                # ----------------------------------------------------

                if unsafe_present:

                    pareto_unsafe_parallel_comparisons += 1

                    existing_dominates = (
                        old_length <= total_length + EPS
                        and old_cost <= total_cost + EPS
                    )

                    compressed_dominates = (
                        total_length <= old_length + EPS
                        and total_cost <= old_cost + EPS
                    )

                    # ------------------------------------------------
                    # Existing unsafe edge is no longer and no more
                    # expensive. Keep it and discard the compressed path.
                    #
                    # If both alternatives are equal within tolerance, this
                    # branch keeps the already existing edge.
                    # ------------------------------------------------

                    if existing_dominates:

                        pareto_existing_unsafe_edge_wins += 1

                        case_name = (
                            "pareto_existing_unsafe_edge_dominates_"
                            "compressed_unsafe_path"
                        )

                    # ------------------------------------------------
                    # Compressed unsafe path is no longer and no more
                    # expensive. Replace the existing unsafe edge by the
                    # compressed alternative.
                    # ------------------------------------------------

                    elif compressed_dominates:

                        pareto_compressed_unsafe_path_wins += 1

                        new_length[
                            compressed_edge
                        ] = total_length

                        new_upgrade_cost[
                            compressed_edge
                        ] = total_cost

                        case_name = (
                            "pareto_compressed_unsafe_path_dominates_"
                            "existing_unsafe_edge"
                        )

                    # ------------------------------------------------
                    # Incomparable alternatives:
                    # keep both by adding a subdivision gadget.
                    # ------------------------------------------------

                    else:

                        x = next_subdivision_vertex
                        next_subdivision_vertex += 1

                        new_vertices.add(
                            x
                        )

                        set_subdivision_coordinates(
                            x,
                            v0,
                            v1,
                        )

                        e1 = BicycleNetwork.canonical_edge(
                            v0,
                            x,
                        )

                        e2 = BicycleNetwork.canonical_edge(
                            x,
                            v1,
                        )

                        if (
                            e1 in new_safe_edges
                            or e1 in new_unsafe_edges
                            or e2 in new_safe_edges
                            or e2 in new_unsafe_edges
                        ):

                            raise RuntimeError(
                                "Subdivision edge already exists. "
                                f"Subdivision vertex={x}"
                            )

                        new_unsafe_edges.add(
                            e1
                        )

                        new_safe_edges.add(
                            e2
                        )

                        new_length[e1] = (
                            total_length
                            /
                            2.0
                        )

                        new_length[e2] = (
                            total_length
                            /
                            2.0
                        )

                        new_upgrade_cost[e1] = total_cost

                        case_name = (
                            "parallel_unsafe_path_to_existing_unsafe_edge"
                        )

                # ----------------------------------------------------
                # CASE 3b:
                # Safe compressed path parallel to an existing unsafe edge.
                # Keep both as before.
                # ----------------------------------------------------

                else:

                    x = next_subdivision_vertex
                    next_subdivision_vertex += 1

                    new_vertices.add(
                        x
                    )

                    set_subdivision_coordinates(
                        x,
                        v0,
                        v1,
                    )

                    e1 = BicycleNetwork.canonical_edge(
                        v0,
                        x,
                    )

                    e2 = BicycleNetwork.canonical_edge(
                        x,
                        v1,
                    )

                    if (
                        e1 in new_safe_edges
                        or e1 in new_unsafe_edges
                        or e2 in new_safe_edges
                        or e2 in new_unsafe_edges
                    ):

                        raise RuntimeError(
                            "Subdivision edge already exists. "
                            f"Subdivision vertex={x}"
                        )

                    new_safe_edges.add(
                        e1
                    )

                    new_safe_edges.add(
                        e2
                    )

                    new_length[e1] = (
                        total_length
                        /
                        2.0
                    )

                    new_length[e2] = (
                        total_length
                        /
                        2.0
                    )

                    case_name = (
                        "parallel_safe_path_to_existing_unsafe_edge"
                    )

            compression_records.append(
                {
                    "endpoints": (
                        v0,
                        v1,
                    ),
                    "compressed_edge": compressed_edge,
                    "path_vertices": list(
                        path_vertices
                    ),
                    "path_edges": list(
                        path_edges
                    ),
                    "total_length": total_length,
                    "unsafe_present": unsafe_present,
                    "total_cost": total_cost,
                    "case": case_name,
                }
            )

        # ------------------------------------------------------------
        # Final consistency checks before updating the network.
        # ------------------------------------------------------------

        if new_safe_edges & new_unsafe_edges:

            overlap = (
                new_safe_edges
                &
                new_unsafe_edges
            )

            raise RuntimeError(
                "PathCompressionReduction produced edges that are both "
                f"safe and unsafe: {list(overlap)[:5]}"
            )

        all_new_edges = (
            new_safe_edges
            |
            new_unsafe_edges
        )

        for e in all_new_edges:

            u, v = e

            if u not in new_vertices or v not in new_vertices:

                raise RuntimeError(
                    "PathCompressionReduction produced an edge with "
                    f"missing endpoint: {e}"
                )

            if e not in new_length:

                raise RuntimeError(
                    "PathCompressionReduction produced an edge without "
                    f"length: {e}"
                )

            if new_length[e] <= 0.0:

                raise RuntimeError(
                    "PathCompressionReduction produced non-positive "
                    f"edge length: {e}, length={new_length[e]}"
                )

        for e in new_unsafe_edges:

            if e not in new_upgrade_cost:

                raise RuntimeError(
                    "PathCompressionReduction produced unsafe edge "
                    f"without upgrade cost: {e}"
                )

            if new_upgrade_cost[e] < 0.0:

                raise RuntimeError(
                    "PathCompressionReduction produced negative upgrade "
                    f"cost: {e}, cost={new_upgrade_cost[e]}"
                )

        for s, t in instance.terminal_pairs:

            if s not in new_vertices or t not in new_vertices:

                raise RuntimeError(
                    "PathCompressionReduction deleted a terminal vertex "
                    f"of pair {(s, t)}"
                )

        # ------------------------------------------------------------
        # Update network.
        # ------------------------------------------------------------

        net.vertices = new_vertices
        net.safe_edges = new_safe_edges
        net.unsafe_edges = new_unsafe_edges
        net.length = new_length
        net.upgrade_cost = new_upgrade_cost

        net.node_x = new_node_x
        net.node_y = new_node_y

        # Geometries are no longer valid after compression.
        net.edge_geometry = {}

        # ------------------------------------------------------------
        # Construct reduced instance.
        # ------------------------------------------------------------

        reduced = BicycleInstance(
            network=net,
            terminal_pairs=list(
                instance.terminal_pairs
            ),
            alpha=instance.alpha,
            budget=instance.budget,
            info=(
                instance.info
                + " [path compressed]"
            ),
        )

        if hasattr(
            reduced,
            "clone_metadata_from",
        ):

            reduced.clone_metadata_from(
                instance
            )

        reduced.path_compression_records = compression_records

        reduced.path_compression_pareto_stats = {
            "unsafe_parallel_comparisons": pareto_unsafe_parallel_comparisons,
            "existing_unsafe_edge_wins": pareto_existing_unsafe_edge_wins,
            "compressed_unsafe_path_wins": pareto_compressed_unsafe_path_wins,
        }

        print(
            f"PCR comparisons: {pareto_unsafe_parallel_comparisons} | edge<path: {pareto_existing_unsafe_edge_wins} | path<edge: {pareto_compressed_unsafe_path_wins} "
        )

        return reduced
