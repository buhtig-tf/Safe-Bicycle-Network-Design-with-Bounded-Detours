import time
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from experimentresult import ExperimentResult
from bicyclenetwork import BicycleNetwork


class ILPSolver:
    """
    Exact solver for the BicycleNetwork instance
    using Gurobi MILP.

    Measures runtime and stores solution.
    """

    def __init__(
            self,
            instance,
            timeout: int | None = None,
            treewidth_dp_cuts=False,
            treewidth_dp_cuts_setup=1,
            #
            tw_dp_max_bag_size=14,
            tw_dp_max_candidate_sides = 5000,
            tw_dp_max_cuts_per_pair = 20,
            tw_dp_max_cut_size = 10,
        ):
        self.instance = instance
        self.model = None
        self.buildtime = None
        self.runtime = None
        self.solution = None
        self.timeout = timeout

        self.treewidth_dp_cuts = treewidth_dp_cuts
        self.num_treewidth_dp_cuts = 0
        self.treewidth_dp_cut_sizes = []

        if treewidth_dp_cuts_setup == 1:
            self.tw_dp_max_cut_size = tw_dp_max_cut_size
            self.tw_dp_max_cuts_per_pair = tw_dp_max_cuts_per_pair
            self.tw_dp_max_bag_size = tw_dp_max_bag_size
            self.tw_dp_max_candidate_sides = tw_dp_max_candidate_sides
        elif treewidth_dp_cuts_setup == 2:
            self.tw_dp_max_cut_size = 2 * tw_dp_max_cut_size
            self.tw_dp_max_cuts_per_pair = 2 * tw_dp_max_cuts_per_pair
            self.tw_dp_max_bag_size = 2 * tw_dp_max_bag_size
            self.tw_dp_max_candidate_sides = 2 * tw_dp_max_candidate_sides
        elif treewidth_dp_cuts_setup == 3:
            self.tw_dp_max_cut_size = int(round(0.5 * tw_dp_max_cut_size))
            self.tw_dp_max_cuts_per_pair = int(round(0.5 * tw_dp_max_cuts_per_pair))
            self.tw_dp_max_bag_size = int(round(0.5 * tw_dp_max_bag_size))
            self.tw_dp_max_candidate_sides = int(round(0.5 * tw_dp_max_candidate_sides))
        else:
            self.tw_dp_max_cut_size = tw_dp_max_cut_size
            self.tw_dp_max_cuts_per_pair = tw_dp_max_cuts_per_pair
            self.tw_dp_max_bag_size = tw_dp_max_bag_size
            self.tw_dp_max_candidate_sides = tw_dp_max_candidate_sides


        self.num_treewidth_dp_candidate_sides = 0

        # can be dropped after monitoring
        self.treewidth_dp_candidate_time = 0.0
        self.treewidth_dp_test_time = 0.0


    # ------------------------------------------------------------
    # HELPER TREEWIDTH DP I
    # ------------------------------------------------------------
    def _compute_tree_decomposition(
        self,
        H,
    ):
        """
        Compute an approximate tree decomposition of H and return it as
        a NetworkX graph whose nodes are frozenset bags.
        """

        if len(H.nodes()) == 0:

            return None

        if len(H.nodes()) == 1:

            only = next(
                iter(H.nodes())
            )

            TD = nx.Graph()

            TD.add_node(
                frozenset(
                    {only}
                )
            )

            return TD

        try:

            _, decomposition = nx.approximation.treewidth_min_fill_in(
                H
            )

        except Exception:

            _, decomposition = nx.approximation.treewidth_min_degree(
                H
            )

        TD = nx.Graph()

        for bag in decomposition.nodes():

            TD.add_node(
                frozenset(
                    bag
                )
            )

        for a, b in decomposition.edges():

            TD.add_edge(
                frozenset(
                    a
                ),
                frozenset(
                    b
                ),
            )

        if len(TD.nodes()) == 0:

            return None

        return TD

    def _tree_decomposition_bag_partition_sides(
        self,
        G,
        max_bag_size=14,
        max_candidate_sides=10000,
    ):
        """
        Generate candidate vertex sides from one global tree decomposition
        by partitioning adhesions of decomposition-tree edges.

        Despite the historical name, this now uses adhesions
        A_ij = B_i ∩ B_j rather than full bag partitions.

        For every tree-decomposition edge ij, removing ij splits the
        decomposition tree into two sides. Let A be the adhesion. For every
        S subset A, generate

            U = (vertices on one TD side outside A) union S.

        The resulting candidate sides are later tested against each
        pair-specific admissible graph.
        """

        candidate_sides = set()

        if len(G.nodes()) == 0:

            return []

        net = self.instance.network

        # ------------------------------------------------------------
        # Helper: rank sides by small boundary and high unsafe fraction.
        # ------------------------------------------------------------

        def boundary_score(U):

            U = set(
                U
            )

            crossing = set()

            for u in U:

                if u not in G:

                    continue

                for v in G.neighbors(
                    u
                ):

                    if v in U:

                        continue

                    e = BicycleNetwork.canonical_edge(
                        u,
                        v,
                    )

                    crossing.add(
                        e
                    )

            unsafe = sum(
                1
                for e in crossing
                if e in net.unsafe_edges
            )

            if crossing:

                unsafe_fraction = (
                    unsafe
                    /
                    len(crossing)
                )

            else:

                unsafe_fraction = 0.0

            return (
                len(crossing),
                -unsafe_fraction,
                len(U),
                tuple(
                    sorted(U)
                ),
            )

        # ------------------------------------------------------------
        # Work component-wise.
        # ------------------------------------------------------------

        connected_components = sorted(
            nx.connected_components(G),
            key=lambda C: min(C),
        )

        for component in connected_components:

            component = set(
                component
            )

            if len(component) <= 1:

                continue

            H = G.subgraph(
                component
            ).copy()

            TD = self._compute_tree_decomposition(
                H
            )

            if TD is None:

                continue

            component_vertices = set(
                H.nodes()
            )

            if len(TD.nodes()) <= 1:

                continue

            # --------------------------------------------------------
            # For each decomposition-tree edge, partition its adhesion.
            # --------------------------------------------------------

            for bag_a, bag_b in sorted(
                TD.edges(),
                key=lambda e: (
                    len(e[0] & e[1]),
                    tuple(sorted(e[0])),
                    tuple(sorted(e[1])),
                ),
            ):

                adhesion = frozenset(
                    set(bag_a)
                    &
                    set(bag_b)
                )

                adhesion_vertices = tuple(
                    sorted(
                        adhesion
                    )
                )

                adhesion_size = len(
                    adhesion_vertices
                )

                if (
                    max_bag_size is not None
                    and adhesion_size > max_bag_size
                ):

                    continue

                # ----------------------------------------------------
                # Split the decomposition tree at this edge.
                # ----------------------------------------------------

                TD_without_edge = TD.copy()

                TD_without_edge.remove_edge(
                    bag_a,
                    bag_b,
                )

                comp_a = nx.node_connected_component(
                    TD_without_edge,
                    bag_a,
                )

                comp_b = nx.node_connected_component(
                    TD_without_edge,
                    bag_b,
                )

                vertices_a = set()
                vertices_b = set()

                for bag in comp_a:

                    vertices_a.update(
                        bag
                    )

                for bag in comp_b:

                    vertices_b.update(
                        bag
                    )

                exclusive_a = (
                    vertices_a
                    -
                    set(adhesion)
                )

                exclusive_b = (
                    vertices_b
                    -
                    set(adhesion)
                )

                # Both exclusive sides empty is useless.
                if not exclusive_a and not exclusive_b:

                    continue

                # ----------------------------------------------------
                # Enumerate S subset adhesion.
                #
                # We generate only one orientation, using exclusive_a.
                # The complement orientation is deduplicated by normalizing
                # U versus component_vertices - U.
                # ----------------------------------------------------

                num_masks = (
                    1
                    <<
                    adhesion_size
                )

                for mask in range(
                    num_masks
                ):

                    S = {
                        adhesion_vertices[i]
                        for i in range(
                            adhesion_size
                        )
                        if (
                            mask
                            >>
                            i
                        )
                        &
                        1
                    }

                    side = set(
                        exclusive_a
                    )

                    side.update(
                        S
                    )

                    if not side:

                        continue

                    if side == component_vertices:

                        continue

                    side = frozenset(
                        side
                    )

                    complement = frozenset(
                        component_vertices
                        -
                        set(side)
                    )

                    if not complement:

                        continue

                    # Normalize U and V\U, since they define the same cut.
                    if (
                        len(complement) < len(side)
                        or (
                            len(complement) == len(side)
                            and tuple(sorted(complement))
                            < tuple(sorted(side))
                        )
                    ):

                        side = complement

                    if not side:

                        continue

                    if len(side) == len(component_vertices):

                        continue

                    candidate_sides.add(
                        side
                    )

        # ------------------------------------------------------------
        # Rank and cap after generation.
        # ------------------------------------------------------------

        candidate_sides = sorted(
            candidate_sides,
            key=boundary_score,
        )

        if max_candidate_sides is not None:

            candidate_sides = candidate_sides[
                :max_candidate_sides
            ]

        return candidate_sides

    # ------------------------------------------------------------
    # HELPER TREEWIDTH DP III
    # ------------------------------------------------------------
    def _add_treewidth_dp_cuts(
        self,
        m,
        net,
        feasible_arcs,
        arc_edge,
        s,
        t,
        x,
        candidate_sides,
        seen_cuts,
        max_cut_size=10,
        max_cuts_per_pair=50,
    ):
        """
        Add unsafe-only s-t cuts found from global tree-decomposition
        bag-partition candidate sides.

        Candidate sides U are generated once from the full graph.

        For the current pair, we test each U against the pair-specific
        alpha-admissible graph H_k induced by feasible_arcs.

        If exactly one of s,t lies in U and all crossing edges in H_k
        are unsafe, we add

            sum_{e in delta_Hk(U)} x_e >= 1.
        """

        if not feasible_arcs:

            return 0, []

        if not candidate_sides:

            return 0, []

        # ------------------------------------------------------------
        # Pair-specific admissible undirected edge set.
        # ------------------------------------------------------------

        admissible_edges = set()
        admissible_nodes = set()

        for a in feasible_arcs:

            e = arc_edge[a]

            admissible_edges.add(
                e
            )

            u, v = e

            admissible_nodes.add(
                u
            )

            admissible_nodes.add(
                v
            )

        admissible_nodes.add(
            s
        )

        admissible_nodes.add(
            t
        )

        if s not in admissible_nodes or t not in admissible_nodes:

            return 0, []

        added = 0
        cut_sizes = []

        # ------------------------------------------------------------
        # Test candidate sides.
        # ------------------------------------------------------------

        for raw_side in candidate_sides:

            if (
                max_cuts_per_pair is not None
                and added >= max_cuts_per_pair
            ):

                break

            # Restrict side to the admissible graph of this pair.
            side = (
                set(raw_side)
                &
                admissible_nodes
            )

            if not side:

                continue

            s_in = (
                s
                in side
            )

            t_in = (
                t
                in side
            )

            # Need exactly one terminal on the side.
            if s_in == t_in:

                continue

            crossing_edges = set()

            for e in admissible_edges:

                u, v = e

                if (
                    u in side
                ) != (
                    v in side
                ):

                    crossing_edges.add(
                        e
                    )

            if not crossing_edges:

                continue

            if (
                max_cut_size is not None
                and len(crossing_edges) > max_cut_size
            ):

                continue

            # --------------------------------------------------------
            # Validity condition:
            # crossing edges in the pair-specific admissible graph must
            # be unsafe-only.
            # --------------------------------------------------------

            has_safe_crossing = any(
                e in net.safe_edges
                for e in crossing_edges
            )

            if has_safe_crossing:

                continue

            unsafe_crossing_edges = {
                e
                for e in crossing_edges
                if e in net.unsafe_edges
            }

            if len(unsafe_crossing_edges) != len(crossing_edges):

                continue

            if not unsafe_crossing_edges:

                continue

            key = frozenset(
                unsafe_crossing_edges
            )

            if key in seen_cuts:

                continue

            if any(
                e not in x
                for e in key
            ):

                continue

            seen_cuts.add(
                key
            )

            m.addConstr(
                gp.quicksum(
                    x[e]
                    for e in key
                )
                >= 1
            )

            added += 1

            cut_sizes.append(
                len(key)
            )

        return added, cut_sizes

    # ------------------------------------------------------------
    # Build ILP (with arc pruning)
    # ------------------------------------------------------------
    def build_model(self):

        from collections import defaultdict

        inst = self.instance
        net = inst.network

        G = net.to_networkx()

        m = gp.Model("bicycle_ilp")
        m.setParam("OutputFlag", False)

        if self.timeout is not None:
            m.setParam(
                "TimeLimit",
                self.timeout,
            )

        EPS = 1e-9

        # --------------------------------------------------------
        # Variables: upgrade decision
        # --------------------------------------------------------

        x = {}

        for e in net.unsafe_edges:

            x[e] = m.addVar(
                vtype=GRB.BINARY,
            )

        # --------------------------------------------------------
        # Objective: minimize total upgrade cost
        # --------------------------------------------------------

        upgrade_expr = gp.quicksum(
            net.upgrade_cost[e] * x[e]
            for e in net.unsafe_edges
        )

        m.setObjective(
            upgrade_expr,
            GRB.MINIMIZE,
        )

        m.ObjCon = float(
            getattr(
                self.instance,
                "total_budget_delta",
                0.0,
            )
        )

        # --------------------------------------------------------
        # Directed arc set and cached edge data
        # --------------------------------------------------------

        directed_edges = []
        arc_edge = {}
        arc_length = {}

        for u, v in G.edges():

            e = net.canonical_edge(u, v)

            for a in [
                (u, v),
                (v, u),
            ]:

                directed_edges.append(a)

                arc_edge[a] = e

                arc_length[a] = net.length[e]


        treewidth_dp_candidate_sides = []

        if self.treewidth_dp_cuts:

            before = time.perf_counter()

            treewidth_dp_candidate_sides = (
                self._tree_decomposition_bag_partition_sides(
                    G,
                    max_bag_size=self.tw_dp_max_bag_size,
                    max_candidate_sides=self.tw_dp_max_candidate_sides,
                )
            )

            after = time.perf_counter()

            self.treewidth_dp_candidate_time = after - before

            self.num_treewidth_dp_candidate_sides = len(
                treewidth_dp_candidate_sides
            )

            print(
                "treewidth DP candidate sides:",
                len(treewidth_dp_candidate_sides),
            )

        # --------------------------------------------------------
        # Terminal pair constraints
        # --------------------------------------------------------

        total_directed_arcs = len(directed_edges)

        # --------------------------------------------------------
        # Dijkstra cache
        # --------------------------------------------------------

        distance_cache = {}

        def distances_from(source):

            if source not in distance_cache:

                distance_cache[source] = nx.single_source_dijkstra_path_length(
                    G,
                    source,
                    weight="length",
                )

            return distance_cache[source]

        treewidth_dp_cut_seen = set()
        num_treewidth_dp_cuts = 0
        treewidth_dp_cut_sizes = []

        for pair_id, (s, t) in enumerate(inst.terminal_pairs):

            if s not in G or t not in G:
                continue

            # ----------------------------------------------------
            # Base distance in full graph
            # ----------------------------------------------------

            try:

                dist_s = distances_from(s)

            except nx.NetworkXNoPath:

                continue

            if t not in dist_s:
                continue

            base_dist = dist_s[t]

            threshold = inst.alpha * base_dist

            # ----------------------------------------------------
            # Distances to target for arc pruning
            # ----------------------------------------------------

            dist_t = distances_from(t)

            # ----------------------------------------------------
            # Keep only arcs that can be part of some
            # s-t path of length at most threshold.
            #
            # Condition:
            #   dist(s,u) + length(u,v) + dist(v,t) <= threshold
            # ----------------------------------------------------

            feasible_arcs = []

            for u, v in directed_edges:

                if u not in dist_s:
                    continue

                if v not in dist_t:
                    continue

                if (
                    dist_s[u]
                    + arc_length[(u, v)]
                    + dist_t[v]
                    <= threshold + EPS
                ):

                    feasible_arcs.append(
                        (u, v)
                    )


            if not feasible_arcs:

                # This should not happen if base_dist exists,
                # because the shortest path itself should survive
                # the pruning. If it happens, make the model infeasible.
                m.addConstr(
                    0 == 1,
                    name=f"infeasible_pair_{pair_id}",
                )

                continue

            if self.treewidth_dp_cuts:

                before = time.perf_counter()

                added_cuts, new_cut_sizes = self._add_treewidth_dp_cuts(
                    m=m,
                    net=net,
                    feasible_arcs=feasible_arcs,
                    arc_edge=arc_edge,
                    s=s,
                    t=t,
                    x=x,
                    candidate_sides=treewidth_dp_candidate_sides,
                    seen_cuts=treewidth_dp_cut_seen,
                    max_cut_size=self.tw_dp_max_cut_size,
                    max_cuts_per_pair=self.tw_dp_max_cuts_per_pair,
                )

                after = time.perf_counter()

                self.treewidth_dp_test_time += after - before

                num_treewidth_dp_cuts += added_cuts

                treewidth_dp_cut_sizes.extend(
                    new_cut_sizes
                )

            # ----------------------------------------------------
            # Flow variables for this terminal pair
            # ----------------------------------------------------

            f = m.addVars(
                feasible_arcs,
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
            )

            # ----------------------------------------------------
            # Precompute in/out arcs for flow conservation
            # ----------------------------------------------------

            in_arcs = defaultdict(list)
            out_arcs = defaultdict(list)
            relevant_nodes = set()

            for u, v in feasible_arcs:

                out_arcs[u].append(
                    (u, v)
                )

                in_arcs[v].append(
                    (u, v)
                )

                relevant_nodes.add(u)
                relevant_nodes.add(v)

            relevant_nodes.add(s)
            relevant_nodes.add(t)

            # ----------------------------------------------------
            # Flow conservation
            # ----------------------------------------------------

            for v in relevant_nodes:

                inflow = gp.quicksum(
                    f[a]
                    for a in in_arcs[v]
                )

                outflow = gp.quicksum(
                    f[a]
                    for a in out_arcs[v]
                )

                if v == s:

                    m.addConstr(
                        outflow - inflow == 1
                    )

                elif v == t:

                    m.addConstr(
                        inflow - outflow == 1
                    )

                else:

                    m.addConstr(
                        inflow == outflow
                    )

            # ----------------------------------------------------
            # Activation constraints
            #
            # One constraint per unsafe UNDIRECTED edge.
            #
            # Instead of:
            #     f[u,v] <= x[e]
            #     f[v,u] <= x[e]
            #
            # use:
            #     f[u,v] + f[v,u] <= x[e]
            #
            # This is exact for one-unit s-t flow and usually gives
            # fewer constraints and a slightly stronger relaxation.
            # ----------------------------------------------------

            unsafe_edge_arcs = defaultdict(list)

            for a in feasible_arcs:

                e = arc_edge[a]

                if e in net.unsafe_edges:

                    unsafe_edge_arcs[e].append(a)

            for e, arcs in unsafe_edge_arcs.items():

                m.addConstr(
                    gp.quicksum(
                        f[a]
                        for a in arcs
                    )
                    <= x[e]
                )

            # ----------------------------------------------------
            # Length constraint
            # ----------------------------------------------------

            m.addConstr(
                gp.quicksum(
                    arc_length[a] * f[a]
                    for a in feasible_arcs
                )
                <= threshold
            )


        m.update()

        self.num_treewidth_dp_cuts = num_treewidth_dp_cuts
        self.treewidth_dp_cut_sizes = treewidth_dp_cut_sizes

        print(f"treewidth DP candidate sides: {self.num_treewidth_dp_candidate_sides} | cuts: {num_treewidth_dp_cuts}")

        if treewidth_dp_cut_sizes:

            print(f"TWDC setup --- Bag size:{self.tw_dp_max_bag_size} | Cand. Sides:{self.tw_dp_max_candidate_sides} | Cuts pP:{self.tw_dp_max_cuts_per_pair} | Cut Size:{self.tw_dp_max_cut_size}")

            print(f"treewidth DP cut size | avg: {sum(treewidth_dp_cut_sizes) / len(treewidth_dp_cut_sizes)} | min: {min(treewidth_dp_cut_sizes)} | max: {max(treewidth_dp_cut_sizes)}")

            print("treewidth DP candidate generation time:", self.treewidth_dp_candidate_time)
            print("treewidth DP pair-test time:", self.treewidth_dp_test_time)

        print("directed arcs total:", total_directed_arcs)

        print(f"vars: {m.NumVars} | constrs: {m.NumConstrs}")
        # --------------------------------------------------------
        # Store model
        # --------------------------------------------------------

        self.model = m
        self.x = x

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------
    def solve(self):

        # ------------------------------------------------------------
        # Build model.
        # ------------------------------------------------------------

        build_start = time.perf_counter()

        if self.model is None:

            self.build_model()

        # Important: include pending Gurobi model updates in build time.
        self.model.update()

        build_end = time.perf_counter()

        self.buildtime = (
            build_end
            -
            build_start
        )


        optimize_start = time.perf_counter()

        self.model.optimize()

        optimize_end = time.perf_counter()

        self.runtime = (
            optimize_end
            -
            optimize_start
        )


        #print("GUROBI runtime:", self.model.Runtime)
        #print("GUROBI work:", getattr(self.model, "Work", None))
        #print("nodes:", self.model.NodeCount)
        #print("iterations:", self.model.IterCount)
        #print("barrier iterations:", self.model.BarIterCount)
        #print("objective:", self.model.ObjVal if self.model.SolCount > 0 else None)
        #print("best bound:", self.model.ObjBound)
        #print("gap:", self.model.MIPGap)

        if self.model.SolCount > 0:

            self.solution = {
                e: self.x[e].X
                for e in self.x
            }

        else:

            self.solution = {}

        selected_edges = [
            e
            for e, val in self.solution.items()
            if val > 0.5
        ]

        result = ExperimentResult(

            instance_filename=
                self.instance.filename(),

            solver="ilp",

            objective=(
                self.model.ObjVal
                if self.model.SolCount > 0
                else None
            ),

            optimal=(
                self.model.Status == GRB.OPTIMAL
            ),

            feasible=(
                self.model.SolCount > 0
            ),

            status=str(self.model.Status),

            selected_edges=selected_edges,

            solver_runtime=self.runtime,

            solver_buildtime = self.buildtime,

            total_runtime=(
                self.buildtime
                +
                self.runtime
            ),
        )

        if self.model.Status == GRB.OPTIMAL:

            result.status = "OPTIMAL"

        elif self.model.Status == GRB.TIME_LIMIT:

            result.status = "TIME_LIMIT"

        else:

            result.status = str(
                self.model.Status
            )

        result.optimal = (
            self.model.Status == GRB.OPTIMAL
        )

        result.termination_reason = (
            result.status
        )

        self.result = result

        return result

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    def summary(self):

        print("\n" + "=" * 50)
        print("ILP SOLVER RESULT")
        print("=" * 50)

        print(f"Runtime: {self.runtime:.4f}s")
        print(f"Objective (min budget): {self.model.ObjVal if self.model else None}")

        print("=" * 50)


    def solution_set(self):
        return [
                e for e, val in self.solution.items()
                if val > 0.5
            ]
