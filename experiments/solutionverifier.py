import networkx as nx


class SolutionVerifier:
    """
    General verifier for BicycleNetwork solutions.

    A solution is simply:
        upgraded_edges = set of edges that become safe

    Checks detour feasibility for all terminal pairs:
        dist_upgraded(s,t) <= alpha * dist_original(s,t)
    """

    def __init__(self, instance, upgraded_edges):

        self.instance = instance
        self.upgraded_edges = set(
            upgraded_edges
        )

        invalid_edges = (
            self.upgraded_edges
            -
            self.instance.network.unsafe_edges
        )

        if invalid_edges:

            raise ValueError(
                "Solution contains edges that are not unsafe "
                "edges of the instance: "
                f"{sorted(invalid_edges)}"
            )

        self.errors = []

    # ------------------------------------------------------------
    # Build upgraded graph
    # ------------------------------------------------------------

    def _build_upgraded_graph(self):

        return self.instance.network.upgraded_graph(
            self.upgraded_edges
        )

    # ------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------

    def check(self, verbose=True):

        self.errors = []

        inst = self.instance
        net = inst.network

        G = net.to_networkx()
        G_up = self._build_upgraded_graph()

        ok = True

        for (s, t) in inst.terminal_pairs:

            try:
                base_dist = nx.shortest_path_length(
                    G, s, t, weight="length"
                )
            except nx.NetworkXNoPath:
                self.errors.append((s, t, "no base path"))
                ok = False
                continue

            try:
                new_dist = nx.shortest_path_length(
                    G_up, s, t, weight="length"
                )
            except nx.NetworkXNoPath:
                self.errors.append((s, t, "no upgraded path"))
                ok = False
                continue

            threshold = inst.alpha * base_dist

            if new_dist > threshold:

                ok = False
                self.errors.append(
                    {
                        "s": s,
                        "t": t,
                        "base": base_dist,
                        "upgraded": new_dist,
                        "threshold": threshold,
                    }
                )

        if verbose:

            print("\n" + "=" * 60)
            #print("SOLUTION VERIFICATION")
            #print("=" * 60)

            if ok:
                print("SOLUTION VERIFICATION: ✔ FEASIBLE SOLUTION")
            else:
                print("SOLUTION VERIFICATION ✘ INFEASIBLE SOLUTION")
                print(f"Violations: {len(self.errors)}")

                for err in self.errors[:20]:
                    print(err)

            print("=" * 60)

        return ok
