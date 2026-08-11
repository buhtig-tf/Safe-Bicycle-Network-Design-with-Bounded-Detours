import time

class ExhaustiveReducer:
    """
    Runs a full reduction pipeline and records statistics:

        - runtime per rule
        - unsafe/safe edge removals
        - budget changes (if applicable)
        - instance size evolution
    """

    def __init__(self, instance):
        self.instance = instance

        self.history = []
        self.total_budget_delta = 0.0
        self.total_runtime = 0.0

        self.silent = True


    # ------------------------------------------------------------
    # Helper: edge diff
    # ------------------------------------------------------------
    def _edge_diff(self, net_before, net_after):

        removed_safe = net_before.safe_edges - net_after.safe_edges
        removed_unsafe = net_before.unsafe_edges - net_after.unsafe_edges

        return removed_safe, removed_unsafe

    # ------------------------------------------------------------
    # Run one rule
    # ------------------------------------------------------------
    def _run_rule(self, rule, inst, *args):

        if not self.silent:
            print(f"Apply {rule.name}")

        # ------------------------------------------------------------
        # Snapshot BEFORE running the rule.
        # Do not keep only references, because a rule might mutate inst.
        # ------------------------------------------------------------

        net_before = inst.network

        safe_before = set(
            net_before.safe_edges
        )

        unsafe_before = set(
            net_before.unsafe_edges
        )

        vertices_before = set(
            net_before.vertices
        )

        budget_before = getattr(
            inst,
            "budget",
            None,
        )

        ntp_before = inst.number_of_pairs()

        n_before = len(
            vertices_before
        )

        m_before = (
            len(safe_before)
            +
            len(unsafe_before)
        )

        # ------------------------------------------------------------
        # Run rule.
        # ------------------------------------------------------------

        start = time.perf_counter()

        new_inst = rule.run(
            inst,
            *args,
        )

        end = time.perf_counter()

        if new_inst is None:

            raise RuntimeError(
                f"Reduction rule {rule.name} returned None"
            )

        runtime = (
            end
            -
            start
        )

        self.total_runtime += runtime

        # ------------------------------------------------------------
        # Snapshot AFTER running the rule.
        # ------------------------------------------------------------

        net_after = new_inst.network

        safe_after = set(
            net_after.safe_edges
        )

        unsafe_after = set(
            net_after.unsafe_edges
        )

        vertices_after = set(
            net_after.vertices
        )

        budget_after = getattr(
            new_inst,
            "budget",
            None,
        )

        ntp_after = new_inst.number_of_pairs()

        n_after = len(
            vertices_after
        )

        m_after = (
            len(safe_after)
            +
            len(unsafe_after)
        )

        # ------------------------------------------------------------
        # Edge changes.
        # ------------------------------------------------------------

        removed_safe = (
            safe_before
            -
            safe_after
        )

        removed_unsafe = (
            unsafe_before
            -
            unsafe_after
        )

        added_safe = (
            safe_after
            -
            safe_before
        )

        added_unsafe = (
            unsafe_after
            -
            unsafe_before
        )

        removed_vertices = (
            vertices_before
            -
            vertices_after
        )

        added_vertices = (
            vertices_after
            -
            vertices_before
        )

        # ------------------------------------------------------------
        # Budget accounting.
        # ------------------------------------------------------------

        budget_delta = None

        if (
            budget_before is not None
            and budget_after is not None
        ):

            budget_delta = (
                budget_before
                -
                budget_after
            )

            self.total_budget_delta += budget_delta

        # ------------------------------------------------------------
        # Basic sanity checks.
        # ------------------------------------------------------------

        if safe_after & unsafe_after:

            overlap = safe_after & unsafe_after

            raise RuntimeError(
                f"{rule.name} produced edges that are both safe and unsafe: "
                f"{list(overlap)[:5]}"
            )

        if budget_delta is not None:

            eps = 1e-6

            if budget_delta < -eps:

                raise RuntimeError(
                    f"{rule.name} increased the budget: "
                    f"{budget_before} -> {budget_after}"
                )

        if ntp_after > ntp_before:

            raise RuntimeError(
                f"{rule.name} increased the number of terminal pairs: "
                f"{ntp_before} -> {ntp_after}"
            )

        # ------------------------------------------------------------
        # History entry.
        # ------------------------------------------------------------

        entry = {
            "rule": rule.name,
            "runtime": runtime,

            "removed_safe": len(removed_safe),
            "removed_unsafe": len(removed_unsafe),
            "added_safe": len(added_safe),
            "added_unsafe": len(added_unsafe),

            "removed_vertices": len(removed_vertices),
            "added_vertices": len(added_vertices),

            "budget_before": budget_before,
            "budget_after": budget_after,
            "budget_delta": budget_delta,

            "n_before": n_before,
            "n_after": n_after,
            "m_before": m_before,
            "m_after": m_after,

            "ntp_before": ntp_before,
            "ntp_after": ntp_after,

            "same_instance_object": (
                new_inst is inst
            ),
            "same_network_object": (
                new_inst.network is inst.network
            ),
        }

        self.history.append(
            entry
        )

        # ------------------------------------------------------------
        # Store reduction metadata.
        # Use a shallow copy of history to avoid accidental aliasing.
        # ------------------------------------------------------------

        new_inst.reduction_history = list(
            self.history
        )

        new_inst.total_reduction_runtime = self.total_runtime
        new_inst.total_budget_delta = self.total_budget_delta
        new_inst.reduced = True

        new_inst.clone_metadata_from(
            inst
        )

        return new_inst

    # ------------------------------------------------------------
    # Publication preprocessing pipeline
    # ------------------------------------------------------------
    def run_kernel1(self):

        from removesatisfiedpairsreduction import RemoveSatisfiedPairsReduction
        from leafpruningreduction import LeafPruningReduction
        from uniqueadmissiblepathreduction import UniqueAdmissiblePathReduction
        from pathcompressionreduction import PathCompressionReduction
        from mandatoryedgereduction import MandatoryEdgeReduction
        from alpharelevantedgereduction import AlphaRelevantEdgeReduction

        inst = self.instance

        if not self.silent:
            print("\n" + "=" * 60)
            print("REDUCTION PIPELINE: kernel1")
            print("=" * 60)
            print(f"BEFORE max degree: {inst.max_degree()}")

        # ------------------------------------------------------------
        # First cleanup.
        # ------------------------------------------------------------

        inst = self._run_rule(
            RemoveSatisfiedPairsReduction(),
            inst,
        )

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            LeafPruningReduction(),
            inst,
            R,
        )

        inst = self._run_rule(
            UniqueAdmissiblePathReduction(),
            inst,
        )

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            LeafPruningReduction(),
            inst,
            R,
        )

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            PathCompressionReduction(),
            inst,
            R,
        )

        inst = self._run_rule(
            MandatoryEdgeReduction(),
            inst,
        )

        inst = self._run_rule(
            RemoveSatisfiedPairsReduction(),
            inst,
        )

        # ------------------------------------------------------------
        # Second cleanup.
        # ------------------------------------------------------------

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            LeafPruningReduction(),
            inst,
            R,
        )

        inst = self._run_rule(
            UniqueAdmissiblePathReduction(),
            inst,
        )

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            LeafPruningReduction(),
            inst,
            R,
        )

        R = inst.network.minimum_feedback_edge_set()

        inst = self._run_rule(
            PathCompressionReduction(),
            inst,
            R,
        )

        # ------------------------------------------------------------
        # Final alpha-relevance cleanup only.
        # ------------------------------------------------------------

        inst = self._run_rule(
            AlphaRelevantEdgeReduction(),
            inst,
        )

        if not self.silent:
            print(f"AFTER max degree: {inst.max_degree()}")
            print("=" * 60)

        inst.reduction_pipeline = "kernel1"

        self.final_instance = inst

        return inst

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    def summary(self):

        print("\n" + "=" * 60)
        print("REDUCTION PIPELINE SUMMARY")
        print("=" * 60)

        for h in self.history:

            print(
                f"{h['rule']:<35} "
                f"time={h['runtime']:.4f}s "
                f"| safe {h['removed_safe']}-{h['added_safe']} "
                f"| unsafe {h['removed_unsafe']}-{h['added_unsafe']} "
                f"| budget={h['budget_before']}->{h['budget_after']} "
                f"(Δ={h['budget_delta']}) "
                f"| n={h['n_before']}->{h['n_after']} "
                f"| m={h['m_before']}->{h['m_after']} "
                f"| ntp={h['ntp_before']}->{h['ntp_after']} "
                f"| same_inst={h['same_instance_object']} "
                f"| same_net={h['same_network_object']}"
            )

        print("-" * 60)
        print(
            f"TOTAL TIME: {self.total_runtime:.4f}s"
        )
        print("=" * 60)
        print("-" * 60)
        print(f"TOTAL BUDGET DELTA: {self.total_budget_delta}")
        print("=" * 60)

        return self.history
