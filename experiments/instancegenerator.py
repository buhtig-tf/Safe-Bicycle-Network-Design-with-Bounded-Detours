from dataclasses import dataclass
import random

from bicycleinstance import BicycleInstance


@dataclass
class InstanceGenerator:

    network: "BicycleNetwork"

    alpha: float

    target_num_pairs: int | None = None

    target_pair_fraction: float | None = None

    budget: float | None = None

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------

    def _random(
        self,
        instance_id,
        seed,
    ):

        if seed is None:

            return random.Random(
                instance_id
            )

        return random.Random(
            seed
        )

    def _num_pairs(self):

        if self.target_num_pairs is not None:

            return self.target_num_pairs

        if self.target_pair_fraction is None:

            raise ValueError(
                "Either target_num_pairs or target_pair_fraction "
                "must be specified."
            )

        return round(
            self.target_pair_fraction
            *
            len(self.network.vertices)
        )

    def _budget(self):

        if self.budget is None:

            self.budget = sum(
                self.network.upgrade_cost[e]
                for e in self.network.unsafe_edges
            )

        return self.budget

    # ------------------------------------------------------------
    # Random OD pairs
    # ------------------------------------------------------------

    def random_pairs(
        self,
        instance_id,
        seed=None,
    ):

        rng = self._random(
            instance_id,
            seed,
        )

        vertices = list(
            self.network.vertices
        )

        num_pairs = self._num_pairs()

        if num_pairs > len(vertices) * (len(vertices) - 1):

            raise ValueError(
                f"Requested {num_pairs} ordered pairs, "
                f"but the network has only {len(vertices)} vertices."
            )

        pairs = []
        seen = set()

        attempts = 0
        max_attempts = 100 * num_pairs + 1000

        while (
            len(pairs) < num_pairs
            and attempts < max_attempts
        ):

            attempts += 1

            s, t = rng.sample(
                vertices,
                2,
            )

            pair = (
                s,
                t,
            )

            if pair in seen:
                continue

            seen.add(
                pair
            )

            pairs.append(
                pair
            )

        if len(pairs) < num_pairs:

            raise RuntimeError(
                f"Could only generate {len(pairs)} "
                f"random pairs out of requested {num_pairs}."
            )

        return BicycleInstance(
            network=self.network,
            terminal_pairs=pairs,
            alpha=self.alpha,
            target_num_pairs=self.target_num_pairs,
            target_pair_fraction=self.target_pair_fraction,
            generator="random-pairs",
            budget=self._budget(),
            info=f"{self.network.info} (random pairs)",
            instance_id=instance_id,
        )
