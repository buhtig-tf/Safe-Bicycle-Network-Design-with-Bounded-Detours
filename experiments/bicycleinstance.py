from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple
import os

@dataclass
class BicycleInstance:
    """
    Instance of the bicycle upgrade problem.

        (network, terminal_pairs, alpha, budget)
    """

    network: "BicycleNetwork"

    terminal_pairs: List[Tuple[Any, Any]]

    #Metadata
    alpha: float

    budget: float

    info: str = ""

    #Metadata
    generator: str = ""

    #Metadata
    seed: int | None = None

    #Metadata
    target_num_pairs: int | None = None

    #Metadata
    target_pair_fraction: float | None = None

    reduction_history: list = field(
        default_factory=list
    )

    total_reduction_runtime: float = 0.0

    total_budget_delta: float = 0.0

    reduced: bool = False

    reduction_pipeline: str = ""

    #Metadata
    instance_id: int = 0

    # ------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------

    @classmethod
    def metadata_fields(cls):

        return [
            "instance_id",
            "generator",
            "target_num_pairs",
            "target_pair_fraction",
        ]

    def get_id(self):
        return self.instance_id

    def set_id(self, new_instance_id: int):
        self.instance_id = new_instance_id

    def number_of_pairs(self) -> int:

        return len(self.terminal_pairs)

    def terminals(self) -> Set:

        T = set()

        for s, t in self.terminal_pairs:

            T.add(s)
            T.add(t)

        return T

    def number_of_terminals(self) -> int:

        return len(self.terminals())

    def number_of_vertices(self) -> int:

        return len(self.network.vertices)

    def number_of_edges(self) -> int:

        return (
            len(self.network.safe_edges)
            + len(self.network.unsafe_edges)
        )

    def number_of_safe_edges(self) -> int:

        return len(self.network.safe_edges)

    def number_of_unsafe_edges(self) -> int:

        return len(self.network.unsafe_edges)

    def max_degree(self) -> int:
        return self.network.max_degree()

    def average_degree(self) -> float:
        return self.network.average_degree()

    # -------------------------------

    def clone_metadata_from(
        self,
        other,
    ):
        for field in self.metadata_fields():

            setattr(
                self,
                field,
                getattr(other, field),
            )

    # ------------------------------------------------------------
    # Cost information
    # ------------------------------------------------------------

    def total_upgrade_cost(self) -> float:

        return sum(
            self.network.upgrade_cost[e]
            for e in self.network.unsafe_edges
        )

    # ------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------

    def check(self):

        G = self.network.to_networkx()

        for s, t in self.terminal_pairs:

            if s not in G:
                raise ValueError(
                    f"Unknown terminal: {s}"
                )

            if t not in G:
                raise ValueError(
                    f"Unknown terminal: {t}"
                )

        if self.alpha < 1:
            raise ValueError(
                "alpha must be at least 1."
            )

        if self.budget < 0:
            raise ValueError(
                "budget must be nonnegative."
            )

        print("All alright")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    def summary(self):

        return {

            "info":
                self.info,

            "vertices":
                self.number_of_vertices(),

            "edges":
                self.number_of_edges(),

            "safe_edges":
                self.number_of_safe_edges(),

            "unsafe_edges":
                self.number_of_unsafe_edges(),

            "terminal_pairs":
                self.number_of_pairs(),

            "terminals":
                self.number_of_terminals(),

            "alpha":
                self.alpha,

            "budget":
                self.budget,

            "total_upgrade_cost":
                self.total_upgrade_cost(),
        }

    def print(self):

        print("\n" + "=" * 60)
        print("BICYCLE INSTANCE")
        print("=" * 60)

        for k, v in self.summary().items():

            print(f"{k}: {v}")

        print("=" * 60)


    # --------------------------------------------------------
    # File naming / storage
    # --------------------------------------------------------

    def filename(self):
        """
        Return the canonical instance filename.

        Examples:

            altenholz_A_random-pairs_tf25_a12_i0
            schoeppingen_C_random-pairs_tf75_a15_i1
        """

        parts = [
            self.network.filename()
        ]

        if getattr(self, "generator", ""):
            parts.append(
                self.generator
            )

        if getattr(self, "target_pair_fraction", None) is not None:
            parts.append(
                f"tf{int(round(100 * self.target_pair_fraction))}"
            )

        if getattr(self, "target_num_pairs", None) is not None:
            parts.append(
                f"tn{self.target_num_pairs}"
            )

        parts.append(
            f"a{int(round(10 * self.alpha))}"
        )

        parts.append(
            f"i{self.instance_id}"
        )

        if self.reduction_pipeline:

            parts.append(
                self.reduction_pipeline
            )

        return "_".join(parts)


    def directory(
        self,
        root="data/instances",
    ):
        subdir = (
            "reduced"
            if self.reduced
            else "original"
        )

        return os.path.join(
            root,
            subdir,
            self.network.origin_type + "s",
        )


    def path(
        self,
        root="data/instances",
        extension=".pkl",
    ):

        return os.path.join(
            self.directory(root),
            self.filename() + extension,
        )


    # --------------------------------------------------------
    # Pickle storage
    # --------------------------------------------------------

    def save(
        self,
        root="data/instances",
    ):

        import os
        import pickle

        path = self.path(
            root=root,
            extension=".pkl",
        )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(path, "wb") as f:

            pickle.dump(
                self,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return path


    @classmethod
    def load(
        cls,
        path,
    ):

        import pickle

        with open(path, "rb") as f:

            return pickle.load(f)


    @classmethod
    def all_saved_instances(
        cls,
        root="data/instances",
    ):

        return [
            cls.load(path)
            for path in cls.all_paths(root)
        ]

    @classmethod
    def all_paths(
        cls,
        root="data/instances/original",
    ):

        import os

        paths = []

        for dirpath, _, filenames in os.walk(root):

            for fn in filenames:

                if fn.endswith(".pkl"):

                    paths.append(
                        os.path.join(
                            dirpath,
                            fn,
                        )
                    )

        return sorted(paths)
