from dataclasses import dataclass, field
import json
import os


@dataclass
class ExperimentResult:
    """
    Result of solving a BicycleInstance.

    Stores both preprocessing and solver statistics.
    """

    # --------------------------------------------------------
    # Identification
    # --------------------------------------------------------

    instance_filename: str

    # now as @property
    # method: str

    solver: str

    preprocessor: str = ""

    # --------------------------------------------------------
    # Solution quality
    # --------------------------------------------------------

    objective: float | None = None

    optimal: bool = False

    feasible: bool = False

    status: str = ""

    selected_edges: list = field(
        default_factory=list
    )

    timeout: int | None = None

    termination_reason: str = ""

    # --------------------------------------------------------
    # Runtime decomposition
    # --------------------------------------------------------

    reduction_runtime: float = 0.0

    solver_buildtime: float = 0.0

    solver_runtime: float = 0.0

    total_runtime: float = 0.0

    # --------------------------------------------------------
    # Reduction statistics
    # --------------------------------------------------------

    budget_delta: float = 0.0

    safe_removed: int = 0

    unsafe_removed: int = 0

    vertices_before: int = 0
    vertices_after: int = 0

    edges_before: int = 0
    edges_after: int = 0

    # --------------------------------------------------------
    # Arbitrary extra information
    # --------------------------------------------------------

    info: str = ""

    extra: dict = field(
        default_factory=dict
    )

    # ========================================================
    # File naming
    # ========================================================

    def filename(self):
        """
        Example:

            bartholomae_pes_f30_i7
        """

        return self.instance_filename

    def directory(
        self,
        root="data/results",
    ):
        """
        Example:

            data/results/ilp

            data/results/kernel_ilp
        """

        return os.path.join(
            root,
            self.method,
        )

    def path(
        self,
        root="data/results",
    ):
        return os.path.join(
            self.directory(root),
            self.filename() + ".json",
        )

    # ========================================================
    # Export
    # ========================================================

    def save(
        self,
        root="data/results",
    ):

        path = self.path(root)

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(path, "w") as f:

            json.dump(
                self.to_dict(),
                f,
                indent=2,
            )

        return path

    # ========================================================
    # Import
    # ========================================================

    @classmethod
    def load(
        cls,
        instance_filename,
        method,
        root="data/results",
    ):

        path = os.path.join(
            root,
            method,
            instance_filename + ".json",
        )

        with open(path, "r") as f:

            data = json.load(f)

        return cls(**data)

    # ========================================================
    # Utilities
    # ========================================================

    @property
    def method(self):

        if self.preprocessor:

            return (
                f"{self.preprocessor}_"
                f"{self.solver}"
            )

        return self.solver


    def to_dict(self):

        return {

            "instance_filename":
                self.instance_filename,

            "solver":
                self.solver,

            "preprocessor":
                self.preprocessor,

            "objective":
                self.objective,

            "optimal":
                self.optimal,

            "feasible":
                self.feasible,

            "status":
                self.status,

            "selected_edges":
                self.selected_edges,

            "timeout":
                self.timeout,

            "termination_reason":
                self.termination_reason,

            "reduction_runtime":
                self.reduction_runtime,

            "solver_buildtime":
                self.solver_buildtime,

            "solver_runtime":
                self.solver_runtime,

            "total_runtime":
                self.total_runtime,

            "budget_delta":
                self.budget_delta,

            "safe_removed":
                self.safe_removed,

            "unsafe_removed":
                self.unsafe_removed,

            "vertices_before":
                self.vertices_before,

            "vertices_after":
                self.vertices_after,

            "edges_before":
                self.edges_before,

            "edges_after":
                self.edges_after,

            "info":
                self.info,

            "extra":
                self.extra,
        }

    def print(self):

        print()
        print("=" * 60)
        print("EXPERIMENT RESULT")
        print("=" * 60)

        print(
            f"instance: {self.instance_filename} | solver: {self.solver} | preprocessor: {self.preprocessor}"
        )

        print(
            f"objective: {self.objective}"
        )

        print(
            f"status: {self.status} | optimal: {self.optimal} | feasible: {self.feasible}"
        )

        print()

        print(
            f"reduction runtime: "
            f"{self.reduction_runtime:.4f}s"
        )

        print(
            f"solver buildtime: {self.solver_buildtime:.4f}s | runtime: {self.solver_runtime:.4f}s"
            f""
        )

        print(
            f"total runtime: "
            f"{self.total_runtime:.4f}s"
        )

        print()

        print(
            f"budget delta: {self.budget_delta}"
        )

        print(
            f"removed | safe: {self.safe_removed} | unsafe: {self.unsafe_removed}"
        )

        print(
            f""
        )

        print(
            f"vertices: "
            f"{self.vertices_before}"
            f" -> "
            f"{self.vertices_after}"
            f" | edges: "
            f"{self.edges_before}"
            f" -> "
            f"{self.edges_after}"
        )

        print("=" * 60)
