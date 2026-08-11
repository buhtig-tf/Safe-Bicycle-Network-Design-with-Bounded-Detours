"""
Data structure and utilities for the Bicycle Network Improvement
Problem on road networks derived from OpenStreetMap.

The network is represented as a simple undirected graph. Parallel
OSM alternatives are handled during conversion from OSMnx graphs.
Edges are classified as safe or unsafe according to one of the
safety models A, B, or C. For unsafe edges, upgrade cost is equal
to edge length.

Core graph operations require NetworkX. Optional network-generation
and plotting functionality has additional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    TYPE_CHECKING,
)
import os
import random

import networkx as nx


if TYPE_CHECKING:
    from villages import Village


# ============================================================
# Types
# ============================================================

Node = int
Edge = Tuple[Node, Node]

# ============================================================
# Bicycle Network
# ============================================================

@dataclass
class BicycleNetwork:
    """
    Bicycle Network Improvement Problem instance.

    The edge set is partitioned into

        safe_edges
        unsafe_edges

    Every edge has a length.

    Every unsafe edge has an upgrade cost.
    """

    vertices: Set[Node]

    safe_edges: Set[Edge]
    unsafe_edges: Set[Edge]

    length: Dict[Edge, float]

    upgrade_cost: Dict[Edge, float]


    node_x: Dict[Node, float]
    node_y: Dict[Node, float]

    edge_geometry: Dict[Edge, object]

    origin_key: str = ""

    origin_type: str = ""

    radius_km: float | None = None

    safety_model: str = ""

    info: str = ""

    MAJOR_ROADS = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
    }

    LOCAL_ROADS = {
        "residential",
        "service",
        "unclassified",
    }

    TERTIARY_ROADS = {
        "tertiary",
        "tertiary_link",
    }

    CYCLEWAY_TAGS = [
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
    ]

    # --------------------------------------------------------
    # edge utilities
    # --------------------------------------------------------

    @staticmethod
    def canonical_edge(u: Node, v: Node) -> Edge:
        return (u, v) if u < v else (v, u)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------


    @staticmethod
    def _highway(data):

        highway = data.get("highway")

        if isinstance(highway, list):
            highway = highway[0]

        return highway


    @staticmethod
    def _is_cycle_path_like(highway):

        return highway in {
            "path",
            "track",
        }


    @staticmethod
    def _is_pedestrian_like(highway):

        return highway == "pedestrian"


    @staticmethod
    def _has_cycle_infrastructure(data) -> bool:

        for tag in BicycleNetwork.CYCLEWAY_TAGS:

            value = data.get(tag)

            if value is None:
                continue

            if value not in {
                "no",
                "none",
            }:
                return True

        return False


    @staticmethod
    def _bicycle_allowed(data) -> bool:

        bicycle = data.get("bicycle")

        return bicycle not in {
            "no",
            "dismount",
        }


    @staticmethod
    def _bicycle_designated(data) -> bool:

        return data.get("bicycle") == "designated"


    @staticmethod
    def _bicycle_explicitly_allowed(data) -> bool:

        return data.get("bicycle") in {
            "yes",
            "designated",
        }


    @staticmethod
    def _has_traffic_calming(data) -> bool:

        return "traffic_calming" in data


    @staticmethod
    def _speed_at_most(
        data,
        threshold,
    ) -> bool:

        speed = data.get("maxspeed")

        if speed is None:
            return False

        try:

            speed = int(
                str(speed).split()[0]
            )

            return speed <= threshold

        except Exception:

            return False


    # ------------------------------------------------------------------
    # SAFE EDGE CLASSIFIER
    # ------------------------------------------------------------------
    @staticmethod
    def _is_safe_edge(data, safety_model="B") -> bool:

        model = str(safety_model).upper()

        # Backward-compatible aliases
        if model == "HIER":
            model = "A"
        elif model == "SEC":
            model = "B"
        elif model == "PES":
            model = "C"

        if model == "A":
            return BicycleNetwork._is_safe_edge_A(data)

        if model == "B":
            return BicycleNetwork._is_safe_edge_B(data)

        if model == "C":
            return BicycleNetwork._is_safe_edge_C(data)

        raise ValueError(
            f"Unknown safety model: {safety_model}. "
            "Use one of: A, B, C."
        )


    # ------------------------------------------------------------------
    # Model A
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_edge_A(data) -> bool:
        """
        Model A: permissive local-access model.

        This model classifies the fewest edges as unsafe.

        Intuition:
            Local village streets are generally considered suitable for
            cycling unless they are explicitly high-hierarchy roads.

        Safe:
            - dedicated cycleways
            - explicit cycle infrastructure
            - living streets
            - traffic-calmed roads
            - bicycle-accessible paths/tracks
            - bicycle-accessible pedestrian streets
            - residential roads
            - service roads
            - unclassified roads
            - tertiary roads only if speed <= 30 or traffic-calmed

        Unsafe:
            - major roads without cycling infrastructure
            - tertiary roads without low-speed evidence
            - unknown or unsupported highway types
        """

        highway = BicycleNetwork._highway(data)

        if highway == "cycleway":
            return True

        if BicycleNetwork._has_cycle_infrastructure(data):
            return True

        if BicycleNetwork._has_traffic_calming(data):
            return True

        if highway == "living_street":
            return True

        if (
            BicycleNetwork._is_cycle_path_like(highway)
            and BicycleNetwork._bicycle_allowed(data)
        ):
            return True

        if (
            BicycleNetwork._is_pedestrian_like(highway)
            and BicycleNetwork._bicycle_allowed(data)
        ):
            return True

        if highway in BicycleNetwork.MAJOR_ROADS:
            return False

        if highway in BicycleNetwork.LOCAL_ROADS:
            return True

        if highway in BicycleNetwork.TERTIARY_ROADS:
            return BicycleNetwork._speed_at_most(
                data,
                threshold=30,
            )

        return False


    # ------------------------------------------------------------------
    # Model B
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_edge_B(data) -> bool:
        """
        Model B: balanced speed-aware model.

        This is the recommended default model.

        Intuition:
            Ordinary local village streets are considered suitable for
            cycling. Ambiguous or higher-hierarchy roads require stronger
            low-speed evidence.

        Safe:
            - dedicated cycleways
            - explicit cycle infrastructure
            - living streets
            - traffic-calmed roads
            - bicycle-accessible paths/tracks
            - bicycle-accessible pedestrian streets
            - residential roads if speed is missing or <= 30
            - service roads if speed is missing or <= 30
            - unclassified roads if speed is missing or <= 30
            - tertiary roads only if speed <= 30 or traffic-calmed

        Unsafe:
            - major roads without cycling infrastructure
            - tertiary roads without low-speed evidence
            - unknown or unsupported highway types
        """

        highway = BicycleNetwork._highway(data)

        if highway == "cycleway":
            return True

        if BicycleNetwork._has_cycle_infrastructure(data):
            return True

        if BicycleNetwork._has_traffic_calming(data):
            return True

        if highway == "living_street":
            return True

        if (
            BicycleNetwork._is_cycle_path_like(highway)
            and BicycleNetwork._bicycle_allowed(data)
        ):
            return True

        if (
            BicycleNetwork._is_pedestrian_like(highway)
            and BicycleNetwork._bicycle_allowed(data)
        ):
            return True

        if highway in BicycleNetwork.MAJOR_ROADS:
            return False

        if highway in BicycleNetwork.LOCAL_ROADS:

            speed = data.get("maxspeed")

            if speed is None:
                return True

            return BicycleNetwork._speed_at_most(
                data,
                threshold=30,
            )

        if highway in BicycleNetwork.TERTIARY_ROADS:
            return BicycleNetwork._speed_at_most(
                data,
                threshold=30,
            )

        return False


    # ------------------------------------------------------------------
    # Model C
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_edge_C(data) -> bool:
        """
        Model C: conservative low-stress model.

        This model classifies the most edges as unsafe, but avoids the
        overly pessimistic behavior of treating ordinary residential roads
        with missing speed tags as unsafe.

        Intuition:
            Roads are considered safe only if they are clearly low-stress:
            explicit bicycle infrastructure, living streets, traffic calming,
            or residential streets. More ambiguous local roads require
            explicit low-speed evidence.

        Safe:
            - dedicated cycleways
            - explicit cycle infrastructure
            - living streets
            - traffic-calmed roads
            - pedestrian streets only if bicycles are explicitly allowed
            - paths/tracks only if bicycles are explicitly allowed
            - residential roads if speed is missing or <= 30
            - service roads only if speed <= 20 or traffic-calmed
            - unclassified roads only if speed <= 30 or traffic-calmed

        Unsafe:
            - major roads without cycling infrastructure
            - tertiary roads without cycling infrastructure
            - service roads without low-speed evidence
            - unclassified roads without low-speed evidence
            - paths/tracks without explicit bicycle permission
            - pedestrian streets without explicit bicycle permission
        """

        highway = BicycleNetwork._highway(data)

        if highway == "cycleway":
            return True

        if BicycleNetwork._has_cycle_infrastructure(data):
            return True

        if BicycleNetwork._has_traffic_calming(data):
            return True

        if highway == "living_street":
            return True

        if (
            BicycleNetwork._is_pedestrian_like(highway)
            and BicycleNetwork._bicycle_explicitly_allowed(data)
        ):
            return True

        if (
            BicycleNetwork._is_cycle_path_like(highway)
            and BicycleNetwork._bicycle_explicitly_allowed(data)
        ):
            return True

        if highway in BicycleNetwork.MAJOR_ROADS:
            return False

        if highway in BicycleNetwork.TERTIARY_ROADS:
            return False

        if highway == "residential":

            speed = data.get("maxspeed")

            if speed is None:
                return True

            return BicycleNetwork._speed_at_most(
                data,
                threshold=30,
            )

        if highway == "service":
            return BicycleNetwork._speed_at_most(
                data,
                threshold=20,
            )

        if highway == "unclassified":
            return BicycleNetwork._speed_at_most(
                data,
                threshold=30,
            )

        return False

    # --------------------------------------------------------
    # constructors
    # --------------------------------------------------------

    @classmethod
    def from_osmnx_graph(
        cls,
        G,
        origin_key: str = "",
        origin_type: str = "",
        radius_km: float | None = None,
        info: str = "",
        safety_model: str = "sec",
    ) -> "BicycleNetwork":
        """
        Construct a BicycleNetwork from an already downloaded OSMnx graph.

        The input graph may be an OSMnx MultiDiGraph, MultiGraph, or a
        simple graph. We build an undirected BicycleNetwork.

        Antiparallel OSMnx arcs representing the same physical street are
        deduplicated. True parallel alternatives between the same unordered
        endpoint pair are preserved by subdivision.

        If multiple OSM alternatives connect the same unordered pair {u,v},
        each alternative is represented by its own two-edge path u--x--v.

        Safe alternatives produce two safe half-edges. Unsafe alternatives
        produce two unsafe half-edges. Since the cost model is length-based,
        each unsafe half-edge gets half the original length and half the
        original upgrade cost.
        """

        try:
            from shapely.geometry import LineString
        except Exception:
            LineString = None

        vertices = set(G.nodes())

        # ------------------------------------------------------------
        # Helper: robust node coordinates.
        # ------------------------------------------------------------

        def node_xy(v):

            return (
                float(G.nodes[v]["x"]),
                float(G.nodes[v]["y"]),
            )

        # ------------------------------------------------------------
        # Helper: canonicalize OSM id into something hashable and
        # orientation-insensitive.
        #
        # OSMnx sometimes stores simplified edges as lists of OSM ids.
        # The reverse direction may contain the same ids in reverse order.
        # Since our model is undirected, that order should not distinguish
        # alternatives.
        # ------------------------------------------------------------

        def normalize_osmid(osmid):

            if isinstance(osmid, list):

                return tuple(
                    sorted(osmid)
                )

            if isinstance(osmid, tuple):

                return tuple(
                    sorted(osmid)
                )

            return osmid

        # ------------------------------------------------------------
        # Helper: orientation-insensitive geometry signature.
        #
        # A LineString traversed from u to v and the same LineString
        # traversed from v to u should have the same signature.
        # ------------------------------------------------------------

        def geometry_signature(geometry):

            if geometry is None:

                return None

            if not hasattr(geometry, "coords"):

                return repr(geometry)

            coords = []

            for coord in geometry.coords:

                coords.append(
                    tuple(
                        round(
                            float(value),
                            7,
                        )
                        for value in coord
                    )
                )

            forward = tuple(
                coords
            )

            backward = tuple(
                reversed(coords)
            )

            return min(
                forward,
                backward,
            )

        # ------------------------------------------------------------
        # Helper: split a LineString-like geometry into two halves.
        #
        # If no usable geometry exists, fall back to straight segments
        # through the endpoint coordinates.
        #
        # The returned midpoint is used as the subdivision-vertex
        # coordinate.
        # ------------------------------------------------------------

        def split_geometry_at_half(
            geometry,
            u,
            v,
        ):

            xu, yu = node_xy(u)
            xv, yv = node_xy(v)

            fallback_mid_x = (
                xu
                +
                xv
            ) / 2.0

            fallback_mid_y = (
                yu
                +
                yv
            ) / 2.0

            def fallback_result():

                if LineString is None:

                    return (
                        None,
                        None,
                        fallback_mid_x,
                        fallback_mid_y,
                    )

                return (
                    LineString(
                        [
                            (
                                xu,
                                yu,
                            ),
                            (
                                fallback_mid_x,
                                fallback_mid_y,
                            ),
                        ]
                    ),
                    LineString(
                        [
                            (
                                fallback_mid_x,
                                fallback_mid_y,
                            ),
                            (
                                xv,
                                yv,
                            ),
                        ]
                    ),
                    fallback_mid_x,
                    fallback_mid_y,
                )

            if (
                geometry is None
                or LineString is None
                or not hasattr(
                    geometry,
                    "coords",
                )
            ):

                return fallback_result()

            coords = list(
                geometry.coords
            )

            if len(coords) < 2:

                return fallback_result()

            total_geom_length = float(
                geometry.length
            )

            if total_geom_length <= 0.0:

                return fallback_result()

            half = (
                total_geom_length
                /
                2.0
            )

            first_coords = [
                coords[0]
            ]

            accumulated = 0.0

            split_point = None
            split_index = None

            for i in range(
                1,
                len(coords),
            ):

                x0, y0 = coords[
                    i - 1
                ][:2]

                x1, y1 = coords[
                    i
                ][:2]

                dx = (
                    x1
                    -
                    x0
                )

                dy = (
                    y1
                    -
                    y0
                )

                segment_length = (
                    dx * dx
                    +
                    dy * dy
                ) ** 0.5

                if segment_length <= 0.0:

                    continue

                if (
                    accumulated
                    +
                    segment_length
                    >= half
                ):

                    ratio = (
                        half
                        -
                        accumulated
                    ) / segment_length

                    mx = (
                        x0
                        +
                        ratio
                        *
                        dx
                    )

                    my = (
                        y0
                        +
                        ratio
                        *
                        dy
                    )

                    split_point = (
                        mx,
                        my,
                    )

                    split_index = i

                    first_coords.append(
                        split_point
                    )

                    break

                first_coords.append(
                    coords[i]
                )

                accumulated += segment_length

            if split_point is None:

                split_point = coords[-1][:2]
                split_index = len(coords) - 1

                first_coords.append(
                    split_point
                )

            second_coords = [
                split_point
            ]

            second_coords.extend(
                coords[split_index:]
            )

            # Avoid degenerate geometries.
            if len(first_coords) < 2:

                first_coords = [
                    coords[0],
                    split_point,
                ]

            if len(second_coords) < 2:

                second_coords = [
                    split_point,
                    coords[-1],
                ]

            geom1 = LineString(
                first_coords
            )

            geom2 = LineString(
                second_coords
            )

            return (
                geom1,
                geom2,
                float(split_point[0]),
                float(split_point[1]),
            )

        # ------------------------------------------------------------
        # Collect model-relevant OSM alternatives for each unordered
        # endpoint pair.
        #
        # The signature deliberately uses only model-relevant properties:
        # OSM id, length, safety class, and geometry up to reversal.
        # Thus, opposite directions of the same physical street collapse
        # unless they differ in a way that matters for our model.
        # ------------------------------------------------------------

        alternatives = {}
        seen_signatures = {}

        raw_parallel_records = 0
        deduplicated_same_street_records = 0

        for u, v, data in G.edges(data=True):

            if u == v:

                continue

            e = cls.canonical_edge(
                u,
                v,
            )

            length = float(
                data.get(
                    "length",
                    0.0,
                )
            )

            if length <= 0.0:

                continue

            safe = cls._is_safe_edge(
                data,
                safety_model=safety_model,
            )

            geometry = data.get(
                "geometry"
            )

            signature = (
                normalize_osmid(
                    data.get(
                        "osmid"
                    )
                ),
                round(
                    length,
                    3,
                ),
                bool(safe),
                geometry_signature(
                    geometry
                ),
            )

            if e not in alternatives:

                alternatives[e] = []
                seen_signatures[e] = set()

            else:

                raw_parallel_records += 1

            if signature in seen_signatures[e]:

                deduplicated_same_street_records += 1

                continue

            seen_signatures[e].add(
                signature
            )

            alternatives[e].append(
                {
                    "u": u,
                    "v": v,
                    "length": length,
                    "safe": safe,
                    "geometry": geometry,
                    "data": data,
                }
            )

        # ------------------------------------------------------------
        # Build BicycleNetwork edge data.
        # ------------------------------------------------------------

        safe_edges = set()
        unsafe_edges = set()

        lengths = {}
        costs = {}
        edge_geometry = {}

        node_x = {
            v: G.nodes[v]["x"]
            for v in G.nodes()
        }

        node_y = {
            v: G.nodes[v]["y"]
            for v in G.nodes()
        }

        if vertices:

            next_subdivision_vertex = (
                max(vertices)
                +
                1
            )

        else:

            next_subdivision_vertex = 0

        endpoint_pairs_with_true_parallel_alternatives = 0
        total_true_parallel_alternatives = 0

        for e, alts in sorted(
            alternatives.items(),
            key=lambda item: item[0],
        ):

            if not alts:

                continue

            # --------------------------------------------------------
            # Single model-relevant alternative: keep direct edge.
            # --------------------------------------------------------

            if len(alts) == 1:

                alt = alts[0]

                length = alt["length"]
                safe = alt["safe"]

                lengths[e] = length
                edge_geometry[e] = alt["geometry"]

                if safe:

                    safe_edges.add(
                        e
                    )

                else:

                    unsafe_edges.add(
                        e
                    )

                    costs[e] = length

                continue

            # --------------------------------------------------------
            # Multiple true alternatives: subdivide each one.
            # --------------------------------------------------------

            endpoint_pairs_with_true_parallel_alternatives += 1
            total_true_parallel_alternatives += len(alts)

            u, v = e

            for alt in alts:

                x = next_subdivision_vertex
                next_subdivision_vertex += 1

                vertices.add(
                    x
                )

                geom1, geom2, mx, my = split_geometry_at_half(
                    alt["geometry"],
                    u,
                    v,
                )

                node_x[x] = mx
                node_y[x] = my

                e1 = cls.canonical_edge(
                    u,
                    x,
                )

                e2 = cls.canonical_edge(
                    x,
                    v,
                )

                if (
                    e1 in safe_edges
                    or e1 in unsafe_edges
                    or e2 in safe_edges
                    or e2 in unsafe_edges
                ):

                    raise RuntimeError(
                        "Subdivision edge already exists while expanding "
                        f"parallel OSM alternatives: {e1}, {e2}"
                    )

                half_length = (
                    alt["length"]
                    /
                    2.0
                )

                if half_length <= 0.0:

                    raise RuntimeError(
                        "Parallel OSM alternative produced non-positive "
                        f"half length for edge {e}: {half_length}"
                    )

                lengths[e1] = half_length
                lengths[e2] = half_length

                edge_geometry[e1] = geom1
                edge_geometry[e2] = geom2

                if alt["safe"]:

                    safe_edges.add(
                        e1
                    )

                    safe_edges.add(
                        e2
                    )

                else:

                    unsafe_edges.add(
                        e1
                    )

                    unsafe_edges.add(
                        e2
                    )

                    costs[e1] = half_length
                    costs[e2] = half_length

        # ------------------------------------------------------------
        # Basic consistency checks.
        # ------------------------------------------------------------

        if safe_edges & unsafe_edges:

            overlap = (
                safe_edges
                &
                unsafe_edges
            )

            raise RuntimeError(
                "from_osmnx_graph produced edges that are both safe and "
                f"unsafe: {list(overlap)[:5]}"
            )

        all_edges = (
            safe_edges
            |
            unsafe_edges
        )

        for e in all_edges:

            u, v = e

            if u not in vertices or v not in vertices:

                raise RuntimeError(
                    "from_osmnx_graph produced an edge with missing "
                    f"endpoint: {e}"
                )

            if e not in lengths:

                raise RuntimeError(
                    "from_osmnx_graph produced an edge without length: "
                    f"{e}"
                )

            if lengths[e] <= 0.0:

                raise RuntimeError(
                    "from_osmnx_graph produced a non-positive edge "
                    f"length: {e}, length={lengths[e]}"
                )

        for e in unsafe_edges:

            if e not in costs:

                raise RuntimeError(
                    "from_osmnx_graph produced unsafe edge without "
                    f"upgrade cost: {e}"
                )

            if costs[e] < 0.0:

                raise RuntimeError(
                    "from_osmnx_graph produced negative upgrade cost: "
                    f"{e}, cost={costs[e]}"
                )

        # ------------------------------------------------------------
        # Diagnostics.
        # ------------------------------------------------------------

        print(
            "from_osmnx_graph raw repeated endpoint records:",
            raw_parallel_records,
        )

        print(
            "from_osmnx_graph deduplicated same-street records:",
            deduplicated_same_street_records,
        )

        print(
            "from_osmnx_graph endpoint pairs with true parallel alternatives:",
            endpoint_pairs_with_true_parallel_alternatives,
        )

        print(
            "from_osmnx_graph total true parallel alternatives:",
            total_true_parallel_alternatives,
        )

        return cls(
            vertices=vertices,
            safe_edges=safe_edges,
            unsafe_edges=unsafe_edges,
            length=lengths,
            upgrade_cost=costs,
            node_x=node_x,
            node_y=node_y,
            edge_geometry=edge_geometry,
            origin_key=origin_key,
            origin_type=origin_type,
            radius_km=radius_km,
            info=info,
            safety_model=safety_model,
        )



    @classmethod
    def from_place(
        cls,
        place: Village,
        network_type: str = "drive",
        safety_model: str = "sec",
    ) -> "BicycleNetwork":

        import osmnx as ox
        import numpy as np
        from geopy.distance import geodesic

        def graph_centroid_distance_km(G, lat, lon):
            """
            Distance between expected village center and graph centroid (km).
            """

            xs = []
            ys = []

            for _, data in G.nodes(data=True):
                if "x" in data and "y" in data:
                    xs.append(data["x"])
                    ys.append(data["y"])

            if not xs:
                return float("inf")

            centroid_lat = float(np.mean(ys))
            centroid_lon = float(np.mean(xs))

            return geodesic(
                (lat, lon),
                (centroid_lat, centroid_lon)
            ).km


        G = ox.graph_from_place(
            place.name,
            network_type=network_type,
            simplify=True,
        )

        # ----------------------------------------------------
        # Keep only largest connected component
        # ----------------------------------------------------

        # convert to undirected for component analysis
        G_undirected = G.to_undirected()

        largest_cc_nodes = max(
            nx.connected_components(G_undirected),
            key=len
        )

        G = G.subgraph(largest_cc_nodes).copy()

        dist_km = graph_centroid_distance_km(G, place.lat, place.lon)

        max_centroid_offset_km = 4.0

        # Reject likely geocoding mismatches if the largest connected
        # component is more than 4 km from the stored municipality center.
        if dist_km > max_centroid_offset_km:

            raise ValueError(
                f"[OSM MISMATCH] {place.name}: "
                f"graph centroid too far ({dist_km:.2f} km)"
            )

        return cls.from_osmnx_graph(
            G,
            origin_key=place.key,
            origin_type="village",
            safety_model=safety_model,
        )

    @staticmethod
    def nearest_vertex_in_osm_graph(
        G,
        x,
        y,
    ):
        return min(
            G.nodes(),
            key=lambda v:
            (
                (G.nodes[v]["x"] - x) ** 2
                +
                (G.nodes[v]["y"] - y) ** 2
            )
        )

    @classmethod
    def from_place_radius(
        cls,
        place: Village,
        radius_km: float,
        safety_model: str = "B",
        network_type: str = "drive",
    ) -> "BicycleNetwork":
        """
        Construct a BicycleNetwork from all roads within radius_km
        of the given place.

        If the Village object provides valid lat/lon coordinates, these are
        used as the region center. If the radius query fails, we try a
        geocoded centroid fallback. This is useful because some stored
        coordinates or some point-based Overpass queries can fail for a few
        places, while place-based village generation still works.

        Only the connected component containing the chosen center is
        retained.
        """

        import math
        import osmnx as ox
        import networkx as nx

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------

        def parse_lat_lon(
            lat,
            lon,
        ):
            """
            Return (lat, lon) as floats if valid, otherwise None.
            """

            if lat is None or lon is None:

                return None

            try:

                lat = float(
                    lat
                )

                lon = float(
                    lon
                )

            except Exception:

                return None

            if not (
                math.isfinite(lat)
                and math.isfinite(lon)
            ):

                return None

            if not (
                -90.0 <= lat <= 90.0
                and -180.0 <= lon <= 180.0
            ):

                return None

            return (
                lat,
                lon,
            )

        def geocoded_centroid():
            """
            Geocode the place name and return centroid coordinates.
            """

            gdf = ox.geocode_to_gdf(
                place.name
            )

            if len(gdf) == 0:

                raise RuntimeError(
                    f"Geocoding returned no result for {place.name}"
                )

            center = gdf.geometry.iloc[0].centroid

            return (
                float(center.y),
                float(center.x),
            )

        def query_graph_from_center(
            lat,
            lon,
            center_source,
            max_attempts=3,
        ):
            """
            Run the OSMnx radius query with retries and useful context.

            The OSMnx 'response' UnboundLocalError is treated as a failed
            Overpass/request attempt and retried.
            """

            import time

            configure_osmnx_timeout(
                timeout_seconds=300
            )

            endpoints = [
                None,  # keep current/default endpoint first
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter",
            ]

            last_error = None

            for endpoint in endpoints:

                if endpoint is not None:

                    print(
                        "[from_place_radius] trying Overpass endpoint:",
                        endpoint,
                    )

                    set_overpass_endpoint(
                        endpoint
                    )

                for attempt in range(
                    1,
                    max_attempts + 1,
                ):

                    try:

                        print(
                            "[from_place_radius] querying",
                            place.key,
                            f"radius_km={radius_km}",
                            f"safety_model={safety_model}",
                            f"center_source={center_source}",
                            f"lat={lat}",
                            f"lon={lon}",
                            f"attempt={attempt}/{max_attempts}",
                        )

                        G = ox.graph_from_point(
                            (
                                lat,
                                lon,
                            ),
                            dist=radius_km * 1000,
                            network_type=network_type,
                            simplify=True,
                        )

                        if len(G.nodes()) == 0:

                            raise RuntimeError(
                                "OSMnx returned an empty graph"
                            )

                        return G

                    except UnboundLocalError as e:

                        if "response" in str(e):

                            last_error = RuntimeError(
                                "OSMnx/Overpass request failed before a "
                                "response object was available. "
                                f"place={place.key}, "
                                f"name={place.name}, "
                                f"radius_km={radius_km}, "
                                f"safety_model={safety_model}, "
                                f"center_source={center_source}, "
                                f"lat={lat}, lon={lon}, "
                                f"attempt={attempt}/{max_attempts}"
                            )

                        else:

                            raise

                    except Exception as e:

                        last_error = RuntimeError(
                            "OSMnx radius query failed. "
                            f"place={place.key}, "
                            f"name={place.name}, "
                            f"radius_km={radius_km}, "
                            f"safety_model={safety_model}, "
                            f"center_source={center_source}, "
                            f"lat={lat}, lon={lon}, "
                            f"attempt={attempt}/{max_attempts}, "
                            f"error={type(e).__name__}: {e}"
                        )

                    print(
                        "[from_place_radius] query attempt failed:",
                        last_error,
                    )

                    if attempt < max_attempts:

                        time.sleep(
                            10 * attempt
                        )

            raise RuntimeError(
                "All OSMnx radius query attempts failed. "
                f"place={place.key}, name={place.name}, "
                f"radius_km={radius_km}, "
                f"safety_model={safety_model}, "
                f"center_source={center_source}, "
                f"lat={lat}, lon={lon}"
            ) from last_error

        def configure_osmnx_timeout(timeout_seconds=300):
            """
            Configure OSMnx timeout settings defensively.

            Important:
            Do not set requests_kwargs["timeout"], because some OSMnx versions
            already pass timeout explicitly to requests.get(...). Setting it
            again causes:
                TypeError: requests.api.get() got multiple values for keyword argument 'timeout'
            """

            if hasattr(ox.settings, "timeout"):

                ox.settings.timeout = max(
                    getattr(
                        ox.settings,
                        "timeout",
                        timeout_seconds,
                    ),
                    timeout_seconds,
                )

            # Remove duplicate timeout if it was set previously.
            if hasattr(ox.settings, "requests_kwargs"):

                requests_kwargs = getattr(
                    ox.settings,
                    "requests_kwargs",
                    {},
                )

                if requests_kwargs is None:

                    requests_kwargs = {}

                requests_kwargs = dict(
                    requests_kwargs
                )

                requests_kwargs.pop(
                    "timeout",
                    None,
                )

                ox.settings.requests_kwargs = requests_kwargs

            if hasattr(ox.settings, "overpass_rate_limit"):

                ox.settings.overpass_rate_limit = True


        def set_overpass_endpoint(endpoint):
            """
            Set Overpass endpoint across OSMnx versions if supported.
            """

            if endpoint is None:

                return

            if hasattr(ox.settings, "overpass_endpoint"):

                ox.settings.overpass_endpoint = endpoint

            elif hasattr(ox.settings, "overpass_url"):

                ox.settings.overpass_url = endpoint

        # ------------------------------------------------------------
        # Build candidate centers.
        # ------------------------------------------------------------

        center_candidates = []

        stored_center = parse_lat_lon(
            getattr(place, "lat", None),
            getattr(place, "lon", None),
        )

        if stored_center is not None:

            center_candidates.append(
                (
                    stored_center[0],
                    stored_center[1],
                    "stored_lat_lon",
                )
            )

        # Add geocoded fallback. We do this even if stored coordinates exist,
        # because for the few failing regions the stored point query may fail
        # while the place centroid query succeeds.
        try:

            geo_lat, geo_lon = geocoded_centroid()

            geo_center = parse_lat_lon(
                geo_lat,
                geo_lon,
            )

            if geo_center is not None:

                # Avoid exact duplicate candidate.
                already_present = any(
                    abs(geo_center[0] - lat) <= 1e-9
                    and abs(geo_center[1] - lon) <= 1e-9
                    for lat, lon, _ in center_candidates
                )

                if not already_present:

                    center_candidates.append(
                        (
                            geo_center[0],
                            geo_center[1],
                            "geocoded_centroid",
                        )
                    )

        except Exception as e:

            # If stored coordinates are valid, geocoding is only a fallback,
            # so do not fail immediately. If no stored coordinates exist, we
            # will fail below because there are no candidates.
            print(
                "[from_place_radius] geocoding fallback failed:",
                place.key,
                type(e).__name__,
                e,
            )

        if not center_candidates:

            raise ValueError(
                "Cannot generate radius network because no valid center "
                f"coordinates are available for place={place.key}, "
                f"name={place.name}, "
                f"lat={getattr(place, 'lat', None)}, "
                f"lon={getattr(place, 'lon', None)}"
            )

        # ------------------------------------------------------------
        # Download network around the first center that works.
        # ------------------------------------------------------------

        last_error = None
        chosen_lat = None
        chosen_lon = None
        chosen_source = None
        G = None

        for lat, lon, center_source in center_candidates:

            try:

                G = query_graph_from_center(
                    lat,
                    lon,
                    center_source,
                )

                chosen_lat = lat
                chosen_lon = lon
                chosen_source = center_source

                break

            except Exception as e:

                last_error = e

                print(
                    "[from_place_radius] center failed:",
                    place.key,
                    f"center_source={center_source}",
                    f"lat={lat}",
                    f"lon={lon}",
                    f"error={type(e).__name__}: {e}",
                )

        if G is None:

            raise RuntimeError(
                "All radius-query center candidates failed for "
                f"place={place.key}, name={place.name}, "
                f"radius_km={radius_km}."
            ) from last_error

        # ------------------------------------------------------------
        # Find node closest to chosen region center.
        # ------------------------------------------------------------

        center_node = cls.nearest_vertex_in_osm_graph(
            G,
            chosen_lon,
            chosen_lat,
        )

        # ------------------------------------------------------------
        # Undirected connectivity graph.
        # ------------------------------------------------------------

        H = nx.Graph()

        H.add_nodes_from(
            G.nodes()
        )

        for u, v in G.edges():

            H.add_edge(
                u,
                v,
            )

        if center_node not in H:

            raise RuntimeError(
                "Center node is not present in connectivity graph. "
                f"place={place.key}, center_node={center_node}"
            )

        # ------------------------------------------------------------
        # Keep only component containing center node.
        # ------------------------------------------------------------

        component = nx.node_connected_component(
            H,
            center_node,
        )

        G = G.subgraph(
            component
        ).copy()

        # ------------------------------------------------------------
        # Convert to BicycleNetwork.
        # ------------------------------------------------------------

        return cls.from_osmnx_graph(
            G,
            origin_key=place.key,
            origin_type="region",
            radius_km=radius_km,
            info=(
                f"{place.key} "
                f"({radius_km} km radius, "
                f"center={chosen_source}, "
                f"lat={chosen_lat}, lon={chosen_lon})"
            ),
            safety_model=safety_model,
        )

    # --------------------------------------------------------
    # graph conversion
    # --------------------------------------------------------

    def all_edges(self) -> Set[Edge]:
        return self.safe_edges | self.unsafe_edges

    def to_networkx(self) -> nx.Graph:

        G = nx.Graph()

        G.add_nodes_from(self.vertices)

        for e in self.safe_edges:

            G.add_edge(
                *e,
                length=self.length[e],
                safe=True,
            )

        for e in self.unsafe_edges:

            G.add_edge(
                *e,
                length=self.length[e],
                safe=False,
                cost=self.upgrade_cost[e],
            )

        return G

    def safe_graph(self) -> nx.Graph:

        G = nx.Graph()

        G.add_nodes_from(
            self.vertices
        )

        for e in self.safe_edges:

            G.add_edge(
                *e,
                length=self.length[e],
            )

        return G

    def unsafe_graph(self) -> nx.Graph:

        G = nx.Graph()

        #G.add_nodes_from(self.vertices)

        for e in self.unsafe_edges:

            G.add_edge(
                *e,
                length=self.length[e],
                cost=self.upgrade_cost[e],
            )

        return G

    # --------------------------------------------------------
    # basic counts
    # --------------------------------------------------------

    def number_of_vertices(self) -> int:
        return len(self.vertices)

    def number_of_edges(self) -> int:
        return len(self.all_edges())

    def number_of_safe_edges(self) -> int:
        return len(self.safe_edges)

    def number_of_unsafe_edges(self) -> int:
        return len(self.unsafe_edges)

    # --------------------------------------------------------
    # shortest path distances
    # --------------------------------------------------------

    def distance(
        self,
        u: Node,
        v: Node,
    ) -> float:

        try:

            return nx.shortest_path_length(
                self.to_networkx(),
                u,
                v,
                weight="length",
            )

        except nx.NetworkXNoPath:

            return float("inf")

    def safe_distance(
        self,
        u: Node,
        v: Node,
    ) -> float:

        try:

            return nx.shortest_path_length(
                self.safe_graph(),
                u,
                v,
                weight="length",
            )

        except nx.NetworkXNoPath:

            return float("inf")

    def detour_factor(
        self,
        u: Node,
        v: Node,
    ) -> float:

        d_all = self.distance(u, v)

        if d_all == float("inf"):
            return float("inf")

        d_safe = self.safe_distance(u, v)

        return d_safe / d_all

    # --------------------------------------------------------
    # structural parameters
    # --------------------------------------------------------

    def max_degree(self) -> int:
        G = self.to_networkx()

        if G.number_of_nodes() == 0:
            return 0

        return max(
            degree
            for _, degree in G.degree()
        )

    def average_degree(self) -> float:
        G = self.to_networkx()

        if G.number_of_nodes() == 0:
            return 0.0

        return sum(
            degree
            for _, degree in G.degree()
        ) / G.number_of_nodes()

    def feedback_edge_number(self) -> int:
        """
        Exact minimum feedback edge set size.

        m - n + c
        """

        G = self.to_networkx()

        n = G.number_of_nodes()
        m = G.number_of_edges()
        c = nx.number_connected_components(G)

        return m - n + c

    def treewidth_upper_bound(self) -> int:
        """
        Min-fill heuristic upper bound.
        """

        tw, _ = nx.approximation.treewidth_min_fill_in(
            self.to_networkx()
        )

        return tw

    def treewidth_bounds(self):
        G = self.to_networkx()

        tw_fill, _ = nx.approximation.treewidth_min_fill_in(G)
        tw_deg, _ = nx.approximation.treewidth_min_degree(G)

        return {
            "min_fill": tw_fill,
            "min_degree": tw_deg,
        }

    def treewidth_decomposition(self):
        """
        Returns:
            (treewidth upper bound,
             decomposition)
        """

        return nx.approximation.treewidth_min_fill_in(
            self.to_networkx()
        )

    def is_planar(self) -> bool:
        """
        Returns True iff the underlying graph is planar.
        """

        G = self.to_networkx()

        planar, _ = nx.check_planarity(G)

        return planar

    def planar_embedding(self):
        """
        Returns a planar embedding if the graph is planar.

        Returns
        -------
        embedding : nx.PlanarEmbedding

        Raises
        ------
        ValueError
            If the graph is not planar.
        """

        G = self.to_networkx()

        planar, embedding = nx.check_planarity(G)

        if not planar:
            raise ValueError(
                "Graph is not planar."
            )

        return embedding

    def nonplanarity_witness(self):
        """
        Returns a Kuratowski subgraph if the graph is non-planar.
        """

        G = self.to_networkx()

        planar, witness = nx.check_planarity(
            G,
            counterexample=True,
        )

        if planar:
            return None

        return witness
    # --------------------------------------------------------
    # OD pair generation
    # --------------------------------------------------------

    def random_od_pairs(
        self,
        r: int,
        seed=None,
    ) -> List[Tuple[Node, Node]]:

        rng = random.Random(seed)

        vertices = list(self.vertices)

        pairs = []

        while len(pairs) < r:

            s, t = rng.sample(vertices, 2)

            if self.distance(s, t) < float("inf"):

                pairs.append((s, t))

        return pairs

    def all_connected_pairs(
        self,
    ) -> List[Tuple[Node, Node]]:

        vertices = list(self.vertices)

        pairs = []

        for i in range(len(vertices)):

            for j in range(i + 1, len(vertices)):

                u = vertices[i]
                v = vertices[j]

                if self.distance(u, v) < float("inf"):

                    pairs.append((u, v))

        return pairs

    # --------------------------------------------------------
    # upgrade simulation
    # --------------------------------------------------------

    def upgraded_graph(
        self,
        F: Set[Edge],
    ) -> nx.Graph:

        G = nx.Graph()

        G.add_nodes_from(self.vertices)

        for e in self.safe_edges:

            G.add_edge(
                *e,
                length=self.length[e],
            )

        for e in F:

            G.add_edge(
                *e,
                length=self.length[e],
            )

        return G

    def distance_after_upgrade(
        self,
        u: Node,
        v: Node,
        F: Set[Edge],
    ) -> float:

        try:

            return nx.shortest_path_length(
                self.upgraded_graph(F),
                u,
                v,
                weight="length",
            )

        except nx.NetworkXNoPath:

            return float("inf")

    def is_pair_served(
        self,
        u: Node,
        v: Node,
        F: Set[Edge],
        alpha: float,
    ) -> bool:

        d_original = self.distance(u, v)

        d_safe = self.distance_after_upgrade(
            u,
            v,
            F,
        )

        return d_safe <= alpha * d_original

    def upgrade_cost_of(
        self,
        F: Set[Edge],
    ) -> float:

        return sum(
            self.upgrade_cost[e]
            for e in F
        )

    def is_feasible_solution(
        self,
        F: Set[Edge],
        od_pairs: List[Tuple[Node, Node]],
        alpha: float,
        budget: float,
    ) -> bool:

        if self.upgrade_cost_of(F) > budget:
            return False

        for s, t in od_pairs:

            if not self.is_pair_served(
                s,
                t,
                F,
                alpha,
            ):
                return False

        return True

    # --------------------------------------------------------
    # File naming / storage
    # --------------------------------------------------------

    def filename(self):
        """
        Canonical filename derived from metadata.

        Examples:

            bartholomae_pes

            bartholomae_r5_pes
        """

        parts = [self.origin_key]

        if self.origin_type == "region":

            parts.append(
                f"r{int(self.radius_km)}"
            )

        parts.append(
            self.safety_model
        )

        return "_".join(parts)


    def directory(
        self,
        root="data/networks",
    ):
        """
        Canonical storage directory.

        Examples:

            data/networks/villages

            data/networks/regions
        """

        return os.path.join(
            root,
            self.origin_type + "s",
        )


    def path(
        self,
        root="data/networks",
        extension=".pkl",
    ):
        """
        Full canonical path.
        """

        return os.path.join(
            self.directory(root),
            self.filename() + extension,
        )


    # --------------------------------------------------------
    # Pickle storage
    # --------------------------------------------------------

    def save(
        self,
        root="data/networks",
    ):
        """
        Save network as pickle using canonical path.
        """

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
        """
        Load network from canonical pickle path.
        """
        import pickle

        with open(path, "rb") as f:

            return pickle.load(f)


    @classmethod
    def load_by_specifics(
        cls,
        origin_key,
        safety_model,
        origin_type="village",
        radius_km=None,
        root="data/networks",
    ):
        """
        Load network from canonical pickle path.
        """

        import os
        import pickle

        parts = [origin_key]

        if radius_km is not None:
            origin_type = "region"

        if origin_type == "region":

            parts.append(
                f"r{int(radius_km)}"
            )

        parts.append(
            safety_model
        )

        filename = "_".join(parts) + ".pkl"

        path = os.path.join(
            root,
            origin_type + "s",
            filename,
        )

        with open(path, "rb") as f:

            return pickle.load(f)


    # --------------------------------------------------------
    # JSON export
    # --------------------------------------------------------

    def to_json(
        self,
        root="data/networks",
    ):
        """
        Export network as JSON using canonical path.
        """

        import os
        import json

        path = self.path(
            root=root,
            extension=".json",
        )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        data = {

            "origin_key":
                self.origin_key,

            "origin_type":
                self.origin_type,

            "radius_km":
                self.radius_km,

            "safety_model":
                self.safety_model,

            "info":
                self.info,

            "vertices":
                list(self.vertices),

            "safe_edges":
                [
                    list(e)
                    for e in self.safe_edges
                ],

            "unsafe_edges":
                [
                    list(e)
                    for e in self.unsafe_edges
                ],

            "length":
                {
                    str(e): self.length[e]
                    for e in self.length
                },

            "upgrade_cost":
                {
                    str(e): self.upgrade_cost[e]
                    for e in self.upgrade_cost
                },

            "node_x":
                self.node_x,

            "node_y":
                self.node_y,
        }

        with open(path, "w") as f:

            json.dump(
                data,
                f,
                indent=2,
            )

        return path


    # --------------------------------------------------------
    # JSON import
    # --------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        origin_key,
        safety_model,
        origin_type="village",
        radius_km=None,
        root="data/networks",
    ):
        """
        Load network from canonical JSON path.
        """

        import os
        import json
        import ast

        parts = [origin_key]

        if origin_type == "region":

            parts.append(
                f"r{int(radius_km)}"
            )

        parts.append(
            safety_model
        )

        filename = "_".join(parts) + ".json"

        path = os.path.join(
            root,
            origin_type + "s",
            filename,
        )

        with open(path, "r") as f:

            data = json.load(f)

        return cls(

            vertices=set(
                data["vertices"]
            ),

            safe_edges={
                tuple(e)
                for e in data["safe_edges"]
            },

            unsafe_edges={
                tuple(e)
                for e in data["unsafe_edges"]
            },

            length={
                ast.literal_eval(k): v
                for k, v in data["length"].items()
            },

            upgrade_cost={
                ast.literal_eval(k): v
                for k, v in data["upgrade_cost"].items()
            },

            node_x=data.get(
                "node_x",
                {},
            ),

            node_y=data.get(
                "node_y",
                {},
            ),

            edge_geometry={},

            origin_key=data.get(
                "origin_key",
                "",
            ),

            origin_type=data.get(
                "origin_type",
                "",
            ),

            radius_km=data.get(
                "radius_km",
                None,
            ),

            safety_model=data.get(
                "safety_model",
                "",
            ),

            info=data.get(
                "info",
                "",
            ),
        )

    # ============================================================
    # Exact minimum feedback edge set
    # ============================================================

    def minimum_feedback_edge_set(self) -> Set[Edge]:
        """
        Returns an exact minimum feedback edge set.

        For an undirected graph:
            E \ spanning forest.
        """

        G = self.to_networkx()

        forest_edges = set()

        for component in nx.connected_components(G):

            H = G.subgraph(component)

            T = nx.minimum_spanning_tree(
                H,
                weight="length",
            )

            for u, v in T.edges():

                forest_edges.add(
                    self.canonical_edge(u, v)
                )

        return self.all_edges() - forest_edges


    def check_feedback_edge_set(self) -> bool:

        F = self.minimum_feedback_edge_set()

        assert len(F) == self.feedback_edge_number()

        G = self.to_networkx().copy()

        G.remove_edges_from(F)

        assert nx.is_forest(G)

        return True


    # ============================================================
    # Unsafe-edge graph structure
    # ============================================================

    def unsafe_connected_components(self):

        G = self.unsafe_graph()

        return list(
            nx.connected_components(G)
        )


    def number_of_unsafe_components(self) -> int:

        return len(
            self.unsafe_connected_components()
        )


    def unsafe_component_sizes(self):

        return sorted(
            (
                len(C)
                for C in self.unsafe_connected_components()
            ),
            reverse=True,
        )


    def unsafe_component_edge_counts(self):

        G = self.unsafe_graph()

        counts = []

        for C in nx.connected_components(G):

            H = G.subgraph(C)

            counts.append(
                H.number_of_edges()
            )

        return sorted(
            counts,
            reverse=True,
        )


    def unsafe_edge_fraction(self) -> float:

        m = self.number_of_edges()

        if m == 0:
            return 0.0

        return (
            self.number_of_unsafe_edges()
            / m
        )


    def largest_unsafe_component_fraction(self) -> float:

        sizes = self.unsafe_component_edge_counts()

        if not sizes:
            return 0.0

        return (
            sizes[0]
            / self.number_of_unsafe_edges()
        )


    # ============================================================
    # Diameter statistics
    # ============================================================

    def diameter(self) -> int:
        """
        Unweighted diameter of the largest connected component.
        """

        G = self.to_networkx()

        if not nx.is_connected(G):

            largest = max(
                nx.connected_components(G),
                key=len,
            )

            G = G.subgraph(largest)

        return nx.diameter(G)


    def weighted_diameter(self) -> float:
        """
        Length-weighted diameter of the largest
        connected component.
        """

        G = self.to_networkx()

        if not nx.is_connected(G):

            largest = max(
                nx.connected_components(G),
                key=len,
            )

            G = G.subgraph(largest)

        max_dist = 0.0

        for _, dist in nx.all_pairs_dijkstra_path_length(
            G,
            weight="length",
        ):

            local_max = max(
                dist.values()
            )

            max_dist = max(
                max_dist,
                local_max,
            )

        return max_dist


    # ============================================================
    # Average shortest-path statistics
    # ============================================================

    def average_shortest_path_length(self) -> float:
        """
        Unweighted average shortest-path length
        on the largest connected component.
        """

        G = self.to_networkx()

        if not nx.is_connected(G):

            largest = max(
                nx.connected_components(G),
                key=len,
            )

            G = G.subgraph(largest)

        return nx.average_shortest_path_length(G)


    def average_weighted_shortest_path_length(
        self,
    ) -> float:
        """
        Length-weighted average shortest-path length
        on the largest connected component.
        """

        G = self.to_networkx()

        if not nx.is_connected(G):

            largest = max(
                nx.connected_components(G),
                key=len,
            )

            G = G.subgraph(largest)

        return nx.average_shortest_path_length(
            G,
            weight="length",
        )

    #-----------------------------
    # PLOTTING
    #-----------------------------
    def plot(
        self,
        figsize=(10, 10),
        cmap="YlOrRd",
        linewidth=2,
        show=True,
    ):
        """
        Geographic plot.

        Safe edges:
            black

        Unsafe edges:
            heat map according to upgrade cost
        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fig, ax = plt.subplots(
            figsize=figsize
        )

        #
        # unsafe edges color scale
        #
        if self.unsafe_edges:

            costs = [
                self.upgrade_cost[e]
                for e in self.unsafe_edges
            ]

            norm = mpl.colors.Normalize(
                vmin=min(costs),
                vmax=max(costs),
            )

            mapper = mpl.cm.ScalarMappable(
                norm=norm,
                cmap=cmap,
            )

        else:

            mapper = None

        #
        # safe edges
        #
        for e in self.safe_edges:

            self._draw_edge(
                ax,
                e,
                color="black",
                linewidth=linewidth,
            )

        #
        # unsafe edges
        #
        for e in self.unsafe_edges:

            color = mapper.to_rgba(
                self.upgrade_cost[e]
            )

            self._draw_edge(
                ax,
                e,
                color=color,
                linewidth=linewidth + 0.5,
            )

        #
        # colorbar
        #
        if mapper is not None:

            plt.colorbar(
                mapper,
                ax=ax,
                label="Upgrade cost",
            )

        ax.set_aspect("equal")

        ax.set_axis_off()

        ax.set_title(f"{self.origin_key} ({self.safety_model})")

        if show:
            plt.show()

        return fig, ax


    def _draw_edge(
        self,
        ax,
        e,
        color,
        linewidth,
    ):

        geom = self.edge_geometry.get(e)

        if geom is not None:

            if hasattr(geom, "xy"):

                x, y = geom.xy

                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=linewidth,
                )

                return

        u, v = e

        ax.plot(
            [self.node_x[u], self.node_x[v]],
            [self.node_y[u], self.node_y[v]],
            color=color,
            linewidth=linewidth,
        )


    # --------------------------------------------------------
    # Path scanning
    # --------------------------------------------------------

    @classmethod
    def village_paths(
        cls,
        root="data/networks",
    ):
        """
        Return paths of all stored village networks.
        """

        paths = []

        directory = os.path.join(
            root,
            "villages",
        )

        if not os.path.isdir(directory):
            return paths

        for filename in sorted(
            os.listdir(directory)
        ):

            if filename.endswith(".pkl"):

                paths.append(
                    os.path.join(
                        directory,
                        filename,
                    )
                )

        return paths


    @classmethod
    def region_paths(
        cls,
        root="data/networks",
    ):
        """
        Return paths of all stored region networks.
        """

        paths = []

        directory = os.path.join(
            root,
            "regions",
        )

        if not os.path.isdir(directory):
            return paths

        for filename in sorted(
            os.listdir(directory)
        ):

            if filename.endswith(".pkl"):

                paths.append(
                    os.path.join(
                        directory,
                        filename,
                    )
                )

        return paths


    @classmethod
    def all_paths(
        cls,
        root="data/networks",
    ):
        """
        Return paths of all stored networks.

        Does not load anything.
        """

        return (
            cls.village_paths(root)
            +
            cls.region_paths(root)
        )


    # --------------------------------------------------------
    # Bulk loading
    # --------------------------------------------------------

    @classmethod
    def all_saved_villages(
        cls,
        root="data/networks",
    ):
        """
        Load all stored village networks.
        """

        return [
            cls.load(path)
            for path in cls.village_paths(root)
        ]


    @classmethod
    def all_saved_regions(
        cls,
        root="data/networks",
    ):
        """
        Load all stored region networks.
        """

        return [
            cls.load(path)
            for path in cls.region_paths(root)
        ]


    @classmethod
    def all_saved_networks(
        cls,
        root="data/networks",
    ):
        """
        Load all stored benchmark networks.
        """

        return [
            cls.load(path)
            for path in cls.all_paths(root)
        ]
