from dataclasses import dataclass

from villages import Village
from bicyclenetwork import BicycleNetwork


@dataclass(frozen=True)
class VillageRegion:

    key: str

    center: Village

    radius_km: float

    def network(
        self,
        safety_model="A",
    ):
        """
        Build the BicycleNetwork corresponding
        to the radius-limited region around the
        center village.

        Only the connected component containing
        the center village is kept.
        """

        return BicycleNetwork.from_place_radius(
            place=self.center,
            radius_km=self.radius_km,
            safety_model=safety_model,
        )

    def info(self):

        print(f"Key:         {self.key}")
        print(f"Center:      {self.center.name}")
        print(f"Population:  {self.center.population}")
        print(f"Radius (km): {self.radius_km}")

    def __str__(self):

        return (
            f"{self.key}\n"
            f"Center: {self.center.name}\n"
            f"Population: {self.center.population}\n"
            f"Radius: {self.radius_km} km"
        )
