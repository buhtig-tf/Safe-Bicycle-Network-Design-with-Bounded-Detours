from dataclasses import dataclass


@dataclass(frozen=True)
class Village:

    key: str

    name: str

    population: int

    state: str

    osm_id: int | None = None

    osm_type: str | None = None

    lat: float | None = None

    lon: float | None = None

    @property
    def short_name(self):

        return self.name.split(
            ","
        )[0]
