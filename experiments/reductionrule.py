#from abc import ABC, abstractmethod


#class ReductionRule(ABC):

    #name = "unnamed"

    #@abstractmethod
    #def apply(self, instance, *args, **kwargs):
        #pass

from abc import ABC, abstractmethod
import time


class ReductionRule(ABC):

    name = "unnamed"

    def __init__(self):

        self.last_runtime = 0.0

        self.total_runtime = 0.0

        self.num_calls = 0

    def run(
        self,
        instance,
        *args,
        **kwargs,
    ):

        start = time.perf_counter()

        reduced = self.apply(
            instance,
            *args,
            **kwargs,
        )

        runtime = (
            time.perf_counter()
            - start
        )

        self.last_runtime = runtime

        self.total_runtime += runtime

        self.num_calls += 1

        return reduced

    @property
    def average_runtime(self):

        if self.num_calls == 0:
            return 0.0

        return (
            self.total_runtime
            / self.num_calls
        )

    @abstractmethod
    def apply(
        self,
        instance,
        *args,
        **kwargs,
    ):
        pass
