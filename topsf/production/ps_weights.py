# coding: utf-8

"""
Producers related to event weights.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


ak = maybe_import("awkward")


@producer(
    uses={
        "PSWeight",
    },
    produces={
        "ISR", "ISR_up", "ISR_down",
        "FSR", "FSR_up", "FSR_down",
    },
)
def ps_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    PS weights producer.
    PS weight [0]: ISR=2 FSR=1
    PS weight [2]: ISR=0.5 FSR=1
    PS weight [1]: ISR=1 FSR=2
    PS weight [3]: ISR=1 FSR=0.5
    """
    events = set_ak_column(events, "ISR_up", events.PSWeight[:, 0])
    events = set_ak_column(events, "FSR_up", events.PSWeight[:, 1])
    events = set_ak_column(events, "ISR_down", events.PSWeight[:, 2])
    events = set_ak_column(events, "FSR_down", events.PSWeight[:, 3])
    events = set_ak_column(events, "ISR", 1.0)
    events = set_ak_column(events, "FSR", 1.0)
    return events
