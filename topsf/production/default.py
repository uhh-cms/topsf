# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import

from topsf.production.weights import weights
from topsf.production.features import features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={features, weights},
    produces={features, weights},
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # weights
    events = self[weights](events, **kwargs)

    # high-level features (probe jet properties, etc.)
    events = self[features](events, **kwargs)

    return events
