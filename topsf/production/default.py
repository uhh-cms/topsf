# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from topsf.production.categories import category_ids
from topsf.production.weights import weights
from topsf.production.features import features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={category_ids, weights},
    produces={category_ids, weights},
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # weights
    events = self[weights](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    # high-level features (probe jet properties, etc.)
    events = self[features](events, **kwargs)

    return events
