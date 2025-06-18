# coding: utf-8

"""
Selection methods defining masks for categories.
"""
from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic categorizers

@categorizer(uses={"event"})
def sel_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Passes every event."""
    return events, ak.ones_like(events.event, dtype=bool)


#
# channel categorizers
#

@categorizer(uses={"event", "channel_id"})
def sel_1m(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the muon channel."""
    ch = self.config_inst.get_channel("mu")
    return events, (events["channel_id"] == ch.id)


@categorizer(uses={"event", "channel_id"})
def sel_1e(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the electron channel."""
    ch = self.config_inst.get_channel("e")
    return events, (events["channel_id"] == ch.id)


# @categorizer(uses={"event", "channel_id"})
# def sel_lepton(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
#     """Combined electron and muon channel, select events either in muon or electron channel."""
#     ch_m = self.config_inst.get_channel("mu")
#     ch_e = self.config_inst.get_channel("e")
#     mask = (events["channel_id"] == ch_m.id) or (events["channel_id"] == ch_e.id)
#     return events, mask
