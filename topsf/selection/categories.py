# coding: utf-8

"""
Selection methods defining masks for categories.
"""
from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

from topsf.production.gen_top import gen_top_decay_n_had

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic selectors

@selector(uses={"event"})
def sel_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Passes every event."""
    return ak.ones_like(events.event, dtype=bool)


#
# channel selectors
#

@selector(uses={"event", "channel_id"})
def sel_1m(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the muon channel."""
    ch = self.config_inst.get_channel("mu")
    return events["channel_id"] == ch.id


@selector(uses={"event", "channel_id"})
def sel_1e(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the electron channel."""
    ch = self.config_inst.get_channel("e")
    return events["channel_id"] == ch.id


#
# top process selectors
#

for n_had in range(2):
    @selector(
        uses={gen_top_decay_n_had},
        cls_name=f"sel_top_{n_had}h",
    )
    def sel_top_n_had(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
        f"""
        Select events with {n_had} hadronically decaying top quarks.
        """
        events = self[gen_top_decay_n_had](events, **kwargs)
        return (events.gen_top_decay_n_had == n_had)
