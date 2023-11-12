# coding: utf-8

"""
Lepton-related selectors.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector
def met_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select events with minimum amount of missing transverse energy (MET).
    """

    # choose jet column
    met = events[self.cfg.column]

    # select met
    sel_met = (met.pt > self.cfg.min_pt)

    # return selection result
    return events, SelectionResult(
        steps={
            "MET": ak.fill_none(sel_met, False),
        },
        objects={},
    )


@met_selection.init
def met_selection_init(self: Selector) -> None:
    # initialize based on config, if it exists
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # get selection parameters from the config
    self.cfg = self.config_inst.x.met_selection.get("default", {})

    # set input columns
    column = self.cfg.get("column", None)
    if column:
        self.uses |= {
            f"{column}.pt",
        }
