# coding: utf-8

"""
Selectors for b-tagged jets.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.lepton import lepton_selection
from topsf.selection.jet import jet_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "event",
        lepton_selection,
        jet_selection,
    },
)
def bjet_lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select events with at least one b-tagged AK4 jet close to the lepton.
    """
    jet_sel_cfg = self.config_inst.x.jet_selection.ak4

    # lepton
    events = self[lepton_selection](events, **kwargs)
    lepton = events.Lepton

    # b-tagged jets
    events, jet_results = self[jet_selection](events, **kwargs)
    bjet_indices = jet_results.objects[jet_sel_cfg.column]["BJet"]
    bjet = events[jet_sel_cfg.column][bjet_indices]

    # closest b-jet to the main lepton
    bjet_lepton_deltar = bjet.delta_r(lepton)
    lepton_closest_bjet_index = ak.firsts(
        bjet_indices[ak.argsort(bjet_lepton_deltar, ascending=True)],
    )

    # select events with b-jet close to the lepton
    sel_jet_lepton_delta_r = (ak.min(bjet_lepton_deltar, axis=1) < 2 / 3 * np.pi)

    # return selection result
    return events, SelectionResult(
        steps={
            "BJetLeptonDeltaR": ak.fill_none(sel_jet_lepton_delta_r, False),
        },
        objects={
            jet_sel_cfg.column: {
                "BJetLeptonDeltaR": ak.singletons(lepton_closest_bjet_index),
            },
        },
    )
