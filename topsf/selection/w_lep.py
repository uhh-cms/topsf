# coding: utf-8

"""
Lepton-related selectors.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

from topsf.production.lepton import choose_lepton

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@selector(
    uses={choose_lepton, "MET.pt", "MET.phi"},
)
def w_lep_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select events with minimum amount of leptonic W pt.
    """

    # get lepton
    events = self[choose_lepton](events, **kwargs)

    # get leptonic W pt from lepton and missing energy
    met = events.MET
    lep = events.Lepton
    w_lep_pt = np.sqrt(
        (met.pt * np.cos(met.phi) + lep.pt * np.cos(lep.phi))**2 +
        (met.pt * np.sin(met.phi) + lep.pt * np.sin(lep.phi))**2,
    )

    # select events with leptonic W above pt threshold
    sel = (w_lep_pt > 120)

    # return selection result
    return events, SelectionResult(
        steps={
            "WLepPt": ak.fill_none(sel, False),
        },
        objects={},
    )
