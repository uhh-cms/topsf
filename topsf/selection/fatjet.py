# coding: utf-8

"""
Selectors for large-radius jets.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.util import masked_sorted_indices
from topsf.selection.lepton import lepton_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "event",
        lepton_selection,
    },
)
def fatjet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select AK8 jets that are well separated from the lepton.
    """

    # get selection parameters from the config
    self.cfg = self.config_inst.x.jet_selection.ak8

    # choose jet column
    fatjet = events[self.cfg.column]

    # select jets
    fatjet_mask = (
        (abs(fatjet.eta) < self.cfg.max_abseta) &
        (fatjet.pt > self.cfg.min_pt)
    )
    fatjet_indices = masked_sorted_indices(fatjet_mask, fatjet.pt)

    # lepton
    events = self[lepton_selection](events, **kwargs)
    lepton = events.Lepton

    # split fatjet collection into near and far from lepton
    fatjet_lepton_deltar = fatjet.delta_r(lepton)
    fatjet_mask_near = (
        fatjet_mask &
        (fatjet_lepton_deltar <= 2 / 3 * np.pi)
    )
    fatjet_mask_far = (
        fatjet_mask &
        (fatjet_lepton_deltar > 2 / 3 * np.pi)
    )
    fatjet_indices_near = masked_sorted_indices(fatjet_mask_near, fatjet.pt)
    fatjet_indices_far = masked_sorted_indices(fatjet_mask_far, fatjet.pt)

    # return selection result
    return events, SelectionResult(
        steps={
            "FatJet": ak.fill_none(ak.num(fatjet_indices_far) >= 1, False),
        },
        objects={
            self.cfg.column: {
                self.cfg.column: fatjet_indices,
                "FatJetNear": fatjet_indices_near,
                "FatJetFar": fatjet_indices_far,
            },
        },
    )


@fatjet_selection.init
def fatjet_selection_init(self: Selector) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = config_inst.x.jet_selection.ak8

    # set input columns
    column = self.cfg.column
    self.uses |= {
        f"{column}.pt",
        f"{column}.eta",
        f"{column}.phi",
        f"{column}.mass",
    }
