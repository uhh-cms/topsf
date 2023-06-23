# coding: utf-8

"""
Selectors for small-radius jets.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.util import masked_sorted_indices

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select AK4 jets with configured pt, eta requirements.
    """

    # get selection parameters from the config
    self.cfg = self.config_inst.x.jet_selection.ak4

    # choose jet column
    jet = events[self.cfg.column]

    # select jets
    jet_mask = (
        (abs(jet.eta) < self.cfg.max_abseta) &
        (jet.pt > self.cfg.min_pt)
    )
    jet_indices = masked_sorted_indices(jet_mask, jet.pt)

    # get b-tagging scores
    btag_val = jet[self.cfg.btag_column]

    # selection masks for b-tagged and non-b-tagged (light) jets
    bjet_mask = (btag_val >= self.btag_wp_value)
    lightjet_mask = (btag_val < self.btag_wp_value)

    # indices of the b-tagged and non-b-tagged (light) jets
    bjet_indices = masked_sorted_indices(bjet_mask, jet.pt)
    lightjet_indices = masked_sorted_indices(lightjet_mask, jet.pt)

    # return selection result
    return events, SelectionResult(
        steps={
            "Jet": ak.fill_none(ak.num(jet_indices) >= 1, False),
        },
        objects={
            self.cfg.column: {
                self.cfg.column: jet_indices,
                "BJet": bjet_indices,
                "LightJet": lightjet_indices,
            },
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = config_inst.x.jet_selection.ak4

    # set b-tagging WP cut value
    self.btag_wp_value = self.config_inst.x.btag_working_points.deepjet[self.cfg.btag_wp]

    # set input columns
    column = self.cfg.column
    self.uses |= {
        f"{column}.pt",
        f"{column}.eta",
        f"{column}.phi",
        f"{column}.mass",
        f"{column}.{self.cfg.btag_column}",
    }
