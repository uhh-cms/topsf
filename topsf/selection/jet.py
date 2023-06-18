# coding: utf-8

"""
Lepton-related selectors.
"""

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.util import masked_sorted_indices
from topsf.selection.lepton import lepton_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Select AK4 jets with configured pt, eta requirements.
    """

    # get selection parameters from the config
    self.cfg = self.config_inst.x.jet_selection["ak4"]

    # choose jet column
    jet = events[self.cfg.column]

    # select jets
    jet_mask = (
        (abs(jet.eta) < self.cfg.max_abseta) &
        (jet.pt > self.cfg.min_pt)
    )
    jet_indices = masked_sorted_indices(jet_mask, jet.pt)

    # return selection result
    return SelectionResult(
        steps={},
        objects={
            self.cfg.column: {
                self.cfg.column: jet_indices,
            },
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # add standard jec and jer for mc, and only jec nominal for dta
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = config_inst.x.jet_selection[self.channel.name]

    # set input columns
    column = self.cfg.column
    self.uses |= {
        f"{column}.pt",
        f"{column}.eta",
    }


@selector(
    uses={
        jet_selection,
    },
)
def bjet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Create BJet and LightJet collections from AK4 jets with configured b-tagging requirements.
    """

    jet = self[jet_selection](events, **kwargs)
    assert not isinstance(jet, tuple), "need to call jet_selection before bjet_selection"

    # get b-tagging scores
    btag_val = jet[self.cfg.btag_column]

    # selection masks for b-tagged and non-b-tagged (light) jets
    bjet_mask = (btag_val >= self.btag_wp_value)
    lightjet_mask = (btag_val < self.btag_wp_value)

    # indices of the b-tagged and non-b-tagged (light) jets
    bjet_indices = masked_sorted_indices(bjet_mask, jet.pt)
    lightjet_indices = masked_sorted_indices(lightjet_mask, jet.pt)

    # return selection result
    return SelectionResult(
        steps={},
        objects={
            self.cfg.column: {
                "BJet": bjet_indices,
                "LightJet": lightjet_indices,
            },
        },
    )


@bjet_selection.init
def bjet_selection_init(self: Selector) -> None:
    # add standard jec and jer for mc, and only jec nominal for dta
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = config_inst.x.jet_selection["ak4"]
    self.btag_wp_value = self.config_inst.x.btag_working_points[self.cfg.btag_wp]
    self.btag_column = self.cfg.btag_column

    # set input columns
    column = self.cfg.column
    self.uses |= {
        f"{column}.pt",
        f"{column}.eta",
        f"{column}.{self.cfg.btag_column}",
    }


@selector(
    uses={
        "event",
        bjet_selection,
        lepton_selection,
    },
)
def bjet_lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Select events with at least one b-tagged AK4 jet close to the lepton.
    """

    # FIXME: cache of selection results needed
    lepton = self[lepton_selection](events, **kwargs)
    assert not isinstance(lepton, tuple), "need to call lepton_selection before bjet_lepton_selection"

    bjet = self[bjet_selection](events, **kwargs)
    assert not isinstance(bjet, tuple), "need to call bjet_selection before bjet_lepton_selection"

    # compute delta_r between lepton and nearest jet
    delta_r_bjet_lepton = ak.min(bjet.metric_table(lepton), axis=1)

    # event selection (TODO: make configurable)
    sel_jet_lepton_delta_r = (delta_r_bjet_lepton < 2 / 3 * np.pi)

    # return selection result
    return SelectionResult(
        steps={
            "BJetLeptonDeltaR": sel_jet_lepton_delta_r,
        },
    )
