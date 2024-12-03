# coding: utf-8

"""
Selectors for small-radius jets.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.util import masked_sorted_indices
from topsf.production.lepton import choose_lepton

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
    self.cfg = self.config_inst.x.jet_selection.get("ak4", "Jet")

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
    bjet_mask = (btag_val >= self.cfg.btag_wp)
    lightjet_mask = (btag_val < self.cfg.btag_wp)

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
    self.cfg = config_inst.x.jet_selection.get("ak4", {})

    # set input columns
    column = self.cfg.get("column", None)
    if column:
        self.uses |= {
            f"{column}.pt",
            f"{column}.eta",
            f"{column}.phi",
            f"{column}.mass",
            f"{column}.{self.cfg.btag_column}",
        }

    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }


@selector(
    uses={
        choose_lepton,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
    },
    produces={
        "Lepton.closest_jet_delta_r",
        "Lepton.closest_jet_pt_rel",
        "Lepton.closest_jet_separation",
    },
    exposed=True,
)
def jet_lepton_2d_selection(
    self: Selector,
    events: ak.Array,
    results: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    BJet/Lepton 2D cut to reduce QCD contribution.

    The 2D requirement is defined as

      delta_r(l, jet) / 0.4 (+) pt_rel(l, jet) / 30 GeV < 1

    where *l* is the main lepton, *jet* refers to the closest AK4
    jet, and (+) is the sum in quadrature.

    The quantity *pt_rel* denotes the magnitude of the perpendicular
    component of the lepton three-vector with respect to the jet axis:

      pt_rel = p_l * sin(angle(p_l, p_jet))

    and can be calculated eqivalently via the cross product of the jet
    and lepton three-momenta as:

      pt_rel = |cross(p_l, p_jet)| / |p_jet|
    """

    # get lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events.Lepton

    # get selected jets
    jet_indices = results.objects.Jet.Jet
    jet = events.Jet[jet_indices]

    # get jet closest to lepton
    jet_lepton_deltar = jet.delta_r(lepton)
    jet_closest_indices = ak.argsort(jet_lepton_deltar, axis=1, ascending=True)
    jet_closest = ak.firsts(jet[jet_closest_indices])

    # calculate 2D separation metric
    delta_r = jet_closest.delta_r(lepton)
    pt_rel = lepton.cross(jet_closest).p / jet_closest.p
    separation = (
        (delta_r / 0.4)**2 +
        (pt_rel / 30)**2
    )

    # select on the metric
    sel = (separation > 1)

    # write out columns
    events = set_ak_column(events, "Lepton.closest_jet_delta_r", delta_r)
    events = set_ak_column(events, "Lepton.closest_jet_pt_rel", pt_rel)
    events = set_ak_column(events, "Lepton.closest_jet_separation", separation)

    # return selection result
    return events, SelectionResult(
        steps={
            "JetLepton2DCut": ak.fill_none(sel, True),
        },
        objects={},
    )
