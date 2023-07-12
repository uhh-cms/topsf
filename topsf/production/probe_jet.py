# coding: utf-8

"""
Column producers related to probe jet.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from topsf.selection.util import masked_sorted_indices
from topsf.production.util import lv_mass
from topsf.production.lepton import choose_lepton
from topsf.production.gen_top import gen_top_decay_products

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        choose_lepton,
        gen_top_decay_products,
        # generator particle kinematics
        "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
    },
    produces={
        "ProbeJet.pt", "ProbeJet.eta", "ProbeJet.phi", "ProbeJet.mass",
        "ProbeJet.msoftdrop",
        "ProbeJet.tau3", "ProbeJet.tau2",
        "ProbeJet.is_hadronic_top",
        "ProbeJet.n_merged",
        "ProbeJet.n_merged_closest",
    },
    check_columns_present={"produces"},  # some used columns optional
)
def probe_jet(
    self: Producer,
    events: ak.Array,
    merged_max_deltar: float = 0.8,
    **kwargs,
) -> ak.Array:
    """
    Produce probe jet and related features (merge category)
    """

    # choose jet column
    fatjet = events[self.cfg.column]

    # get lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events.Lepton

    # TODO: code below duplicated, move to common producer (?)

    # get fatjets on the far side of lepton
    fatjet_lepton_deltar = lv_mass(fatjet).delta_r(lepton)
    fatjet_mask = (
        (fatjet_lepton_deltar > 2 / 3 * np.pi)
    )
    fatjet_indices = masked_sorted_indices(fatjet_mask, fatjet.pt, ascending=False)
    fatjet_far = fatjet[fatjet_indices]

    # probe jet is leading fat jet on the opposite side of the lepton
    probejet = ak.firsts(fatjet_far, axis=1)

    # unite subJetIdx* into ragged column
    subjet_idxs = []
    for i in (1, 2):
        subjet_idx = probejet[f"subJetIdx{i}"]
        subjet_idx = ak.singletons(ak.mask(subjet_idx, subjet_idx >= 0))
        subjet_idxs.append(subjet_idx)
    subjet_idxs = ak.concatenate(subjet_idxs, axis=1)

    # get btag score for probejet subjets
    subjet_btag = events[self.cfg.subjet_column][self.cfg.subjet_btag]
    probejet_subjet_btag_scores = subjet_btag[subjet_idxs]

    # default values for non-top samples
    n_merged = 0
    n_merged_closest = 0
    is_t_hadronic_closest = False

    # get decay products of top quark
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_decay_products](events, **kwargs)

        # obtain decay products of all top quarks
        t = events.GenTopDecay[:, :, 0]  # t quark
        b = events.GenTopDecay[:, :, 1]  # b quark
        q1_or_l = events.GenTopDecay[:, :, 3]  # light quark 1 / lepton
        q2_or_n = events.GenTopDecay[:, :, 4]  # light quark 2 / neutrino

        #
        # method 1: n_merged calculation based on all top quarks
        #

        # check merging criteria for each top quark
        is_b_merged = probejet.delta_r(b) < merged_max_deltar
        is_q1_or_l_merged = probejet.delta_r(q1_or_l) < merged_max_deltar
        is_q2_or_n_merged = probejet.delta_r(q2_or_n) < merged_max_deltar

        # number of decay products merged to probe jet for each top quark
        n_merged = (
            ak.zeros_like(t, dtype=np.uint8) +
            is_b_merged +
            is_q1_or_l_merged +
            is_q2_or_n_merged
        )

        # non-hadronic tops are considered not merged
        is_t_hadronic = (abs(q1_or_l.pdgId) <= 4)
        n_merged = ak.where(is_t_hadronic, n_merged, 0)

        # choose largest number of merged products
        n_merged = ak.max(n_merged, axis=-1)

        #
        # method 2: n_merged calculation based on top quark closest to probe jet
        #

        # get decay products of top quark closest to probe jet
        t_probejet_deltar = probejet.delta_r(t)
        t_probejet_indices = ak.argsort(t_probejet_deltar, axis=1, ascending=True)
        b_closest = ak.firsts(b[t_probejet_indices])
        q1_or_l_closest = ak.firsts(q1_or_l[t_probejet_indices])
        q2_or_n_closest = ak.firsts(q2_or_n[t_probejet_indices])

        # quantities for top quark closest to probe jet
        is_b_closest_merged = (probejet.delta_r(b_closest) < merged_max_deltar)
        is_q1_or_l_closest_merged = (probejet.delta_r(q1_or_l_closest) < merged_max_deltar)
        is_q2_or_n_closest_merged = (probejet.delta_r(q2_or_n_closest) < merged_max_deltar)

        # non-hadronic tops are considered not merged
        is_t_hadronic_closest = (abs(q1_or_l_closest.pdgId) <= 4)
        n_merged_closest = ak.where(is_t_hadronic_closest, n_merged, 0)

        # number of decay products merged to jet
        n_merged_closest = (
            ak.zeros_like(events.event, dtype=np.uint8) +
            is_b_closest_merged +
            is_q1_or_l_closest_merged +
            is_q2_or_n_closest_merged
        )

    # write out columns
    events = set_ak_column(events, "ProbeJet.n_merged", n_merged)
    events = set_ak_column(events, "ProbeJet.n_merged_closest", n_merged_closest)
    events = set_ak_column(events, "ProbeJet.is_hadronic_top", is_t_hadronic_closest)
    for v in ("pt", "eta", "phi", "mass", "tau3", "tau2", "msoftdrop"):
        events = set_ak_column(events, f"ProbeJet.{v}", probejet[v])

    events = set_ak_column(
        events,
        f"ProbeJet.subjet_btag_scores_{self.cfg.subjet_btag}",
        probejet_subjet_btag_scores,
    )

    return events


@probe_jet.init
def probe_jet_init(self: Producer) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = self.config_inst.x.jet_selection.ak8

    # set input columns
    self.uses |= {
        # kinematics
        f"{self.cfg.column}.pt",
        f"{self.cfg.column}.eta",
        f"{self.cfg.column}.phi",
        f"{self.cfg.column}.mass",
        f"{self.cfg.column}.msoftdrop",
        # n-subjettiness variables
        f"{self.cfg.column}.tau3",
        f"{self.cfg.column}.tau2",
        # subjet indices
        f"{self.cfg.column}.subJetIdx1",
        f"{self.cfg.column}.subJetIdx2",
        f"{self.cfg.subjet_column}.{self.cfg.subjet_btag}",
    }

    self.produces |= {
        f"ProbeJet.subjet_btag_scores_{self.cfg.subjet_btag}",
    }
