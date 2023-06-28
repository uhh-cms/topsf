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
    },
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
    is_hadronic_top = False

    # get decay products of top quark
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_decay_products](events, **kwargs)

        t = events.GenTopDecay[:, :, 0]  # t quark
        b = events.GenTopDecay[:, :, 1]  # b quark
        #w = events.GenTopDecay[:, :, 2]  # W boson  # noqa
        q1_or_l = events.GenTopDecay[:, :, 3]  # light quark 1 / lepton
        q2_or_n = events.GenTopDecay[:, :, 4]  # light quark 2 / neutrino

        # top quark + decay products mapped to leading fat jet
        t_probejet_deltar = probejet.delta_r(t)
        t_probejet_indices = ak.argsort(t_probejet_deltar, axis=1, ascending=True)
        #t_probejet = ak.firsts(t[t_probejet_indices])  # noqa
        b_probejet = ak.firsts(b[t_probejet_indices])
        q1_or_l_probejet = ak.firsts(q1_or_l[t_probejet_indices])
        q2_or_n_probejet = ak.firsts(q2_or_n[t_probejet_indices])

        # leading fat jet properties
        is_hadronic_top = (abs(q1_or_l_probejet.pdgId) <= 4)
        is_b_merged = (probejet.delta_r(b_probejet) < merged_max_deltar)
        is_q1_or_l_merged = (probejet.delta_r(q1_or_l_probejet) < merged_max_deltar)
        is_q2_or_n_merged = (probejet.delta_r(q2_or_n_probejet) < merged_max_deltar)

        # number of decay products merged to jet
        n_merged = (
            ak.zeros_like(events.event, dtype=np.uint8) +
            is_q2_or_n_merged +
            is_q1_or_l_merged +
            is_b_merged
        )

    # write out columns
    events = set_ak_column(events, "ProbeJet.is_hadronic_top", is_hadronic_top)
    events = set_ak_column(events, "ProbeJet.n_merged", n_merged)
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
