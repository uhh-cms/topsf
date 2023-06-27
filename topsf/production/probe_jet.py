# coding: utf-8

"""
Column producers related to probe jet.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from topsf.production.gen_top import gen_top_decay_products

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        gen_top_decay_products,
        # generator particle kinematics
        "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
        # fat (AK8) jet kinematics
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.msoftdrop",
        # n-subjettiness variables
        "FatJet.tau3", "FatJet.tau2",
        # subjets
        "FatJet.subJetIdx1", "FatJet_subJetIdx2",
        "SubJet.btagDeepB",  # DeepCSV (TODO: DeepJet)
    },
    produces={
        "ProbeJet.pt", "ProbeJet.eta", "ProbeJet.phi", "ProbeJet.mass",
        "ProbeJet.msoftdrop",
        "ProbeJet.tau3", "ProbeJet.tau2",
        "ProbeJet.is_hadronic_top",
        "ProbeJet.n_merged",
        "ProbeJet.subjet_1_btag_score_deepcsv",
        "ProbeJet.subjet_2_btag_score_deepcsv",
    },
)
def probe_jet(
    self: Producer,
    events: ak.Array,
    results: ak.Array,
    merged_max_deltar: float = 0.8,
    **kwargs,
) -> ak.Array:
    """
    Produce probe jet and related features (merge category)
    """

    # obtain fat jets on opposite side of lepton
    fatjet_far_indices = results.objects.FatJet.FatJetFar
    fatjet_far = events.FatJet[fatjet_far_indices]

    # probe jet is leading fat jet on the opposite side of the lepton
    probejet = ak.firsts(fatjet_far, axis=1)

    # fill subjet btag scores
    for i in (1, 2):
        subjet_idx = probejet[f"subJetIdx{i}"]
        max_idx = ak.max(subjet_idx, axis=0)
        probejet[f"subjet_{i}_btag_score_deepcsv"] = ak.pad_none(
            events.SubJet["btagDeepB"],
            max_idx,
        )[subjet_idx]

    # default values for non-top samples
    n_merged = 0
    is_hadronic_top = False

    # get decay products of top quark
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_decay_products](events, **kwargs)

        t = events.gen_top_decay[:, :, 0]  # t quark
        b = events.gen_top_decay[:, :, 1]  # b quark
        #w = events.gen_top_decay[:, :, 2]  # W boson  # noqa
        q1_or_l = events.gen_top_decay[:, :, 3]  # light quark 1 / lepton
        q2_or_n = events.gen_top_decay[:, :, 4]  # light quark 2 / neutrino

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

    for i in (1, 2):
        events = set_ak_column(
            events,
            f"ProbeJet.subjet_{i}_btag_score_deepcsv",
            ak.fill_none(probejet[f"subjet_{i}_btag_score_deepcsv"], -1),
        )

    return events