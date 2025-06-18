# coding: utf-8

"""
Column producers related to probe jet.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, optional_column as optional

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
        optional("GenPart.pt"),
        optional("GenPart.eta"),
        optional("GenPart.phi"),
        optional("GenPart.mass"),
    },
    produces={
        "ProbeJet.pt", "ProbeJet.eta", "ProbeJet.phi", "ProbeJet.mass",
        "ProbeJet.msoftdrop",
        "ProbeJet.tau3", "ProbeJet.tau2",
        # true if decay used to compute merging is hadronic
        "ProbeJet.is_hadronic_decay",
        # true if particles used to compute merging originate
        # from the decay of associated W boson and/or b quarks
        "ProbeJet.is_associated_decay",
        # number of decay products merged to probe jet
        "ProbeJet.n_merged",
        # merging decisions for each decay product
        "ProbeJet.merged_b",
        "ProbeJet.merged_w1",
        "ProbeJet.merged_w2",
    },
)
def probe_jet(
    self: Producer,
    events: ak.Array,
    merged_max_deltar: float = 0.8,
    top_selection_mode: str = "closest_hadronic",
    **kwargs,
) -> ak.Array:
    """
    Produce probe jet and related features (e.g. number of merged decay products).

    For processes involving one or two top quarks (`st` and `tt`, respectively),
    the delta-R separation between the top quark decay products and the probe jet
    is checked and the decay products are considered merged if the distance is smaller
    than *merged_max_deltar*.

    The probe jet used for the above comparison corresponds to the highest-pt
    large-radius jet in the hemisphere opposite the main lepton.

    The choice of top quark for the above comparison is controlled by *top_selection_mode*,
    which can take the following values:

    * ``closest_hadronic``: only hadronically-decaying top quarks are considered; if
                            multiple such top quarks are found, the one closest to the
                            probe jet will be used.
    * ``closest``: the top quark closest to the probe jet in delta-R will be used
    * ``largest``: the top quark with the largest number of decay products merged to the
                   probe jet will be used

    If no top quark is found satisfying the above criteria, the probe jet features that
    rely on merging information will be masked.

    For single-top production in association with a W boson, the top quark may
    decay leptonically, but the associated W boson decay may be hadronic. In this
    case, the number of merged quarks is determined not from the top quark decay
    products, but from the decay products of the associated hadronic W boson.
    The corresponding subprocess is determined as follows:

    Process     Top decay  Assoc. W decay    # of merged quarks  Subprocess
    -------     ---------  --------------    ------------------  ----
    st_tW       leptonic   hadronic          3                   st_bkg*
                                             2                   st_2q
                                             1 or 0              st_1o0q
    st_tW       leptonic   leptonic          any                 st_bkg
    st_other    leptonic   --                any                 st_bkg

    *) This case arises only if an additional associated b quark is found in close
       proximity to the probe jet. Since this b quark is not related to the actual
       top process, it is classified as background.

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
    has_probejet = ~ak.is_none(probejet, axis=0)

    # helper function for filling in missing values, but keeping
    # the positions corresponding to a missing probe jet masked
    def fill_none_mask_no_probejet(arr, value):
        return ak.mask(
            ak.fill_none(arr, value),
            has_probejet,
        )

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
    is_had_w = False
    is_assoc = False
    merged_b = False
    merged_w1 = False
    merged_w2 = False

    # get decay products of top quark
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_decay_products](events, **kwargs)

        def get_merged(decay, mode="closest_hadronic"):
            # check valid top selection mode
            if mode not in ("closest", "largest", "closest_hadronic"):
                raise ValueError(f"invalid mode '{mode}', expected one of: closest, closest_hadronic, largest")

            # obtain decay products of all top quarks
            t = decay[:, :, 0]  # t quark
            b = decay[:, :, 1]  # b quark
            w1 = decay[:, :, 3]  # light quark 1 / lepton
            w2 = decay[:, :, 4]  # light quark 2 / neutrino
            is_w_had = (abs(w1.pdgId) <= 5)

            # -- helper functions for filtering decay products

            def filter_closest(*args):
                # attach proper behavior to args which was lost in new Vector version
                t = ak.with_name(args[0], "PtEtaPhiMLorentzVector")
                t_probejet_deltar = probejet.delta_r(t)
                t_probejet_indices = ak.argsort(t_probejet_deltar, axis=1, ascending=True)
                return tuple(
                    ak.firsts(arg[t_probejet_indices], axis=1)
                    for arg in args
                )

            def filter_closest_hadronic(*args):
                # first, filter out non-hadronic decays
                args = tuple(arg[is_w_had] for arg in args)

                return filter_closest(*args)

            def is_merged(*args):
                return tuple(
                    probejet.delta_r(ak.with_name(arg, "PtEtaPhiMLorentzVector")) < merged_max_deltar
                    for arg in args
                )

            # method 1: merging calculation based on top quark with
            #           largest number of merged decay products
            if mode == "largest":
                # check merging criteria for each top quark
                merged_b, merged_w1, merged_w2 = is_merged(b, w1, w2)

                # number of decay products merged to probe jet for each top quark
                n_merged = (
                    ak.zeros_like(t.pt, dtype=np.uint8) +
                    merged_b +
                    merged_w1 +
                    merged_w2
                )

                # choose result by largest number of merged products
                idx_largest = ak.argmax(n_merged, keepdims=True, axis=-1)
                is_w_had = ak.firsts(is_w_had[idx_largest], axis=-1)
                merged_b = ak.firsts(merged_b[idx_largest], axis=-1)
                merged_w1 = ak.firsts(merged_w1[idx_largest], axis=-1)
                merged_w2 = ak.firsts(merged_w2[idx_largest], axis=-1)
                n_merged = ak.firsts(n_merged[idx_largest], axis=-1)

                # fill missing values, but keep events w/o probejet masked
                n_merged = fill_none_mask_no_probejet(n_merged, 0)
                is_w_had = fill_none_mask_no_probejet(is_w_had, False)
                merged_b = fill_none_mask_no_probejet(merged_b, False)
                merged_w1 = fill_none_mask_no_probejet(merged_w1, False)
                merged_w2 = fill_none_mask_no_probejet(merged_w2, False)

            # merging calculation based on top quark closest to probe jet,
            # with or w/o requiring it to decay hadronically
            elif mode in ("closest", "closest_hadronic"):
                _filter = filter_closest if mode == "closest" else filter_closest_hadronic

                # choose decay products for top quark closest to probe jet
                t, b, w1, w2, is_w_had = _filter(t, b, w1, w2, is_w_had)

                # quantities for top quark closest to probe jet
                merged_b, merged_w1, merged_w2 = is_merged(b, w1, w2)

                # fill missing values, but keep events w/o probejet masked
                is_w_had = fill_none_mask_no_probejet(is_w_had, False)
                merged_b = fill_none_mask_no_probejet(merged_b, False)
                merged_w1 = fill_none_mask_no_probejet(merged_w1, False)
                merged_w2 = fill_none_mask_no_probejet(merged_w2, False)

                # number of decay products merged to jet
                n_merged = (
                    ak.zeros_like(events.event, dtype=np.uint8) +
                    merged_b +
                    merged_w1 +
                    merged_w2
                )

            # return number of merged decay products
            # and individual bool arrays
            return n_merged, is_w_had, merged_b, merged_w1, merged_w2

        # info on top decay products merged to probe jet
        n_merged, is_had_w, merged_b, merged_w1, merged_w2 = get_merged(
            events.GenTopDecay,
            mode=top_selection_mode,
        )

        # if tW decay, also consider associated W boson and b quark
        if self.dataset_inst.has_tag("has_top_associated_w"):
            # info on top associated decay products merged to probe jet
            n_merged_assoc, is_had_w_assoc, merged_b_assoc, merged_w1_assoc, merged_w2_assoc = get_merged(
                events.GenTopAssociatedDecay,
                mode=top_selection_mode,
            )
            is_assoc = fill_none_mask_no_probejet((is_had_w_assoc & (~is_had_w)), False)

            def merge_assoc(arrs, arrs_assoc):
                assert len(arrs) == len(arrs_assoc)
                return tuple(
                    ak.where(
                        is_assoc,
                        arr_assoc,
                        arr,
                    )
                    for arr, arr_assoc in zip(arrs, arrs_assoc)
                )

            # use associated decay products if main top decay is not hadronic
            is_had_w, merged_b, merged_w1, merged_w2, n_merged = merge_assoc(
                [
                    is_had_w, merged_b, merged_w1,
                    merged_w2, n_merged,
                ],
                [
                    is_had_w_assoc, merged_b_assoc, merged_w1_assoc,
                    merged_w2_assoc, n_merged_assoc,
                ],
            )

    # write out columns
    events = set_ak_column(events, "ProbeJet.is_associated_decay", is_assoc)
    events = set_ak_column(events, "ProbeJet.is_hadronic_decay", is_had_w)

    events = set_ak_column(events, "ProbeJet.merged_b", merged_b)
    events = set_ak_column(events, "ProbeJet.merged_w1", merged_w1)
    events = set_ak_column(events, "ProbeJet.merged_w2", merged_w2)

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
