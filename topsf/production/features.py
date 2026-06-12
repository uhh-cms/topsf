# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from topsf.util import has_tag

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer
def jet_energy_shifts(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Pseudo-producer that registers jet energy shifts.
    """
    return events


@jet_energy_shifts.init
def jet_energy_shifts_init(self: Producer) -> None:
    """
    Register shifts.
    """
    self.shifts |= {
        f"jec_{junc_name}_{junc_dir}"
        for junc_name in self.config_inst.x.jec.Jet.uncertainty_sources
        for junc_dir in ("up", "down")
    } | {"jer_up", "jer_down"}


@producer(
    uses={
        attach_coffea_behavior,
        "event",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        # "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "Muon.pt",
        "Electron.pt",
        "MET.phi", "MET.pt",
    },
    produces={
        attach_coffea_behavior,
        "dummy",
        "n_jet",
        "n_fatjet",
        # "n_bjet",
        "n_muon",
        "n_electron",
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Producer for all high-level features."""

    # dummy to ensure at least one field
    events = set_ak_column(events, "dummy", ak.ones_like(events.event))

    # count jets and fatjets
    jet = ak.with_name(events.Jet, "Jet")
    fatjet = ak.without_parameters(events["FatJet"])
    # bjet = ak.without_parameters(events["BJet"])
    muon = ak.without_parameters(events["Muon"])
    electron = ak.without_parameters(events["Electron"])
    events = set_ak_column(events, "n_jet", ak.num(jet.pt, axis=-1))
    events = set_ak_column(events, "n_fatjet", ak.num(fatjet.pt, axis=-1))
    # events = set_ak_column(events, "n_bjet", ak.num(bjet.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(muon.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(electron.pt, axis=-1))

    jet = events.Jet[ak.argsort(events.Jet.pt, axis=1, ascending=False)]
    # bjet = events.BJet[ak.argsort(events.BJet.pt, axis=1, ascending=False)]
    fatjet = events.FatJet[ak.argsort(events.FatJet.pt, axis=1, ascending=False)]
    jet_phi_padded = ak.pad_none(jet.phi, 3, axis=1, clip=True)
    fatjet_phi_padded = ak.pad_none(fatjet.phi, 3, axis=1, clip=True)
    # bjet_phi_padded = ak.pad_none(bjet.phi, 3, axis=1, clip=True)

    for i in range(3):
        dphi_jet_met = np.abs(jet_phi_padded[:, i] - events.MET.phi)
        dphi_jet_met = ak.where(dphi_jet_met > np.pi, 2 * np.pi - dphi_jet_met, dphi_jet_met)
        events = set_ak_column(events, f"Jet_{i}_MET_delta_phi", dphi_jet_met)
    
    for i in range(3):
        dphi_fatjet_met = np.abs(fatjet_phi_padded[:, i] - events.MET.phi)
        dphi_fatjet_met = ak.where(dphi_fatjet_met > np.pi, 2 * np.pi - dphi_fatjet_met, dphi_fatjet_met)
        events = set_ak_column(events, f"FatJet_{i}_MET_delta_phi", dphi_fatjet_met)

    # for i in range(3):
    #     dphi_bjet_met = np.abs(bjet_phi_padded[:, i] - events.MET.phi)
    #     dphi_bjet_met = ak.where(dphi_bjet_met > np.pi, 2 * np.pi - dphi_bjet_met, dphi_bjet_met)
    #     events = set_ak_column(events, f"BJet_{i}_MET_delta_phi", dphi_bjet_met)

    # btag score for AK4 jets
    # get btagging working points for the given column from config
    if has_tag("skip_btag_weights", self.config_inst):
        print("Skipping btag weight features as 'skip_btag_weights' tag is set in config.")
        btag_col = self.config_inst.x.jet_selection.ak4.btag_column
        wp_dict = self.config_inst.x.btag_working_points[btag_col].fixed_wp
        edges = sorted(wp_dict.values())

        scores = jet[btag_col]

        # count how many WPs are passed
        buckets = sum(scores >= edge for edge in edges)

        jet = ak.with_field(jet, buckets, f"{btag_col}_buckets")
        events = set_ak_column(events, f"Jet.{btag_col}_buckets", buckets)

        # store btag pass/fail decision as boolean for each WP as well
        # 1: pass, 0: fail, -1: undefined (e.g. no jet or no score)
        for wp, edge in wp_dict.items():
            pass_fail = ak.where(scores >= edge, 1, 0)
            pass_fail = ak.where(ak.is_none(scores), -1, pass_fail)
            jet = ak.with_field(jet, pass_fail, f"{btag_col}_pass_{wp}")

        # set some default value for undefined btag scores, and the number of jets to store the features
        DEFAULT_VAL = -10.0
        N_JETS = 5

        # pad per-jet arrays to N_JETS so indexing per jet is safe even if events have fewer jets
        padded_scores = ak.pad_none(ak.nan_to_none(scores), N_JETS, axis=1)
        padded_buckets = ak.pad_none(buckets, N_JETS, axis=1)

        # precompute pass/fail arrays for each WP and pad them
        padded_pass_fail = {}
        for wp, edge in wp_dict.items():
            print(f"Computing pass/fail for WP {wp} with edge {edge}")
            pf_int = ak.where(scores >= edge, 1, 0)
            padded_pass_fail[wp] = ak.pad_none(pf_int, N_JETS, axis=1)
            print(f"  Pass/fail for WP {wp}: {pf_int}")

        for i in range(0, N_JETS):
            # take the i-th jet across events (padded with None where missing) and fill defaults
            score_i = padded_scores[:, i]
            score_i = ak.fill_none(score_i, DEFAULT_VAL)
            events = set_ak_column(events, f"Jet_{i}_{btag_col}", score_i)

            buckets_i = padded_buckets[:, i]
            buckets_i = ak.fill_none(buckets_i, 0)
            events = set_ak_column(events, f"Jet_{i}_{btag_col}_buckets", buckets_i)

            for wp in wp_dict.keys():
                pf_i = padded_pass_fail[wp][:, i]
                # convert missing -> -1 (undefined), keep 0/1 otherwise
                pf_i = ak.fill_none(pf_i, -1)
                events = set_ak_column(events, f"Jet_{i}_{btag_col}_pass_{wp}", pf_i)

    return events


@features.init
def features_init(self: Producer) -> None:
    self.produces |= {
        f"Jet_{i}_MET_delta_phi" for i in range(3)
    } | {
        f"FatJet_{i}_MET_delta_phi" for i in range(3)
    }
    # } | {
    #     f"BJet_{i}_MET_delta_phi" for i in range(3)
    # }
    if has_tag("skip_btag_weights", self.config_inst):
        btag_col = self.config_inst.x.jet_selection.ak4.btag_column
        fixed_wps = self.config_inst.x.btag_working_points[btag_col].fixed_wp.keys()
        self.produces |= {
            f"Jet_{i}_{btag_col}" for i in range(5)
        } | {
            f"Jet_{i}_{btag_col}_buckets" for i in range(5)
        } | {
            f"Jet_{i}_{btag_col}_pass_{wp}" for i in range(5) for wp in fixed_wps
        } | {
            f"Jet.{btag_col}_buckets"
        } | {
            f"Jet.{btag_col}"
        }
        self.uses |= {
            f"Jet.{btag_col}",
            }
