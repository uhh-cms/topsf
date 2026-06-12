
# coding: utf-8

"""
Producers related to event weights.
"""
import law

from columnflow.production import Producer, producer
from columnflow.production.cms.electron import electron_weights, ElectronSFConfig
# from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights, MuonSFConfig
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.cms.btag import btag_weights, btag_wp_weights
from columnflow.production.cms.pdf import pdf_weights
from columnflow.util import maybe_import
from columnflow.selection import SelectionResult
from columnflow.columnar_util import fill_at, set_ak_column

from topsf.production.normalization import normalization_weights
from topsf.production.gen_top import top_pt_weight
from topsf.production.gen_v import vjets_weight
# from topsf.production.ps_weights import ps_weights
from columnflow.production.cms.parton_shower import ps_weights
from topsf.production.normalized_weights import normalized_weight_factory, normalized_btag_weights
from topsf.util import has_tag, record_calls

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


def high_pt_muon_reco(producer, corrector, variable_map):
    # for muon reco SFs, use pT and eta as variables and apply a mask to only compute weights for high pT muons
    momentum = variable_map["pt"] * np.cosh(variable_map["eta"])
    variable_map["p"] = momentum
    return variable_map


muon_reco_weights = muon_weights.derive(
    "muon_reco_weights",
    cls_dict={
        "weight_name": "muon_reco_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_reco_sf_config)),
        "update_corrector_variables": (lambda self, corrector, variables: high_pt_muon_reco(self, corrector, variables)),
    }
)
muon_id_weights = muon_weights.derive(
    "muon_id_weights",
    cls_dict={
        "weight_name": "muon_id_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_iso_sf_config)),
    }
)
muon_iso_weights = muon_weights.derive(
    "muon_iso_weights",
    cls_dict={
        "weight_name": "muon_iso_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_id_sf_config)),
    }
)
muon_trigger_weights = muon_weights.derive(
    "muon_trigger_weights",
    cls_dict={
        "weight_name": "muon_trigger_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_trigger_sf_config)),
    }
)

electron_reco_weights = electron_weights.derive(
    "electron_reco_weights",
    cls_dict={
        "weight_name": "electron_reco_weight",
        "get_electron_config": (lambda self: ElectronSFConfig.new(self.config_inst.x.electron_reco_sf_config)),
    }
)
electron_id_iso_weights = electron_weights.derive(
    "electron_id_iso_weights",
    cls_dict={
        "weight_name": "electron_id_iso_weight",
        "get_electron_config": (lambda self: ElectronSFConfig.new(self.config_inst.x.electron_id_iso_sf_config)),
    }
)

electron_trigger_weights = electron_weights.derive(
    "electron_trigger_weights",
    cls_dict={
        "weight_name": "electron_trigger_weight",
        "get_electron_config": (lambda self: ElectronSFConfig.new(self.config_inst.x.electron_trigger_sf_config)),
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
    }
)


@producer(
    uses={"Electron.pt"},
    produces={"electron_norm_fix_weight"},
    mc_only=True,
)
def electron_norm_fix_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Add a column with a flat 1.2 SF.
    """
    electron_mask = ak.num(events.Electron) == 1
    electron_norm_weight = ak.where(electron_mask, 1.2, 1.0)
    events = set_ak_column(events, "electron_norm_fix_weight", electron_norm_weight)
    return events


@producer(
    uses={
        muon_reco_weights,
        muon_id_weights,
        muon_iso_weights
    },
    produces={
        muon_reco_weights,
        muon_id_weights,
        muon_iso_weights
    },
    mc_only=True,
)
def muon_reco_id_iso_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute muon ID and isolation weights separately.
    """
    muon_mask = (events.Muon["pt"] >= 30) & (abs(events.Muon["eta"]) < 2.4)
    events = self[muon_reco_weights](events, muon_mask=muon_mask, **kwargs)
    events = self[muon_id_weights](events, muon_mask=muon_mask, **kwargs)
    events = self[muon_iso_weights](events, muon_mask=muon_mask, **kwargs)
    return events


@producer(
    uses={electron_reco_weights, electron_id_iso_weights},
    produces={electron_reco_weights, electron_id_iso_weights},
    mc_only=True,
)
def electron_reco_id_iso_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute electron reconstruction, ID and isolation weights separately.
    """
    if self.config_inst.x.year in [2022, 2023]:
        electron_mask = (events.Electron["pt"] >= 35)
    elif self.config_inst.x.year == 2024:
        electron_mask = ((events.Electron["pt"] >= 20.0) & (events.Electron["pt"] < 1000.0))
    events = self[electron_reco_weights](events, electron_mask=electron_mask, **kwargs)
    events = self[electron_id_iso_weights](events, electron_mask=electron_mask, **kwargs)
    return events


@producer
def weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Main event weight producer (e.g. MC generator, scale factors, normalization).
    """
    run_list = []
    with record_calls(self, run_list):
        if self.dataset_inst.is_mc:
            # compute normalization weights
            events = self[normalization_weights](events, **kwargs)

            # compute top pT weights
            if self.dataset_inst.has_tag("is_ttbar"):
                events = self[top_pt_weight](events, **kwargs)

            # compute V+jets K factor weights
            if not has_tag("skip_kfactor_weights", self.config_inst, self.dataset_inst, operator=any) and self.dataset_inst.has_tag("is_v_jets"):
                events = self[vjets_weight](events, **kwargs)

            if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
                events = self[electron_reco_id_iso_weights](events, **kwargs)
                events = self[electron_trigger_weights](events, **kwargs)
                events = self[electron_norm_fix_weights](events, **kwargs)

            if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
                events = self[muon_reco_id_iso_weights](events, **kwargs)
                events = self[muon_trigger_weights](events, **kwargs)

            # FIXME add trigger SF here

            # normalize event weights using stats
            events = self[normalized_pu_weights](events, **kwargs)

            if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying PS weights and normalizing them.")
                events = self[normalized_ps_weights](events, **kwargs)

            if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying scale weights and normalizing them.")
                events = self[normalized_scale_weights](events, **kwargs)

            if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying pdf weights and normalizing them.")
                events = self[normalized_pdf_weights](events, **kwargs)

            # compute btag weights
            if (
                has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any) and
                not has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any)
            ):
                logger.debug("Skipping shape based btag weights and applying fixed WP SF instead.")
                # skip shape based btag weights and apply fixed WP SF instead (for 2024)
                jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
                events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
            elif (
                has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
                not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
            ):
                logger.debug("Skipping fixed WP btag weights and applying shape based SF instead.")
                # apply shape based btag weights (for 2022/23)
                # and normalize
                jet_mask = (events.Jet["pt"] >= 100) & (abs(events.Jet["eta"]) < 2.5)
                events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)
                events = self[normalized_btag_weights](events, jet_mask=jet_mask, **kwargs)
            else:
                logger.warning("No btag weights applied.")

            # # compute MC weights
            # # already run in selection, not needed here?
            # events = self[mc_weight](events, **kwargs)

    logger.info_once(
        "Finished computing event weights:\n" +
        "\n".join(run_list)
    )

    return events


@weights.init
def weights_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        # dynamically add dependencies if running on MC
        if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {
                electron_reco_id_iso_weights,
                electron_trigger_weights,
                electron_norm_fix_weights,
                "Electron.{pt,eta}"
            }
            self.produces |= {
                electron_reco_id_iso_weights,
                electron_trigger_weights,
                electron_norm_fix_weights
            }

        if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {
                muon_reco_id_iso_weights,
                muon_trigger_weights,
                "Muon.{pt,eta,phi}"
            }
            self.produces |= {
                muon_reco_id_iso_weights,
                muon_trigger_weights
            }

        if not self.dataset_inst.has_tag("is_qcd"):
            self.uses |= {ps_weights}
            self.produces |= {ps_weights}

        if self.dataset_inst.has_tag("is_ttbar"):
            self.uses |= {top_pt_weight}
            self.produces |= {top_pt_weight}

        if not has_tag("skip_kfactor_weights", self.config_inst, self.dataset_inst, operator=any) and self.dataset_inst.has_tag("is_v_jets"):
            self.uses |= {vjets_weight}
            self.produces |= {vjets_weight}

        self.uses |= {normalization_weights, normalized_pu_weights}
        self.produces |= {normalization_weights, normalized_pu_weights}

        if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_ps_weights}
            self.produces |= {normalized_ps_weights}

        if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_scale_weights}
            self.produces |= {normalized_scale_weights}

        if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_pdf_weights}
            self.produces |= {normalized_pdf_weights}

        if (
            has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any)
        ):
            logger.warning_once("Using fixed wp b-tagging weights for 2024.")
            self.uses |= {btag_wp_weights}
            self.produces |= {btag_wp_weights}
        elif (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
        ):
            logger.warning_once("Using shape based b-tagging weights for 2022/2023.")
            self.uses |= {btag_weights, normalized_btag_weights}
            self.produces |= {btag_weights, normalized_btag_weights}
        else:
            logger.warning_once("No btag weights producer loaded.")


@producer(
    uses={
        pu_weight,
    },
    # produces={
    #     pu_weight,
    # },
    mc_only=True,
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """

    # compute pu weights
    events = self[pu_weight](events, **kwargs)
    # if self.has_dep(ps_weights):
    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        logger.debug("Compute PS weights for normalization")
        events = self[ps_weights](events, **kwargs)

    if (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
    ):
        # compute btag SF weights (for renormalization tasks)
        logger.debug("Compute btag weights for normalization")
        events = self[btag_weights](
            events,
            jet_mask=results.aux["jet_mask"],
            negative_b_score_action="ignore",
            negative_b_score_log_mode="debug",
            **kwargs,
        )

    # skip scale/pdf weights for some datasets (missing columns)
    # if self.has_dep(murmuf_envelope_weights):
    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(murmuf_envelope_weights):
        # compute scale weights
        logger.debug("Compute scale weights for normalization")
        events = self[murmuf_envelope_weights](events, **kwargs)

    # if self.has_dep(murmuf_weights):
    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(murmuf_weights):
        # read out mur and weights
        logger.debug("Compute murmuf weights for normalization")
        events = self[murmuf_weights](events, **kwargs)

    # if self.has_dep(pdf_weights):
    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(pdf_weights):
        # compute pdf weights
        logger.debug("Compute pdf weights for normalization")
        events = self[pdf_weights](
            events,
            outlier_threshold=0.99,
            outlier_action="remove",
            outlier_log_mode="debug",
            invalid_weights_action="ignore" if self.dataset_inst.has_tag("partial_lhe_weights") else "raise",
            **kwargs,
        )

    return events


@event_weights_to_normalize.init
def event_weights_to_normalize_init(self) -> None:
    # used Producers need to be set in the init or decorator
    if (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
    ):
        self.uses |= {btag_weights}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {murmuf_envelope_weights, murmuf_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {pdf_weights}

    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {ps_weights}


@event_weights_to_normalize.post_init
def event_weights_to_normalize_post_init(self, task: law.Task) -> None:
    # produced columns can be set in post_init to choose stored columns based on the shift
    for _cls in self.uses:
        if _cls == btag_weights and task.shift == "nominal":
            self.produces |= {btag_weights}
        elif _cls == btag_weights:
            self.produces |= self.deps[btag_weights].produced_columns
        elif task.shift == "nominal":
            self.produces |= self.deps[_cls].produced_columns
        else:
            self.produces |= {
                route for route in self.deps[_cls].produced_columns
                if not route.nano_column.endswith("_up") and not route.nano_column.endswith("_down")
            }


# renormalized weights
normalized_scale_weights = normalized_weight_factory(
    producer_name="normalized_scale_weights",
    weight_producers={murmuf_envelope_weights, murmuf_weights},
)
normalized_pdf_weights = normalized_weight_factory(
    producer_name="normalized_pdf_weights",
    weight_producers={pdf_weights},
)
normalized_pu_weights = normalized_weight_factory(
    producer_name="normalized_pu_weights",
    weight_producers={pu_weight},
)
normalized_ps_weights = normalized_weight_factory(
    producer_name="normalized_ps_weights",
    weight_producers={ps_weights},
)


@producer(
    uses={"mc_weight", "genWeight"},
    produces={"mc_weight", "genWeight"},
    mc_only=True,
)
def large_weights_killer(self: Producer, events: ak.Array, stats: dict, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # set mc_weight to zero when genWeight is > 0.5 for powheg HH events
    if self.dataset_inst.has_tag("is_hh") and self.dataset_inst.name.endswith("powheg"):
        # TODO: this feels very unsafe because genWeight can also be just 1 for all events. To be revisited
        weight_too_large = abs(events.genWeight) > 0.5
        logger.warning(f"found {ak.sum(weight_too_large)} HH events with genWeight > 0.5")

        events = fill_at(events, weight_too_large, "mc_weight", 0.0, value_type=np.float32)

    # check for anomalous weights and store in stats
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    anomalous_weights_mask = abs(events.mc_weight) > 1000 * median_weight
    if ak.any(anomalous_weights_mask):
        logger.warning(f"found {ak.sum(anomalous_weights_mask)} events with weights > 1000 * median weight")
        stats["num_events_anomalous_weights"] += ak.sum(anomalous_weights_mask)

    return events
