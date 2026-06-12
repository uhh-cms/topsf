# coding: utf-8

"""
Event weight producer.
"""

from topsf.util import call_once_on_config
import law

from columnflow.util import maybe_import
from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route, set_ak_column
from topsf.util import has_tag

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# extend columnflow's default hist producer
@cf_default.hist_producer(uses={"mc_weight"}, mc_only=True)
def mc_weight(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.mc_weight


@cf_default.hist_producer(uses={"normalization_weight"}, mc_only=True)
def norm(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.normalization_weight


@cf_default.hist_producer(mc_only=True)
def no_weights(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.Array(np.ones(len(events), dtype=np.float32))


@cf_default.hist_producer(
    # uses={
    #     "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
    #     "Jet.btagUParTAK4B",
    #     "FatJet.pt",
    #     "Muon.pt",
    #     "Electron.pt",
    # },
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=False,
    # optional categorizer to obtain baseline event mask
    categorizer_cls=None,
    pre_label="",
)
def base(self: HistProducer, events: ak.Array, task: law.Task, **kwargs) -> ak.Array:
    # apply behavior (for variable reconstruction)

    # jet = ak.without_parameters(events["Jet"])
    # fatjet = ak.without_parameters(events["FatJet"])
    # muon = ak.without_parameters(events["Muon"])
    # electron = ak.without_parameters(events["Electron"])

    # apply mask
    if self.categorizer_cls:
        events, mask = self[self.categorizer_cls](events, **kwargs)
        events = events[mask]

    if self.dataset_inst.is_data:
        logger.debug(f"Dataset {self.dataset_inst} is data, not applying any weights")
        return events, ak.Array(np.ones(len(events), dtype=np.float32))

    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float64))
    logger.info(
        f"HistProducer '{self.cls_name}' (dataset {self.dataset_inst}) uses weight columns: \n"
        f"{', '.join(self.local_weight_columns.keys())}",
    )
    for column in self.local_weight_columns.keys():
        new_weight = weight * Route(column).apply(events)
        if not ak.any(new_weight != weight):
            logger.debug(f"Weight column {column} does not change the weight (all values are 1), skipping multiplication")
        else:
            weight = new_weight
    
    try:
        wmin = float(ak.min(weight))
        wmax = float(ak.max(weight))
        wmean = float(ak.mean(weight))
        logger.debug(f"Applied weight columns {', '.join(self.local_weight_columns.keys())}; weight stats: min={wmin:.3g}, mean={wmean:.3g}, max={wmax:.3g}")
    except Exception:
        logger.warning("Applied weight columns, but couldn't compute summary stats for weights")

    # implement dummy shift by varying weight by factor of 2
    if "dummy" in task.local_shift_inst.name:
        logger.warning("Applying dummy weight shift (should never be use for real analysis)")
        variation = task.local_shift_inst.name.split("_")[-1]
        weight = weight * {"up": 2.0, "down": 0.5}[variation]

    # special case: if only "weight_unweighted" is requested, we do not want to apply any weight at all
    if (
        hasattr(task, "variables") and
        len(task.variables) == 1 and
        task.variables[0].startswith("weight_unweighted")
    ):
        events = set_ak_column(events, "weight", weight, value_type=np.float32)
        return events, ak.Array(np.ones(len(events), dtype=np.float32))

    return events, weight


# @base.setup
# def base_setup(
#     self: HistProducer,
#     reqs: dict,
#     task: law.Task,
#     inputs: dict,
#     reader_targets: law.util.InsertableDict,
# ) -> None:
#     logger.debug(
#         f"HistProducer '{self.cls_name}' (dataset {self.dataset_inst}) uses weight columns: \n"
#         f"{', '.join(self.weight_columns.keys())}",
#     )
#     # if "dy_correction_weight" in self.local_weight_columns.keys():
#     #     # add dy_correction_weight to the reader targets
#     #     reader_targets["dy_correction_weight"] = inputs["dy_correction_weight_producer"]["columns"]

#     # if self.require_hbb_sf_producer and not task.dataset_inst.is_data:
#     #     # add hbb_sf_weights to the reader targets
#     #     reader_targets["hbb_sf_weights"] = inputs["hbb_sf_weights"]["columns"]

#     # if self.require_msd_nonclosure_producer:
#     #     # add msd_nonclosure_uncertainty to the reader targets
#     #     reader_targets["msd_nonclosure_uncertainty"] = inputs["msd_nonclosure_uncertainty"]["columns"]


# @base.requires
# def base_requires(self: HistProducer, task: law.Task, reqs: law.util.InsertableDict) -> None:
#     """
#     Define the requirements for the base HistProducer.
#     This is called before the setup method.
#     """
#     logger.debug("Currently no custom requirements for HistProducer base, only weight columns defined in init")
#     # from columnflow.tasks.production import ProduceColumns
#     # if "dy_correction_weight" in self.local_weight_columns.keys():
#     #     if not self.dy_correction_weight_producer:
#     #         raise Exception(
#     #             "dy_correction_weight_producer must be set if dy_correction_weight is used "
#     #             "in weight_columns",
#     #         )
#     #     reqs["dy_correction_weight_producer"] = ProduceColumns.req(
#     #         task,
#     #         producer=self.dy_correction_weight_producer,
#     #     )

#     # if self.require_hbb_sf_producer and not self.dataset_inst.is_data:
#     #     reqs["hbb_sf_weights"] = ProduceColumns.req(
#     #         task,
#     #         producer="hbb_sf_weights",
#     #     )

#     # self.require_msd_nonclosure_producer = False
#     # if "msd_nonclosure_weight" in self.local_weight_columns.keys():
#     #     msd_nonclosure_shifts = get_shifts_from_sources(
#     #         self.config_inst,
#     #         self.local_weight_columns["msd_nonclosure_weight"],
#     #     )
#     #     if self.dataset_inst.is_mc and task.shift in msd_nonclosure_shifts:
#     #         reqs["msd_nonclosure_uncertainty"] = ProduceColumns.req(
#     #             task,
#     #             producer="msd_nonclosure_uncertainty",
#     #         )
#     #         self.require_msd_nonclosure_producer = True
#     #     else:
#     #         self.local_weight_columns.pop("msd_nonclosure_weight", None)
#     #         self.uses.discard("msd_nonclosure_weight")


@base.init
def base_init(self: HistProducer) -> None:

    if not getattr(self, "config_inst"):
        return

    def update_cat_label(config_inst, pre_label):
        for cat_inst, _, _ in config_inst.walk_categories():
            if pre_label not in cat_inst.label:
                cat_inst.label = "\n".join([cat_inst.label, pre_label])
            else:
                logger.debug(f"Category {cat_inst} already includes pre_label '{pre_label}'")

    if self.pre_label:
        update_cat_label(self.config_inst, self.pre_label)

    if self.categorizer_cls:
        self.uses.add(self.categorizer_cls)

    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst and dataset_inst.is_data:
        # if we are on data, we do not need any weights
        self.local_weight_columns = {}
        return

    year = self.config_inst.campaign.x.year
    cpn_tag = self.config_inst.x.cpn_tag

    if not self.weight_columns:
        raise Exception("weight_columns not set")
    self.local_weight_columns = self.weight_columns.copy()

    if has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst):
        logger.info_once(f"Config {self.config_inst.name} has tag 'skip_btag_wp_weights', removing btag weight columns and use normalized_ht_njet_nhf_btag_weight instead")
        self.local_weight_columns.pop("btag_weight", None)
    elif has_tag("skip_btag_weights", self.config_inst, self.dataset_inst):
        logger.info_once(f"Config {self.config_inst.name} has tag 'skip_btag_weights', removing normalized_ht_njet_nhf_btag_weight columns and use btag weight instead")
        self.local_weight_columns.pop("normalized_ht_njet_nhf_btag_weight", None)
    else:
        # throw error if no tag is set
        if not any(has_tag(tag, self.config_inst, self.dataset_inst) for tag in ["skip_btag_wp_weights", "skip_btag_weights"]):
            raise Exception(
                "No tag set for btag weights. Please set either 'skip_btag_wp_weights' or 'skip_btag_weights' tag in the config or dataset.",
            )

    if dataset_inst and dataset_inst.has_tag("skip_scale"):
        # remove dependency towards mur/muf weights
        for column in [
            "normalized_mur_weight", "normalized_muf_weight", "normalized_murmuf_envelope_weight",
            "mur_weight", "muf_weight", "murmuf_envelope_weight",
        ]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and dataset_inst.has_tag("no_ps_weights"):
        self.local_weight_columns.pop("normalized_isr_weight", None)
        self.local_weight_columns.pop("normalized_fsr_weight", None)

    if dataset_inst and dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and not dataset_inst.has_tag("is_ttbar"):
        # remove dependency towards top pt weights
        self.local_weight_columns.pop("top_pt_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_v_jets") and not dataset_inst.has_tag("is_vv"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("vjets_weight", None)

    # if dataset_inst and not dataset_inst.has_tag("is_dy"):
    #     # remove dependency towards vjets weights
    #     self.local_weight_columns.pop("dy_correction_weight", None)
    #     self.local_weight_columns.pop("dy_weight", None)

    # if dataset_inst and not dataset_inst.has_tag("is_dy"):
    #     # remove dependency towards dy weights
    #     self.local_weight_columns.pop("dy_weight", None)

    self.shifts = set()

    # when jec sources are known btag SF source, then propagate the shift to the HistProducer
    # TODO: we should do this somewhere centrally
    btag_sf_jec_sources = (
        (set(self.config_inst.x.btag_sf_jec_sources) | {"Total"}) &
        set(self.config_inst.x.jec.Jet["uncertainty_sources"])
    )
    self.shifts |= set(get_shifts_from_sources(
        self.config_inst,
        *[f"jec_{jec_source}" for jec_source in btag_sf_jec_sources],
    ))

    for weight_column, shift_sources in self.local_weight_columns.items():
        shift_sources = law.util.make_list(shift_sources)
        shift_sources = [s.format(year=year, cpn_tag=cpn_tag) for s in shift_sources]
        shifts = get_shifts_from_sources(self.config_inst, *shift_sources)
        for shift in shifts:
            if weight_column not in shift.x("column_aliases").keys():
                # make sure that column aliases are implemented
                raise Exception(
                    f"Weight column {weight_column} implements shift {shift}, but does not use it "
                    f"in 'column_aliases' aux {shift.x('column_aliases')}",
                )

        # declare shifts that the produced event weight depends on
        self.shifts |= set(shifts)

    # remove dummy column from weight columns and uses
    self.local_weight_columns.pop("dummy_weight", "")

    # store column names referring to weights to multiply
    self.uses |= self.local_weight_columns.keys()
    # self.uses = {"*"}


@base.post_init
def base_post_init(self: HistProducer, task: law.Task):
    if self.dataset_inst.is_data:
        return
    if "isr" not in task.shift:
        # no nominal ISR weight --> remove it from uses and local_weight_columns
        self.uses.discard("normalized_isr_weight")
        self.local_weight_columns.pop("normalized_isr_weight", None)
    if "fsr" not in task.shift:
        # no nominal FSR weight --> remove it from uses and local_weight_columns
        self.uses.discard("normalized_fsr_weight")
        self.local_weight_columns.pop("normalized_fsr_weight", None)


btag_uncs = [
    "hf", "lf",
    "cferr1", "cferr2",
    "hfstats1", "lfstats1",
    "hfstats2", "lfstats2",
]


btag_uncs_bc = [
    "correlated",
    "uncorrelated",
    "bfragmentation",
    "fsrdef",
    "hdamp",
    "isrdef",
    "jer",
    "jes",
    "muf",
    "mur",
    "pdfas",
    "pileup",
    "statistic",
    "topmass",
    "type3p",
]
btag_uncs_bc_full = [f"{unc}_bc" for unc in btag_uncs_bc] + ["bc"]
btag_uncs_light = [
    "",
    "correlated", "uncorrelated",
]
btag_uncs_light_full = [f"{unc}_light" for unc in btag_uncs_light] + ["light"]

all_btag_uncs = btag_uncs_bc_full + btag_uncs_light_full

default_correction_weights = {
    "normalization_weight": [],
    # "dummy_weight": ["dummy_{cpn_tag}"],
    "normalized_pu_weight": ["minbias_xs"],
    # muon SF
    "muon_reco_weight": ["muon_reco"],
    "muon_trigger_weight": ["muon_trigger"],
    "muon_id_weight": ["muon_id"],
    "muon_iso_weight": ["muon_iso"],
    # electron SF
    "electron_trigger_weight": ["electron_trigger"],
    "electron_id_iso_weight": ["electron_id_iso"],
    "electron_reco_weight": ["electron_reco"],
    # store all btag weights, pop them later depending on the dataset, year, and cpn tag
    "btag_weight": [f"btag_{unc}" for unc in all_btag_uncs],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    # "normalized_murmuf_envelope_weight": ["murmuf_envelope"],
    # "normalized_murmuf_weight": ["murmuf"]
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "normalized_isr_weight": ["isr"],
    "normalized_fsr_weight": ["fsr"],
    "top_pt_weight": ["top_pt"],
    # "vjets_weight": ["vjets"]
}


weight_columns_except_trigger = default_correction_weights.copy()
weight_columns_except_trigger.pop("muon_trigger_weight")
weight_columns_except_trigger.pop("electron_trigger_weight")

weight_columns_except_btag = default_correction_weights.copy()
weight_columns_except_btag.pop("btag_weight")
weight_columns_except_btag.pop("normalized_ht_njet_nhf_btag_weight")

weight_columns_except_btag_and_trigger = default_correction_weights.copy()
weight_columns_except_btag_and_trigger.pop("btag_weight")
weight_columns_except_btag_and_trigger.pop("normalized_ht_njet_nhf_btag_weight")
weight_columns_except_btag_and_trigger.pop("muon_trigger_weight")
weight_columns_except_btag_and_trigger.pop("electron_trigger_weight")

default_hist_producer = base.derive("default", cls_dict={
    "weight_columns": default_correction_weights
})

default_fixed_hist_producer = base.derive("default_fixed", cls_dict={
    # use default weights plus the electron normalization fix weight, which applies a flat SF of 1.2 in the 1e channel for 2024 to account for the missing trigger SFs and other lepton SFs in that channel for 2024
    "weight_columns": {
        **default_correction_weights,
        "electron_norm_fix_weight": [],
    },
})

no_btag_weights = default_hist_producer.derive("no_btag_weights", cls_dict={
    "pre_label": "Without b-tagging weight",
    # "weight_columns": weight_columns_except_btag_and_trigger,  # no trigger weights available yet
    "weight_columns": weight_columns_except_btag,
})

no_trigger_weights = default_hist_producer.derive("no_trigger_weights", cls_dict={
    "pre_label": "Without trigger SF",
    "weight_columns": weight_columns_except_trigger,
})

no_lepton_weights = default_hist_producer.derive("no_lepton_weights", cls_dict={
    "pre_label": "Without lepton SFs",
    "weight_columns": {
        k: v for k, v in default_correction_weights.items() if not k.startswith("muon_") and not k.startswith("electron_")
    },
})

no_scale_weights = default_hist_producer.derive("no_scale_weights", cls_dict={
    "pre_label": "Without scale weights",
    "weight_columns": {
        k: v for k, v in default_correction_weights.items() if not k.startswith("normalized_mur") and not k.startswith("normalized_muf") and not k.startswith("normalized_murmuf") and not k.startswith("mur") and not k.startswith("muf") and not k.startswith("murmuf")
    },
})

no_pdf_weights = default_hist_producer.derive("no_pdf_weights", cls_dict={
    "pre_label": "Without PDF weights",
    "weight_columns": {
        k: v for k, v in default_correction_weights.items() if not k.startswith("normalized_pdf") and not k.startswith("pdf")
    },
})

no_ps_weights = default_hist_producer.derive("no_ps_weights", cls_dict={
    "pre_label": "Without PS weights",
    "weight_columns": {
        k: v for k, v in default_correction_weights.items() if not k.startswith("normalized_isr") and not k.startswith("normalized_fsr") and not k.startswith("isr") and not k.startswith("fsr")
    },
})

#
# HistProducers with masks via categorization
# not working, but kept as reference if we ever want to implement something like this
#

# from hbw.categorization.categories import (
#     mask_fn_mbb80, catid_ge2b_loose, catid_njet2, mask_fn_met70,
#     mask_fn_met_geq40,
# )
# met_geq40_with_dy_corr = with_dy_corr.derive("met_geq40_with_dy_corr", cls_dict={
#     "pre_label": "\n".join([r"$p_{T}^{miss} \geq 40$ GeV"]),
#     "nondy_hist_producer": "met_geq40_no_dycorr",
#     "categorizer_cls": mask_fn_met_geq40,
#     "dy_correction_weight_producer": "dy_correction_weight",
# })

# other btag normalization modes
# base.derive("btag_njet_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_njet_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_njet_nhf_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
