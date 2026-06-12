# coding: utf-8

"""
++++++++++ WIP ++++++++++
Configuration creation for top-tagging scale factor
derivation using Run3 samples.
++++++++++ WIP ++++++++++
"""

from __future__ import annotations

import functools
import os
import law

import order as od
import yaml

from scinum import Number

from columnflow.util import DotDict
from columnflow.cms_util import CATInfo, CATSnapshot
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)
from columnflow.selection.cms.btag import BTagWPCountConfig
from columnflow.production.cms.btag import BTagSFConfig, BTagWPSFConfig
from columnflow.production.cms.electron import ElectronSFConfig
from columnflow.production.cms.muon import MuonSFConfig
from columnflow.production.cms.jet import JetIdConfig

from topsf.config.variables import add_variables
from topsf.config.categories import add_categories
from topsf.config.datasets import add_datasets_from_yaml
from topsf.config.taggers import btag_wps, toptag_wps
from topsf.util import has_tag


thisdir = os.path.dirname(os.path.abspath(__file__))
logger = law.logger.get_logger(__name__)


def add_new_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    """
    Configurable function for creating a config for a run3 analysis given
    a base *analysis* object and a *campaign* (i.e. set of datasets).
    """
    # validation
    assert campaign.x.year in [2022, 2023, 2024]
    if campaign.x.year == 2022:
        assert campaign.x.EE in ["pre", "post"]
    elif campaign.x.year == 2023:
        assert campaign.x.BPix in ["pre", "post"]

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = ""
    if year == 2022:
        corr_postfix = f"{campaign.x.EE}EE"
    elif year == 2023:
        corr_postfix = f"{campaign.x.BPix}BPix"

    implemented_years = [2022, 2023, 2024]

    if year not in implemented_years:
        raise NotImplementedError("For now, only 2022, 2023, and 2024 campaigns are fully implemented")

    # create a config by passing the campaign
    # (if id and name are not set they will be taken from the campaign)
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # add tags to config
    cfg.x.run = 3
    cfg.x.cpn_tag = f"{year}{corr_postfix}"
    cfg.x.year = year
    vnano = campaign.x.version
    logger.info(f"Creating config '{cfg.name}' for campaign '{campaign.name}' with year {year} and version {vnano}")
    cfg.add_tag("skip_kfactor_weights")  # FIXME temporary, remove when kfactors are available for all processes
    logger.warning_once(
        "K factor reweighting for v+jets datasets currently disabled for all configs, as k factors are not available."
    )

    #
    # configure processes
    #

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create parent processes for w_lnu, dy and qcd
    for i_proc, (proc_name, proc_label, child_procs) in enumerate([
        ("vx", "V+jets, VV", ["dy", "w_lnu", "vv"]),  # FIXME Why does dy not work anymore?
        ("mj", "Multijet", ["qcd"]),
    ]):
        proc = od.Process(
            name=proc_name,
            id=int(1e7 + (i_proc + 1)),
            label=proc_label,
        )
        for child_proc in child_procs:
            procs.n(child_proc).add_parent_process(proc)

    # get all root processes (including newly added ones)
    procs = get_root_processes_from_campaign(campaign)

    # create sub-processes for st, tt
    # (defined via cuts on gen-level objects; will be normalized
    # to xs of parent process)
    top_subprocess_cfg = DotDict.wrap({
        "0o1q": {
            "index": 1,
            "label": "not merged (0q or 1q)",
            "colors": {
                "tt": "#A80068",
                "st": "#FF9300",
            },
        },
        "2q": {
            "index": 2,
            "label": "semi-merged (2q)",
            "colors": {
                "tt": "#FF58D0",
                "st": "#FFFF00",
            },
        },
        "3q": {
            "index": 3,
            "label": "fully merged (3q)",
            "colors": {
                "tt": "#FF0064",
                "st": "#FFC900",
            },
        },
        "bkg": {
            "index": 4,
            "label": "background",
            "colors": {
                "tt": "#700034",
                "st": "#A62800",
            },
        },
    })

    # helper function for adding subprocesses
    def add_subprocesses(proc, color_key):
        """Add subprocesses to an existing process."""
        subprocs = {}
        for subproc_name, subproc_cfg in top_subprocess_cfg.items():
            subprocs[subproc_name] = subproc = proc.add_process(
                name=f"{proc.name}_{subproc_name}",
                id=int(proc.id + 1e6 * (subproc_cfg.index)),
                label=f"{proc.label}, {subproc_cfg.label}",
                color=subproc_cfg.colors[color_key],
                aux={
                    "subprocess_name": subproc_name,
                },
            )
            subproc.add_tag("is_subprocess")

            # mark process as signal (used by inference model)
            if subproc_name != "bkg":
                subproc.add_tag("is_topsf_signal")

        proc.add_tag("has_subprocesses")
        return subprocs

    # add subprocesses to processes with top quarks
    for root_proc in ("st", "tt"):
        root_proc_inst = getattr(procs.n, root_proc)
        subprocs = {}  # [depth][subproc_key] -> od.Process
        for proc, depth, children in root_proc_inst.walk_processes(
            algo="bfs",
            include_self=True,
        ):
            # add subprocesses to top-level process (tt, st)
            subprocs[depth] = add_subprocesses(proc, color_key=root_proc)

            # mark subprocesses as children of parent subprocesses
            parent_subprocs = subprocs.get(depth - 1, {})
            if not parent_subprocs:
                continue
            for subproc_name, subproc_inst in subprocs[depth].items():
                parent_subprocs[subproc_name].add_process(subproc_inst)

    # set color of some processes
    colors = {
        "data": "#000000",  # black
        "tt": "#E04F21",  # red
        "qcd": "#5E8FFC",  # blue
        "w_lnu": "#82FF28",  # green
        "st": "#3E00FB",  # dark purple
        "dy": "#FBFF36",  # yellow
        "vv": "#B900FC",  # pink
        "other": "#999999",  # grey
        # christopher's color scheme
        "vx": "#00FF00",
        "mj": "#00D0FF",
    }

    # add processes we are interested in
    # remove processes we don't need from list and following dict!
    process_names = [
        "data",
        "tt",
        "st",
        # "dy",
        # "w_lnu",
        # "vv",
        # "qcd",
        "vx",
        "mj",
    ]

    cfg.x.process_rates = {
        "tt": 1.05,
        "st": 1.5,
        "vx": 1.2,
        "mj": 2.0,
    }

    cfg.x.inference_processes = [
        f"{base_proc}_{subproc_suffix}"
        for base_proc in ("tt", "st")
        for subproc_suffix in ("3q", "2q", "0o1q", "bkg")
    ] + [
        "vx",
        "mj",
    ]

    # setup for fit
    # TODO make configurable from CLI (as params of inference model)
    cfg.x.fit_setup = {
        "channels": [
            "1m",
            "1e",
        ],
        "pt_bins": [
            "pt_300_400",
            "pt_400_480",
            "pt_480_600",
            "pt_600_inf",
        ],
        "wp_names": [
            "very_tight",
            "tight",
            "medium",
            "loose",
            "very_loose",
        ],
        "fit_vars": [
            # "probejet_msoftdrop_inf_rebin",   # use inference mSD
            # "probejet_msoftdrop_widebins",
            "probejet_msoftdrop_inf_rebin_fix",
        ],
        "shape_unc": [
            "fsr",
            "isr",
            # "electron",
            "electron_reco",
            "electron_id_iso",
            "electron_trigger",
            # "muon",
            "muon_reco",
            "muon_id",
            "muon_iso",
            "muon_trigger",
            "minbias_xs",
            # "top_pt",
            "jec_Total",
            "mur",
            "muf",
            # "btag_bc",
            # "btag_light",
        ],
    }
    if year == 2024:
        cfg.x.fit_setup["shape_unc"] += [
            "btag_bc",
            "btag_light",
        ]
    else:
        cfg.x.fit_setup["shape_unc"] += [
            "btag_hf",
            "btag_lf",
        ]

    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))

        # mark the presence of a top quark
        if any(proc.name.startswith(s) for s in ("tt", "st")):
            proc.add_tag("has_top")

        # mark ttbar processes (needed for top pt reweighting)
        if proc.name.startswith("tt"):
            proc.add_tag("is_ttbar")

        # configuration of colors, labels, etc. can happen here
        proc.color = colors.get(proc.name, "#aaaaaa")

    #
    # datasets
    #
    dataset_names = add_datasets_from_yaml(
        cfg,
        limit_dataset_files=limit_dataset_files,
        dataset_types=[
            "data",
            "tt",
            "st",
            "dy",
            "w_lnu",
            "vv",
            "qcd",
        ],
        log=False,
    )

    for dataset in cfg.datasets:
        # update JECera information
        if dataset.is_data and (dataset.name.endswith("c") or dataset.name.endswith("d")):
            dataset.x.jec_era = "RunCD"
        if "twchannel" in dataset.name:
            dataset.add_tag("has_top_associated_w")

    # verify that the root processes of each dataset (or one of their
    # ancestor processes) are registered in the config
    verify_config_processes(cfg, warn=True)
    logger.info(f"Added {len(cfg.processes)} processes and {len(cfg.datasets)} datasets to config '{cfg.name}'")

    #
    # defaults
    #

    # default objects, such as calibrator, selector, producer,
    # ml model, inference model, etc
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default"
    cfg.x.default_reducer = "cf_default"
    cfg.x.default_producer = "default"
    cfg.x.default_hist_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "default"  # "uhh2"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = (
        "probejet_pt",
        "probejet_mass",
        "probejet_msoftdrop_widebins",
        "probejet_tau32",
        "probejet_max_subjet_btag_score_btagDeepB",
        "probejet_msoftdrop_inf_rebin_fix",
    )

    #
    # parameter groups
    #

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "all": process_names,
        "all_subprocs": cfg.x.inference_processes,
    }

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {
        "all": dataset_names,
        "data": ["data_*"],
        "dy": ["dy*"],
        "w_lnu": ["w_lnu*"],
        "qcd_mu": ["qcd_mu*"],
        "qcd_em": ["qcd_em*"],
        "qcd": ["qcd*"],
        "st": ["st*"],
        "tt": ["tt*"],
        "vv": ["ww_pythia", "wz_pythia", "zz_pythia"],
        "vx": ["w_lnu*", "dy*", "ww_pythia", "wz_pythia", "zz_pythia"],
        "mj": ["qcd*"],
        "mc": ["dy*", "w_lnu*", "ww_pythia", "wz_pythia", "zz_pythia", "st*", "tt*", "qcd*"],
        "testing": [
            "tt_sl_powheg",
            "st_tchannel_t_4f_powheg",
            "dy_m4to50_ht800to1500_madgraph",
            "w_lnu_mlnu0to120_ht1500to2500_madgraph",
            "ww_pythia",
            "qcd_mu_pt600to800_pythia",
            # "qcd_em_pt120to170_pythia",
        ],
    }
    if cfg.x.cpn_tag == "2022preEE":
        cfg.x.dataset_groups["testing"] += ["data_egamma_c", "data_mu_c"]
    elif cfg.x.cpn_tag == "2022postEE":
        cfg.x.dataset_groups["testing"] += ["data_egamma_f", "data_mu_f"]
    elif cfg.x.cpn_tag == "2023preBPix":
        cfg.x.dataset_groups["testing"] += ["data_egamma_c", "data_mu_c"]
    elif cfg.x.cpn_tag == "2023postBPix":
        cfg.x.dataset_groups["testing"] += ["data_egamma_d", "data_mu_d"]
    elif cfg.x.cpn_tag == "2024":
        cfg.x.dataset_groups["testing"] += ["data_e_c", "data_mu_c"]

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {
        "default": ["1m"],
        "1m_wp_very_tight_pass": [
            "1m__pt_300_400__tau32_wp_very_tight_pass",
            "1m__pt_400_480__tau32_wp_very_tight_pass",
            "1m__pt_480_600__tau32_wp_very_tight_pass",
            "1m__pt_600_inf__tau32_wp_very_tight_pass",
        ],
        "1m_wp_very_tight_fail": [
            "1m__pt_300_400__tau32_wp_very_tight_fail",
            "1m__pt_400_480__tau32_wp_very_tight_fail",
            "1m__pt_480_600__tau32_wp_very_tight_fail",
            "1m__pt_600_inf__tau32_wp_very_tight_fail",
        ],
        "1m_wp_tight_pass": [
            "1m__pt_300_400__tau32_wp_tight_pass",
            "1m__pt_400_480__tau32_wp_tight_pass",
            "1m__pt_480_600__tau32_wp_tight_pass",
            "1m__pt_600_inf__tau32_wp_tight_pass",
        ],
        "1m_wp_tight_fail": [
            "1m__pt_300_400__tau32_wp_tight_fail",
            "1m__pt_400_480__tau32_wp_tight_fail",
            "1m__pt_480_600__tau32_wp_tight_fail",
            "1m__pt_600_inf__tau32_wp_tight_fail",
        ],
        "1m_wp_medium_pass": [
            "1m__pt_300_400__tau32_wp_medium_pass",
            "1m__pt_400_480__tau32_wp_medium_pass",
            "1m__pt_480_600__tau32_wp_medium_pass",
            "1m__pt_600_inf__tau32_wp_medium_pass",
        ],
        "1m_wp_medium_fail": [
            "1m__pt_300_400__tau32_wp_medium_fail",
            "1m__pt_400_480__tau32_wp_medium_fail",
            "1m__pt_480_600__tau32_wp_medium_fail",
            "1m__pt_600_inf__tau32_wp_medium_fail",
        ],
        "1m_wp_loose_pass": [
            "1m__pt_300_400__tau32_wp_loose_pass",
            "1m__pt_400_480__tau32_wp_loose_pass",
            "1m__pt_480_600__tau32_wp_loose_pass",
            "1m__pt_600_inf__tau32_wp_loose_pass",
        ],
        "1m_wp_loose_fail": [
            "1m__pt_300_400__tau32_wp_loose_fail",
            "1m__pt_400_480__tau32_wp_loose_fail",
            "1m__pt_480_600__tau32_wp_loose_fail",
            "1m__pt_600_inf__tau32_wp_loose_fail",
        ],
        "1e_wp_loose_pass": [
            "1e__pt_300_400__tau32_wp_loose_pass",
            "1e__pt_400_480__tau32_wp_loose_pass",
            "1e__pt_480_600__tau32_wp_loose_pass",
            "1e__pt_600_inf__tau32_wp_loose_pass",
        ],
        "1e_wp_loose_fail": [
            "1e__pt_300_400__tau32_wp_loose_fail",
            "1e__pt_400_480__tau32_wp_loose_fail",
            "1e__pt_480_600__tau32_wp_loose_fail",
            "1e__pt_600_inf__tau32_wp_loose_fail",
        ],
        "1m_wp_very_loose_pass": [
            "1m__pt_300_400__tau32_wp_very_loose_pass",
            "1m__pt_400_480__tau32_wp_very_loose_pass",
            "1m__pt_480_600__tau32_wp_very_loose_pass",
            "1m__pt_600_inf__tau32_wp_very_loose_pass",
        ],
        "1m_wp_very_loose_fail": [
            "1m__pt_300_400__tau32_wp_very_loose_fail",
            "1m__pt_400_480__tau32_wp_very_loose_fail",
            "1m__pt_480_600__tau32_wp_very_loose_fail",
            "1m__pt_600_inf__tau32_wp_very_loose_fail",
        ],
        "1m_all_pt_wp_pass_fail": [
            "1m__pt_300_400__tau32_wp_very_tight_pass",
            "1m__pt_400_480__tau32_wp_very_tight_pass",
            "1m__pt_480_600__tau32_wp_very_tight_pass",
            "1m__pt_600_inf__tau32_wp_very_tight_pass",
            "1m__pt_300_400__tau32_wp_very_tight_fail",
            "1m__pt_400_480__tau32_wp_very_tight_fail",
            "1m__pt_480_600__tau32_wp_very_tight_fail",
            "1m__pt_600_inf__tau32_wp_very_tight_fail",
            "1m__pt_300_400__tau32_wp_tight_pass",
            "1m__pt_400_480__tau32_wp_tight_pass",
            "1m__pt_480_600__tau32_wp_tight_pass",
            "1m__pt_600_inf__tau32_wp_tight_pass",
            "1m__pt_300_400__tau32_wp_tight_fail",
            "1m__pt_400_480__tau32_wp_tight_fail",
            "1m__pt_480_600__tau32_wp_tight_fail",
            "1m__pt_600_inf__tau32_wp_tight_fail",
            "1m__pt_300_400__tau32_wp_medium_pass",
            "1m__pt_400_480__tau32_wp_medium_pass",
            "1m__pt_480_600__tau32_wp_medium_pass",
            "1m__pt_600_inf__tau32_wp_medium_pass",
            "1m__pt_300_400__tau32_wp_medium_fail",
            "1m__pt_400_480__tau32_wp_medium_fail",
            "1m__pt_480_600__tau32_wp_medium_fail",
            "1m__pt_600_inf__tau32_wp_medium_fail",
            "1m__pt_300_400__tau32_wp_loose_pass",
            "1m__pt_400_480__tau32_wp_loose_pass",
            "1m__pt_480_600__tau32_wp_loose_pass",
            "1m__pt_600_inf__tau32_wp_loose_pass",
            "1m__pt_300_400__tau32_wp_loose_fail",
            "1m__pt_400_480__tau32_wp_loose_fail",
            "1m__pt_480_600__tau32_wp_loose_fail",
            "1m__pt_600_inf__tau32_wp_loose_fail",
            "1m__pt_300_400__tau32_wp_very_loose_pass",
            "1m__pt_400_480__tau32_wp_very_loose_pass",
            "1m__pt_480_600__tau32_wp_very_loose_pass",
            "1m__pt_600_inf__tau32_wp_very_loose_pass",
            "1m__pt_300_400__tau32_wp_very_loose_fail",
            "1m__pt_400_480__tau32_wp_very_loose_fail",
            "1m__pt_480_600__tau32_wp_very_loose_fail",
            "1m__pt_600_inf__tau32_wp_very_loose_fail",
        ],
        "1e_all_pt_wp_pass_fail": [
            "1e__pt_300_400__tau32_wp_very_tight_pass",
            "1e__pt_400_480__tau32_wp_very_tight_pass",
            "1e__pt_480_600__tau32_wp_very_tight_pass",
            "1e__pt_600_inf__tau32_wp_very_tight_pass",
            "1e__pt_300_400__tau32_wp_very_tight_fail",
            "1e__pt_400_480__tau32_wp_very_tight_fail",
            "1e__pt_480_600__tau32_wp_very_tight_fail",
            "1e__pt_600_inf__tau32_wp_very_tight_fail",
            "1e__pt_300_400__tau32_wp_tight_pass",
            "1e__pt_400_480__tau32_wp_tight_pass",
            "1e__pt_480_600__tau32_wp_tight_pass",
            "1e__pt_600_inf__tau32_wp_tight_pass",
            "1e__pt_300_400__tau32_wp_tight_fail",
            "1e__pt_400_480__tau32_wp_tight_fail",
            "1e__pt_480_600__tau32_wp_tight_fail",
            "1e__pt_600_inf__tau32_wp_tight_fail",
            "1e__pt_300_400__tau32_wp_medium_pass",
            "1e__pt_400_480__tau32_wp_medium_pass",
            "1e__pt_480_600__tau32_wp_medium_pass",
            "1e__pt_600_inf__tau32_wp_medium_pass",
            "1e__pt_300_400__tau32_wp_medium_fail",
            "1e__pt_400_480__tau32_wp_medium_fail",
            "1e__pt_480_600__tau32_wp_medium_fail",
            "1e__pt_600_inf__tau32_wp_medium_fail",
            "1e__pt_300_400__tau32_wp_loose_pass",
            "1e__pt_400_480__tau32_wp_loose_pass",
            "1e__pt_480_600__tau32_wp_loose_pass",
            "1e__pt_600_inf__tau32_wp_loose_pass",
            "1e__pt_300_400__tau32_wp_loose_fail",
            "1e__pt_400_480__tau32_wp_loose_fail",
            "1e__pt_480_600__tau32_wp_loose_fail",
            "1e__pt_600_inf__tau32_wp_loose_fail",
            "1e__pt_300_400__tau32_wp_very_loose_pass",
            "1e__pt_400_480__tau32_wp_very_loose_pass",
            "1e__pt_480_600__tau32_wp_very_loose_pass",
            "1e__pt_600_inf__tau32_wp_very_loose_pass",
            "1e__pt_300_400__tau32_wp_very_loose_fail",
            "1e__pt_400_480__tau32_wp_very_loose_fail",
            "1e__pt_480_600__tau32_wp_very_loose_fail",
            "1e__pt_600_inf__tau32_wp_very_loose_fail",
        ],
        "1m_all_pt": [
            "1m__pt_300_400",
            "1m__pt_400_480",
            "1m__pt_480_600",
            "1m__pt_600_inf",
        ],
        "1e_all_pt": [
            "1e__pt_300_400",
            "1e__pt_400_480",
            "1e__pt_480_600",
            "1e__pt_600_inf",
        ],
        "1m_all_wp": [
            "1m__tau32_wp_very_tight_pass",
            "1m__tau32_wp_very_tight_fail",
            "1m__tau32_wp_tight_pass",
            "1m__tau32_wp_tight_fail",
            "1m__tau32_wp_medium_pass",
            "1m__tau32_wp_medium_fail",
            "1m__tau32_wp_loose_pass",
            "1m__tau32_wp_loose_fail",
            "1m__tau32_wp_very_loose_pass",
            "1m__tau32_wp_very_loose_fail",
        ],
        "1e_all_wp": [
            "1e__tau32_wp_very_tight_pass",
            "1e__tau32_wp_very_tight_fail",
            "1e__tau32_wp_tight_pass",
            "1e__tau32_wp_tight_fail",
            "1e__tau32_wp_medium_pass",
            "1e__tau32_wp_medium_fail",
            "1e__tau32_wp_loose_pass",
            "1e__tau32_wp_loose_fail",
            "1e__tau32_wp_very_loose_pass",
            "1e__tau32_wp_very_loose_fail",
        ],
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": [
            "LeptonTrigger", "Lepton", "AddLeptonVeto", "MET", "BJet", "METFilters",
        ],
    }

    # Exception: no weight producer configured for task. cf.MergeShiftedHistograms.
    # As of 02.05.2024, it is required to pass a weight_producer for tasks creating histograms.
    # You can add a 'default_weight_producer' to your config or directly add the weight_producer
    # on command line via the '--weight_producer' parameter. To reproduce results from before this date,
    # you can use the 'all_weights' weight_producer defined in columnflow.weight.all_weights:
    # With cf 0.3.x, the 'weight_producer' has been renamed to 'hist_producer'.

    # custom labels for selector steps
    cfg.x.selector_step_labels = {}

    # plotting settings groups
    cfg.x.general_settings_groups = {}
    cfg.x.process_settings_groups = {}
    cfg.x.variable_settings_groups = {}

    #
    # dataset customization
    #

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    #
    # tagger working points
    #

    # full b-tag working points dict
    cfg.x.btag_working_points = btag_wps(cfg, full=True)
    # store only upart wps for fixed wp sf producer
    cfg.x.btag_working_points.btagUParTAK4B.fixed_wp = btag_wps(cfg, full=False)

    # top-tag working points
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
    cfg.x.toptag_working_points = toptag_wps(era=cfg.x.cpn_tag)

    #
    # selector configurations
    # FIXME: adapt for Run3?
    #

    # lepton selection parameters
    cfg.x.lepton_selection = DotDict.wrap({
        "mu": {
            "column": "Muon",
            "min_pt": 55,
            "max_abseta": 2.4,
            "triggers": {
                # # FIXME: adapt for UL17
                # "IsoMu24",
                # "IsoTkMu24",
                # as mttbar:
                # "IsoMu27",
                # updated Run3 trigger recommendations
                "Mu50",
                "HighPtTkMu100",
                "CascadeMu100",
            },
            "id": {
                # "column": "tightId",
                # "value": True,
                # Run 3 high pT muon ID
                "column": "highPtId",
                "value": 2, # (1 = tracker high pT, 2 = global high pT, which includes tracker high pT)
            },
            # "rel_iso": "pfRelIso03_all",
            # "max_rel_iso": 1.5,
            # Run 3 high pt muon iso
            "rel_iso": "tkRelIso",
            "max_rel_iso": 0.05,
            # veto events with additional leptons passing looser cuts
            "min_pt_addveto": 30,
            "id_addveto": {
                "column": "looseId",
                "value": True,
            },
        },
        "e": {
            "column": "Electron",
            "min_pt": 55,
            "max_abseta": 2.4,
            "triggers": {
                # FIXME: adapt for UL17
                # "Ele27_WPTight_Gsf",
                # "Ele115_CaloIdVT_GsfTrkIdT",
                # as mttbar:
                # "Ele35_WPTight_Gsf",
                # updated Run3 trigger recommendations
                "Ele30_WPTight_Gsf",
            },
            #"id": "mvaFall17V2Iso_WP90",  # noqa
            "id": {
                "column": "cutBased",
                "value": 4,
            },
            # veto events with additional leptons passing looser cuts
            "min_pt_addveto": 30,
            #"id_addveto": "mvaFall17V2Iso_WPL",  # noqa
            "id_addveto": {
                "column": "cutBased",
                "value": 1,
            },
        },
    })

    # jet selection parameters
    subjet_wp_key = "btagUParTAK4B" if year == 2024 else "deepcsv"
    subjet_btag_wp = getattr(cfg.x.btag_working_points, subjet_wp_key).loose
    if subjet_btag_wp < 0:
        raise ValueError(f"Invalid subjet b-tag working point for year {year}. Please check the configuration.")

    ak4_btag_wp_key = "btagUParTAK4B" if year == 2024 else "deepjet"
    ak4_btag_wp = getattr(cfg.x.btag_working_points, ak4_btag_wp_key).medium
    if ak4_btag_wp < 0:
        raise ValueError(f"Invalid AK4 b-tag working point for year {year}. Please check the configuration.")

    cfg.x.jet_selection = DotDict.wrap({
        "ak8": {
            "column": "FatJet",
            "min_pt": 300,
            "max_abseta": 2.5,
            "msoftdrop_range": (105, 210),
            # https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV
            "jetId": 2,  # bit2 (2): pass tight ID, fail tightLepVeto, bit3 (6): pass tight and tightLepVeto ID
            # probe jet pt bins (used by category builder)
            "pt_bins": [300, 400, 480, 600, None],
            # parameters for b-tagged subjets
            "subjet_column": "SubJet",
            "subjet_btag": "btagDeepB" if not year == 2024 else "btagUParTAK4B",
            "subjet_btag_wp": subjet_btag_wp,
        },
        # TODO: implement (requires custom nano)
        "hotvr": {
            "column": "HOTVRJetForTopTagging",
            "min_pt": 200,
            "max_abseta": 2.5,
            # clustering parameters (not needed for analysis, added for reference)
            # https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
            "r_min_max": (0.1, 1.5),
            "rho": 600,  # GeV
            "mu": 30,  # Gev, mass jump threshold
            "theta": 0.7,  # mass jump strength
            "min_pt_subjet": 30,  # min pt of subject
        },
        "ak4": {
            "column": "Jet",
            # https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV
            "jetId": 2,  # bit2 (2): pass tight ID, fail tightLepVeto, bit3 (6): pass tight and tightLepVeto ID
            "min_pt": 15,  # TODO: check UHH2
            "max_abseta": 2.5,  # TODO: check UHH2
            "btag_column": "btagDeepFlavB" if not year == 2024 else "btagUParTAK4B",
            "btag_wp": ak4_btag_wp,
        },
    })

    # MET selection parameters
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections?rev=79#xy_Shift_Correction_MET_phi_modu
    cfg.x.met_selection = DotDict.wrap({
        "default": {
            "column": "PuppiMET",
            "min_pt": 50,
        },
    })

    #
    # luminosity
    #

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    if year == 2022:
        if campaign.x.EE == "pre":
            cfg.x.luminosity = Number(7971, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
        elif campaign.x.EE == "post":
            cfg.x.luminosity = Number(26337, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
    elif year == 2023:
        if campaign.has_tag("preBPix"):
            cfg.x.luminosity = Number(17794, {
                "lumi_13TeV_2023": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
        elif campaign.has_tag("postBPix"):
            cfg.x.luminosity = Number(9451, {
                "lumi_13TeV_2023": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
    elif year == 2024:
        # Total - EraB
        cfg.x.luminosity = Number(109_950.0 - 130.0, {
            "lumi_13p6TeV_2024": 0.016j,  # CERN-CMS-DP-2026-003
        })
    else:
        raise NotImplementedError(f"Luminosity for year {year} is not defined.")

    #
    # cross sections
    #

    # cross sections for diboson samples; taken from:
    # - ww (NNLO): https://arxiv.org/abs/1408.5243
    # - wz (NLO): https://arxiv.org/abs/1105.0020
    # - zz (NNLO): https://www.sciencedirect.com/science/article/pii/S0370269314004614?via%3Dihub
    diboson_xsecs_13 = {
        "ww": Number(118.7, {"scale": (0.025j, 0.022j)}),
        "wz": Number(46.74, {"scale": (0.041j, 0.033j)}),
        # "wz": Number(28.55, {"scale": (0.041j, 0.032j)}) + Number(18.19, {"scale": (0.041j, 0.033j)}),  # (W+Z) + (W-Z)  # noqa
        "zz": Number(16.99, {"scale": (0.032j, 0.024j)}),
    }
    # TODO Use 14 TeV xs for Run 3?
    diboson_xsecs_14 = {
        "ww": Number(131.1, {"scale": (0.026j, 0.022j)}),
        "wz": Number(67.06, {"scale": (0.039j, 0.031j)}),
        # "wz": Number(31.50, {"scale": (0.039j, 0.030j)}) + Number(20.32, {"scale": (0.039j, 0.031j)}),  # (W+Z) + (W-Z)  # noqa
        "zz": Number(18.77, {"scale": (0.032j, 0.024j)}),
    }

    # linear interpolation between 13 and 14 TeV
    diboson_xsecs_13_6 = {
        ds: diboson_xsecs_13[ds] + (13.6 - 13.0) * (diboson_xsecs_14[ds] - diboson_xsecs_13[ds]) / (14.0 - 13.0)
        for ds in diboson_xsecs_13.keys()  # ww: 125.8 wz: 58.932 zz: 18.058  noqa
    }

    for ds in diboson_xsecs_14:
        procs.n(ds).set_xsec(13.6, diboson_xsecs_13_6[ds])

    #
    # MET filters
    #

    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
    cfg.x.met_filters = {
        "Flag.goodVertices",
        "Flag.globalSuperTightHalo2016Filter",
        "Flag.EcalDeadCellTriggerPrimitiveFilter",
        "Flag.BadPFMuonFilter",
        "Flag.BadPFMuonDzFilter",
        "Flag.eeBadScFilter",
        "Flag.ecalBadCalibFilter",
    }
    if year == 2024:
        cfg.x.met_filters.add("Flag.hfNoisyHitsFilter")

    #
    # JEC & JER  # FIXME: Taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    #

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=2017#Jet_Energy_Corrections_in_Run2

    # jec configuration taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = campaign.x.postfix
    if jerc_postfix not in ("", "EE", "BPix"):
        raise ValueError(f"Invalid JERC postfix '{jerc_postfix}' for campaign {campaign.name}.")
    if year == 2022:
        jer_campaign = jec_campaign = f"Summer{year2}{jerc_postfix}_22Sep2023"
    elif year == 2023:
        era = "Cv1234" if campaign.has_tag("preBPix") else "D"
        jer_campaign = f"Summer{year2}{jerc_postfix}Prompt{year2}_Run{era}"
        jec_campaign = f"Summer{year2}{jerc_postfix}Prompt{year2}"
    elif year == 2024:
        jec_campaign = "Summer24Prompt24"
        jer_campaign = "Summer23BPixPrompt23_RunD"  # no 2024 JER yet, use 2023 BPix: https://cms-jerc.web.cern.ch/Recommendations/#2024_1 # noqa
    else:
        raise NotImplementedError(f"JEC/JER configuration for year {year} is not defined.")

    jet_type = "AK4PFPuppi"
    fatjet_type = "AK8PFPuppi"
    jec_ak4_version = jec_ak8_version = {
        2022: "V3",
        2023: "V2" if not year == 2023 else "V3",
        2024: "V2",
    }[year]

    cfg.x.jec = DotDict.wrap({
        "Jet": {
            "campaign": jec_campaign,
            "version": jec_ak4_version,
            "jet_type": jet_type,
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "uncertainty_sources": [
                # "AbsoluteStat",
                # "AbsoluteScale",
                # "AbsoluteSample",
                # "AbsoluteFlavMap",
                # "AbsoluteMPFBias",
                # "Fragmentation",
                # "SinglePionECAL",
                # "SinglePionHCAL",
                # "FlavorQCD",
                # "TimePtEta",
                # "RelativeJEREC1",
                # "RelativeJEREC2",
                # "RelativeJERHF",
                # "RelativePtBB",
                # "RelativePtEC1",
                # "RelativePtEC2",
                # "RelativePtHF",
                # "RelativeBal",
                # "RelativeSample",
                # "RelativeFSR",
                # "RelativeStatFSR",
                # "RelativeStatEC",
                # "RelativeStatHF",
                # "PileUpDataMC",
                # "PileUpPtRef",
                # "PileUpPtBB",
                # "PileUpPtEC1",
                # "PileUpPtEC2",
                # "PileUpPtHF",
                # "PileUpMuZero",
                # "PileUpEnvelope",
                # "SubTotalPileUp",
                # "SubTotalRelative",
                # "SubTotalPt",
                # "SubTotalScale",
                # "SubTotalAbsolute",
                # "SubTotalMC",
                "Total",
                # "TotalNoFlavor",
                # "TotalNoTime",
                # "TotalNoFlavorNoTime",
                # "FlavorZJet",
                # "FlavorPhotonJet",
                # "FlavorPureGluon",
                # "FlavorPureQuark",
                # "FlavorPureCharm",
                # "FlavorPureBottom",
                # "TimeRunA",
                # "TimeRunB",
                # "TimeRunC",
                # "TimeRunD",
                # "CorrelationGroupMPFInSitu",
                # "CorrelationGroupIntercalibration",
                # "CorrelationGroupbJES",
                # "CorrelationGroupFlavor",
                # "CorrelationGroupUncorrelated",
            ],
            # "data_per_era": True if year == 2022 else False,
            "data_per_era": False,
        },
        "FatJet": {
            "campaign": jec_campaign,
            "version": jec_ak8_version,
            "jet_type": fatjet_type,
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "uncertainty_sources": [
                # "AbsoluteStat",
                # "AbsoluteScale",
                # "AbsoluteSample",
                # "AbsoluteFlavMap",
                # "AbsoluteMPFBias",
                # "Fragmentation",
                # "SinglePionECAL",
                # "SinglePionHCAL",
                # "FlavorQCD",
                # "TimePtEta",
                # "RelativeJEREC1",
                # "RelativeJEREC2",
                # "RelativeJERHF",
                # "RelativePtBB",
                # "RelativePtEC1",
                # "RelativePtEC2",
                # "RelativePtHF",
                # "RelativeBal",
                # "RelativeSample",
                # "RelativeFSR",
                # "RelativeStatFSR",
                # "RelativeStatEC",
                # "RelativeStatHF",
                # "PileUpDataMC",
                # "PileUpPtRef",
                # "PileUpPtBB",
                # "PileUpPtEC1",
                # "PileUpPtEC2",
                # "PileUpPtHF",
                # "PileUpMuZero",
                # "PileUpEnvelope",
                # "SubTotalPileUp",
                # "SubTotalRelative",
                # "SubTotalPt",
                # "SubTotalScale",
                # "SubTotalAbsolute",
                # "SubTotalMC",
                "Total",
                # "TotalNoFlavor",
                # "TotalNoTime",
                # "TotalNoFlavorNoTime",
                # "FlavorZJet",
                # "FlavorPhotonJet",
                # "FlavorPureGluon",
                # "FlavorPureQuark",
                # "FlavorPureCharm",
                # "FlavorPureBottom",
                # "TimeRunA",
                # "TimeRunB",
                # "TimeRunC",
                # "TimeRunD",
                # "CorrelationGroupMPFInSitu",
                # "CorrelationGroupIntercalibration",
                # "CorrelationGroupbJES",
                # "CorrelationGroupFlavor",
                # "CorrelationGroupUncorrelated",
            ],
            # "data_per_era": True if year == 2022 else False,
            "data_per_era": False,
        },
        "SubJet": {
            "campaign": jec_campaign,
            "version": jec_ak4_version,
            "jet_type": jet_type,
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "uncertainty_sources": [
                # "AbsoluteStat",
                # "AbsoluteScale",
                # "AbsoluteSample",
                # "AbsoluteFlavMap",
                # "AbsoluteMPFBias",
                # "Fragmentation",
                # "SinglePionECAL",
                # "SinglePionHCAL",
                # "FlavorQCD",
                # "TimePtEta",
                # "RelativeJEREC1",
                # "RelativeJEREC2",
                # "RelativeJERHF",
                # "RelativePtBB",
                # "RelativePtEC1",
                # "RelativePtEC2",
                # "RelativePtHF",
                # "RelativeBal",
                # "RelativeSample",
                # "RelativeFSR",
                # "RelativeStatFSR",
                # "RelativeStatEC",
                # "RelativeStatHF",
                # "PileUpDataMC",
                # "PileUpPtRef",
                # "PileUpPtBB",
                # "PileUpPtEC1",
                # "PileUpPtEC2",
                # "PileUpPtHF",
                # "PileUpMuZero",
                # "PileUpEnvelope",
                # "SubTotalPileUp",
                # "SubTotalRelative",
                # "SubTotalPt",
                # "SubTotalScale",
                # "SubTotalAbsolute",
                # "SubTotalMC",
                "Total",
                # "TotalNoFlavor",
                # "TotalNoTime",
                # "TotalNoFlavorNoTime",
                # "FlavorZJet",
                # "FlavorPhotonJet",
                # "FlavorPureGluon",
                # "FlavorPureQuark",
                # "FlavorPureCharm",
                # "FlavorPureBottom",
                # "TimeRunA",
                # "TimeRunB",
                # "TimeRunC",
                # "TimeRunD",
                # "CorrelationGroupMPFInSitu",
                # "CorrelationGroupIntercalibration",
                # "CorrelationGroupbJES",
                # "CorrelationGroupFlavor",
                # "CorrelationGroupUncorrelated",
            ],
            # "data_per_era": True if year == 2022 else False,
            "data_per_era": False,
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": jet_type,
        },
        "FatJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": fatjet_type,
        },
        "SubJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": jet_type,
        },
    })

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
        "",  # total
        "Absolute",
        "AbsoluteMPFBias",
        "AbsoluteScale",
        "AbsoluteStat",
        f"Absolute_{year}",
        "BBEC1",
        f"BBEC1_{year}",
        "EC2",
        f"EC2_{year}",
        "FlavorQCD",
        "Fragmentation",
        "HF",
        f"HF_{year}",
        "PileUpDataMC",
        "PileUpPtBB",
        "PileUpPtEC1",
        "PileUpPtEC2",
        "PileUpPtHF",
        "PileUpPtRef",
        "RelativeBal",
        "RelativeFSR",
        "RelativeJEREC1",
        "RelativeJEREC2",
        "RelativeJERHF",
        "RelativePtBB",
        "RelativePtEC1",
        "RelativePtEC2",
        "RelativePtHF",
        "RelativeSample",
        f"RelativeSample_{year}",
        "RelativeStatEC",
        "RelativeStatFSR",
        "RelativeStatHF",
        "SinglePionECAL",
        "SinglePionHCAL",
        "TimePtEta",
    ]

    if cfg.x.run == 2:
        cfg.x.met_phi_correction_set = "{variable}_metphicorr_pfmet_{data_source}"
    else:
        from columnflow.calibration.cms.met import METPhiConfig
        met_column = cfg.x.met_selection.default.column
        cfg.x.met_phi_correction = METPhiConfig(
            met_name=met_column,
            met_type=met_column,
            correction_set="met_xy_corrections",
            keep_uncorrected=True,  # TODO do we need this?
            pt_phi_variations={
                "stat_xdn": "metphi_statx_down",
                "stat_xup": "metphi_statx_up",
                "stat_ydn": "metphi_staty_down",
                "stat_yup": "metphi_staty_up",
            },
            variations={
                "pu_dn": "minbias_xs_down",
                "pu_up": "minbias_xs_up",
            },
        )

    #
    # producer configurations
    #

    # lepton sf taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L338C1-L352C85
    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    if cfg.x.cpn_tag == "2022postEE":
        sf_campaign = "2022Re-recoE+PromptFG"
        # TODO: we need to use different SFs for control regions
    elif cfg.x.cpn_tag == "2022preEE":
        sf_campaign = "2022Re-recoBCD"
    elif cfg.x.cpn_tag == "2023preBPix":
        sf_campaign = "2023PromptC"
    elif cfg.x.cpn_tag == "2023postBPix":
        sf_campaign = "2023PromptD"
    elif cfg.x.cpn_tag == "2024":
        sf_campaign = "2024Prompt"
    else:
        raise ValueError(f"Invalid campaign tag '{cfg.x.cpn_tag}' for electron SF configuration.")

    cfg.x.electron_reco_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=sf_campaign,
        working_point={
            "RecoBelow20": (lambda variables: variables["pt"] < 20),
            "Reco20to75": (lambda variables: (variables["pt"] >= 20) & (variables["pt"] < 75.0)),
            "RecoAbove75": (lambda variables: variables["pt"] >= 75.0),
        },
    )
    cfg.x.electron_id_iso_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=sf_campaign,
        # working_point={
        #     "wp80iso": (lambda variables: variables["pt"] > 10),  # NOTE: probably the wrong SF
        # },
        working_point="Tight",
    )
    cfg.x.electron_trigger_sf_config = ElectronSFConfig(
        correction="Electron-HLT-SF",
        campaign=sf_campaign,
        hlt_path="HLT_SF_Ele30_TightID",
    )

    # names of muon correction sets and working points
    # (used in the muon producer)
    # TODO: we need to use different SFs for control regions
    cfg.x.muon_reco_sf_config = MuonSFConfig(
        correction="NUM_GlobalMuons_DEN_TrackerMuonProbes",
    )
    cfg.x.muon_id_sf_config = MuonSFConfig(
        correction="NUM_HighPtID_DEN_GlobalMuonProbes",
    )
    cfg.x.muon_iso_sf_config = MuonSFConfig(
        correction="NUM_probe_TightRelTkIso_DEN_HighPtProbes",
    )
    cfg.x.muon_trigger_sf_config = MuonSFConfig(
        correction="NUM_HLT_DEN_TrkHighPtTightRelIsoProbes",
    )

    if year == 2024:
        cfg.x.jet_id = JetIdConfig(
            corrections={"AK4PUPPI_Tight": 2, "AK4PUPPI_TightLeptonVeto": 3},
        )
        cfg.x.fatjet_id = JetIdConfig(
            corrections={"AK8PUPPI_Tight": 2, "AK8PUPPI_TightLeptonVeto": 3},
        )
    else:
        logger.debug(f"(Fat)Jet ID recalculation not configured for {cfg.x.cpn_tag} campaign. Will be skipped.")
        cfg.add_tag("skip_jet_ids")

    # b tagging SF configuration
    discr = "btagDeepFlavB" if year != 2024 else "btagUParTAK4B"

    btag_uncs = {
        # combined(?) uncertainties
        # uncertainties to b/c jets
        "down_bc": "bc_down",
        "up_bc": "bc_up",
        # uncertainties to light jets
        "down_light": "light_down",
        "up_light": "light_up",
        # split uncertainties(?) (all needed?)
        # uncertainties to b/c jets
        "up_correlated_bc": "correlated_bc_up",
        "up_uncorrelated_bc": "uncorrelated_bc_up",
        "up_bfragmentation_bc": "bfragmentation_bc_up",
        "up_fsrdef_bc": "fsrdef_bc_up",
        "up_hdamp_bc": "hdamp_bc_up",
        "up_isrdef_bc": "isrdef_bc_up",
        "up_jer_bc": "jer_bc_up",
        "up_jes_bc": "jes_bc_up",
        "up_muf_bc": "muf_bc_up",
        "up_mur_bc": "mur_bc_up",
        "up_pdfas_bc": "pdfas_bc_up",
        "up_pileup_bc": "pileup_bc_up",
        "up_statistic_bc": "statistic_bc_up",  # to be decorrelated between years
        "up_topmass_bc": "topmass_bc_up",
        "up_type3_bc": "type3_bc_up",
        "down_correlated_bc": "correlated_bc_down",
        "down_uncorrelated_bc": "uncorrelated_bc_down",
        "down_bfragmentation_bc": "bfragmentation_bc_down",
        "down_fsrdef_bc": "fsrdef_bc_down",
        "down_hdamp_bc": "hdamp_bc_down",
        "down_isrdef_bc": "isrdef_bc_down",
        "down_jer_bc": "jer_bc_down",
        "down_jes_bc": "jes_bc_down",
        "down_muf_bc": "muf_bc_down",
        "down_mur_bc": "mur_bc_down",
        "down_pdfas_bc": "pdfas_bc_down",
        "down_pileup_bc": "pileup_bc_down",
        "down_statistic_bc": "statistic_bc_down",  # to be decorrelated between years
        "down_topmass_bc": "topmass_bc_down",
        "down_type3_bc": "type3_bc_down",
        # uncertainties to light jets
        "down_correlated_light": "correlated_light_down",
        "up_correlated_light": "correlated_light_up",
        "down_uncorrelated_light": "uncorrelated_light_down",
        "up_uncorrelated_light": "uncorrelated_light_up",
    }
    if year == 2024:
        cfg.add_tag("skip_btag_weights")
        logger.debug("Setting up fixed WP based btag SFs for 2024, as shape based SFs are not yet available. Please switch to shape based SFs as soon as they are available.")  # noqa
        # NOTE: switch to shape based SF also for 2024 as soon as they are available
        cfg.x.btag_sf = BTagSFConfig(
            correction_set="Dummy",
            jec_sources=cfg.x.btag_sf_jec_sources,
            discriminator=discr,
        )
        # implementation from hbt analysis:
        # https://github.com/uhh-cms/hh2bbtautau/blob/4b2f1bc57a9c2ada18776e5ac6f0372269e1e26c/hbt/config/configs_hbt.py#L1410 # noqa
        cfg.x.btag_wp_count_config = BTagWPCountConfig(
            jet_name="Jet",
            btag_column=discr,
            btag_wps=cfg.x.btag_working_points.btagUParTAK4B.fixed_wp,
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
            abs_eta_edges=(0.0, 1.0, 1.5, 2.0, 5.0),
        )

        def dataset_groups(dataset_inst: od.Dataset) -> list[od.Dataset]:
            # check which group the dataset belongs to
            for group_index in range(0, len(cfg.x.btag_wp_eff_groups)):
                group_tag = f"btag_wp_eff_group_{group_index}"
                if dataset_inst.has_tag(group_tag):
                    return [
                        _dataset_inst
                        for _dataset_inst in cfg.datasets
                        if _dataset_inst.has_tag(group_tag)
                    ]
            raise NotImplementedError(f"btag WP efficiency group not implemented for dataset {dataset_inst.name}")

        cfg.x.btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Jet",
            btag_column=discr,
            correction_set="UParTAK4_merged",
            btag_wps=cfg.x.btag_working_points.btagUParTAK4B.fixed_wp,
            dataset_groups=dataset_groups,
            systs=btag_uncs,
            # further merge eta bins for sufficient statistics in each bin
            abs_eta_edges=(0.0, 1.5, 5.0),
            wp_merging={
                # remove xxtight for better stats
                "loose": ["loose"],
                "medium": ["medium"],
                "tight": ["tight"],
                "xtight": ["xtight"],
                # "xxtight": ["xxtight"],
            },
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000) if not limit_dataset_files == 2 else (0, 10_000),  # no pt binning for testing with limited files # noqa
        )
    else:
        cfg.add_tag("skip_btag_wp_weights")  # skip fixed WP based btag weights for 2022/2023, apply shape based SF
        logger.debug("Setting up shape based btag SFs for 2022/2023.")
        logger.warning_once("Evaluate used processes for normalized btag SFs for 2022/2023, set to 'tt' + 'st' for now.")
        cfg.x.btag_sf = BTagSFConfig(
            correction_set="deepJet_shape",
            jec_sources=cfg.x.btag_sf_jec_sources,
            discriminator=discr,
        )
        # implementation from hbt analysis:
        # https://github.com/uhh-cms/hh2bbtautau/blob/4b2f1bc57a9c2ada18776e5ac6f0372269e1e26c/hbt/config/configs_hbt.py#L1410 # noqa
        cfg.x.btag_wp_count_config = BTagWPCountConfig(
            jet_name="Dummy",
        )
        cfg.x.btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Dummy",
        )

    # top pt reweighting parameters
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat?rev=31
    cfg.x.top_pt_reweighting_params = {
        "a": 0.0615,
        "b": -0.0005,
    }

    # V+jets reweighting
    # FIXME update to Run 3 k-factors
    cfg.x.vjets_reweighting = DotDict.wrap({
        "w": {
            "value": "wjets_kfactor_value",
            "error": "wjets_kfactor_error",
        },
        "z": {
            "value": "zjets_kfactor_value",
            "error": "zjets_kfactor_error",
        },
    })

    #
    # systematic shifts
    #

    # read in JEC sources from file
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]
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

    # declare the shifts
    def add_shifts(cfg):
        # nominal shift
        cfg.add_shift(name="nominal", id=0)

        # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
        # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
        cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
        cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

        cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
        cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

        # pileup / minimum bias cross section variations
        cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
        cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
        add_shift_aliases(
            cfg,
            "minbias_xs",
            {
                "normalized_pu_weight": "normalized_pu_weight_{name}",
                "pu_weight": "pu_weight_{name}",
            },
        )

        # top pt reweighting
        cfg.add_shift(name="top_pt_up", id=9, type="shape")
        cfg.add_shift(name="top_pt_down", id=10, type="shape")
        add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

        # renormalization scale
        cfg.add_shift(name="mur_up", id=901, type="shape")
        cfg.add_shift(name="mur_down", id=902, type="shape")

        # factorization scale
        cfg.add_shift(name="muf_up", id=903, type="shape")
        cfg.add_shift(name="muf_down", id=904, type="shape")

        # combined renormalization and factorization scale variation
        cfg.add_shift(name="murmuf_up", id=907, type="shape")
        cfg.add_shift(name="murmuf_down", id=908, type="shape")
        cfg.add_shift(name="murmuf_envelope_up", id=909, type="shape")
        cfg.add_shift(name="murmuf_envelope_down", id=910, type="shape")
        add_shift_aliases(cfg, "murmuf", {"murmuf_weight": "murmuf_weight_{direction}"})
        add_shift_aliases(cfg, "murmuf", {"murmuf_envelope_weight": "murmuf_envelope_weight_{direction}"})

        # scale variation (?)
        cfg.add_shift(name="scale_up", id=905, type="shape")
        cfg.add_shift(name="scale_down", id=906, type="shape")

        # pdf variations
        cfg.add_shift(name="pdf_up", id=951, type="shape")
        cfg.add_shift(name="pdf_down", id=952, type="shape")

        # alpha_s variation
        cfg.add_shift(name="alpha_up", id=961, type="shape")
        cfg.add_shift(name="alpha_down", id=962, type="shape")

        # PSWeight variations
        cfg.add_shift(name="isr_up", id=7001, type="shape")  # PS weight [0] ISR=2 FSR=1
        cfg.add_shift(name="isr_down", id=7002, type="shape")  # PS weight [2] ISR=0.5 FSR=1
        add_shift_aliases(cfg, "isr", {"isr": "isr_{direction}"})
        cfg.add_shift(name="fsr_up", id=7003, type="shape")  # PS weight [1] ISR=1 FSR=2
        cfg.add_shift(name="fsr_down", id=7004, type="shape")  # PS weight [3] ISR=1 FSR=0.5
        add_shift_aliases(cfg, "fsr", {"fsr": "fsr_{direction}"})

        for unc in ["mur", "muf", "murmuf_envelope", "pdf", "isr", "fsr"]:
            col = unc
            add_shift_aliases(
                cfg,
                unc,
                {
                    f"normalized_{col}_weight": f"normalized_{col}_weight_" + "{direction}",
                    f"{col}_weight": f"{col}_weight_" + "{direction}",
                },
            )

        # event weights due to muon scale factors
        if not cfg.has_tag("skip_muon_weights"):
            # cfg.add_shift(name="muon_up", id=111, type="shape")
            # cfg.add_shift(name="muon_down", id=112, type="shape")
            # add_shift_aliases(cfg, "muon", {"muon_weight": "muon_weight_{direction}"})
            cfg.add_shift(name="muon_reco_up", id=113, type="shape")
            cfg.add_shift(name="muon_reco_down", id=114, type="shape")
            add_shift_aliases(cfg, "muon_reco", {"muon_reco_weight": "muon_reco_weight_{direction}"})
            cfg.add_shift(name="muon_id_up", id=115, type="shape")
            cfg.add_shift(name="muon_id_down", id=116, type="shape")
            add_shift_aliases(cfg, "muon_id", {"muon_id_weight": "muon_id_weight_{direction}"})
            cfg.add_shift(name="muon_iso_up", id=117, type="shape")
            cfg.add_shift(name="muon_iso_down", id=118, type="shape")
            add_shift_aliases(cfg, "muon_iso", {"muon_iso_weight": "muon_iso_weight_{direction}"})
            cfg.add_shift(name="muon_trigger_up", id=119, type="shape")
            cfg.add_shift(name="muon_trigger_down", id=120, type="shape")
            add_shift_aliases(cfg, "muon_trigger", {"muon_trigger_weight": "muon_trigger_weight_{direction}"})

        # event weights due to electron scale factors
        if not cfg.has_tag("skip_electron_weights"):
            # cfg.add_shift(name="electron_up", id=121, type="shape")
            # cfg.add_shift(name="electron_down", id=122, type="shape")
            # add_shift_aliases(cfg, "electron", {"electron_weight": "electron_weight_{direction}"})
            cfg.add_shift(name="electron_reco_up", id=123, type="shape")
            cfg.add_shift(name="electron_reco_down", id=124, type="shape")
            add_shift_aliases(cfg, "electron_reco", {"electron_reco_weight": "electron_reco_weight_{direction}"})
            cfg.add_shift(name="electron_id_iso_up", id=125, type="shape")
            cfg.add_shift(name="electron_id_iso_down", id=126, type="shape")
            add_shift_aliases(cfg, "electron_id_iso", {"electron_id_iso_weight": "electron_id_iso_weight_{direction}"})
            cfg.add_shift(name="electron_trigger_up", id=127, type="shape")
            cfg.add_shift(name="electron_trigger_down", id=128, type="shape")
            add_shift_aliases(cfg, "electron_trigger", {"electron_trigger_weight": "electron_trigger_weight_{direction}"})

        # V+jets reweighting
        cfg.add_shift(name="vjets_up", id=201, type="shape")
        cfg.add_shift(name="vjets_down", id=202, type="shape")
        add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

        # b-tagging shifts
        if year != 2024:
            logger.debug("adding shape based btag SF shifts for 2022/2023")
            btag_uncs = [
                "hf", "lf",
                "hfstats1", "hfstats2",
                "lfstats1", "lfstats2",
                "cferr1", "cferr2",
            ]
            for i, unc in enumerate(btag_uncs):
                logger.debug(
                    f"adding btag SF shift for unc. source '{unc}' with id {500 + 2 * i} (up) and {501 + 2 * i} (down)"
                )
                cfg.add_shift(name=f"btag_{unc}_up", id=500 + 2 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_down", id=501 + 2 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}",
                    {
                        btag_weight: f"{btag_weight}_{unc}_" + "{direction}"
                        for btag_weight in (
                            "btag_weight",
                            # "normalized_btag_weight",
                            # "normalized_njet_btag_weight",
                            # "normalized_ht_njet_btag_weight",
                            "normalized_ht_njet_nhf_btag_weight",
                            # "normalized_ht_btag_weight",
                        )
                    },
                )
        else:
            # https://cms-analysis-corrections.docs.cern.ch/corrections_era/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/BTV/2025-08-19/#btagging_preliminaryjsongz  # noqa
            btag_uncs_bc = [
                "fsrdef", "isrdef",
                "hdamp", "jer", "jes",
                "mass", "statistic",
                "tune",
            ]
            btag_uncs_light = [
                "correlated", "uncorrelated",
            ]
            for i, unc in enumerate(btag_uncs_bc):
                cfg.add_shift(name=f"btag_{unc}_bc_up", id=501 + 4 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_bc_down", id=502 + 4 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}_bc",
                    {
                        f"btag_weight": f"btag_weight_{unc}_bc_" + "{direction}",
                    },
                )
            for i, unc in enumerate(btag_uncs_light):
                cfg.add_shift(name=f"btag_{unc}_light_up", id=503 + 4 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_light_down", id=504 + 4 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}_light",
                    {
                        f"btag_weight": f"btag_weight_{unc}_light_" + "{direction}",
                    },
                )

            cfg.add_shift(name="btag_bc_up", id=501 + 4 * len(btag_uncs_bc), type="shape")
            cfg.add_shift(name="btag_bc_down", id=502 + 4 * len(btag_uncs_bc), type="shape")
            cfg.add_shift(name="btag_light_up", id=503 + 4 * len(btag_uncs_light), type="shape")
            cfg.add_shift(name="btag_light_down", id=504 + 4 * len(btag_uncs_light), type="shape")
            add_shift_aliases(
                cfg,
                "btag_bc",
                {
                    "btag_weight": "btag_weight_bc_" + "{direction}",
                },
            )
            add_shift_aliases(
                cfg,
                "btag_light",
                {
                    "btag_weight": "btag_weight_light_" + "{direction}",
                },
            )

        # jet energy scale (JEC) uncertainty variations
        for jec_source in cfg.x.jec.Jet.uncertainty_sources:
            idx = all_jec_sources.index(jec_source)
            cfg.add_shift(
                name=f"jec_{jec_source}_up",
                id=5000 + 2 * idx,
                type="shape",
                tags={"jec"},
                aux={
                    "jec_source": jec_source,
                    "version": 1,
                },
            )
            cfg.add_shift(
                name=f"jec_{jec_source}_down",
                id=5001 + 2 * idx,
                type="shape",
                tags={"jec"},
                aux={
                    "jec_source": jec_source,
                    "version": 1,
                },
            )
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    "Jet.pt": "Jet.pt_{name}",
                    "Jet.mass": "Jet.mass_{name}",
                    "PuppiMET.pt": "PuppiMET.pt_{name}",
                    "PuppiMET.phi": "PuppiMET.phi_{name}",
                    "FatJet.pt": "FatJet.pt_{name}",
                    "FatJet.mass": "FatJet.mass_{name}",
                },
            )

            if jec_source in ["Total", *cfg.x.btag_sf_jec_sources]:
                # when jec_source is a known btag SF source, add aliases for btag weight column
                add_shift_aliases(
                    cfg,
                    f"jec_{jec_source}",
                    {
                        btag_weight: f"{btag_weight}_jec_{jec_source}_" + "{direction}"
                        for btag_weight in (
                            "btag_weight",
                            # "normalized_btag_weight",
                            # "normalized_njet_btag_weight",
                            # "normalized_ht_njet_btag_weight",
                            "normalized_ht_njet_nhf_btag_weight",
                            # "normalized_ht_btag_weight",
                        )
                    },
                )

        # jet energy resolution (JER) scale factor variations
        cfg.add_shift(name="jer_up", id=6000, type="shape")
        cfg.add_shift(name="jer_down", id=6001, type="shape")
        add_shift_aliases(
            cfg,
            "jer",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                "PuppiMET.pt": "PuppiMET.pt_{name}",
                "PuppiMET.phi": "PuppiMET.phi_{name}",
                "FatJet.pt": "FatJet.pt_{name}",
                "FatJet.mass": "FatJet.mass_{name}",
            },
        )

    # add the shifts
    add_shifts(cfg)

    #
    # external files
    # setup taken from https://github.com/uhh-cms/hh2bbtautau/blob/ed8f363ac239b0257fc7f470b96f5c09a0572c34/hbt/config/configs_hbt.py#L1574  # noqa: E501
    # https://cms-analysis-corrections.docs.cern.ch
    #

    cfg.x.external_files = DotDict()

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    # prepare run/era/nano meta data info to determine files in the CAT metadata structure
    # see https://cms-analysis-corrections.docs.cern.ch
    cat_info = {
        (2022, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="22CDSep23-Summer22",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-12-15", jme="2026-04-13", lum="2024-01-31", muo="2026-04-28", tau="2025-12-25"),  # noqa: E501
        ),
        (2022, "EE", 12): CATInfo(
            run=3,
            vnano=12,
            era="22EFGSep23-Summer22EE",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-12-15", jme="2026-04-13", lum="2024-01-31", muo="2026-04-28", tau="2025-12-25"),  # noqa: E501
        ),
        (2023, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="23CSep23-Summer23",
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-12-15", jme="2026-04-13", lum="2024-01-31", muo="2026-04-28", tau="2025-12-25"),  # noqa: E501
        ),
        (2023, "BPix", 12): CATInfo(
            run=3,
            vnano=12,
            era="23DSep23-Summer23BPix",
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-12-15", jme="2026-04-13", lum="2024-01-31", muo="2026-04-28", tau="2025-12-25"),  # noqa: E501
        ),
        (2024, "", 15): CATInfo(
            run=3,
            vnano=15,
            era="24CDEReprocessingFGHIPrompt-Summer24",
            pog_directories={"dc": "Collisions24"},
            snapshot=CATSnapshot(btv="2026-03-10", dc="2025-07-25", egm="2025-12-15", jme="2025-12-02", muo="2026-04-28", lum="2026-04-15"),  # noqa: E501
        ),
    }[(year, campaign.x.postfix, vnano)]
    cfg.x.cat_info = cat_info

    # common files
    # (versions in the end are for hashing in cases where file contents changed but paths did not)
    add_external("lumi", {
        "golden": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: (cat_info.get_file("dc", "Cert_Collisions2022_355100_362760_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: (cat_info.get_file("dc", "Cert_Collisions2023_366442_370790_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            # not yet available at CAT space
            # 2024: (cat_info.get_file("dc", "Cert_Collisions2024_378981_386951_Golden.json"), "v1"),
            2024: ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json", "v1"),  # noqa: E501
        }[year],
        "normtag": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            2024: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),  # TODO: correct?
        }[year],
    })

    # pileup weight corrections
    if year != 2024:  # TODO: not yet available, see https://cms-analysis-corrections.docs.cern.ch
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights.json.gz"), "v1"))
    elif year == 2024:
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights_CDEFGHI.json.gz"), "v1"))

    # jet energy correction
    add_external("jet_jerc", (cat_info.get_file("jme", "jet_jerc.json.gz"), "v1"))

    # fat jet energy correction
    add_external("fat_jet_jerc", (cat_info.get_file("jme", "fatJet_jerc.json.gz"), "v1"))  # noqa: E501

    # jet veto map
    add_external("jet_veto_map", (cat_info.get_file("jme", "jetvetomaps.json.gz"), "v1"))

    # btag scale factor
    if year != 2024:
        add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging.json.gz"), "v1"))
    else:
        # keep this in case we want to switch back to the fixed wp for 2024
        add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging.json.gz"), "v1"))  # noqa: E501
        # use custom file with merged SF for both b/c and light jets
        add_external("btag_wp_sf_corr", ("/data/dust/user/matthiej/topsf/topsf/config/run3/btagging_merged.json.gz", "v1"))  # noqa: E501

    # updated jet id
    add_external("jet_id", (cat_info.get_file("jme", "jetid.json.gz"), "v1"))

    # muon scale factors
    add_external("muon_sf", (cat_info.get_file("muo", "muon_HighPt.json.gz"), "v1"))

    # met phi correction
    if year != 2024:  # TODO: not yet available for 2024
        add_external("met_phi_corr", (cat_info.get_file("jme", f"met_xyCorrections_{year}_{year}{campaign.x.postfix}.json.gz"), "v1"))  # noqa: E501

    # electron scale factors
    add_external("electron_sf", (cat_info.get_file("egm", "electron.json.gz"), "v1"))
    add_external("electron_trigger_sf", (cat_info.get_file("egm", "electronHlt.json.gz"), "v1"))
    # electron energy correction and smearing
    add_external("electron_ss", (cat_info.get_file("egm", "electronSS_EtDependent.json.gz"), "v1"))  # FIXME correct for us? # noqa: E501

    # # top-tagging scale factors (TODO)
    # "toptag_sf": (f"{sources['jet']}/JMAR/???/???.json", "v1"),  # noqa

    # # V+jets reweighting
    # "vjets_reweighting": f"{sources['local_repo']}/data/json/vjets_reweighting.json",

    #
    # event reduction configuration
    #

    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            #
            # NanoAOD columns
            #

            # general event info
            "run", "luminosityBlock", "event",

            # weights
            "genWeight",
            "LHEWeight.*",
            "LHEPdfWeight", "LHEScaleWeight",
            "PSWeight",

            # muons
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.tunepRelPt", "Muon.rawPt",
            "Muon.pdgId",
            "Muon.jetIdx",
            "Muon.nStations",
            "Muon.pfRelIso03_all", "Muon.pfRelIso04_all", "Muon.tkRelIso",

            # electrons
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "Electron.pdgId",
            "Electron.jetIdx",
            "Electron.deltaEtaSC",
            "Electron.pfRelIso03_all",

            # photons (for L1 prefiring)
            "Photon.pt", "Photon.eta", "Photon.phi", "Photon.mass",
            "Photon.jetIdx",

            # columns for btag reweighting crosschecks
            "njet", "ht", "nhf",

            # AK4 jets
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
            "Jet.rawFactor",
            "Jet.btagDeepFlavB", "Jet.hadronFlavour", "Jet.btagUParTAK4B",
            # # optional, enable if needed
            # "Jet.area",
            # "Jet.hadronFlavour", "Jet.partonFlavour",
            # "Jet.jetId", "Jet.puId", "Jet.puIdDisc",
            # # cleaning
            # "Jet.cleanmask",
            # "Jet.muonSubtrFactor",
            # # indices to other collections
            # "Jet.electronIdx*",
            # "Jet.muonIdx*",
            # "Jet.genJetIdx*",
            # # number of jet constituents
            # "Jet.nConstituents",
            # "Jet.nElectrons",
            # "Jet.nMuons",
            # # PF energy fractions
            # "Jet.chEmEF",
            # "Jet.chHEF",
            # "Jet.neEmEF",
            # "Jet.neHEF",
            # "Jet.muEF",
            # # taggers
            # "Jet.qgl",
            # "Jet.btag*",

            # AK8 jets
            "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass", "FatJet.msoftdrop",
            "FatJet.rawFactor",
            "FatJet.tau1", "FatJet.tau2", "FatJet.tau3", "FatJet.tau4",
            "FatJet.subJetIdx1", "FatJet.subJetIdx2",
            # # optional, enable if needed
            # "FatJet.area", "FatJet.jetId", "FatJet.hadronFlavour",
            # "FatJet.genJetAK8Idx",
            # "FatJet.muonIdx3SJ", "FatJet.electronIdx3SJ",
            # "FatJet.nBHadrons", "FatJet.nCHadrons",
            # # taggers
            # "FatJet.btag*", "FatJet.deepTag*", "FatJet.particleNet*",

            # subjets
            "SubJet.btagDeepB", "SubJet.btagUParTAK4B"

            # generator quantities
            "Generator.*",

            # # generator particles
            # "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
            # "GenPart.pdgId",
            # "GenPart.*",

            # missing transverse momentum
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
            "PuppiMET.phi", "PuppiMET.pt",

            # number of primary vertices
            "PV.npvs",
            "PV.npvsGood",

            # average number of pileup interactions
            "Pileup.nTrueInt",

            #
            # columns added during selection
            #

            # generator particle info
            "GenTopDecay.*",
            "GenTopAssociatedDecay.*",
            "GenPartonTop.*",
            "GenVBoson.*",

            # generic leptons (merger of Muon/Electron)
            "Lepton.*",

            # probe jet
            "ProbeJet.*",

            # columns for PlotCutflowVariables
            "cutflow.*",

            # other columns, required by various tasks
            "channel_id", "category_ids", "process_id",
            "deterministic_seed",
            "mc_weight",
            "pu_weight*",
            "pdf_weight*", "fsr_weight*", "isr_weight*",
            "muf_weight*", "mur_weight*", "murmuf_weight*", "murmuf_envelope*",
        },
        "cf.MergeSelectionMasks": {
            "channel_id", "process_id", "category_ids",
            "normalization_weight",
            "cutflow.*",
            "mc_weight",
        },
        "cf.UniteColumns": {
            "*",
        },
    })

    #
    # event weights
    #

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    # add b tagging weights
    btag_shifts = ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2", "cferr1", "cferr2"]
    full_btag_uncs = btag_uncs_bc_full + btag_uncs_light_full
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "normalized_pu_weight": get_shifts("minbias_xs"),
        "muon_reco_weight": get_shifts("muon_reco"),
        "muon_id_weight": get_shifts("muon_id"),
        "muon_iso_weight": get_shifts("muon_iso"),
        "muon_trigger_weight": get_shifts("muon_trigger"),
        "electron_id_iso_weight": get_shifts("electron_id_iso"),
        "electron_reco_weight": get_shifts("electron_reco"),
    })
    if cfg.has_tag("use_non_normalized_weights"):
        logger.debug("Using non-normalized event weights.")
        # store non normalized for future checks
        cfg.x.event_weights["btag_weight"] = get_shifts("btag")
        cfg.x.event_weights["fsr_weight"] = get_shifts("fsr")
        cfg.x.event_weights["isr_weight"] = get_shifts("isr")
    else:
        logger.debug("Using normalized event weights.")
        if not cfg.x.year == 2024:
            logger.debug("Use normalized btag weights for 2022/2023, with shape-based SF and uncertainties.")
            cfg.x.event_weights["normalized_ht_njet_nhf_btag_weight"] = get_shifts(
                *(f"btag_{unc}" for unc in btag_shifts)
            )
            # cfg.x.event_weights["normalized_njet_btag_weight"] = get_shifts("btag")
            # cfg.x.event_weights["normalized_ht_btag_weight"] = get_shifts("btag")

    if cfg.x.year == 2024:
        logger.debug("Using fixed wp btag weights for 2024, with separate uncertainties for b/c and light jets.")
        cfg.x.event_weights["btag_weight"] = get_shifts(*(f"btag_{unc}" for unc in full_btag_uncs))

    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if dataset.has_tag("is_ttbar"):
            # top pt reweighting
            dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")
        if not has_tag("skip_kfactor_weights", cfg, dataset, operator=any) and dataset.has_tag("is_v_jets"):
            # V+jets QCD NLO reweighting
            dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")
        # add PSWeight variations for all datasets but qcd
        if not cfg.has_tag("use_non_normalized_weights"):
            if not dataset.has_tag("is_qcd") and dataset.is_mc:
                logger.debug_once("Use normalized ps weights.")
                dataset.x.event_weights["normalized_isr_weight"] = get_shifts("isr")
                dataset.x.event_weights["normalized_fsr_weight"] = get_shifts("fsr")
            if dataset.has_tag("has_top"):
                logger.debug_once("Use normalized scale variation weights.")
                dataset.x.event_weights["normalized_mur_weight"] = get_shifts("mur")
                dataset.x.event_weights["normalized_muf_weight"] = get_shifts("muf")
                # switch to combined if needed
                # dataset.x.event_weights["normalized_murmuf_envelope_weight"] = get_shifts("murmuf_envelope")
                # dataset.x.event_weights["normalized_murmuf_weight"] = get_shifts("murmuf")
            if not has_tag("skip_pdf", cfg, dataset):
                logger.debug_once("Use normalized pdf weights.")
                dataset.x.event_weights["normalized_pdf_weight"] = get_shifts("pdf")

        # group datasets together for btag WP efficiency calculation in 2024
        if year == 2024:
            # TODO figure out which datasets should be grouped together;
            # for now, group all datasets together
            cfg.x.btag_wp_eff_groups = [
                ["tt_*", "st_*", "qcd_*", "ww_*", "dy_*", "w_lnu_*", "wz_*", "zz_*"],
                # ["tt_*"],
                # ["st_*"],
                # ["dy_*"],
                # ["w_lnu_*"],
                # ["ww_*", "wz_*", "zz_*"],
                # ["qcd_*"],
                # ["dy_*", "w_lnu_*", "wz_*", "zz_*", "ww_*", "qcd_*"],
                # ["tt_*", "st_*"],
            ]
            group_matched = False
            for i, dataset_pattern in enumerate(cfg.x.btag_wp_eff_groups):
                if law.util.multi_match(dataset.name, dataset_pattern):
                    if group_matched:
                        raise ValueError(
                            f"dataset '{dataset.name}' already has a btag WP group assigned! Cannot assign it to more "
                            "than one group",
                        )
                    group_matched = True
                    dataset.add_tag(f"btag_wp_eff_group_{i}")
            if not group_matched and dataset.is_mc:
                raise ValueError(f"no btag_wp_eff_group_* assigned to dataset '{dataset.name}'")
            if group_matched and dataset.is_data:
                raise ValueError(f"must not assign btag_wp_eff_group_* to dataset '{dataset.name}'")

    # #
    # # versions
    # #
    # cfg.x.versions = {
    #     "tt_*": "test_v7",
    #     "st_*": "test_v7",
    #     "dy_*": "test_v7",
    #     "w_*": "test_v7",
    #     "ww_*": "test_v7",
    #     "wz_*": "test_v7",
    #     "zz_*": "test_v7",
    #     "data_*": "test_v7",
    #     "topsf.CreateDatacards": "test_v8",
    # }

    # # named references to actual versions to use for certain sets of tasks
    # main_ver = "test_v4"
    # cfg.x.named_versions = DotDict.wrap({
    #     "default": f"{main_ver}",
    #     "calibrate": "test_v4",
    #     "select": "test_v4",
    #     "reduce": f"{main_ver}",
    #     "merge": f"{main_ver}",
    #     "produce": f"{main_ver}",
    #     "hist": f"{main_ver}",
    #     "plot": f"{main_ver}",
    #     "datacards": f"{main_ver}",
    # })

    # # versions per task family and optionally also dataset and shift
    # # None can be used as a key to define a default value
    # cfg.x.versions = {
    #     None: cfg.x.named_versions["default"],
    #     # CSR tasks
    #     "cf.CalibrateEvents": cfg.x.named_versions["calibrate"],
    #     "cf.SelectEvents": cfg.x.named_versions["select"],
    #     "cf.ReduceEvents": cfg.x.named_versions["reduce"],
    #     # merging tasks
    #     "cf.MergeSelectionStats": cfg.x.named_versions["merge"],
    #     "cf.MergeSelectionMasks": cfg.x.named_versions["merge"],
    #     "cf.MergeReducedEvents": cfg.x.named_versions["merge"],
    #     "cf.MergeReductionStats": cfg.x.named_versions["merge"],
    #     # column production
    #     "cf.ProduceColumns": cfg.x.named_versions["produce"],
    #     # histogramming
    #     "cf.CreateCutflowHistograms": cfg.x.named_versions["hist"],
    #     "cf.CreateHistograms": cfg.x.named_versions["hist"],
    #     "cf.MergeHistograms": cfg.x.named_versions["hist"],
    #     "cf.MergeShiftedHistograms": cfg.x.named_versions["hist"],
    #     # plotting
    #     "cf.PlotVariables1D": cfg.x.named_versions["plot"],
    #     "cf.PlotVariables2D": cfg.x.named_versions["plot"],
    #     "cf.PlotVariablesPerProcess2D": cfg.x.named_versions["plot"],
    #     "cf.PlotShiftedVariables1D": cfg.x.named_versions["plot"],
    #     "cf.PlotShiftedVariablesPerProcess1D": cfg.x.named_versions["plot"],
    #     #
    #     "cf.PlotCutflow": cfg.x.named_versions["plot"],
    #     "cf.PlotCutflowVariables1D": cfg.x.named_versions["plot"],
    #     "cf.PlotCutflowVariables2D": cfg.x.named_versions["plot"],
    #     "cf.PlotCutflowVariablesPerProcess2D": cfg.x.named_versions["plot"],
    #     # datacards
    #     "cf.CreateDatacards": cfg.x.named_versions["datacards"],
    # }

    #
    # finalization
    #

    # add categories
    add_categories(cfg)

    # add variables
    add_variables(cfg)

    # add channels
    cfg.add_channel("e", id=1)
    cfg.add_channel("mu", id=2)

    return cfg
