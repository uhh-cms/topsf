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

import order as od
import yaml

from scinum import Number

from columnflow.util import DotDict
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)

from topsf.config.variables import add_variables
from topsf.config.categories import add_categories

thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
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
    assert campaign.x.year in [2022, 2023]
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

    implemented_years = [2022]

    if year not in implemented_years:
        raise NotImplementedError("For now, only 2022 campaign is fully implemented")

    # create a config by passing the campaign
    # (if id and name are not set they will be taken from the campaign)
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # add some important tags to the config
    cfg.x.cpn_tag = f"{year}{corr_postfix}"

    if year in (2022, 2023):
        cfg.x.run = 3

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

    # add datasets we need to study
    dataset_names = [
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop 2022 v12 preEE datasets
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # DY 2022 v12 preEE datasets
        # "dy_m4to50_ht40to70_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "dy_m4to50_ht70to100_madgraph",  # FIXME AssertionError in preEE (full stat.)
        "dy_m4to50_ht100to400_madgraph",
        "dy_m4to50_ht400to800_madgraph",
        "dy_m4to50_ht800to1500_madgraph",
        "dy_m4to50_ht1500to2500_madgraph",
        "dy_m4to50_ht2500toinf_madgraph",
        # "dy_m50to120_ht40to70_madgraph",  # FIXME AssertionError in preEE (full stat.)
        "dy_m50to120_ht70to100_madgraph",
        "dy_m50to120_ht100to400_madgraph",
        "dy_m50to120_ht400to800_madgraph",
        # WJets 2022 v12 preEE datasets
        # "w_lnu_mlnu0to120_ht40to100_madgraph",  # FIXME cramjam.DecompressionError: lzma data error
        "w_lnu_mlnu0to120_ht100to400_madgraph",
        "w_lnu_mlnu0to120_ht400to800_madgraph",
        "w_lnu_mlnu0to120_ht800to1500_madgraph",
        "w_lnu_mlnu0to120_ht1500to2500_madgraph",
        "w_lnu_mlnu0to120_ht2500toinf_madgraph",
        # Diboson
        "ww_pythia",
        "wz_pythia",
        "zz_pythia",
        # QCD 2022 v12 preEE datasets
        # "qcd_ht70to100_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "qcd_ht100to200_madgraph",  # FIXME no xs for 13.6 in https://xsdb-temp.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=DAS%3DQCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8  # noqa
        # "qcd_ht200to400_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "qcd_ht400to600_madgraph",  # FIXME AssertionError in preEE (full stat.)
        "qcd_ht600to800_madgraph",
        "qcd_ht800to1000_madgraph",
        "qcd_ht1000to1200_madgraph",
        "qcd_ht1200to1500_madgraph",
        "qcd_ht1500to2000_madgraph",
        "qcd_ht2000toinf_madgraph",
    ]
    if campaign.x.EE == "pre":
        dataset_names += [
            # "data_egamma_a",  # FIXME JEC setup fails
            # "data_egamma_b",  # FIXME JEC setup fails
            "data_egamma_c",
            "data_egamma_d",
            "data_mu_c",
            "data_mu_d",
        ]
    if campaign.x.EE == "post":
        dataset_names += [
            "data_egamma_e",
            "data_egamma_f",
            "data_egamma_g",
            "data_mu_e",
            "data_mu_f",
            "data_mu_g",
        ]

    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # update JECera information
        if dataset.is_data and (dataset_name.endswith("c") or dataset_name.endswith("d")):
            dataset.x.era = "CD"

        # mark the presence of a top quark
        if any(dataset_name.startswith(s) for s in ("tt", "st")):
            dataset.add_tag("has_top")
            if "twchannel" in dataset_name:
                dataset.add_tag("has_top_associated_w")

        # mark ttbar (for top pT reweighting)
        if dataset_name.startswith("tt"):
            dataset.add_tag("is_ttbar")

        # mark v+jets processes (for NLO reweighting)
        if dataset_name.startswith("w_lnu"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_w_jets")
        if dataset_name.startswith("dy"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_z_jets")

        # for testing purposes, limit the number of files per dataset TODO make #files variable depending on dataset
        if limit_dataset_files:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

    # verify that the root processes of each dataset (or one of their
    # ancestor processes) are registered in the config
    verify_config_processes(cfg, warn=True)

    #
    # defaults
    #

    # default objects, such as calibrator, selector, producer,
    # ml model, inference model, etc
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "default"  # "uhh2"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = (
        "probejet_pt",
        "probejet_mass",
        "probejet_msoftdrop_widebins",
        "probejet_tau32",
        "probejet_max_subjet_btag_score_btagDeepB",
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
        "qcd": ["qcd_ht*"],
        "st": ["st*"],
        "tt": ["tt*"],
        "vv": ["ww_pythia", "wz_pythia", "zz_pythia"],
        "vx": ["w_lnu*", "dy*", "ww_pythia", "wz_pythia", "zz_pythia"],
        "mj": ["qcd_ht*"],
    }

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
    cfg.x.default_weight_producer = "all_weights"

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
    # pileup
    #

    # # minimum bias cross section in mb (milli) for creating PU weights, values from
    # # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    # cfg.x.minbias_xs = Number(69.2, 0.046j)

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

    #
    # JEC & JER  # FIXME: Taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    #

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=2017#Jet_Energy_Corrections_in_Run2

    # jec configuration taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = ""
    if year == 2022 and campaign.x.EE == "post":
        jerc_postfix = "EE"

    if cfg.x.run == 3:
        jerc_campaign = f"Summer{year2}{jerc_postfix}_22Sep2023"
        jet_type = "AK4PFPuppi"

    cfg.x.jec = DotDict.wrap({
        "campaign": jerc_campaign,
        "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2"}[year],
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
            "CorrelationGroupMPFInSitu",
            "CorrelationGroupIntercalibration",
            "CorrelationGroupbJES",
            "CorrelationGroupFlavor",
            "CorrelationGroupUncorrelated",
        ],
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    # TODO: get jerc working for Run3
    cfg.x.jer = DotDict.wrap({
        "campaign": jerc_campaign,
        "version": {2022: "JRV1"}[year],
        "jet_type": jet_type,
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

    #
    # tagger working points
    #

    # b-tag working points
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    # TODO: add correct 2022 + 2022preEE WP for deepcsv if needed
    btag_key = f"2022{campaign.x.EE}EE" if year == 2022 else year
    cfg.x.btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {
                "2022preEE": 0.0583, "2022postEE": 0.0614,
            }[btag_key],
            "medium": {
                "2022preEE": 0.3086, "2022postEE": 0.3196,
            }[btag_key],
            "tight": {
                "2022preEE": 0.7183, "2022postEE": 0.7300,
            }[btag_key],
        },
        "deepcsv": {
            "loose": {
                "2022preEE": 0.1208, "2022postEE": 0.1208,
            }[btag_key],
            "medium": {
                "2022preEE": 0.4168, "2022postEE": 0.4168,
            }[btag_key],
            "tight": {
                "2022preEE": 0.7665, "2022postEE": 0.7665,
            }[btag_key],
        },
    })

    # top-tag working points
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
    # FIXME use my own WPs here?
    cfg.x.toptag_working_points = DotDict.wrap({
        "tau32": {
            "very_loose": 0.69,
            "loose": 0.61,
            "medium": 0.52,
            "tight": 0.47,
            "very_tight": 0.38,
        },
    })

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
                "IsoMu27",
            },
            "id": {
                "column": "tightId",
                "value": True,
            },
            "rel_iso": "pfRelIso03_all",
            "max_rel_iso": 1.5,
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
                "Ele35_WPTight_Gsf",
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
    cfg.x.jet_selection = DotDict.wrap({
        "ak8": {
            "column": "FatJet",
            "min_pt": 300,
            "max_abseta": 2.5,
            "msoftdrop_range": (105, 210),
            # probe jet pt bins )used by category builder)
            "pt_bins": [300, 400, 480, 600, None],
            # parameters for b-tagged subjets
            "subjet_column": "SubJet",
            "subjet_btag": "btagDeepB",
            "subjet_btag_wp": cfg.x.btag_working_points.deepcsv.loose,
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
            "min_pt": 15,  # TODO: check UHH2
            "max_abseta": 2.5,  # TODO: check UHH2
            "btag_column": "btagDeepFlavB",  # nano v9: "DeepJet b+bb+lepb tag discriminator"
            "btag_wp": cfg.x.btag_working_points.deepjet.medium,
        },
    })

    # MET selection parameters
    cfg.x.met_selection = DotDict.wrap({
        "default": {
            "column": "MET",
            "min_pt": 50,
        },
    })

    #
    # producer configurations
    #

    if cfg.x.run == 3:
        # TODO: check that everyting is setup as intended

        # btag weight configuration
        cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

        # lepton sf taken from
        # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L338C1-L352C85
        # names of electron correction sets and working points
        # (used in the electron_sf producer)
        if cfg.x.cpn_tag == "2022postEE":
            # TODO: we need to use different SFs for control regions
            cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoE+PromptFG", "Tight")
        elif cfg.x.cpn_tag == "2022preEE":
            cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoBCD", "Tight")

        # names of muon correction sets and working points
        # (used in the muon producer)
        # TODO: we need to use different SFs for control regions
        cfg.x.muon_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{cfg.x.cpn_tag}")
        cfg.x.muon_id_sf_names = ("NUM_TightID_DEN_TrackerMuons", f"{cfg.x.cpn_tag}")
        cfg.x.muon_iso_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{cfg.x.cpn_tag}")

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
        add_shift_aliases(cfg, "minbias_xs", {"pu_weight": "pu_weight_{name}"})

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

        # scale variation (?)
        cfg.add_shift(name="scale_up", id=905, type="shape")
        cfg.add_shift(name="scale_down", id=906, type="shape")

        # pdf variations
        cfg.add_shift(name="pdf_up", id=951, type="shape")
        cfg.add_shift(name="pdf_down", id=952, type="shape")

        # alpha_s variation
        cfg.add_shift(name="alpha_up", id=961, type="shape")
        cfg.add_shift(name="alpha_down", id=962, type="shape")

        # TODO: murf_envelope?
        for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
            add_shift_aliases(cfg, unc, {
                # TODO: normalized?
                f"{unc}_weight": f"{unc}_weight_{{direction}}",
            })

        # event weights due to muon scale factors
        if not cfg.has_tag("skip_muon_weights"):
            cfg.add_shift(name="muon_up", id=111, type="shape")
            cfg.add_shift(name="muon_down", id=112, type="shape")
            add_shift_aliases(cfg, "muon", {"muon_weight": "muon_weight_{direction}"})

        # event weights due to electron scale factors
        if not cfg.has_tag("skip_electron_weights"):
            cfg.add_shift(name="electron_up", id=121, type="shape")
            cfg.add_shift(name="electron_down", id=122, type="shape")
            add_shift_aliases(cfg, "electron", {"electron_weight": "electron_weight_{direction}"})

        # V+jets reweighting
        cfg.add_shift(name="vjets_up", id=201, type="shape")
        cfg.add_shift(name="vjets_down", id=202, type="shape")
        add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

        # b-tagging shifts
        btag_uncs = [
            "hf", "lf",
            f"hfstats1_{year}", f"hfstats2_{year}",
            f"lfstats1_{year}", f"lfstats2_{year}",
            "cferr1", "cferr2",
        ]
        for i, unc in enumerate(btag_uncs):
            cfg.add_shift(name=f"btag_{unc}_up", id=501 + 2 * i, type="shape")
            cfg.add_shift(name=f"btag_{unc}_down", id=502 + 2 * i, type="shape")
            add_shift_aliases(
                cfg,
                f"btag_{unc}",
                {
                    # taken from
                    # https://github.com/uhh-cms/hh2bbww/blob/c6d4ee87a5c970660497e52aed6b7ebe71125d20/hbw/config/config_run2.py#L421
                    "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                    "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
                    "btag_weight": f"btag_weight_{unc}_" + "{direction}",
                    "njet_btag_weight": f"njet_btag_weight_{unc}_" + "{direction}",
                },
            )

        # jet energy scale (JEC) uncertainty variations
        for jec_source in cfg.x.jec.uncertainty_sources:
            idx = all_jec_sources.index(jec_source)
            cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
            cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    "Jet.pt": "Jet.pt_{name}",
                    "Jet.mass": "Jet.mass_{name}",
                    "MET.pt": "MET.pt_{name}",
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
                "MET.pt": "MET.pt_{name}",
            },
        )

    # add the shifts
    add_shifts(cfg)

    #
    # external files
    # taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L535C7-L579C84
    #

    # external files
    json_mirror = "/afs/cern.ch/user/j/jmatthie/public/mirrors/jsonpog-integration-49ddc547"
    local_repo = "/nfs/dust/cms/user/matthiej/topsf"  # TODO: avoid hardcoding path

    if cfg.x.run == 3:
        corr_tag = f"{year}_Summer22{jerc_postfix}"

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{json_mirror}/POG/LUM/{corr_tag}/puWeights.json.gz", "v1"),

        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{corr_tag}/jet_jerc.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{corr_tag}/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{corr_tag}/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{corr_tag}/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{json_mirror}/POG/JME/{corr_tag}/met.json.gz", "v1"),

        # V+jets reweighting
        "vjets_reweighting": f"{local_repo}/data/json/vjets_reweighting.json.gz",
    })

    # temporary fix due to missing corrections in run 3
    if cfg.x.run == 3:
        # cfg.add_tag("skip_electron_weights")
        # cfg.add_tag("skip_muon_weights")
        cfg.x.external_files.pop("met_phi_corr")

    if year == 2022 and campaign.x.EE == "pre":
        cfg.x.external_files.update(DotDict.wrap({
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "lumi": {
                "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },
            "pu": {
                # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileup_JSON.txt", "v1"),  # noqa
                "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
                "data_profile": {
                    # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))
    elif year == 2022 and campaign.x.EE == "post":
        cfg.x.external_files.update(DotDict.wrap({
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "lumi": {
                "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },
            "pu": {
                # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileup_JSON.txt", "v1"),  # noqa
                "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
                "data_profile": {
                    # data profiles were produced with 99 bins instead of 100 --> use custom produced data profiles instead  # noqa
                    # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))
    else:
        raise NotImplementedError(f"No lumi and pu files provided for year {year}")

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

            # muons
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
            "Muon.pdgId",
            "Muon.jetIdx",
            "Muon.nStations",
            "Muon.pfRelIso03_all", "Muon.pfRelIso04_all",

            # electrons
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "Electron.pdgId",
            "Electron.jetIdx",
            "Electron.deltaEtaSC",
            "Electron.pfRelIso03_all",

            # photons (for L1 prefiring)
            "Photon.pt", "Photon.eta", "Photon.phi", "Photon.mass",
            "Photon.jetIdx",

            # AK4 jets
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
            "Jet.rawFactor",
            "Jet.btagDeepFlavB", "Jet.hadronFlavour",
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
            "SubJet.btagDeepB",

            # generator quantities
            "Generator.*",

            # # generator particles
            # "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
            # "GenPart.pdgId",
            # "GenPart.*",

            # missing transverse momentum
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",

            # number of primary vertices
            "PV.npvs",

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
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pu_weight": get_shifts("minbias_xs"),
        "muon_weight": get_shifts("muon"),
    })

    if not cfg.has_tag("skip_electron_weights"):
        cfg.x.event_weights["electron_weight"] = get_shifts("electron")
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if dataset.has_tag("is_ttbar"):
            # top pt reweighting
            dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")
        if dataset.has_tag("is_v_jets"):
            # V+jets QCD NLO reweighting
            dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")

    # #
    # # versions
    # #

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
