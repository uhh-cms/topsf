# coding: utf-8

"""
Configuration creation for top-tagging working point
measurement using Run3 samples.
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
from topsf.config.categories_wp import add_categories
from topsf.util import has_tag

thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    """
    Configurable function for creating a config for a run2 analysis given
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

    # set color of some processes
    colors = {
        "tt": "#E04F21",  # red
        "qcd": "#5E8FFC",  # blue
    }

    # add processes we are interested in
    process_names = [
        "tt",
        "qcd",
        "qcd_mu_pt15to20",
        "qcd_mu_pt20to30",
        "qcd_mu_pt30to50",
        "qcd_mu_pt50to80",
        "qcd_mu_pt80to120",
        "qcd_mu_pt120to170",
        "qcd_mu_pt170to300",
        "qcd_mu_pt300to470",
        "qcd_mu_pt470to600",
        "qcd_mu_pt600to800",
        "qcd_mu_pt800to1000",
        "qcd_mu_pt1000toinf",
        "qcd_em_pt10to30",
        "qcd_em_pt30to50",
        "qcd_em_pt50to80",
        "qcd_em_pt80to120",
        "qcd_em_pt120to170",
        "qcd_em_pt170to300",
        "qcd_em_pt300toinf",
    ]

    blue_shades_mu = [
        "#ADD8E6", "#87CEEB", "#5CB3FF", "#4682B4",
        "#4169E1", "#0000FF", "#0000CD", "#00008B",
        "#191970", "#0F52BA", "#082567", "#001F3F",
    ]
    green_shades_em = [
        "#98FB98", "#7CFC00", "#32CD32",
        "#228B22", "#006400", "#013220", "#002B00",
    ]

    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))

        # mark the presence of a top quark
        if proc.name.startswith("tt"):
            proc.add_tag({"has_top", "is_ttbar"})

        # assign colors of qcd_mu processes using the blue shades
        if proc.name.startswith("qcd_mu"):
            proc.color = blue_shades_mu.pop(0)

        # assign colors of qcd_em processes using the green shades
        if proc.name.startswith("qcd_em"):
            proc.color = green_shades_em.pop(0)

        # assign colors of tt processes
        if proc.name == "tt":
            proc.color = colors["tt"]

        # assign colors of qcd processes
        if proc.name == "qcd":
            proc.color = colors["qcd"]

    #
    # datasets
    #

    # add datasets we need to study
    dataset_names = [
        # ttbar (fully hadronic decays only)
        "tt_fh_powheg",
        # # QCD 2022 v12 datasets
        # "qcd_ht70to100_madgraph",
        # # "qcd_ht100to200_madgraph",  # FIXME no xs for 13.6 in https://xsdb-temp.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=DAS%3DQCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8  # noqa
        # "qcd_ht200to400_madgraph",
        # "qcd_ht400to600_madgraph",
        # "qcd_ht600to800_madgraph",
        # "qcd_ht800to1000_madgraph",
        # "qcd_ht1000to1200_madgraph",
        # "qcd_ht1200to1500_madgraph",
        # "qcd_ht1500to2000_madgraph",
        # "qcd_ht2000toinf_madgraph",
        # QCD mu enriched
        # "qcd_mu_pt15to20_pythia",  # FIXME AssertionError for lim. stats. (preEE)
        # # "qcd_mu_pt20to30_pythia",  # FIXME AssertionError for lim. stats. (preEE)  # FIXME broken Root file?
        # "qcd_mu_pt30to50_pythia",  # FIXME AssertionError for lim. stats. (preEE)
        # "qcd_mu_pt50to80_pythia",
        # "qcd_mu_pt80to120_pythia",
        "qcd_mu_pt120to170_pythia",
        "qcd_mu_pt170to300_pythia",
        "qcd_mu_pt300to470_pythia",
        "qcd_mu_pt470to600_pythia",
        "qcd_mu_pt600to800_pythia",
        "qcd_mu_pt800to1000_pythia",
        "qcd_mu_pt1000toinf_pythia",
        # QCD em enriched
        # "qcd_em_pt10to30_pythia",
        # "qcd_em_pt30to50_pythia",
        # "qcd_em_pt50to80_pythia",
        # "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toinf_pythia",
    ]
    # if campaign.x.EE == "post":
    #     dataset_names += [
    #         "qcd_mu_pt20to30_pythia",
    #     ]
    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # mark ttbar
        if dataset_name.startswith("tt"):
            dataset.add_tag({"has_top", "is_ttbar"})

        # mark QCD
        if dataset_name.startswith("qcd"):
            dataset.add_tag({"is_qcd"})

        # for testing purposes, limit the number of files per dataset
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
    cfg.x.default_selector = "wp"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = None
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = (
        "fatjet_pt",
        "fatjet_tau32",
    )

    #
    # parameter groups
    #

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "all": process_names,
        "qcd": ["qcd*"],
        "tt": ["tt*"],
    }

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {
        "all": dataset_names,
        "qcd": ["qcd*"],
        "tt": ["tt*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}

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
            "FatJet", "METFilters",
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
    # pileup
    #

    # # minimum bias cross section in mb (milli) for creating PU weights, values from
    # # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    # not used after moving to correctionlib based PU weights
    # cfg.x.minbias_xs = Number(69.2, 0.046j)

    #
    # MET filters
    #

    # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
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
        fatjet_type = "AK8PFPuppi"

    cfg.x.jec = DotDict.wrap({
        "Jet": {
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
        },
        "FatJet": {
            "campaign": jerc_campaign,
            "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2"}[year],
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
                "CorrelationGroupMPFInSitu",
                "CorrelationGroupIntercalibration",
                "CorrelationGroupbJES",
                "CorrelationGroupFlavor",
                "CorrelationGroupUncorrelated",
            ],
        },
        "SubJet": {
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
                # "CorrelationGroupMPFInSitu",
                # "CorrelationGroupIntercalibration",
                # "CorrelationGroupbJES",
                # "CorrelationGroupFlavor",
                # "CorrelationGroupUncorrelated",
            ],
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    # TODO: get jerc working for Run3
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": jerc_campaign,
            "version": {2022: "JRV1"}[year],
            "jet_type": jet_type,
        },
        "FatJet": {
            "campaign": jerc_campaign,
            "version": {2022: "JRV1"}[year],
            "jet_type": jet_type,
            "external_file_key": "jet_jerc",
        },
        "SubJet": {
            "campaign": jerc_campaign,
            "version": {2022: "JRV1"}[year],
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
            "max_abseta": 2.4,  # note: SF analysis has 2.5
            "msoftdrop_range": (105, 210),
            # probe jet pt bins (used by category builder)
            "pt_bins": [300, 400, 480, 600, None],
            # parameters for b-tagged subjets
            "subjet_column": "SubJet",
            "subjet_btag": "btagDeepB",
            "subjet_btag_wp": cfg.x.btag_working_points.deepcsv.loose,
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
        "a_up": 0.0615 * 1.5,
        "a_down": 0.0615 * 0.5,
        "b": -0.0005,
        "b_up": -0.0005 * 1.5,
        "b_down": -0.0005 * 0.5,
    }

    # V+jets reweighting
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
    # FIXME same as Run2?
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
        if not has_tag("skip_muon_weights", cfg, operator=any):
            cfg.add_shift(name="muon_up", id=111, type="shape")
            cfg.add_shift(name="muon_down", id=112, type="shape")
            add_shift_aliases(cfg, "muon", {"muon_weight": "muon_weight_{direction}"})

        # event weights due to electron scale factors
        cfg.add_shift(name="electron_up", id=121, type="shape")
        cfg.add_shift(name="electron_down", id=122, type="shape")
        add_shift_aliases(cfg, "electron", {"electron_weight": "electron_weight_{direction}"})

        # V+jets reweighting
        cfg.add_shift(name="vjets_up", id=201, type="shape")
        cfg.add_shift(name="vjets_down", id=202, type="shape")
        add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

        # prefiring weights
        cfg.add_shift(name="l1_ecal_prefiring_up", id=301, type="shape")
        cfg.add_shift(name="l1_ecal_prefiring_down", id=302, type="shape")
        add_shift_aliases(
            cfg,
            "l1_ecal_prefiring",
            {
                "l1_ecal_prefiring_weight": "l1_ecal_prefiring_weight_{direction}",
            },
        )

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
                    # TODO: normalized?
                    #"normalized_btag_weight": f"normalized_btag_weight_{unc}_{{direction}}",  # noqa
                    #"normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_{{direction}}",  # noqa
                    "btag_weight": f"btag_weight_{unc}_{{direction}}",
                },
            )

        # jet energy scale (JEC) uncertainty variations
        for jec_source in cfg.x.jec.Jet.uncertainty_sources:
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

        # PSWeight variations
        cfg.add_shift(name="ISR_up", id=7001, type="shape")  # PS weight [0] ISR=2 FSR=1
        cfg.add_shift(name="ISR_down", id=7002, type="shape")  # PS weight [2] ISR=0.5 FSR=1
        add_shift_aliases(cfg, "ISR", {"ISR": "ISR_{direction}"})
        cfg.add_shift(name="FSR_up", id=7003, type="shape")  # PS weight [1] ISR=1 FSR=2
        cfg.add_shift(name="FSR_down", id=7004, type="shape")  # PS weight [3] ISR=1 FSR=0.5
        add_shift_aliases(cfg, "FSR", {"FSR": "FSR_{direction}"})

    # add the shifts
    add_shifts(cfg)

    #
    # external files
    # taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L535C7-L579C84
    #

    # external files
    json_mirror = "/afs/cern.ch/user/j/jmatthie/public/mirrors/jsonpog-integration-49ddc547"
    local_repo = "/data/dust/user/matthiej/topsf"  # TODO: avoid hardcoding path

    if cfg.x.run == 3:
        corr_tag = f"{year}_Summer22{jerc_postfix}"

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{json_mirror}/POG/LUM/{corr_tag}/puWeights.json.gz", "v1"),

        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{corr_tag}/jet_jerc.json.gz", "v1"),

        # jet veto map
        "jet_veto_map": (f"{json_mirror}/POG/JME/{corr_tag}/jetvetomaps.json.gz", "v1"),

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
    # electron and met still missing
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
            "PSWeight",

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

            # generator particles
            # "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
            # "GenPart.pdgId",
            # "GenPart.*",

            # number of primary vertices
            "PV.npvs",

            # average number of pileup interactions
            "Pileup.nTrueInt",

            #
            # columns added during selection
            #

            # generator particle info
            "GenPartonTop.*",

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

    # event weights only present in certain datasets or configs
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
        # add PSWeight variations for all datasets but qcd
        if not dataset.has_tag("is_qcd"):
            dataset.x.event_weights["ISR"] = get_shifts("ISR")
            dataset.x.event_weights["FSR"] = get_shifts("FSR")

    # #
    # # versions
    # #
    # cfg.x.versions = {
    #     "tt_*": "test_v7",
    # }

    # # named references to actual versions to use for certain sets of tasks
    # main_ver = "v1"
    # cfg.x.named_versions = DotDict.wrap({
    #     "default": f"{main_ver}",
    #     "calibrate": f"{main_ver}",
    #     "select": f"{main_ver}",
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

    return cfg
