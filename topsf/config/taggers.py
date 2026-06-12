# coding: utf-8

"""
Stores the taggers information for the top-tagging scale factor analysis.
"""
from __future__ import annotations
import law

from columnflow.util import DotDict

logger = law.logger.get_logger(__name__)


def btag_wps(
    config,
    full=True,
) -> DotDict:
    # b-tag working points
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    # TODO: use PNet? -> not available for SubJet tagging, only DeepCSV for v12, and UnifiedParT for v15
    btag_key = config.x.cpn_tag
    btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {
                "2022preEE": 0.0583, "2022postEE": 0.0614, "2023preBPix": 0.0479, "2023postBPix": 0.048, "2024": -10.0,
            }[btag_key],
            "medium": {
                "2022preEE": 0.3086, "2022postEE": 0.3196, "2023preBPix": 0.2431, "2023postBPix": 0.2435, "2024": -10.0,
            }[btag_key],
            "tight": {
                "2022preEE": 0.7183, "2022postEE": 0.7300, "2023preBPix": 0.6553, "2023postBPix": 0.6563, "2024": -10.0,
            }[btag_key],
        },
        "deepcsv": {
            "loose": {
                "2022preEE": 0.1208, "2022postEE": 0.1208, "2023preBPix": 0.1208, "2023postBPix": 0.1208, "2024": -10.0,
            }[btag_key],
            "medium": {
                "2022preEE": 0.4168, "2022postEE": 0.4168, "2023preBPix": 0.4168, "2023postBPix": 0.4168, "2024": -10.0,
            }[btag_key],
            "tight": {
                "2022preEE": 0.7665, "2022postEE": 0.7665, "2023preBPix": 0.7665, "2023postBPix": 0.7665, "2024": -10.0,
            }[btag_key],
        },
        "btagUParTAK4B": {
            "loose": {
                "2022preEE": -10.0, "2022postEE": -10.0, "2023preBPix": -10.0, "2023postBPix": -10.0, "2024": 0.0246
            }[btag_key],
            "medium": {
                "2022preEE": -10.0, "2022postEE": -10.0, "2023preBPix": -10.0, "2023postBPix": -10.0, "2024": 0.1272
            }[btag_key],
            "tight": {
                "2022preEE": -10.0, "2022postEE": -10.0, "2023preBPix": -10.0, "2023postBPix": -10.0, "2024": 0.4648
            }[btag_key],
            "xtight": {
                "2022preEE": -10.0, "2022postEE": -10.0, "2023preBPix": -10.0, "2023postBPix": -10.0, "2024": 0.6298
            }[btag_key],
            "xxtight": {
                "2022preEE": -10.0, "2022postEE": -10.0, "2023preBPix": -10.0, "2023postBPix": -10.0, "2024": 0.9739
            }[btag_key],
        },
    })
    # store upart wp different for fixed wp sf producer
    btagUParTAK4B__fixed_wp = DotDict.wrap({
        "loose": 0.0246,
        "medium": 0.1272,
        "tight": 0.4648,
        "xtight": 0.6298,
        # "xxtight": 0.9739,
    })
    result = btag_working_points if full else btagUParTAK4B__fixed_wp

    return result


def toptag_wps(era) -> DotDict:
    # top-tag working points
    toptag_working_points = DotDict.wrap({
        "tau32_run2": {
            # stored here for reference
            # https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
            "very_loose": 0.69,
            "loose": 0.61,
            "medium": 0.52,
            "tight": 0.47,
            "very_tight": 0.38,
        },
        "tau32_v7": {
            # v7, 2223 values
            # with mass constraint
            "very_loose": 0.761,
            "loose": 0.680,
            "medium": 0.579,
            "tight": 0.514,
            "very_tight": 0.395,
        },
        "tau32_v8_run3": {
            # with mass constraint
            "very_loose": 0.73,
            "loose": 0.63,
            "medium": 0.53,
            "tight": 0.47,
            "very_tight": 0.36,
        },
        "tau32_v11_run3": {
            # with mass constraint
            # [0.36944236, 0.47045506, 0.52725393, 0.62049409, 0.71313679]
            "very_loose": 0.71,
            "loose": 0.62,
            "medium": 0.53,
            "tight": 0.47,
            "very_tight": 0.37,
        }
    })
    # era-specific working points (v8)
    toptag_working_points_eras = DotDict.wrap({
        "2022preEE": {
            "tau32": {
                # 0.36660745, 0.47563951, 0.53662783, 0.63603016, 0.73238039
                "very_loose": 0.73,
                "loose": 0.64,
                "medium": 0.54,
                "tight": 0.48,
                "very_tight": 0.37,
            },
        },
        "2022postEE": {
            "tau32": {
                # 0.36733566, 0.47297406, 0.53398852, 0.63312435, 0.72942649
                "very_loose": 0.73,
                "loose": 0.63,
                "medium": 0.53,
                "tight": 0.47,
                "very_tight": 0.37,
            },
        },
        "2023preBPix": {
            "tau32": {
                # 0.36037594, 0.46878631, 0.5314992, 0.63098395, 0.72912963
                "very_loose": 0.73,
                "loose": 0.63,
                "medium": 0.53,
                "tight": 0.47,
                "very_tight": 0.36,
            },
        },
        "2023postBPix": {
            "tau32": {
                # 0.35929614, 0.47072125, 0.53208042, 0.63179965, 0.72908621
                "very_loose": 0.73,
                "loose": 0.63,
                "medium": 0.53,
                "tight": 0.47,
                "very_tight": 0.36,
            },
        },
        "2024": {
            "tau32": {
                # 0.37950085, 0.49627566, 0.55961938, 0.66063325, 0.75729332
                "very_loose": 0.76,
                "loose": 0.66,
                "medium": 0.56,
                "tight": 0.50,
                "very_tight": 0.38,
            },
        },
        "222324": {
            "tau32": {
                # 0.36372085, 0.47182575, 0.53341095, 0.63278459, 0.72981009
                "very_loose": 0.73,
                "loose": 0.63,
                "medium": 0.53,
                "tight": 0.47,
                "very_tight": 0.36,
            },
        },
    })
    # logger.warning_once("Reminder: As of v8, the topwp values deviate in between the eras. Consider using era-specific WPs if we want to be precise. -> update numbers when rehistogramming!")
    # return toptag_working_points_eras[era]
    logger.info_once("Reminder: As of v11, the topwp values are the same between the eras. Using the same WPs again for all Eras.")
    return toptag_working_points["tau32_v11_run3"]
