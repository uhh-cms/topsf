# coding: utf-8

"""
Stores the taggers information for the top-tagging scale factor analysis.
"""

from __future__ import annotations
from columnflow.util import DotDict


def btag_wps(
    config,
    full=True,
) -> DotDict:
    # b-tag working points
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    # TODO: use PNet?
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
        "xxtight": 0.9739,
    })
    result = btag_working_points if full else btagUParTAK4B__fixed_wp

    return result


def toptag_wps() -> DotDict:
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
        "tau32": {
            # with mass constraint
            "very_loose": 0.761,
            "loose": 0.680,
            "medium": 0.579,
            "tight": 0.514,
            "very_tight": 0.395,
        }
    })
    return toptag_working_points
