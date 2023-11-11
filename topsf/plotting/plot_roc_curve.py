# coding: utf-8

"""
Plot function for drawing ROC curve
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import remove_residual_axis

from topsf.plotting.plot_all import plot_all


hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def plot_roc_curve(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_inst: od.Variable,
    style_config: dict | None = None,
    # hide_errors: bool | None = None,
    # variable_settings: dict | None = None,
    binning_variable_labels: list | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Given a dictionary of *hists* with identical binning, compute the corresponding ROC curve
    showing the background rejection as a function of the signal efficiency.

    *hists* should contain the keys *signal* and *background*, which should correspond to the
    distribution of the discriminating variable for signal and background events, respectively.
    """

    # remove residual `shift` axis
    remove_residual_axis(hists, "shift")

    if "signal" not in hists or "background" not in hists:
        hists_keys_str = ", ".join(hists)
        print(
            f"WARNING: `hists` should contain the keys 'signal' and 'background', got: {hists_keys_str}"
        )

    # plot config with a single entry for drawing the ROC curve
    plot_config = {
        "roc_curve": {
            "method": "draw_efficiency",
            "hist": hists,
            "kwargs": {
                "plot_mode": "roc",
            },
        },
    }

    # style config for setting axis ranges, labels, etc.
    default_style_config = {
        "ax_cfg": {
            "xlim": (0.0, 1),
            #"ylim": (0.0, 1),
            "ylim": (1e-4, 1),
            #"xlim": (0.5, 1),
            #"ylim": (0.5, 1),
            #"xlim": (0.8, 1),
            #"ylim": (0.8, 1),
            "xlabel": "Signal efficiency ($\\varepsilon$)",
            "ylabel": "1 $-$ background rejection",
            "xscale": "linear",
            #"yscale": "linear",
            #"xscale": "log",
            "yscale": "log",
        },
        "legend_cfg": {},
        "annotate_cfg": {
            "text": "\n".join(
                [category_inst.label] +
                (binning_variable_labels or [])
            ),
        },
        "cms_label_cfg": {
            # "lumi": config_inst.x.luminosity.get("nominal") / 1000,  # pb -> fb
        },
    }
    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(
        plot_config,
        style_config,
        skip_ratio=True,
        **kwargs,
    )


def plot_efficiency(
    hists: OrderedDict,
    totals: dict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_inst: od.Variable,
    plot_mode: str | None = "signal",  # 'signal', 'background' or 'roc'
    style_config: dict | None = None,
    # hide_errors: bool | None = None,
    # variable_settings: dict | None = None,
    binning_variable_labels: list | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Given a dictionary of *hists* with identical binning, compute and plot the background
    rejection / signal efficiency as a function of the discriminanting variable.

    *hists* should contain the keys *signal* and *background*, which should correspond to the
    distribution of the discriminating variable for signal and background events, respectively.
    """
    assert plot_mode in ("signal", "background")

    # remove residual `shift` axis
    remove_residual_axis(hists, "shift")

    if plot_mode == "roc" and (
        "signal" not in hists and
        "background" not in hists
    ):
        hists_keys_str = ", ".join(hists)
        print(
            f"WARNING: `hists` should contain the keys 'signal' and 'background', got: {hists_keys_str}"
        )
    elif plot_mode not in hists:
        hists_keys_str = ", ".join(hists)
        print(
            f"WARNING: `hists` should contain the key '{plot_mode}', got: {hists_keys_str}"
        )

    # plot config with a single entry for drawing the ROC curve
    plot_config = {
        "roc_curve": {
            "method": "draw_efficiency",
            "hist": hists,
            "kwargs": {
                "totals": totals,
                "plot_mode": plot_mode,
            },
        },
    }

    labels = {
        "signal": "Signal efficiency ($\\varepsilon_{S}$)",
        "background": "1 $-$ background rejection ($\\varepsilon_{B}$)",
        "variable": variable_inst.x_title,
    }

    # style config for setting axis ranges, labels, etc.
    default_style_config = {
        "ax_cfg": {
            "xlim": (0.0, 1) if plot_mode == "roc" else None,
            "ylim": (1e-4, 1),
            "xlabel": labels["signal"] if plot_mode == "roc" else labels["variable"],
            "ylabel": labels["background"] if plot_mode == "roc" else labels[plot_mode],
            "xscale": "linear",
            "yscale": "log",
        },
        "legend_cfg": {},
        "annotate_cfg": {
            "text": "\n".join(
                #[category_inst.label] +
                (binning_variable_labels or [])
            ),
        },
        "cms_label_cfg": {
            # "lumi": config_inst.x.luminosity.get("nominal") / 1000,  # pb -> fb
        },
    }
    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(
        plot_config,
        style_config,
        skip_ratio=True,
        **kwargs,
    )
