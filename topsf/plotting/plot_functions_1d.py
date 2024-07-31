# coding: utf-8

"""
Example plot functions for one-dimensional plots.
"""

from __future__ import annotations

from collections import OrderedDict
import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import (
    prepare_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density_to_hists,
)
from columnflow.plotting.plot_all import plot_all


hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def plot_variable_per_process(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = prepare_plot_config(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
    )

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    default_style_config["legend_cfg"]["fontsize"] = 12
    default_style_config["annotate_cfg"]["fontsize"] = 16
    default_style_config["annotate_cfg"]["xy"] = (0.5, 0.7)
    default_style_config["annotate_cfg"]["xycoords"] = "axes fraction"

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    skip_legend = kwargs.pop("skip_legend", False)
    fig, axs = plot_all(plot_config, style_config, skip_legend=True, **kwargs)
    ax = axs[0]

    # custom legend
    if not skip_legend:
        # resolve legend kwargs
        legend_kwargs = {
            "ncol": 3,
            "loc": "upper right",
        }
        legend_kwargs.update(style_config.get("legend_cfg", {}))

        # retrieve the legend handles and their labels
        handles, labels = ax.get_legend_handles_labels()

        # assume all `StepPatch` objects are part of MC stack
        in_stack = [
            isinstance(handle, mpl.patches.StepPatch)
            for handle in handles
        ]

        # reverse order of entries that are part of the stack
        if any(in_stack):
            def shuffle(entries, mask):
                entries = np.array(entries, dtype=object)
                entries[mask] = entries[mask][::-1]
                return list(entries)

            handles = shuffle(handles, in_stack)
            labels = shuffle(labels, in_stack)

        # make legend using ordered handles/labels
        ax.legend(handles, labels, **legend_kwargs)

    return fig, axs
