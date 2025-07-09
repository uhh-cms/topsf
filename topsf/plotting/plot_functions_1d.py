# coding: utf-8

"""
Example plot functions for one-dimensional plots.
"""

from __future__ import annotations

from collections import OrderedDict
import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import (
    prepare_stack_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density,
)
from columnflow.plotting.plot_all import plot_all


hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def plot_variable_stack(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift] | None,
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
    hists, variable_style_config = apply_variable_settings(hists, variable_insts, variable_settings)
    hists, process_style_config = apply_process_settings(hists, process_settings)
    # TODO: check 'hists = apply_process_scaling(hists)' in new cf implementation
    # https://github.com/columnflow/columnflow/commit/ab1372be66040767d14426fe5ecce192e8b90f9e#diff-c42ce639168b3f538b75f7c78404ba4be180f525393b3ff14e9100f847c6cbadR64
    if density:
        hists = apply_density(hists, density)

    plot_config = prepare_stack_plot_config(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
        shift_insts=shift_insts,
        **kwargs,
    )

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    default_style_config["legend_cfg"]["fontsize"] = 12
    default_style_config["legend_cfg"]["labelspacing"] = 0.2
    default_style_config["annotate_cfg"]["fontsize"] = 16
    default_style_config["annotate_cfg"]["xy"] = (0.5, 0.65)
    default_style_config["annotate_cfg"]["xycoords"] = "axes fraction"
    default_style_config["rax_cfg"]["ylim"] = (0.3, 1.7)

    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        variable_style_config[variable_inst],
        style_config,
        deep=True,
    )
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = "Normalized entries"

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

        pretty_legend = kwargs.pop("pretty_legend")
        if pretty_legend:
            import matplotlib.patches as patches

            """
            Define the custom legend entries and their positions:
            X and Y positions are given in axes fraction
            ┌──────────────────────────────────────────────────────────────┐
            │                                                              │
            │── tt/st merge categories: ───────────────────────────────────│  Y[0]
            │   ┌─────┐   ┌─────┐                 ┌─────┐                  │
            │───│     │─/─│     │─── 'annot' ─────│     │─── 'annot' ──────│  Y[1]
            │   └─────┘   └─────┘                 └─────┘                  │
            │   :     :   :     :                    :                     │   :
            │   ┌─────┐   ┌─────┐                    ˌ                     │
            │───│     │─/─│     │─── 'annot' ─────   o   ─── 'annot' ──────│  Y[4]
            │   └─────┘   └─────┘                    ˈ                     │
            └──────────────────────────────────────────────────────────────┘
                X[1]      X[2]                    X[3]
                                <--->
                                 ANNOT_SEP
            """

            TOP_LABELS = ["fully merged (3q)", "semi-merged (2q)", "not merged (0q or 1q)", "background"]
            OTHER_LABELS = ["V+jets, VV", "Multijet", "MC stat. unc.", "Data"]
            X_POSITIONS = [0.05, 0.11, 0.55]  # of (colored) rectangles
            Y_POSITIONS = [0.9, 0.85, 0.8, 0.75, 0.7]  # of rows
            RECT_WIDTH = 0.04
            RECT_HEIGHT = 0.03
            ANNOT_SEP = 0.05  # annotation separation
            ANNOTATION_FONT_SIZE = 13

            def add_rectangle(ax, x_pos, y_pos, color, hatch=None):
                """ Adds a rectangle to the axis with the specified position and color """
                rect = patches.Rectangle(
                    (x_pos, y_pos),
                    RECT_WIDTH,
                    RECT_HEIGHT,
                    transform=ax.transAxes,
                    facecolor=color, hatch=hatch,
                )
                ax.add_patch(rect)

            def add_annotation(ax, x_pos, y_pos, text, font_size=ANNOTATION_FONT_SIZE):
                """ Adds an annotation to the axis at the specified position """
                ax.annotate(text, xy=(x_pos, y_pos), xycoords="axes fraction", fontsize=font_size)

            def handle_top_labels(ax, handles, indices, x_pos, y_pos, label):
                """ Handles the top like labels by adding rectangles and annotations in a first column """
                # get the labels of the top like entries
                tt_label = handles[indices[0]].get_label().split(",")[0]
                st_label = handles[indices[1]].get_label().split(",")[0]
                add_annotation(
                    ax,
                    X_POSITIONS[0],
                    Y_POSITIONS[0],
                    f"{tt_label} / {st_label} merge categories:",
                    font_size=ANNOTATION_FONT_SIZE + 1,
                )

                # add rectangles for the top like entries
                for i in range(2):
                    add_rectangle(
                        ax,
                        x_pos[i],
                        y_pos,
                        handles[indices[i]].get_facecolor(),
                    )

                # add slash ('/') between the two rectangles
                add_annotation(
                    ax,
                    0.097,
                    y_pos,
                    "/",
                )
                # add annotation for the label
                x_pos_annot = X_POSITIONS[1] + ANNOT_SEP
                add_annotation(
                    ax,
                    x_pos_annot,
                    y_pos,
                    label,
                )

            def handle_other_labels(ax, handles, indices, x_pos, y_pos, label):
                """ Handles the other like labels by adding rectangles and annotations in a second column """
                # try getting the color of the handle to use it for the rectangle
                try:
                    color = handles[indices[0]].get_facecolor()
                    add_rectangle(
                        ax,
                        x_pos,
                        y_pos,
                        color,
                    )
                    # add annotation at the same distance as in first column
                    x_pos_annot = X_POSITIONS[2] + ANNOT_SEP
                    add_annotation(
                        ax,
                        x_pos_annot,
                        y_pos,
                        label,
                    )
                # if the color cannot be retrieved, we need to handle the special cases
                except (AttributeError, IndexError):
                    if "MC stat. unc." in label:
                        add_rectangle(
                            ax,
                            x_pos,
                            y_pos,
                            "none",
                            hatch="//",
                        )
                        x_pos_annot = X_POSITIONS[2] + ANNOT_SEP
                        add_annotation(
                            ax,
                            x_pos_annot,
                            y_pos,
                            label,
                        )
                    elif "Data" in label:
                        handle_data_label(
                            ax,
                            x_pos,
                            y_pos,
                            label,
                        )
                    else:
                        raise NotImplementedError(f"Pretty legend is not implemented for label {label}")

            def handle_data_label(ax, x_pos, y_pos, label):
                """ Converts relative positions to plot coordinates and creates Data entry """
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_pos_transformed = x_pos * (x_max - x_min) + x_min
                y_pos_transformed = y_pos * (y_max - y_min) + y_min

                rect_width_transformed = RECT_WIDTH * (x_max - x_min)
                rect_height_transformed = RECT_HEIGHT * (y_max - y_min)

                # get the center of the rectangle to place the dot
                x_pos_data_point = x_pos_transformed + rect_width_transformed / 2
                y_pos_data_point = y_pos_transformed + rect_height_transformed / 2
                # get length of the vertical error bars
                yerr_data_point = rect_height_transformed / 2

                ax.errorbar(x_pos_data_point, y_pos_data_point,
                            yerr=yerr_data_point, fmt="o", color="black")

                x_pos_annot = X_POSITIONS[2] + ANNOT_SEP
                add_annotation(
                    ax,
                    x_pos_annot,
                    y_pos,
                    label,
                )

            def create_rectangle_and_annotation(ax, handles, indices, x_pos, y_pos, label, top_mode):
                """ Create rectangle and annotation for both top like and other legend entries """
                if top_mode:
                    handle_top_labels(
                        ax,
                        handles,
                        indices,
                        x_pos,
                        y_pos,
                        label,
                    )
                else:
                    handle_other_labels(
                        ax,
                        handles,
                        indices,
                        x_pos,
                        y_pos,
                        label,
                    )

            def plot_legend(ax, handles, labels):
                """ Main function to plot the custom legend based on the provided labels """
                if any("merged" in label for label in labels):
                    top_label_indices = {
                        label: [i for i, lbl in enumerate(labels) if label in lbl] for label in TOP_LABELS
                    }
                    other_label_indices = {
                        label: [i for i, lbl in enumerate(labels) if label in lbl] for label in OTHER_LABELS
                    }

                    for i, label in enumerate(TOP_LABELS):
                        create_rectangle_and_annotation(
                            ax,
                            handles,
                            top_label_indices[label],
                            [X_POSITIONS[0], X_POSITIONS[1]],
                            Y_POSITIONS[i + 1],
                            label,
                            True,
                        )

                    for i, label in enumerate(OTHER_LABELS):
                        create_rectangle_and_annotation(
                            ax,
                            handles,
                            other_label_indices[label],
                            X_POSITIONS[2],
                            Y_POSITIONS[i + 1],
                            label,
                            False,
                        )
                else:
                    raise NotImplementedError("Pretty legend is only implemented for merged samples")

                return fig, ax

            plot_legend(ax, handles, labels)

        else:
            # make legend using ordered handles/labels
            ax.legend(handles, labels, **legend_kwargs)

    return fig, axs
