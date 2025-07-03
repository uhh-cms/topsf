# coding: utf-8
"""
Custom tasks related to top-tagging ROC curves
for WP derivation.
"""

import luigi
import law
from collections import OrderedDict

from columnflow.util import dict_add_strict, maybe_import
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.plotting import PlotBase
from columnflow.tasks.framework.mixins import VariablesMixin
from columnflow.tasks.framework.parameters import MultiSettingsParameter
from columnflow.tasks.plotting import PlotVariablesBaseSingleShift

from topsf.tasks.base import TopSFTask

np = maybe_import("numpy")


class EfficiencyVariablesMixin(VariablesMixin):

    efficiency_variable = luigi.Parameter(
        description="variable in which to measure efficiency",
    )
    binning_variables = MultiSettingsParameter(
        default=(),
        description="map of variable names with 'min' and 'max' keys to indicate binning in those "
        "variables",
    )

    @classmethod
    def resolve_param_values_post_init(cls, params):
        """
        Resolve `signal_variable` and `binning_variables` and set `variables` param
        to be passed to dependent histogram task.
        """
        params = super().resolve_param_values_post_init(params)

        # no-op if config is not set
        if "config_insts" not in params:
            return params

        # required parameters not present, do nothing
        if "binning_variables" not in params or "efficiency_variable" not in params:
            return params

        # efficiency variables already resolved, do nothing
        if params.get("_eff_vars_resolved", False):
            return params

        # get configuration instance
        config_inst = params["config_insts"][0]

        # handle binning variables
        for var_name in list(params["binning_variables"]):
            # retrieve variable from config
            variable_inst = config_inst.get_variable(var_name)
            bv_dict = params["binning_variables"][var_name]
            bv_dict["variable_inst"] = variable_inst

            # find bin edges, rounding min/max down/up as needed
            min_val = bv_dict.get("min", -np.inf)
            max_val = bv_dict.get("max", np.inf)
            binning_inf = [-np.inf] + variable_inst.binning + [np.inf]
            min_bin_edge = binning_inf[::-1][np.digitize(min_val, binning_inf[::-1])]
            max_bin_edge = binning_inf[np.digitize(max_val, binning_inf, right=True)]
            bv_dict["_bin_edges"] = (min_bin_edge, max_bin_edge)

            # compute plot label from bin edges
            short_label = variable_inst.x("short_label", variable_inst.x_title)
            if all(np.isfinite(e) for e in bv_dict["_bin_edges"]):
                bv_dict["_plot_label"] = (
                    f"{min_bin_edge:g} < "
                    f"{short_label} < "
                    f"{max_bin_edge:g} {variable_inst.unit}"
                )
            elif np.isfinite(min_bin_edge):
                bv_dict["_plot_label"] = (
                    f"{short_label} > "
                    f"{min_bin_edge:g} {variable_inst.unit}"
                )
            elif np.isfinite(max_bin_edge):
                bv_dict["_plot_label"] = (
                    f"{short_label} < "
                    f"{max_bin_edge:g} {variable_inst.unit}"
                )

        # single multi-dimensional variable containing binning
        params["variables"] = law.util.make_tuple(
            cls.join_multi_variable(
                list(params["binning_variables"]) +
                [params["efficiency_variable"]],
            ),
        )
        params["_eff_vars_resolved"] = True

        return params

    @property
    def variables_repr(self):
        elems = [self.efficiency_variable]
        for bv, bv_dict in self.binning_variables.items():
            bin_edges_repr = "_".join(
                {np.inf: "max", -np.inf: "min"}.get(
                    bin_edge,
                    f"{bin_edge:g}".replace(".", "p"),
                )
                for bin_edge in bv_dict._bin_edges
            )

            elems.append(f"{bv}_{bin_edges_repr}")

        if len(elems) <= 2:
            return "_".join(elems)

        return f"{elems[0]}_{len(elems)-1}_{law.util.create_hash(sorted(elems))}"


class PlotEfficiencyBase(
    TopSFTask,
    EfficiencyVariablesMixin,
    PlotVariablesBaseSingleShift,
    PlotBase,
):
    """
    Calculate and plot the signal efficiency, background rejection (or ROC curve) resulting
    from a cut on discriminating variables.

    Accepts multiple *datasets*, which are classed into signal or background based on the
    ``is_signal`` attribute of the underlying :py:class:`order.Datasets` instance. The choice
    of datasets considered for the ROC curve measurement may further be restricted by setting
    the *processes* parameter.
    """

    # upstream requirements
    reqs = Requirements(
        PlotVariablesBaseSingleShift.reqs,
    )

    plot_function = PlotBase.plot_function.copy(
        default="topsf.plotting.plot_roc_curve.plot_efficiency",
        add_default_to_description=True,
    )
    plot_mode = None  # set in derived class

    def output(self):
        # add 'data' output containing ROC curve data points
        b = self.branch_data

        branch_repr = f"proc_{self.processes_repr}__cat_{b.category}__var_{self.variables_repr}"
        output = {
            "plots": [
                self.target(name)
                for name in self.get_plot_names(f"plot__{branch_repr}")
            ],
            "data": self.target(f"data__{branch_repr}.json"),
        }

        return output

    def get_hists_key(self, dataset_inst):
        return None

    def process_hists(self, hists):
        return hists

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # retrieve config object instances
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
        for i, config_inst in enumerate(self.config_insts):
            process_insts = [config_inst.get_process(p) for p in self.processes[i]]
        leaf_process_insts = {
            leaf_proc
            for proc in process_insts
            for leaf_proc in proc.get_leaf_processes()
        }

        # histogram data, summed up for background and
        # signal processes
        hists = {}

        with self.publish_step(f"plotting ROC curve for variable {self.branch_data.variable} in category {category_inst.name}"):  # noqa
            for i, config_inst in enumerate(self.config_insts):
                # histogram data per process
                hists_config = {}
                hists[config_inst] = hists_config
                for config, ds_inp in self.input().items():
                    if config_inst.name == config:
                        for dataset, inp in ds_inp.items():
                            dataset_inst = config_inst.get_dataset(dataset)
                            # skip when the dataset does not contain any leaf process
                            if not any(map(dataset_inst.has_process, leaf_process_insts)):
                                continue
                            h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                            # work on a copy
                            h = h_in.copy()

                            # axis selections
                            h = h[{
                                "process": [
                                    hist.loc(p.name)
                                    for p in leaf_process_insts
                                    if p.name in h.axes["process"]
                                ],
                                "category": [
                                    hist.loc(c.name)
                                    for c in leaf_category_insts
                                    if c.name in h.axes["category"]
                                ],
                                "shift": [
                                    hist.loc(s.name)
                                    for s in plot_shifts
                                    if s.name in h.axes["shift"]
                                ],
                            }]

                            # axis reductions
                            assert len(h.axes["shift"]) == 1, f"expected exactly one shift axis, got: {h.axes['shift']}"
                            h = h[{"process": sum, "category": sum, "shift": 0}]

                            # add the histogram
                            hists_key = self.get_hists_key(dataset_inst)
                            if hists_key in hists:
                                hists_config[hists_key] += h
                            else:
                                hists_config[hists_key] = h
                        for key in ['signal', 'background']:
                            try:
                                hists[config_inst][key] = hists_config[key]
                            except KeyError:
                                # if the key is not present, skip it
                                continue

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n" +
                    "  - requested variable requires columns that were missing during histogramming\n" +
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # merge configs if multiconfig
            if len(self.config_insts) != 1:
                plot_mode_memory = {}
                merged_hists = {}
                for _hists in hists.values():
                    for plot_mode, h in _hists.items():
                        if plot_mode in merged_hists:
                            merged_hists[plot_mode] += h
                        else:
                            merged_hists[plot_mode] = h
                            plot_mode_memory[plot_mode] = plot_mode

                plot_modes = list(plot_mode_memory.values())
                hists = {plot_mode_memory[plot_mode]: h for plot_mode, h in merged_hists.items()}
            else:
                hists = hists[self.config_inst]
                plot_modes = list(hists.keys())

            # apply binning variables and ranges to histograms,
            # keeping track of total values
            totals = {}
            for key in list(hists):
                # calculate totals to use as the denominator for the efficiency calculation
                # (for binning variables marked as 'flow', values outside the given range
                # will also count towards this)
                sel_totals = {
                    bv: slice(
                        hist.loc(bv_dict._bin_edges[0]),
                        hist.loc(bv_dict._bin_edges[1]),
                        # don't count under/overflow for totals unless requested
                        None if bv_dict.get("flow") else sum,
                    )
                    for bv, bv_dict in self.binning_variables.items()
                }
                totals[key] = hists[key][sel_totals].sum(flow=True).value

                # select and sum over the specified bin range for binning variables
                sel_hist = {
                    bv: slice(bv_slice.start, bv_slice.stop, sum)
                    for bv, bv_slice in sel_totals.items()
                }
                hists[key] = hists[key][sel_hist]

            # post-process histograms
            # FIXME: what does this do?
            hists = self.process_hists(hists)

            # temporarily use a merged luminostiy value, assigned to the first config
            config_inst = self.config_insts[0]
            lumi = sum([_config_inst.x.luminosity for _config_inst in self.config_insts])
            with law.util.patch_object(config_inst.x, "luminosity", lumi):
                # call the plot function
                fig, axs, data = self.call_plot_func(
                    self.plot_function,
                    hists=hists,
                    totals=totals,
                    config_inst=config_inst,
                    category_inst=category_inst.copy_shallow(),
                    **self.get_plot_parameters(),
                )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")

            # save the ROC curve data
            self.output()["data"].dump(data, formatter="json")

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()

        # retrieve plot label
        labels = [
            label
            for bv in self.binning_variables.values()
            if (label := bv.get("_plot_label", None))
        ]

        dict_add_strict(params, "binning_variable_labels", labels)
        variable_inst = self.config_inst.get_variable(self.efficiency_variable)
        dict_add_strict(params, "variable_inst", variable_inst.copy())
        dict_add_strict(params, "plot_mode", self.plot_mode)
        return params


class PlotEfficiency(
    PlotEfficiencyBase,
):
    """
    Calculate and plot the efficiency resulting from a cut on a discriminating variable.
    """

    efficiency_type = luigi.ChoiceParameter(
        description="signal efficiency or background rejection",
        choices=("signal", "background"),
    )

    @property
    def plot_mode(self):
        return self.efficiency_type

    def get_hists_key(self, dataset_inst):
        return self.efficiency_type


class PlotROCCurve(
    PlotEfficiencyBase,
):
    """
    Calculate and plot the ROC curve (background rejection vs. signal efficiency) resulting
    from a cut on discriminating variables.

    Accepts multiple *datasets*, which are classed into signal or background depending on the
    presence of a tag ``signal_tag`` of the underlying :py:class:`order.Datasets` instance. The
    choice of datasets considered for the ROC curve measurement may further be restricted by
    setting the *processes* parameter.
    """

    signal_tag = luigi.Parameter(
        description="datasets marked with this tag are considered signal, otherwise background",
    )

    def get_hists_key(self, dataset_inst):
        return (
            "signal"
            if dataset_inst.has_tag(self.signal_tag)
            else "background"
        )


class PlotROCCurveByVariable(
    PlotEfficiencyBase,
):
    """
    Calculate and plot the ROC curve (background rejection vs. signal efficiency) resulting
    from a cut on discriminating variables.

    An event is identified as a signal event based on the value of the variable ``signal_variable``.

    Accepts multiple *datasets*. The choice of datasets considered for the ROC curve measurement
    may further be restricted by setting the *processes* parameter.
    """

    signal_variable = luigi.Parameter(
        description="events where this variable is 0 (1) are considered background (signal)",
    )

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        params = super().resolve_param_values(params)

        # add `signal_variables` to variables for histograms
        if "variables" in params:
            if len(params["variables"]) != 1:
                raise ValueError(f"only 1 variable supported, got {len(params['variables'])}")
            discriminant_variable = params["variables"][0]
            signal_variable = params["signal_variable"]
            if not discriminant_variable.endswith(f"-{signal_variable}"):
                params["variables"] = law.util.make_tuple(f"{discriminant_variable}-{signal_variable}")

        return params

    def get_hists_key(self, dataset_inst):
        return "all"

    def process_hists(self, hists):
        """Split 'all' histogram into signal and background."""
        import hist
        hists_key = self.get_hists_key(None)
        h = hists[hists_key]
        return {
            "signal": h[{self.signal_variable: hist.loc(1)}],
            "background": h[{self.signal_variable: hist.loc(0)}],
        }
