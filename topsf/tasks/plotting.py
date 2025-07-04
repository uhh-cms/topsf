# coding: utf-8
"""
Custom plotting tasks.
"""

import luigi
import law
import order as od
from collections import OrderedDict

import columnflow.tasks.plotting
from columnflow.tasks.framework.decorators import view_output_plots
# from columnflow.util import dict_add_strict

from topsf.tasks.base import TopSFTask


class PlotVariables1D(
    columnflow.tasks.plotting.PlotVariables1D,
    TopSFTask,
):
    plot_function = columnflow.tasks.plotting.PlotVariables1D.plot_function.copy(
        default="topsf.plotting.plot_functions_1d.plot_variable_stack",
        add_default_to_description=True,
    )

    pretty_legend = luigi.BoolParameter(
        default=False,
        description="Use pretty legend",
    )

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]

        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

        for i, config_inst in enumerate(self.config_insts):
            process_insts = [config_inst.get_process(p) for p in self.processes[i]]
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }
        # get assignment of processes to datasets and shifts
        config_process_map, process_shift_map = self.get_config_process_map()

        message = f"plotting {self.branch_data.variable} in {category_inst.name}"
        message += " with pretty legend" if self.pretty_legend else ""
        # following implementation in columnflow:
        # https://github.com/columnflow/columnflow/blob/master/columnflow/tasks/plotting.py#L165
        hists: dict[od.Config, dict[od.Process, hist.Hist]] = {}
        with self.publish_step(message):
            for i, config_inst in enumerate(self.config_insts):
                # histogram data per process
                hists_config = {}
                for config, ds_inp in self.input().items():
                    if config_inst.name == config:
                        for dataset, inp in ds_inp.items():
                            dataset_inst = config_inst.get_dataset(dataset)
                            h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                            # loop and extract one histogram per process
                            for process_inst in process_insts:
                                # skip when the dataset is already known to not contain any sub process
                                if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                                    continue

                                # work on a copy
                                h = h_in.copy()

                                # axis selections
                                h = h[{
                                    "process": [
                                        hist.loc(p.name)
                                        for p in sub_process_insts[process_inst]
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
                                    ]
                                }]

                                # axis reductions
                                h = h[{"process": sum, "category": sum}]

                                # add the histogram
                                if process_inst in hists_config:
                                    hists_config[process_inst] += h
                                else:
                                    hists_config[process_inst] = h
                        hists[config_inst] = {
                            proc_inst: hists_config[proc_inst]
                            for proc_inst in sorted(
                                hists_config.keys(), key=list(config_process_map[config_inst].keys()).index,
                            )
                        }

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n" +
                    "  - requested variable requires columns that were missing during histogramming\n" +
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # merge configs if multiconfig
            if len(self.config_insts) != 1:
                process_memory = {}
                merged_hists = {}
                for _hists in hists.values():
                    for process_inst, h in _hists.items():
                        if process_inst.id in merged_hists:
                            merged_hists[process_inst.id] += h
                        else:
                            merged_hists[process_inst.id] = h
                            process_memory[process_inst.id] = process_inst

                process_insts = list(process_memory.values())
                hists = {process_memory[process_id]: h for process_id, h in merged_hists.items()}
            else:
                hists = hists[self.config_inst]
                process_insts = list(hists.keys())

            # sort hists by process order
            hists = OrderedDict(
                (process_inst.copy_shallow(), hists[process_inst])
                for process_inst in sorted(hists, key=process_insts.index)
            )

            # temporarily use a merged luminostiy value, assigned to the first config
            config_inst = self.config_insts[0]
            lumi = sum([_config_inst.x.luminosity for _config_inst in self.config_insts])
            with law.util.patch_object(config_inst.x, "luminosity", lumi):
                # call the plot function
                fig, _ = self.call_plot_func(
                    self.plot_function,
                    hists=hists,
                    config_inst=config_inst,
                    category_inst=category_inst.copy_shallow(),
                    variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                    shift_insts=plot_shifts,
                    pretty_legend=self.pretty_legend,
                    **self.get_plot_parameters(),
                )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")

        def get_plot_parameters(self):
            # convert parameters to usable values during plotting
            params = super().get_plot_parameters()
            # dict_add_strict(params, "foo", "bar")
            return params
