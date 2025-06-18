# coding: utf-8
"""
Custom plotting tasks.
"""

import luigi
import law
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
        default="topsf.plotting.plot_functions_1d.plot_variable_per_process",
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
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = {}

        message = f"plotting {self.branch_data.variable} in {category_inst.name}"
        message += " with pretty legend" if self.pretty_legend else ""
        with self.publish_step(message):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
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
                            hist.loc(p.id)
                            for p in sub_process_insts[process_inst]
                            if p.id in h.axes["process"]
                        ],
                        "category": [
                            hist.loc(c.id)
                            for c in leaf_category_insts
                            if c.id in h.axes["category"]
                        ],
                        "shift": [
                            hist.loc(s.id)
                            for s in plot_shifts
                            if s.id in h.axes["shift"]
                        ],
                    }]

                    # axis reductions
                    h = h[{"process": sum, "category": sum}]

                    # add the histogram
                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n" +
                    "  - requested variable requires columns that were missing during histogramming\n" +
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # sort hists by process order
            hists = OrderedDict(
                (process_inst.copy_shallow(), hists[process_inst])
                for process_inst in sorted(hists, key=process_insts.index)
            )

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
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
