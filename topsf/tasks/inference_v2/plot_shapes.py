# coding: utf-8

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow

from topsf.tasks.inference import CreateDatacards
from columnflow.util import dev_sandbox, DotDict
from topsf.tasks.inference_v2.combine_base import CombineBaseTask
from topsf.tasks.inference_v2.workspace import CreateWorkspaceV2
from topsf.tasks.inference_v2.gen_toys import GenToysV2
from topsf.tasks.inference_v2.post_fit_shapes import PostFitShapesFromWorkspaceV2


class PlotShapesV2(
    CombineBaseTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    file_types = law.CSVParameter(
        default="pdf",
        significant=False,
        description="File types to create",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
        CreateWorkspace=CreateWorkspaceV2,
        GenToys=GenToysV2,
        PostFitShapesFromWorkspace=PostFitShapesFromWorkspaceV2,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        if self.mode_inst == "exp":
            reqs["pfsfw_exp"] = self.reqs.PostFitShapesFromWorkspace.req(self)

        return reqs

    def requires(self):
        if self.mode_inst == "exp":
            reqs = {
                "pfsfw_exp": self.reqs.PostFitShapesFromWorkspace.req(self),
            }
        return reqs

    def create_branch_map(self):
        shapes_list = ["TotalBkg", "TotalProcs", "TotalSig", "data_obs"]
        map = []
        map_per_bin = [
            DotDict({"channel": channel, "shape": shape, "pt_bin": pt_bin, "year": year, "region": region})
            for channel in ["1e", "1m"]
            for shape in shapes_list
            for pt_bin in self.pt_bins_inst
            for year in self.years_inst
            for region in ["pass", "fail"]
        ]
        map += map_per_bin
        map_total = [
            DotDict({"total_shape": shape})
            for shape in ["TotalBkg", "TotalProcs", "TotalSig", "data_obs"]
        ]
        map += map_total
        return map

    def get_plot_names(self, name: str) -> list[str]:
        """
        Returns a list of basenames for created plots given a file *name* for all configured file
        types.
        """

        return [
            f"{name}.{ft}"
            for ft in self.file_types
        ]

    def output(self):
        output_dict = {}
        b = self.branch_data
        if self.mode_inst == "exp":
            output_dict["shapes_exp_plot_log"] = []
            output_dict["plots"] = []
            if hasattr(b, "channel"):
                output_dict["shapes_exp_plot_log"] += [
                    self.target(name)
                    for name in [f"shapes_exp_{b.channel}_{b.shape}_{b.pt_bin}_{b.year}_{b.region}.log"]
                ]
                output_dict["plots"] += [
                    self.target(name)
                    for name in self.get_plot_names(f"plot__shapes_{b.channel}_{b.shape}_{b.pt_bin}_{b.year}_{b.region}")  # noqa: E501
                ]
            elif hasattr(b, "total_shape"):
                output_dict["shapes_exp_plot_log"] += [
                    self.target(name)
                    for name in [f"shapes_exp_{b.total_shape}.log"]
                ]
                output_dict["plots"] += [
                    self.target(name)
                    for name in self.get_plot_names(f"plot__shapes_{b.total_shape}")
                ]
        return output_dict

    @property
    def plot_shapes_name(self):
        name = f"shape_plots_{self.mode_inst}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        pfsfw_name = self.reqs.PostFitShapesFromWorkspace.req(self).pfsfw_name
        parts.insert_after("fit_v2", "pfsfw_name", pfsfw_name)
        parts.insert_after("pfsfw_name", "plot_shapes", self.plot_shapes_name)

        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import uproot
        import mplhep as hep
        import matplotlib.pyplot as plt

        if self.mode_inst == "exp":
            b = self.branch_data
            input_shapes = self.input()["pfsfw_exp"]["pfsfw_exp"].path

            if hasattr(b, "channel"):
                key = f"{b.channel}_{b.shape}_{b.pt_bin}_{b.year}_{b.region}"
                shape_key = (
                    f"bin_{b.channel}__{b.year}__{b.pt_bin}__tau32_wp_{self.wp_name_inst}_{b.region}_prefit/{b.shape}",
                    f"bin_{b.channel}__{b.year}__{b.pt_bin}__tau32_wp_{self.wp_name_inst}_{b.region}_postfit/{b.shape}",
                )

                # load root file
                file = uproot.open(input_shapes)
                prefit = file[shape_key[0]].to_hist()
                postfit = file[shape_key[1]].to_hist()

                print(f"Running plot shapes for {key}...")

                fig, ax = plt.subplots()
                hep.histplot(prefit, ax=ax, label="prefit", color="blue")
                hep.histplot(postfit, ax=ax, label="postfit", color="red")
                ax.set_title(key)
                ax.legend()
                self.output()["plots"][0].dump(fig, formatter="mpl")
                message = f"Plotted pre- and postfit shapes for {key} in {self.output()['plots'][0].path}"

                self.output()["shapes_exp_plot_log"][0].dump(message, formatter="text")
            elif hasattr(b, "total_shape"):
                key = b.total_shape
                shape_key = (
                    f"prefit/{b.total_shape}",
                    f"postfit/{b.total_shape}",
                )

                # load root file
                file = uproot.open(input_shapes)
                prefit = file[shape_key[0]].to_hist()
                postfit = file[shape_key[1]].to_hist()

                print(f"Running plot shapes for {key}...")

                fig, ax = plt.subplots()
                hep.histplot(prefit, ax=ax, label="prefit", color="blue")
                hep.histplot(postfit, ax=ax, label="postfit", color="red")
                ax.set_title(key)
                ax.legend()
                self.output()["plots"][0].dump(fig, formatter="mpl")
                message = f"Plotted pre- and postfit shapes for {key} in {self.output()['plots'][0].path}"

                self.output()["shapes_exp_plot_log"][0].dump(message, formatter="text")
