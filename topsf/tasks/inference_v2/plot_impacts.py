# coding: utf-8

import luigi
import law
import os

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow

from topsf.tasks.inference_v2.inference_base import InferenceBaseTask
from topsf.tasks.inference_v2.impacts import ImpactsExpV2, ImpactsObsV2
from topsf.tasks.inference_v2.mixins import ModeMixin, ToysMixin


class PlotImpactsBaseV2(
    InferenceBaseTask,
    ModeMixin,
):
    per_page = luigi.IntParameter(
        default=30,
        significant=False,
        description="Number of parameters to show per page",
    )

    height = luigi.IntParameter(
        default=600,
        significant=False,
        description="Height of the canvas, in pixels",
    )

    left_margin = luigi.FloatParameter(
        default=0.4,
        significant=False,
        description="Left margin of the canvas, expressed as a fraction",
    )

    label_size = luigi.FloatParameter(
        default=0.021,
        significant=False,
        description="Parameter name label size",
    )

    sort = luigi.ChoiceParameter(
        choices=["impact", "constraint", "pull"],
        default="impact",
        significant=False,
        description="The metric to sort the list of parameters",
    )

    relative = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Show impacts relative to the uncertainty on the POI",
    )

    plot_name = luigi.Parameter(
        default="",
        significant=False,
        description="Name of the plot",
    )

    poi = law.CSVParameter(
        significant=False,
        description="POI to plot",
    )

    cms_label = luigi.Parameter(
        default="simpw",
        significant=False,
        description="CMS label",
    )

    summary = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Print summary of the impacts",
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs[f"impacts_{self.mode}"] = self.requires_from_branch()
        return reqs

    def requires(self):
        reqs = {
            f"impacts_{self.mode}": self.reqs.Impacts.req(self),
        }
        return reqs

    def output(self):
        output_dict = {}
        for i in range(0, len(self.poi)):
            output_dict[f"plot_impacts_{self.mode}__{self.poi[i]}"] = self.target(
                f"plot_impacts_{self.mode}__{self.poi[i]}.pdf",
            )
            output_dict[f"impacts_{self.mode}__{self.poi[i]}_log"] = self.target(
                f"plot_impacts_{self.mode}__{self.poi[i]}.log",
            )
            if self.summary:
                output_dict[f"impacts_{self.mode}__{self.poi[i]}_summary"] = self.target(
                    f"plot_impacts_{self.mode}__{self.poi[i]}_summary.pdf",
                )
        return output_dict

    @property
    def plot_impacts_name(self):
        name = f"plots__{self.mode}__{self.plot_name}" if len(self.plot_name) > 0 else f"plots__{self.mode}"  # noqa: E501
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        impacts_name = self.reqs.Impacts.req(self).impacts_name
        parts.insert_after("fit_v2", "impacts", impacts_name)
        parts.insert_after("impacts", "plot_impacts", self.plot_impacts_name)

        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_impacts = self.input()[f"impacts_{self.mode}"][f"impacts_{self.mode}"].path

        for i in range(0, len(self.poi)):
            output_file = self.output()[f"plot_impacts_{self.mode}__{self.poi[i]}"].path
            output_basename = os.path.basename(output_file).replace(".pdf", "")
            output_dirname = os.path.dirname(output_file) + "/"
            # touch output_dirname
            os.makedirs(output_dirname, exist_ok=True)
            command = "plotImpacts.py"
            command += f" -i {input_impacts}"
            command += f" --POI {self.poi[i]}"
            command += f" -o {output_basename}"
            command += " --summary" if self.summary else ""
            command += f" --cms-label '{str(self.cms_label)}'"
            command += f" --per-page {self.per_page}"
            command += f" --height {self.height}"
            command += f" --left-margin {self.left_margin}"
            command += f" --label-size {self.label_size}"
            command += f" --sort {self.sort}"
            command += " --relative" if self.relative else ""
            self.publish_message(f"Plot impacts for parameter of interest: {self.poi[i]}")
            p, output = self.run_command(command, echo=True, cwd=output_dirname)
            self.output()[f"impacts_{self.mode}__{self.poi[i]}_log"].dump(output, formatter="text")


class PlotImpactsExpV2(
    PlotImpactsBaseV2,
    ToysMixin,
):

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        Impacts=ImpactsExpV2,
    )


class PlotImpactsObsV2(
    PlotImpactsBaseV2,
):

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        Impacts=ImpactsObsV2,
    )
