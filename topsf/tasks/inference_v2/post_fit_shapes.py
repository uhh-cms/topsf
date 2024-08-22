# coding: utf-8

import luigi
import law
import os

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference import CreateDatacards

from topsf.tasks.inference_v2.combine_base import CombineBaseTask
from topsf.tasks.inference_v2.workspace import CreateWorkspaceV2
from topsf.tasks.inference_v2.gen_toys import GenToysV2
from topsf.tasks.inference_v2.multi_dim_fit import MultiDimFitV2


class PostFitShapesFromWorkspaceV2(
    CombineBaseTask,
):
    postfit = luigi.BoolParameter(
        significant=False,
        description="Create postfit shapes",
    )

    sampling = luigi.BoolParameter(
        significant=False,
        description="Sample the fit results",
    )

    print_fit = luigi.BoolParameter(
        significant=False,
        description="Print the fit results",
    )

    total_shapes = luigi.BoolParameter(
        significant=False,
        description="Create total shapes",
    )

    covariance = luigi.BoolParameter(
        significant=False,
        description="Create covariance matrix",
    )

    mode = luigi.ChoiceParameter(
        choices=["exp", "obs"],
        default="exp",
        significant=True,
        description="Mode of the combine tool",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
        CreateWorkspace=CreateWorkspaceV2,
        GenToys=GenToysV2,
        MultiDimFit=MultiDimFitV2,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        reqs["fit_result"] = self.reqs.MultiDimFit.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "fit_result": self.reqs.MultiDimFit.req(self),
        }
        return reqs

    def output(self):
        output_dict = {}
        output_dict[f"pfsfw_{self.mode}"] = self.target(f"pfsfw_{self.mode}.root")
        output_dict[f"pfsfw_{self.mode}_log"] = self.target(f"pfsfw_{self.mode}.log")
        return output_dict

    @property
    def pfsfw_name(self):
        name = f"pfswf__{self.mode}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "pfsfw_name", self.pfsfw_name)

        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_workspace = self.input()["workspace"]["workspace"].path
        input_fit_result = self.input()["fit_result"][f"fit_{self.mode}_result"].path

        output_file = self.output()[f"pfsfw_{self.mode}"].path
        output_dirname = os.path.dirname(output_file) + "/"
        # touch output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        command_to_run = "PostFitShapesFromWorkspace"
        command_to_run += f" -w {input_workspace}"
        command_to_run += f" --output {output_file}"
        command_to_run += f" -f {input_fit_result}:fit_mdf"
        command_to_run += " --sampling" if self.sampling else ""
        command_to_run += " --print" if self.print_fit else ""
        command_to_run += " --total-shapes" if self.total_shapes else ""
        command_to_run += " --covariance" if self.covariance else ""
        command_to_run += " --postfit" if self.postfit else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()[f"pfsfw_{self.mode}_log"].dump("\n".join(outp), formatter="text")
