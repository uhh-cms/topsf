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

    @property
    def postfit_inst(self):
        return self.postfit

    @property
    def sampling_inst(self):
        return self.sampling

    @property
    def print_fit_inst(self):
        return self.print_fit

    @property
    def total_shapes_inst(self):
        return self.total_shapes

    @property
    def covariance_inst(self):
        return self.covariance

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
        output_dict = {
            "pfsfw_log": self.target("pfsfw.log"),
        }
        if self.mode_inst == "exp":
            output_dict["pfsfw_exp"] = self.target("pfsfw_exp.root")
        return output_dict

    @property
    def pfsfw_name(self):
        name = f"pfswf__{self.mode_inst}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "pfsfw_name", self.pfsfw_name)

        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_workspace = self.input()["workspace"]["workspace"].path
        if self.mode_inst == "exp":
            input_fit_result = self.input()["fit_result"]["fit_exp_result"].path
            output_file = self.output()["pfsfw_exp"].path
        output_dirname = os.path.dirname(output_file) + "/"
        # touch output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        command_to_run = "PostFitShapesFromWorkspace"
        command_to_run += f" -w {input_workspace}"
        command_to_run += f" -o {output_file}"
        command_to_run += f" -f {input_fit_result}:fit_mdf"
        command_to_run += " --sampling" if self.sampling_inst else ""
        command_to_run += " --print" if self.print_fit_inst else ""
        command_to_run += " --total-shapes" if self.total_shapes_inst else ""
        command_to_run += " --covariance" if self.covariance_inst else ""
        command_to_run += " --postfit" if self.postfit_inst else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["pfsfw_log"].dump("\n".join(outp), formatter="text")
