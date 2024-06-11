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


class ImpactsV2(
    CombineBaseTask,
):
    mass = luigi.IntParameter(
        significant=True,
        description="Mass point",
    )

    robust_fit = luigi.IntParameter(
        significant=False,
        description="Run a robust fit",
    )

    combine_parallel = luigi.IntParameter(
        significant=False,
        description="Run the fits in parallel",
    )

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data for the fit",
    )

    @property
    def mass_inst(self):
        return self.mass

    @property
    def do_initial_fit_inst(self):
        return self.do_initial_fit

    @property
    def robust_fit_inst(self):
        return self.robust_fit

    @property
    def do_fits_inst(self):
        return self.do_fits

    @property
    def combine_parallel_inst(self):
        return self.combine_parallel

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
        CreateWorkspace=CreateWorkspaceV2,
        GenToys=GenToysV2,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        reqs["gen_toys"] = self.reqs.GenToys.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "gen_toys": self.reqs.GenToys.req(self),
        }
        return reqs

    def output(self):
        output_dict = {
            "impacts_log": self.target("impacts.log"),
        }
        if self.mode_inst == "exp":
            output_dict["impacts_exp"] = self.target("impacts_exp.json")
        return output_dict

    @property
    def impacts_name(self):
        name = f"impacts__m{self.mass}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "impacts", self.impacts_name)
        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_workspace = self.input()["workspace"]["workspace"].path
        input_toys = self.input()["gen_toys"]["toy_file"].path
        if self.mode_inst == "exp":
            output_impact_file = self.output()["impacts_exp"].path
        output_dirname = os.path.dirname(output_impact_file) + "/"
        # touch output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        # do initial fit
        command_initial = "combineTool.py -M Impacts"
        command_initial += f" -m {self.mass}"
        command_initial += f" -d {input_workspace}"
        command_initial += " --doInitialFit"
        command_initial += f" --robustFit {self.robust_fit}"
        command_initial += " -t -1" if self.asimov_data else ""
        command_initial += f" --toysFile {input_toys}"
        self.publish_message("Running initial fit using command:")
        self.publish_message(command_initial)
        p_initial, output_initial = self.run_command(command_initial, echo=True, cwd=output_dirname)

        # calculate impacts for all nuisance
        command_impacts = "combineTool.py -M Impacts"
        command_impacts += f" -m {self.mass}"
        command_impacts += f" -d {input_workspace}"
        command_impacts += " --doFits"
        command_impacts += f" --robustFit {self.robust_fit}"
        command_impacts += " -t -1" if self.asimov_data else ""
        command_impacts += f" --toysFile {input_toys}"
        command_impacts += f" --parallel {self.combine_parallel}"
        self.publish_message("Running impacts using command:")
        self.publish_message(command_impacts)
        p_impacts, output_impacts = self.run_command(command_impacts, echo=True, cwd=output_dirname)

        # collecting ouput, converting to .json format
        self.output()["impacts_exp"].touch()
        command_collect = "combineTool.py -M Impacts"
        command_collect += f" -m {self.mass}"
        command_collect += f" -d {input_workspace}"
        command_collect += f" -o {output_impact_file}"
        self.publish_message("Collecting impacts using command:")
        self.publish_message(command_collect)
        p_collect, output_collect = self.run_command(command_collect, echo=True, cwd=output_dirname)

        # store all outputs in log file
        output = output_initial + output_impacts + output_collect
        self.output()["impacts_log"].dump("\n".join(output), formatter="text")
