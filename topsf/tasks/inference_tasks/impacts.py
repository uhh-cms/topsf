# coding: utf-8

"""
initial_fit_file="higgsCombine_initialFit_Test.MultiDimFit.mH0.root"
    if [ ! -f "${out_dir}/${initial_fit_file}" ]; then
        combineTool.py -M Impacts \
            -m 0 \
            -d "${out_dir}/Workspace.root" \
            --doInitialFit \
            --robustFit 1 \
            -t -1 \
            --toysFile "${out_dir}/${toy_file}"
    fi

    lumiUncorrelated17_file="higgsCombine_paramFit_Test_lumiUncorrelated17.MultiDimFit.mH0.root"
    if [ ! -f "${out_dir}/${lumiUncorrelated17_file}" ]; then
        combineTool.py -M Impacts \
            -m 0 \
            -d "${out_dir}/Workspace.root" \
            --doFits \
            --robustFit 1 \
            -t -1 \
            --toysFile "${out_dir}/${toy_file}" \
            --parallel 12
    fi

    if [ ! -f "${out_dir}/impacts_exp.json" ]; then
        combineTool.py -M Impacts \
            -m 0 \
            -d "${out_dir}/Workspace.root" \
            -o "${out_dir}/impacts_exp.json"
    fi
    echo "Done."

    for poi in SF__FullyMerged__UL17__pt_300to400; do
        if [ ! -f "plots/impacts_exp_${poi}.pdf" ]; then
            echo "Plotting impacts for parameter: ${poi}"
            plotImpacts.py \
                -i ${out_dir}/impacts_exp.json \
                --POI ${poi} \
                --translate ${out_dir}/translate.json \
                -o plots/impacts_exp_${poi}
            echo "Done with ${poi}."
        fi
    done
"""

import luigi
import law
import os

from columnflow.tasks.framework.base import Requirements, CommandTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference_tasks.create_workspace import CreateWorkspace
from topsf.tasks.inference_tasks.combine_task import MultiDimFit, GenToys
from topsf.tasks.inference_tasks.postfitshapes import PostFitShapesFromWorkspace
from columnflow.util import dev_sandbox

from topsf.tasks.base import TopSFTask


class Impacts(
    TopSFTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    CommandTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "combine_sandbox"))

    # FIXME add descriptions of combine specific parameters

    combine_help = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Print help message for combine.",
    )

    verbosity = luigi.IntParameter(
        default=0,
        significant=False,
        description="Verbosity level for combine.",
    )

    mass = luigi.IntParameter(
        default=120,
        significant=False,
        description="Higgs mass to store in the output tree",
    )

    do_initial_fit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Perform initial fit.",
    )

    robust_fit = luigi.IntParameter(
        default=1,
        significant=False,
        description="Perform a robust fit.",
    )

    asimov_data = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Use Asimov data.",
    )

    n_toys = luigi.IntParameter(
        default=-1,
        significant=False,
        description="Number of toys to use.",
    )

    do_fits = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Perform fits.",
    )

    combine_parallel = luigi.IntParameter(
        default=1,
        significant=False,
        description="Number of parallel fits.",
    )

    store_impact_fits = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Store impact fits.",
    )

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateWorkspace=CreateWorkspace,
        MultiDimFit=MultiDimFit,
        GenToys=GenToys,
        PostFitShapesFromWorkspace=PostFitShapesFromWorkspace,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        reqs["gentoys"] = self.reqs.GenToys.req(self)
        reqs["fit_result"] = self.reqs.MultiDimFit.req(self)
        reqs["PrePostFitShapes_exp"] = self.reqs.PostFitShapesFromWorkspace.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "gentoys": self.reqs.GenToys.req(self),
            "fit_result": self.reqs.MultiDimFit.req(self),
            "PrePostFitShapes_exp": self.reqs.PostFitShapesFromWorkspace.req(self),
        }
        return reqs

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model
        return [
            {
                "categories": cats,
            },
        ]

    def output(self):
        output_dict = {"impacts_log": self.target("impacts.log")}
        if not self.store_impact_fits:
            output_dict["impacts_exp"] = self.target("impacts_exp.json")
        else:
            raise NotImplementedError("Storing impact fits is not yet implemented.")
        return output_dict

    @property
    def impacts_name(self):
        name = f"Impacts_{'initialFit_Test' if self.do_initial_fit else ''}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        workspace_name = self.reqs.CreateWorkspace.req(self).workspace_name
        gentoys_name = self.reqs.GenToys.req(self).gentoys_name
        mdf_name = self.reqs.MultiDimFit.req(self).mdf_name
        pfsfw_name = self.reqs.PostFitShapesFromWorkspace.req(self).pfsfw_name

        parts.insert_after("inf_model", "workspace", workspace_name)
        parts.insert_after("workspace", "gentoys", gentoys_name)
        parts.insert_after("gentoys", "mdf", mdf_name)
        parts.insert_after("mdf", "pfsfw", pfsfw_name)
        parts.insert_after("pfsfw", "impacts", self.impacts_name)

        return parts

    def run_command(self, cmd, optional=False, echo=False, **kwargs):
        # proper command encoding
        cmd = (law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd).strip()

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path
        self.publish_message("cwd: {}".format(kwargs.get("cwd", os.getcwd())))

        # call it
        output = []
        with self.publish_step("running '{}' ...".format(law.util.colored(cmd, "cyan"))):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            # q = __import__("functools").partial(__import__("os")._exit, 0)
            # __import__("IPython").embed()
            for line in lines:
                if echo:
                    print(line)
                output.append(line)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            raise Exception(f"command failed with exit code {p.returncode}: {cmd}")

        return p, output

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        if not self.store_impact_fits:
            self.publish_message("Storing impact fits is not yet implemented, proceeding with temporary storage.")
            input_workspace = self.input()["workspace"]["workspace"].path
            input_toys = self.input()["gentoys"]["toy_file"].path
            output_impact_file = self.output()["impacts_exp"].path

            # do initial fit
            command_inital = f"combineTool.py -M Impacts -d {input_workspace} --toysFile {input_toys}"
            command_inital += f" -m {self.mass}"
            command_inital += " --doInitialFit"
            command_inital += f" --robustFit {self.robust_fit}"
            command_inital += " -t -1" if self.asimov_data else ""
            self.publish_message("Running initial fit using command:")
            self.publish_message(command_inital)
            p_initial, output_initial = self.run_command(command_inital, echo=True)

            # calculate impacts for all nuisance
            command_impacts = f"combineTool.py -M Impacts -d {input_workspace} --toysFile {input_toys}"
            command_impacts += f" -m {self.mass}"
            command_impacts += " --doFits"
            command_impacts += f" --robustFit {self.robust_fit}"
            command_impacts += " -t -1" if self.asimov_data else ""
            command_impacts += " --parallel 12"
            self.publish_message("Running impacts using command:")
            self.publish_message(command_impacts)
            p_impacts, output_impacts = self.run_command(command_impacts, echo=True)

            # collecting ouput, converting to .json format
            self.output()["impacts_exp"].touch()
            command_collect = f"combineTool.py -M Impacts -d {input_workspace} -o {output_impact_file}"
            command_collect += f" -m {self.mass}"
            self.publish_message("Collecting impacts using command:")
            self.publish_message(command_collect)
            p_collect, output_collect = self.run_command(command_collect, echo=True)

            # remove every .root file in the current directory
            for file in os.listdir("."):
                if file.endswith(".root"):
                    os.remove(file)

            # store all outputs in log file
            output = output_initial + output_impacts + output_collect
            self.output()["impacts_log"].dump("\n".join(output), formatter="text")

        else:
            self.publish_message("Storing impact fits is not yet implemented.")
            raise NotImplementedError("Storing impact fits is not yet implemented.")  # FIXME use cwd in run_command


class PlotImpacts(
    TopSFTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    CommandTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "combine_sandbox"))

    poi = law.CSVParameter(
        default=[],
        significant=False,
        description="Parameter of interest to plot impacts for.",
    )

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateWorkspace=CreateWorkspace,
        MultiDimFit=MultiDimFit,
        GenToys=GenToys,
        PostFitShapesFromWorkspace=PostFitShapesFromWorkspace,
        Impacts=Impacts,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        reqs["gentoys"] = self.reqs.GenToys.req(self)
        reqs["fit_result"] = self.reqs.MultiDimFit.req(self)
        reqs["PrePostFitShapes_exp"] = self.reqs.PostFitShapesFromWorkspace.req(self)
        reqs["impacts_exp"] = self.reqs.Impacts.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "gentoys": self.reqs.GenToys.req(self),
            "fit_result": self.reqs.MultiDimFit.req(self),
            "PrePostFitShapes_exp": self.reqs.PostFitShapesFromWorkspace.req(self),
            "impacts_exp": self.reqs.Impacts.req(self),
        }
        return reqs

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)
        return [
            {
                "categories": cats,
            },
        ]

    def output(self):
        output_dict = {}
        for i in range(0, len(self.poi)):
            output_dict[f"plot_impacts_{self.poi[i]}"] = self.target(f"impacts_exp_{self.poi[i]}.pdf")
            output_dict[f"impacts_{self.poi[i]}_log"] = self.target(f"plot_impacts_{self.poi[i]}.log")
        return output_dict

    @property
    def plot_impacts_name(self):
        name = "PlotImpacts"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        workspace_name = self.reqs.CreateWorkspace.req(self).workspace_name
        gentoys_name = self.reqs.GenToys.req(self).gentoys_name
        mdf_name = self.reqs.MultiDimFit.req(self).mdf_name
        pfsfw_name = self.reqs.PostFitShapesFromWorkspace.req(self).pfsfw_name
        impacts_name = self.reqs.Impacts.req(self).impacts_name

        parts.insert_after("inf_model", "workspace", workspace_name)
        parts.insert_after("workspace", "gentoys", gentoys_name)
        parts.insert_after("gentoys", "mdf", mdf_name)
        parts.insert_after("mdf", "pfsfw", pfsfw_name)
        parts.insert_after("pfsfw", "impacts", impacts_name)
        parts.insert_after("impacts", "plot_impacts", self.plot_impacts_name)

        return parts

    def run_command(self, cmd, optional=False, echo=False, **kwargs):
        # proper command encoding
        cmd = (law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd).strip()

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path
        self.publish_message("cwd: {}".format(kwargs.get("cwd", os.getcwd())))

        # call it
        output = []
        with self.publish_step("running '{}' ...".format(law.util.colored(cmd, "cyan"))):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            # q = __import__("functools").partial(__import__("os")._exit, 0)
            # __import__("IPython").embed()
            for line in lines:
                if echo:
                    print(line)
                output.append(line)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            raise Exception(f"command failed with exit code {p.returncode}: {cmd}")

        return p, output

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_impacts = self.input()["impacts_exp"]["impacts_exp"].path

        for i in range(0, len(self.poi)):
            output_file = self.output()[f"plot_impacts_{self.poi[i]}"].path
            output_basename = os.path.basename(output_file).strip(".pdf")
            print(output_basename)
            print(os.path.dirname(output_file) + "/")
            output_dirname = os.path.dirname(output_file) + "/"
            # touch output_dirname
            if not os.path.exists(output_dirname):
                os.makedirs(output_dirname)
            command = f"plotImpacts.py -i {input_impacts} --POI {self.poi[i]} -o {output_basename}"
            self.publish_message(f"Running plot impacts for parameter of interest: {self.poi[i]}")
            self.publish_message(f"Using command: {command}")
            p, output = self.run_command(command, echo=True, cwd=output_dirname)
            self.output()[f"impacts_{self.poi[i]}_log"].dump("\n".join(output), formatter="text")
            self.publish_message(f"Done with {self.poi[i]}.")
