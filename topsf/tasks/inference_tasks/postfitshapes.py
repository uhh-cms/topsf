# coding: utf-8

"""
# calculate prefit and postfit shapes: expected
    echo "________________________________________________________"
    echo "Calculating prefit and postfit shapes: expected"
    if [ ! -f "${out_dir}/PrePostFitShapes_exp.root" ]; then
        PostFitShapesFromWorkspace \
            -w "${out_dir}/Workspace.root" \
            -o "${out_dir}/PrePostFitShapes_exp.root" \
            -f multidimfit_exp.root:fit_mdf \
            --postfit \
            --sampling \
            --print \
            --total-shapes \
            --covariance || return $?
    fi
    echo "Done."
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
from columnflow.util import dev_sandbox

from topsf.tasks.base import TopSFTask


class PostFitShapesFromWorkspace(
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

    postfit = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Create post-fit histograms in addition to pre-fit",
    )

    sampling = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use the cov. matrix sampling method for the post-fit uncertainty (deprecated!)",
    )

    print = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Print tables of background shifts and relative uncertainties",
    )

    total_shapes = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save signal- and background shapes added for all channels/categories",
    )

    covariance = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save the covariance and correlation matrices of the process yields",
    )

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateWorkspace=CreateWorkspace,
        MultiDimFit=MultiDimFit,
        GenToys=GenToys,
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

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model
        return [
            {
                "categories": cats,
            },
        ]

    def output(self):
        return {
            "PrePostFitShapes_exp": self.target("PrePostFitShapes_exp.root"),
            "pfsfw_log": self.target("pfsfw.log"),
        }

    @property
    def pfsfw_name(self):
        return "PostFitShapesFromWorkspace"

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        workspace_name = self.reqs.CreateWorkspace.req(self).workspace_name
        gentoys_name = self.reqs.GenToys.req(self).gentoys_name
        mdf_name = self.reqs.MultiDimFit.req(self).mdf_name

        parts.insert_after("inf_model", "workspace", workspace_name)
        parts.insert_after("workspace", "gentoys", gentoys_name)
        parts.insert_after("gentoys", "mdf", mdf_name)
        parts.insert_after("mdf", "pfsfw", self.pfsfw_name)

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

        input_workspace = self.input()["workspace"]["workspace"].path
        input_fit_result = self.input()["fit_result"]["fit_exp_result"].path

        options = {
            "postfit": "--postfit" if self.postfit else "",
            "sampling": "--sampling" if self.sampling else "",
            "print": "--print" if self.print else "",
            "total_shapes": "--total-shapes" if self.total_shapes else "",
            "covariance": "--covariance" if self.covariance else "",
        }

        command_to_run = (
            f"PostFitShapesFromWorkspace -w {input_workspace} "
            f"-o {self.output()['PrePostFitShapes_exp'].path} "
            f"-f {input_fit_result}:fit_mdf {' '.join(options.values())}"
        )
        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["pfsfw_log"].dump("\n".join(outp), formatter="text")
