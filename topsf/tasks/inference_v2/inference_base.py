# coding: utf-8

import luigi
import law
import os

from columnflow.tasks.framework.base import Requirements, CommandTask
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference import CreateDatacards
from columnflow.util import dev_sandbox
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)

from topsf.tasks.inference_v2.mixins import FitMixin
from topsf.tasks.base import TopSFTask


class InferenceBaseTask(
    FitMixin,
    TopSFTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    CommandTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "combine_sandbox"))

    verbosity = luigi.IntParameter(
        default=2,
        significant=False,
        description="Verbosity of the combine tool",
    )

    print_help = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Show the help message of the combine tool",
    )

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["datacards"] = self.requires_from_branch()

        return reqs

    def requires(self):
        reqs = {
            "datacards": self.reqs.CreateDatacards.req(self),
        }
        return reqs

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
        run_message = f"running '{cmd}' ..."
        output.append(run_message)
        output.append("cwd: {}".format(kwargs.get("cwd", os.getcwd())))
        output.append("")
        with self.publish_step(law.util.colored(run_message, "cyan")):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            for line in lines:
                if echo:
                    print(line)
                output.append(line)

        output_str = "\n".join(output)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            raise Exception(f"command failed with exit code {p.returncode}: {cmd}")

        return p, output_str

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model

        return [
            {
                "categories": cats,
            },
        ]
