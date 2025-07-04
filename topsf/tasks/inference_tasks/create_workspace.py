# coding: utf-8

"""
    # first, create workspace from datacards (also reads in templates)
    echo "________________________________________________________"
    echo "Creating workspace"
    if [ ! -f "${out_dir}/Workspace.root" ]; then
        text2workspace.py \
            "${out_dir}/DataCard.dat" \
            -o "${out_dir}/Workspace.root" \
            -P topsf.inference.combine_physics_model:topsf_model \
            --PO sf_naming_scheme=SF__{msc}__{year}__{pt_bin} \
            --PO sf_range=[1,0.2,2.0] \
            --PO merge_scenarios=FullyMerged:TagAndProbe,SemiMerged:ThetaLike,NotMerged:ThetaLike \
            --PO years=${year} \
            --PO pt_bins=${pt_bin} || return $?
    fi
    echo "Done."
"""

import luigi
import law
import os

from columnflow.tasks.framework.base import Requirements, CommandTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference import CreateDatacards
from columnflow.util import dev_sandbox

from topsf.tasks.base import TopSFTask


class CreateWorkspace(
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

    physics_model = luigi.Parameter(
        default="default",
        significant=True,
        description="Name of the physics model to use.",
    )

    sf_naming_scheme = luigi.Parameter(
        default="SF__{msc}__{year}__{pt_bin}",
        significant=True,
        description="Naming scheme for scale factors.",
    )

    wp_names = law.CSVParameter(
        default="tau32_wp_very_tight",
        significant=True,
        description="Working point name.",
    )

    sf_range = luigi.Parameter(
        default="[1,0.2,2.0]",
        significant=True,
        description="Range for scale factors.",
    )

    merge_scenarios = luigi.Parameter(
        default="3q:TagAndProbe,2q:ThetaLike,0o1q:ThetaLike",
        significant=True,
        description="List of merge scenarios.",
    )

    year = law.CSVParameter(
        default="UL17",
        significant=True,
        description="Year of data taking.",
    )

    pt_bins = law.CSVParameter(
        default="pt_300_400",
        significant=True,
        description="Pt bins.",
    )

    verbosity = luigi.IntParameter(
        default=0,
        significant=False,
        description="Verbosity level for combine.",
    )

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model
        # if self.per_category:
        #     return [
        #         {
        #             "categories": [cat],
        #         }
        #         for cat in cats
        #     ]
        # else:
        #     return [
        #         {
        #             "categories": cats,
        #         },
        #     ]
        return [
            {
                "categories": cats,
            },
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["datacards"] = self.reqs.CreateDatacards.req(self)

        return reqs

    def requires(self):
        reqs = {
            "datacards": self.reqs.CreateDatacards.req(self),
        }
        return reqs

    @property
    def workspace_name(self):
        if self.physics_model:
            p_m_part = f"{law.util.create_hash(self.physics_model)}"
        if self.year:
            if len(self.year) < 2:
                year_part = "__".join(self.year)
            else:
                year_part = f"{len(self.year)}__{law.util.create_hash(sorted(self.year))}"
        if self.pt_bins:
            if len(self.pt_bins) < 2:
                pt_bin_part = "__".join(self.pt_bins)
            else:
                pt_bin_part = f"{len(self.pt_bins)}__{law.util.create_hash(sorted(self.pt_bins))}"
        if self.wp_names:
            if len(self.wp_names) < 2:
                wp_names_part = "__".join(self.wp_names)
            else:
                wp_names_part = f"{len(self.wp_names)}__{law.util.create_hash(sorted(self.wp_names))}"
        # add merge scenarios (e.g. 3q_2q_0o1q from 3q:TagAndProbe,2q:ThetaLike,0o1q:ThetaLike or 2q_0o1q from 2q:ThetaLike,0o1q:ThetaLike)  # noqa: E501
        if self.merge_scenarios:
            merge_scenarios = self.merge_scenarios.split(",")
            merge_scenarios = [msc.split(":")[0] for msc in merge_scenarios]
            merge_scenarios = "_".join(merge_scenarios)
            merge_scenarios_part = f"{merge_scenarios}"

        name = f"physics_model__{p_m_part or 'none'}" + f"__years__{year_part or 'none'}" + f"__pt_bins__{pt_bin_part or 'none'}" + f"__wp_names__{wp_names_part or 'none'}" + f"__msc__{merge_scenarios_part or 'none'}"  # FIXME: too much information? maybe another meta hash or don't include all params # noqa: E501
        return f"workspace__{name}"

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        parts.insert_after("inf_model", "workspace", self.workspace_name)

        return parts

    def output(self):
        return {
            "workspace": self.target("workspace.root"),
            "t2w_log": self.target("t2w.log"),
        }

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

        input_datacards = self.input()["datacards"]["card"].path
        output_workspace = self.output()["workspace"].path

        # Build string representation of command to run
        options = {
            "--PO sf_naming_scheme": self.sf_naming_scheme,
            "--PO sf_range": self.sf_range,
            "--PO merge_scenarios": self.merge_scenarios,
            "--PO years": ",".join(self.year),
            "--PO pt_bins": ",".join(self.pt_bins),
            "--PO wp_names": ",".join(self.wp_names),
        }

        command_to_run = (
            f"text2workspace.py {input_datacards} -o {output_workspace} "
            f"-P {self.physics_model} -v {str(self.verbosity)}"
        )
        for option, value in options.items():
            command_to_run += f" {option}={value}"

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["t2w_log"].dump("\n".join(outp), formatter="text")
