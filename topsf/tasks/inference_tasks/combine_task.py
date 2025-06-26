# coding: utf-8

"""
    # create toys
    echo "________________________________________________________"
    echo "Creating toys"
    toy_file="higgsCombine_toy.GenerateOnly.mH120.123456.root"
    if [ ! -f "${out_dir}/${toy_file}" ]; then
        combine -M GenerateOnly \
            -t -1 \
            --setParameters SF__NotMerged__${year}__${pt_bin}=1.,SF__FullyMerged__${year}__${pt_bin}=1.,SF__SemiMerged__${year}__${pt_bin}=1. \  # noqa
            --freezeParameters SF__NotMerged__${year}__${pt_bin},SF__FullyMerged__${year}__${pt_bin},SF__SemiMerged__${year}__${pt_bin} \  # noqa
            --saveToys \
            -n _toy \
            "${out_dir}/Workspace.root" || return $?
    fi
    echo "Done."

    # performing maximum-likelihood fit (MultiDimFit): expected
    echo "________________________________________________________"
    echo "Performing maximum-likelihood fit (MultiDimFit): expected"
    mdf_file="higgsCombine_exp.MultiDimFit.mH120.root"
    if [ ! -f "${out_dir}/${mdf_file}" ]; then
        combine -M MultiDimFit \
            -v 2 \
            --cminSingleNuisFit \
            --cminFallbackAlgo Minuit2,Simplex,0:0.1 \
            --datacard "${out_dir}/Workspace.root" \
            -n _exp \
            -t -1 \
            --toysFile "${out_dir}/${toy_file}" \
            --algo singles \
            --saveFitResult \
            --saveWorkspace  || return $?
    fi
    echo "Done."

    # perform maximum-likelihood fit (MultiDimFit): expected, with frozen systematics
    echo "________________________________________________________"
    echo "Performing maximum-likelihood fit (MultiDimFit): expected, with frozen systematics"
    mdf_freeze_file="higgsCombine_exp_freeze_syst.MultiDimFit.mH120.root"
    if [ ! -f "${out_dir}/${mdf_freeze_file}" ]; then
        combine -M MultiDimFit \
            -v 2 \
            --cminSingleNuisFit \
            --cminFallbackAlgo Minuit2,Simplex,0:0.1 \
            --datacard "${out_dir}/${mdf_file}" \
            -n _exp_freeze_syst \
            -t -1 \
            --toysFile "${out_dir}/${toy_file}" \
            --algo singles \
            --freezeParameters allConstrainedNuisances \
            --snapshotName MultiDimFit \
            --saveFitResult \
            --saveWorkspace || return $?
    fi
    echo "Done."
"""

import luigi
import law
import os
import shutil

from columnflow.tasks.framework.base import Requirements, CommandTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference_tasks.create_workspace import CreateWorkspace
from columnflow.util import dev_sandbox

from topsf.tasks.base import TopSFTask


class CombineTask(
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

    # method = luigi.Parameter(
    #     default="MultiDimFit",
    #     significant=True,
    #     description="Name of the method to use.",
    # )

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

    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateWorkspace=CreateWorkspace,
    )

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model
        return [
            {
                "categories": cats,
            },
        ]

    ### Left here for documentation purposes if needed later:  # noqa: E266
    # def workflow_requires(self):
    #     reqs = super().workflow_requires()

    #     reqs["workspace"] = [self.reqs.CreateWorkspace.req(self, sf_naming_scheme=sf_naming_scheme) for sf_naming_scheme in self.sf_naming_schemes]  # noqa: E501

    #     return reqs

    # @property
    # def sf_naming_schemes(self):
    #     return self.freeze_parameters.split(",")

    # def requires(self):
    #     reqs = {
    #         "workspace": [self.reqs.CreateWorkspace.req(self, sf_naming_scheme=sf_naming_scheme) for sf_naming_scheme in self.sf_naming_schemes],  # noqa: E501
    #     }
    #     return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
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


class GenToys(
    CombineTask,
):

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data.",
    )

    n_toys = luigi.IntParameter(
        default=0,
        significant=False,
        description="Number of toys to generate.",
    )

    set_parameters = law.CSVParameter(
        default="",
        significant=False,
        description="Set parameters as a list with 'param1=value1,param2=value2,...'.",
    )

    freeze_gen_parameters = law.CSVParameter(
        default="",
        significant=False,
        description="Freeze parameters given as a list with 'param1,param2,...'.",
    )

    save_toys = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save toys.",
    )

    gen_name = luigi.Parameter(
        default="_toy",
        significant=False,
        description="Name of the job, affects the name of the output tree",
    )

    @property
    def gentoys_name(self):
        if self.freeze_gen_parameters:
            if len(self.freeze_gen_parameters) < 3:
                fgp_part = "__".join(self.freeze_gen_parameters[:3])
            else:
                sorted_fgp = sorted(self.freeze_gen_parameters[3:])
                fgp_part = f"{len(self.freeze_gen_parameters)}__{law.util.create_hash(sorted_fgp)}"
        if self.asimov_data:
            as_part = "asimov_data"
        name = f"gen_toys__{as_part or 'none'}__fgp__{fgp_part or 'none'}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        workspace_name = self.reqs.CreateWorkspace.req(self).workspace_name

        parts.insert_after("inf_model", "workspace", workspace_name)
        parts.insert_after("workspace", "gentoys", self.gentoys_name)

        return parts

    def output(self):
        return {
            "toy_file": self.target(f"higgsCombine{self.gen_name}.GenerateOnly.mH120.123456.root"),
            "gen_toys_log": self.target("gen_toys.log"),
        }

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_workspace = self.input()["workspace"]["workspace"].path
        output_toy_file = self.output()["toy_file"].path

        EMPTY_STRING = ""

        # turn inputs into strings understandable by combine
        new_set_parameters = ",".join(self.set_parameters)
        new_freeze_gen_parameters = ",".join(self.freeze_gen_parameters)

        options = {
            "asimov_data": "-t -1" if self.asimov_data else EMPTY_STRING,
            "n_toys": f"-t {self.n_toys}" if self.n_toys > 0 else EMPTY_STRING,
            "set_parameters": f"--setParameters {new_set_parameters or ''}",
            "freeze_gen_parameters": f"--freezeParameters {new_freeze_gen_parameters or ''}",
            "save_toys": "--saveToys" if self.save_toys else EMPTY_STRING,
            "gen_name": f"-n {self.gen_name}" if len(str(self.gen_name)) else EMPTY_STRING,
            "combine_help": "-h" if self.combine_help else EMPTY_STRING,
            "verbosity": f"-v {self.verbosity}" if self.verbosity > 0 else EMPTY_STRING,
        }

        command_to_run = "combine -M GenerateOnly" + f" {input_workspace}"
        # add options
        for option, value in options.items():
            command_to_run += f" {value}"

        p, outp = self.run_command(command_to_run, echo=True)

        # move output to target
        if self.save_toys:
            self.output()["toy_file"].parent.touch()
            shutil.move(f"higgsCombine{self.gen_name}.GenerateOnly.mH120.123456.root", output_toy_file)

        self.output()["gen_toys_log"].dump("\n".join(outp), formatter="text")


class MultiDimFit(
    CombineTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "combine_sandbox"))

    cminSingleNuisFit = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use cminSingleNuisFit.",
    )

    cminFallbackAlgo = luigi.Parameter(
        default="Minuit2,Simplex,0:0.1",
        significant=False,
        description="Fallback algorithm.",
    )

    fit_name = luigi.Parameter(
        default="",
        significant=False,
        description="Name of the job, affects the name of the output tree",
    )

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data.",
    )

    algo = luigi.Parameter(
        default="singles",
        significant=False,
        description="Algorithm to use.",
    )

    save_fit_result = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save fit result.",
    )

    save_workspace = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save workspace.",
    )

    snapshot_name = luigi.Parameter(
        default="",
        significant=False,
        description="Name of the snapshot.",
    )

    freeze_fit_parameters = luigi.Parameter(
        default="",
        significant=False,
        description="Freeze parameters given as a list with 'param1,param2,...'.",
    )

    @property
    def mdf_name(self):
        if self.asimov_data:
            as_part = "asimov_data"
        if self.fit_name:
            if self.fit_name[0] == "_":
                fn_part = self.fit_name[1:]
        gentoys_name = self.reqs.GenToys.req(self).gentoys_name
        gt_part = law.util.create_hash(gentoys_name)
        mdf_name = f"mdf__{fn_part or 'none'}__{as_part or 'none'}__{gt_part}"
        return mdf_name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        workspace_name = self.reqs.CreateWorkspace.req(self).workspace_name
        gentoys_name = self.reqs.GenToys.req(self).gentoys_name

        parts.insert_after("inf_model", "workspace", workspace_name)
        parts.insert_after("workspace", "gentoys", gentoys_name)
        parts.insert_after("gentoys", "mdf", self.mdf_name)

        return parts

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateWorkspace=CreateWorkspace,
        GenToys=GenToys,
    )

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)  # categories defined in inference model
        return [
            {
                "categories": cats,
            },
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        reqs["toy_file"] = self.reqs.GenToys.req(self)

        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "toy_file": self.reqs.GenToys.req(self),
        }
        return reqs

    def output(self):
        return {
            "mdf_exp_file": self.target("higgsCombine_exp.MultiDimFit.mH120.root"),
            "mdf_exp_frozen_file": self.target("higgsCombine_exp_frozen.MultiDimFit.mH120.root"),
            "fit_exp_result": self.target("multidimfit_exp.root"),
            "fit_exp_frozen_result": self.target("multidimfit_exp_frozen.root"),
            "mdf_exp_log": self.target("mdf_exp.log"),
            "mdf_exp_frozen_log": self.target("mdf_exp_frozen.log"),
        }

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_workspace = self.input()["workspace"]["workspace"].path
        toy_file = self.input()["toy_file"]["toy_file"].path
        output_mdf_exp_file = self.output()["mdf_exp_file"].path
        output_fit_exp_result = self.output()["fit_exp_result"].path
        output_mdf_exp_frozen_file = self.output()["mdf_exp_frozen_file"].path
        output_fit_exp_frozen_result = self.output()["fit_exp_frozen_result"].path

        # perform fit: expected
        command_expected = f"combine -M MultiDimFit --datacard {input_workspace}"
        command_expected += f" --toysFile {toy_file}"
        command_expected += " -t -1" if self.asimov_data else ""
        command_expected += f" --algo {self.algo}" if len(str(self.algo)) else ""
        command_expected += " --saveFitResult" if self.save_fit_result else ""
        command_expected += " --saveWorkspace" if self.save_workspace else ""
        command_expected += " --cminSingleNuisFit" if self.cminSingleNuisFit else ""
        command_expected += f" --cminFallbackAlgo {self.cminFallbackAlgo}" if len(str(self.cminFallbackAlgo)) else ""
        command_expected += " -n _exp"
        command_expected += f" -v {self.verbosity}" if self.verbosity > 0 else ""
        self.publish_message(f"running command: {command_expected}")
        p_exp, outp_exp = self.run_command(command_expected, echo=True)

        # move output to target
        self.output()["mdf_exp_file"].parent.touch()
        self.output()["fit_exp_result"].parent.touch()
        shutil.move("higgsCombine_exp.MultiDimFit.mH120.root", output_mdf_exp_file)
        shutil.move("multidimfit_exp.root", output_fit_exp_result)

        self.output()["mdf_exp_log"].dump("\n".join(outp_exp), formatter="text")

        # perform fit: expected, with frozen systematics
        new_input_workspace = self.output()["mdf_exp_file"].path
        command_expected_frozen = f"combine -M MultiDimFit --datacard {new_input_workspace}"
        command_expected_frozen += f" --toysFile {toy_file}"
        command_expected_frozen += " -t -1" if self.asimov_data else ""
        command_expected_frozen += f" --algo {self.algo}" if len(str(self.algo)) else ""
        command_expected_frozen += " --saveFitResult" if self.save_fit_result else ""
        command_expected_frozen += " --saveWorkspace" if self.save_workspace else ""
        command_expected_frozen += " --freezeParameters allConstrainedNuisance"
        command_expected_frozen += " --snapshotName MultiDimFit"
        command_expected_frozen += " -n _exp_frozen"
        command_expected_frozen += f" -v {self.verbosity}" if self.verbosity > 0 else ""
        command_expected_frozen += " --cminSingleNuisFit" if self.cminSingleNuisFit else ""
        command_expected_frozen += f" --cminFallbackAlgo {self.cminFallbackAlgo or ''}"
        self.publish_message(f"running command: {command_expected_frozen}")
        p_exp_frozen, outp_exp_frozen = self.run_command(command_expected_frozen, echo=True)

        # move output to target
        self.output()["mdf_exp_frozen_file"].parent.touch()
        self.output()["fit_exp_frozen_result"].parent.touch()
        shutil.move("higgsCombine_exp_frozen.MultiDimFit.mH120.root", output_mdf_exp_frozen_file)
        shutil.move("multidimfit_exp_frozen.root", output_fit_exp_frozen_result)

        self.output()["mdf_exp_frozen_log"].dump("\n".join(outp_exp_frozen), formatter="text")
