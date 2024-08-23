# coding: utf-8

import luigi
import law
import os

from topsf.tasks.inference_v2.run_combine import CombineBaseTask


class GenToysV2(
    CombineBaseTask,
):
    n_toys = luigi.IntParameter(
        default=-1,
        significant=False,
        description="Number of toys to generate",
    )

    set_parameters = law.CSVParameter(
        significant=False,
        description="Set parameters for the toys",
    )

    freeze_gen_parameters = law.CSVParameter(
        significant=False,
        description="Freeze parameters for the generation",
    )

    save_toys = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save the generated toys",
    )

    gen_name = luigi.Parameter(
        significant=True,
        description="Name of the generated toys",
    )

    @property
    def gen_toys_name(self):
        name = f"gen_{self.gen_name}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "gen_toys", self.gen_toys_name)
        return parts

    def output(self):
        output_dict = {
            "toy_file": self.target(f"higgsCombine{self.gen_name}.GenerateOnly.mH120.123456.root"),
            "gen_toys_log": self.target("gen_toys.log"),
        }
        return output_dict

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        self.combine_method = "GenerateOnly"
        workspace = self.input()["workspace"]["workspace"].path
        output_toy_file = self.output()["toy_file"].path
        output_dirname = os.path.dirname(output_toy_file) + "/"
        print(self.combine_method)
        # touch output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        # turn inputs into strings understandable by combine
        new_set_parameters = ",".join(self.set_parameters)
        new_freeze_gen_parameters = ",".join(self.freeze_gen_parameters)

        command_to_run = f"combine -M {self.combine_method}"
        command_to_run += " -t -1" if self.asimov_data else f" -t {self.n_toys}"
        command_to_run += f" --setParameters {new_set_parameters}"
        command_to_run += f" --freezeParameters {new_freeze_gen_parameters}"
        command_to_run += " --saveToys" if self.save_toys else ""
        command_to_run += f" -n {self.gen_name}"
        command_to_run += f" {workspace}"

        p, outp = self.run_command(command_to_run, echo=True, cwd=output_dirname)

        self.output()["gen_toys_log"].dump("\n".join(outp), formatter="text")
