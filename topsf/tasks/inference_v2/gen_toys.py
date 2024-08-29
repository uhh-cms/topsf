# coding: utf-8

import law
import os

from topsf.tasks.inference_v2.combine_base import CombineBaseTask
from topsf.tasks.inference_v2.mixins import ToysMixin


class GenToysV2(
    CombineBaseTask,
    ToysMixin,
):
    def output(self):
        output_dict = {
            "toy_file": self.target(f"higgsCombine{self.gen_name}.GenerateOnly.mH120.123456.root"),
            "gen_toys_log": self.target("gen_toys.log"),
        }
        return output_dict

    @property
    def gen_toys_name(self):
        name = f"toys_{self.gen_name}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "gen_toys", self.gen_toys_name)
        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        self.combine_method = "GenerateOnly"
        workspace = self.input()["workspace"]["workspace"].path
        output_toy_file = self.output()["toy_file"].path
        output_dirname = os.path.dirname(output_toy_file) + "/"
        print(self.combine_method)
        # touch output_dirname
        os.makedirs(output_dirname, exist_ok=True)

        # turn inputs into strings understandable by combine
        new_set_parameters = ",".join(self.set_gen_parameters)
        new_freeze_gen_parameters = ",".join(self.freeze_gen_parameters)

        command_to_run = f"combine -M {self.combine_method}"
        command_to_run += " -t -1" if self.asimov_data else f" -t {self.n_toys}"
        command_to_run += f" --setParameters {new_set_parameters}"
        command_to_run += f" --freezeParameters {new_freeze_gen_parameters}"
        command_to_run += " --saveToys" if self.save_toys else ""
        command_to_run += f" -n {self.gen_name}"
        command_to_run += f" {workspace}"

        p, outp = self.run_command(command_to_run, echo=True, cwd=output_dirname)

        self.output()["gen_toys_log"].dump(outp, formatter="text")
