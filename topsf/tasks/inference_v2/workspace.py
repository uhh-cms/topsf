# coding: utf-8

import law
import os

from topsf.tasks.inference_v2.inference_base import InferenceBaseTask


class CreateWorkspaceV2(
    InferenceBaseTask,
):
    @property
    def workspace_name(self):
        name = "workspace__initial"  # indicates that this is the initial workspace
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("fit_v2", "workspace_v2", self.workspace_name)
        return parts

    def output(self):
        output_dict = {
            "workspace": self.target("workspace.root"),
            "t2w_log": self.target("text2workspace.log"),
        }
        return output_dict

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_datacards = self.input()["datacards"]["card"].path
        output_workspace = self.output()["workspace"].path
        output_dirname = os.path.dirname(output_workspace)
        # touch the output_dirname
        os.makedirs(output_dirname, exist_ok=True)

        years = ",".join(self.years)
        pt_bins = ",".join(self.pt_bins)
        fit_modes = ""
        for key, value in self.fit_modes.items():
            fit_modes += f"{key}:{value},"
        fit_modes = fit_modes[:-1]

        command_to_run = "text2workspace.py"
        command_to_run += f" {input_datacards}"
        command_to_run += f" -o {output_workspace}"
        command_to_run += f" -P {self.physics_model}"
        command_to_run += " --PO sf_naming_scheme=SF__{msc}__{year}__{pt_bin}"
        command_to_run += f" --PO sf_range={self.sf_range}"
        command_to_run += f" --PO merge_scenarios={fit_modes}"
        command_to_run += f" --PO years={years}"
        command_to_run += f" --PO pt_bins={pt_bins}"
        command_to_run += f" -v {self.verbosity}"
        command_to_run += " -h" if self.print_help else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["t2w_log"].dump(outp, formatter="text")
