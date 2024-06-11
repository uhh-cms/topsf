# coding: utf-8

import law
import os

from topsf.tasks.inference_v2.combine_base import CombineBaseTask


class CreateWorkspaceV2(
    CombineBaseTask,
):
    @property
    def workspace_name(self):
        name = f"workspace__{self.mode_inst}"
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
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        years = ",".join(self.years_inst)
        pt_bins = ",".join(self.pt_bins_inst)

        command_to_run = "text2workspace.py"
        command_to_run += f" {input_datacards}"
        command_to_run += f" -o {output_workspace}"
        command_to_run += f" -P {self.physics_model_inst}"
        command_to_run += " --PO sf_naming_scheme=SF__{msc}__{year}__{pt_bin}"
        command_to_run += f" --PO sf_range={self.sf_range_inst}"
        command_to_run += f" --PO merge_scenarios={self.fit_modes_inst}"
        command_to_run += f" --PO years={years}"
        command_to_run += f" --PO pt_bins={pt_bins}"
        command_to_run += f" -v {self.combine_verbosity_inst}"
        command_to_run += " -h" if self.combine_help_inst else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["t2w_log"].dump("\n".join(outp), formatter="text")
