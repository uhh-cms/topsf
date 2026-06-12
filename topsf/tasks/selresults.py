# coding: utf-8
"""
Custom tasks checking selection results.
"""

from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin,
)
from columnflow.tasks.selection import MergeSelectionStats
import json

class CheckSelectionResults(
    TopSFTask,
    CalibratorsMixin,
    SelectorMixin,
):
    run_command_in_tmp = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["selection_stats"] = self.requires_from_branch()

        return reqs

    def requires(self):
        reqs = {
            "selection_stats": self.reqs.MergeSelectionStats.req(self),
        }
        return reqs

    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)

        return [
            {
                "categories": cats,
            },
        ]

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_stats = self.input()["selection_stats"]["stats"].path
        with open(input_stats, "r") as f:
            stats = json.load(f)
        q = __import__('functools').partial(__import__('os')._exit, 0)
        __import__('IPython').embed()
