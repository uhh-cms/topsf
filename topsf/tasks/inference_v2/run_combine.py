# coding: utf-8

import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from topsf.tasks.inference import CreateDatacards

from topsf.tasks.inference_v2.combine_base import InferenceBaseTask
from topsf.tasks.inference_v2.workspace import CreateWorkspaceV2


class CombineBaseTask(
    InferenceBaseTask,
):
    # set general parameters available for all combine commands of the form
    # 'combine -M method -t -l'
    combine_method = luigi.Parameter(
        default=None,
        significant=False,
        description="Method to run with combine tool",
    )

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data for the fit",
    )

    mode = luigi.ChoiceParameter(
        choices=["exp", "obs"],
        default="exp",
        significant=True,
        description="Mode of the combine tool",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
        CreateWorkspace=CreateWorkspaceV2,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["workspace"] = self.reqs.CreateWorkspace.req(self)
        return reqs

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
        }
        return reqs
