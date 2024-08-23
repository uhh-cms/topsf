# coding: utf-8

import luigi
import law

from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.parameters import SettingsParameter


class FitMixin(
    # CreateDatacards,
    ConfigTask,  # FIXME should inherit from what?
):

    physics_model = luigi.Parameter(
        significant=True,
        description="The physics model to use for the fit.",
    )

    wp_name = luigi.Parameter(
        significant=True,
        description="Working point to perform fit for.",
    )

    fit_modes = SettingsParameter(
        significant=True,
        description="Fit modes for each subprocess type (3q=TagAndProbe,2q=ThetaLike,0o1q=ThetaLike).",
    )

    years = law.CSVParameter(
        significant=True,
        description="Years to use for the fit.",
    )

    pt_bins = law.CSVParameter(
        significant=True,
        description="Pt bins to use for the fit.",
    )

    sf_range = luigi.Parameter(
        significant=False,
        description="Scale factor range for the fit.",
    )

    @property
    def fit_name(self):
        basename = "fit"

        pm_name = f"{law.util.create_hash(self.physics_model)}"

        wp_name = f"{self.wp_name}"

        msc_name_elems = []
        for subproc in ("3q", "2q", "0o1q"):
            fit_mode = self.fit_modes.get(subproc, "TagAndProbe")
            fit_mode_short = {
                "TagAndProbe": "TP",
                "ThetaLike": "TL",
            }.get(fit_mode, None)
            if fit_mode_short is None:
                fit_modes_available = ",".join(fit_mode_short)
                raise ValueError(
                    f"unknown fit mode {fit_mode}; expected one of: {fit_modes_available}"
                )
            msc_name_elems.append(f"{subproc}{fit_mode_short}")
        msc_name = "_".join(msc_name_elems)
        if len(self.years) < 4:
            years_name = "__".join(self.years)
        else:
            years_name = f"{len(self.years)}__{law.util.create_hash(sorted(self.years))}"

        if len(self.pt_bins) < 4:
            pt_bins_name = "__".join(self.pt_bins)
        else:
            pt_bins_name = f"{len(self.pt_bins)}__{law.util.create_hash(sorted(self.pt_bins))}"

        name = f"{basename}__pm__{pm_name}__wp__{wp_name}__msc__{msc_name}__years__{years_name}__pt_bins__{pt_bins_name}"  # noqa: E501
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("inf_model", "fit_v2", self.fit_name)
        return parts
