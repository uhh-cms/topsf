# coding: utf-8

import luigi
import law

from columnflow.tasks.framework.base import ConfigTask


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

    fit_modes = luigi.Parameter(
        significant=True,
        description="Fit mode for the fit (3q:TagAndProbe,2q:ThetaLike,0o1q:ThetaLike).",
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
    def physics_model_inst(self):
        return self.physics_model

    @property
    def wp_name_inst(self):
        return self.wp_name

    @property
    def fit_modes_inst(self):
        return self.fit_modes

    @property
    def years_inst(self):
        return self.years

    @property
    def pt_bins_inst(self):
        return self.pt_bins

    @property
    def sf_range_inst(self):
        return self.sf_range

    @property
    def fit_name(self):
        basename = "fit"

        pm_name = f"{law.util.create_hash(self.physics_model)}"

        wp_name = f"{self.wp_name}"

        # add msc and mode information to the name of the following form:
        # 3q:TagAndProbe,2q:ThetaLike,0o1q:ThetaLike -> 3qTP_2qTL_0o1qTL
        fit_modes = self.fit_modes.split(",")
        # ['3q:TagAndProbe', '2q:ThetaLike', '0o1q:ThetaLike']
        msc_name = ""
        for msc in fit_modes:
            merge_scenario = msc.split(":")[0]
            fit_mode = msc.split(":")[1]
            if fit_mode == "TagAndProbe":
                fit_mode = "TP"
            elif fit_mode == "ThetaLike":
                fit_mode = "TL"
            else:
                raise ValueError(f"Unknown fit mode {fit_mode}")
            msc_name += f"{merge_scenario}{fit_mode}_"
        msc_name = msc_name[:-1]

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
