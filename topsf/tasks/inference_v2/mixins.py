# coding: utf-8

import luigi
import law

from columnflow.tasks.framework.parameters import SettingsParameter


class FitMixin():

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
        default="[1,0.2,2.0]",
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
                    f"unknown fit mode {fit_mode}; expected one of: {fit_modes_available}",
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


class ImpactsMixin():
    mass = luigi.IntParameter(
        default=0,
        significant=False,
        description="Mass point",
    )

    robust_fit = luigi.IntParameter(
        default=1,
        significant=False,
        description="Run a robust fit",
    )

    combine_parallel = luigi.IntParameter(
        default=12,
        significant=False,
        description="Run the fits in parallel",
    )

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data for the fit",
    )


class PlotImpactsMixin():
    per_page = luigi.IntParameter(
        default=30,
        significant=False,
        description="Number of parameters to show per page",
    )

    height = luigi.IntParameter(
        default=600,
        significant=False,
        description="Height of the canvas, in pixels",
    )

    left_margin = luigi.FloatParameter(
        default=0.4,
        significant=False,
        description="Left margin of the canvas, expressed as a fraction",
    )

    label_size = luigi.FloatParameter(
        default=0.021,
        significant=False,
        description="Parameter name label size",
    )

    sort = luigi.ChoiceParameter(
        choices=["impact", "constraint", "pull"],
        default="impact",
        significant=False,
        description="The metric to sort the list of parameters",
    )

    relative = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Show impacts relative to the uncertainty on the POI",
    )

    plot_name = luigi.Parameter(
        default="",
        significant=False,
        description="Name of the plot",
    )

    poi = law.CSVParameter(
        significant=False,
        description="POI to plot",
    )

    cms_label = luigi.Parameter(
        default="simpw",
        significant=False,
        description="CMS label",
    )

    summary = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Print summary of the impacts",
    )


class ModeMixin():
    mode = luigi.ChoiceParameter(
        choices=["exp", "obs"],
        # default="obs",
        significant=True,
        description="Mode of the combine tool",
    )


class MultiDimFitMixin():
    cminSingleNuisFit = luigi.BoolParameter(  # FIXME Why does this parameter missing not cause an error?
        significant=False,
        description="Run a single nuisance fit for each channel",
    )

    cminFallbackAlgo = luigi.Parameter(
        default="Minuit2,Simplex,0:0.1",
        significant=False,
        description="Fallback algorithm for the cminimizer",
    )

    algo = luigi.Parameter(
        default="singles",
        significant=False,
        description="Algorithm to use for the fit",
    )

    save_fit_result = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save the fit results",
    )

    save_workspace = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Save the workspace",
    )

    snapshot_name = luigi.Parameter(
        default="",
        significant=False,
        description="Name of the snapshot",
    )

    freeze_fit_parameters = law.CSVParameter(
        significant=False,
        description="Freeze parameters for the fit",
    )


class ToysMixin(
    # ConfigTask,  # FIXME should inherit from what?
):
    n_toys = luigi.IntParameter(
        default=-1,
        significant=False,
        description="Number of toys to generate",
    )

    set_gen_parameters = law.CSVParameter(
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
