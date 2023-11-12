# coding: utf-8

"""
Example inference model.
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation

from columnflow.config_util import get_datasets_from_process

@inference_model
def default(self):

    #
    # categories
    #

    pt_cat_name = "pt_300_400"
    tau32_wp = "very_tight"  # 0.1% efficiency

    for pass_fail in ("pass", "fail"):
        for lep in ("1e", "1m"):
            cat_name = f"{lep}__{pt_cat_name}__tau32_wp_{tau32_wp}_{pass_fail}"
            self.add_category(
                f"cat_{cat_name}",
                config_category=cat_name,
                config_variable="probejet_msoftdrop_widebins",
                # fake data from sum of MC processes
                data_from_processes=[
                    "st_3q", "st_2q", "st_0o1q", "st_bkg",
                    "tt_3q", "tt_2q", "tt_0o1q", "tt_bkg",
                    "vx",
                    "mj",
                ],
                # real data (TODO)
                # config_data_datasets=[
                #     "data_mu_b",
                # ],
                mc_stats=True,
                # empty bins should stay empty!!
                empty_bin_value=0.0,
            )

    #
    # processes
    #

    def get_dataset_names_from_process(proc: str):
        dataset_insts = get_datasets_from_process(
           self.config_inst,
           proc,
           check_deep=True,
        )
        return [dataset_inst.name for dataset_inst in dataset_insts]

    top_subprocesses = [
        "3q",
        "2q",
        "0o1q",
        "bkg",
    ]

    for proc in ("st", "tt"):
        for subproc in top_subprocesses:
            full_proc = f"{proc}_{subproc}"
            self.add_process(
                full_proc,
                is_signal=(subproc != "bkg"),
                config_process=full_proc,
                config_mc_datasets=get_dataset_names_from_process(full_proc),
            )

    for proc in ("vx", "mj"):
        self.add_process(
            proc,
            config_process=proc,
            # config_mc_datasets=[
            # ],
        )

    #
    # parameters
    #

    # # groups
    # self.add_parameter_group("experiment")
    # self.add_parameter_group("theory")

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # # tune uncertainty
    # self.add_parameter(
    #     "tune",
    #     process="TT",
    #     type=ParameterType.shape,
    #     config_shift_source="tune",
    # )

    # # muon weight uncertainty
    # self.add_parameter(
    #     "mu",
    #     process=["ST", "TT"],
    #     type=ParameterType.shape,
    #     config_shift_source="mu",
    # )

    # # jet energy correction uncertainty
    # self.add_parameter(
    #     "jec",
    #     process=["ST", "TT"],
    #     type=ParameterType.shape,
    #     config_shift_source="jec",
    # )

    #
    # post-processing
    #

    self.cleanup()


@inference_model
def default_no_shapes(self):
    # same initialization as "test" above
    default.init_func.__get__(self, self.__class__)()

    #
    # remove all shape parameters
    #

    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape:
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    #
    # post-processing
    #

    self.cleanup()
