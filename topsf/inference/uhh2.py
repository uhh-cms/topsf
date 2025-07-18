# coding: utf-8

"""
TopSF inference model, based on UHH2 LegacyTopTagging analysis
"""
from collections import defaultdict

from columnflow.inference import inference_model, ParameterType, ParameterTransformation
from columnflow.config_util import get_datasets_from_process


def get_process_datasets(config_inst, process):
    """
    Helper function to retrieve all datasets for a process.
    """
    process_insts = [
        p for p, _, _ in config_inst.get_process(process).walk_processes(include_self=True)  # noqa
    ]
    datasets = [
        dataset_inst.name for dataset_inst in config_inst.datasets
        if any(map(dataset_inst.has_process, process_insts))
    ]
    return datasets


@inference_model
def uhh2(self):
    """
    Inference model intended to reproduce the fits in UHH2 LegacyTopTagging analysis.
    """
    processes = set()
    for config_inst in self.config_insts:
        processes.update(config_inst.x.inference_processes)

    #
    # regions/categories
    #
    # Validate all configs have a fit setup defined
    for config_inst in self.config_insts:
        if config_inst.x.fit_setup is None:
            raise ValueError(
                f"Config {config_inst.name} does not have a fit setup defined.",
            )

    # Use the first config as the reference
    ref_setup = self.config_insts[0].x.fit_setup
    ref_channels = ref_setup["channels"]
    ref_pt_bins = ref_setup["pt_bins"]
    ref_wp_names = ref_setup["wp_names"]
    ref_fit_vars = ref_setup["fit_vars"]
    ref_shape_unc = ref_setup["shape_unc"]

    # Compare the rest against the reference
    for config_inst in self.config_insts[1:]:
        setup = config_inst.x.fit_setup
        if setup["channels"] != ref_channels:
            raise ValueError(f"Config {config_inst.name} has different channels.")
        if setup["pt_bins"] != ref_pt_bins:
            raise ValueError(f"Config {config_inst.name} has different pt_bins.")
        if setup["wp_names"] != ref_wp_names:
            raise ValueError(f"Config {config_inst.name} has different wp_names.")
        if setup["fit_vars"] != ref_fit_vars:
            raise ValueError(f"Config {config_inst.name} has different fit_vars.")
        if setup["shape_unc"] != ref_shape_unc:
            raise ValueError(f"Config {config_inst.name} has different shape_unc.")

        # Add check to make sure only one fit variable is used
        if len(ref_fit_vars) != 1:
            raise ValueError(
                "Only one fit variable is supported. "
                f"Found {len(ref_fit_vars)} variables in {config_inst.name}: {', '.join(ref_fit_vars)}.",
            )

    ref_fit_vars = ref_fit_vars[0]  # use the first fit variable in the given list

    # Now safe to assign
    channels = ref_channels
    pt_bins = ref_pt_bins
    wp_names = ref_wp_names
    fit_vars = ref_fit_vars
    shape_unc = ref_shape_unc

    # tuples of inference categories and
    # corresponding config category names
    categories = [
        # (columnflow_name, combine_name)
        (f"{channel}__{pt_bin}__tau32_wp_{wp_name}_{region}",
        f"bin_{channel}__{pt_bin}__tau32_wp_{wp_name}_{region}")
        for channel in channels
        for pt_bin in pt_bins
        for wp_name in wp_names
        for region in ("pass", "fail")
    ]

    for config_cat, inference_cat in categories:

        var_name = fit_vars
        if not all(has_var := [config_inst.has_variable(var_name) for config_inst in self.config_insts]):
            missing_var_configs = [
                config_inst.name for config_inst, has_var in zip(self.config_insts, has_var) if not has_var
            ]
            raise ValueError(
                f"Variable {var_name} not found in configs {', '.join(missing_var_configs)} "
                f"for {config_cat}. Please ensure that {var_name} is part of all configs.",
            )
        cat_name = inference_cat
        cat_kwargs = dict(
            config_data={
                config_inst.name: self.category_config_spec(
                    category=config_cat,
                    variable=var_name,
                    data_datasets=[
                        dataset_inst.name for dataset_inst in
                        get_datasets_from_process(config_inst, "data", strategy="all", only_first=False)
                    ],
                )
                for config_inst in self.config_insts
            },
            mc_stats="0 1 1",  # FIXME: make configurable
            flow_strategy="move",
            empty_bin_value=0.0,  # NOTE: remove this when removing custom rebin task
        )

        # add the category to the inference model
        # print(f"Adding category {inference_cat} with config category {config_cat} and variable {var_name}.")
        self.add_category(cat_name, **cat_kwargs)

    #
    # processes
    #
    inference_processes = {}

    used_datasets = defaultdict(set)
    for proc in processes:
        if any(missing_proc := [not config.has_process(proc) for config in self.config_insts]):
            config_missing = [
                config_inst.name for config_inst, missing in zip(self.config_insts, missing_proc)
                if missing
            ]
            raise Exception(f"Process {proc} is not defined in configs {', '.join(config_missing)}.")

        # get datasets corresponding to this process
        datasets = {config_inst.name: [
            d.name for d in
            get_datasets_from_process(config_inst, proc, strategy="all", check_deep=True, only_first=False)
        ] for config_inst in self.config_insts}

        for config, _datasets in datasets.items():
            if not _datasets:
                raise Exception(
                    f"No datasets found for process {proc} in config {config}. "
                    "Please check your configuration.",
                )
            # print(f"Process {proc} in config {config} was assigned datasets: {datasets}")

            used_datasets[config] |= set(_datasets)

        # print(f"Adding process {proc} with datasets: {datasets}")
        self.add_process(
            name=inference_processes.get(proc, proc),
            config_data={
                config_inst.name: self.process_config_spec(
                    process=proc,
                    mc_datasets=datasets[config_inst.name],
                )
                for config_inst in self.config_insts
            },
            is_signal=self.config_insts[0].get_process(proc).has_tag("is_topsf_signal"),
            # is_dynamic=??,
        )

    #
    # parameters
    # FIXME: assumes all configs have the same uncertainties assigned to the same processes
    #

    # lumi
    lumi_uncertainties = {
        lumi_unc: config_inst.x.luminosity.get(names=lumi_unc, direction=("down", "up"), factor=True)
        for config_inst in self.config_insts
        for lumi_unc in config_inst.x.luminosity.uncertainties
    }
    for lumi_unc_name, effect in lumi_uncertainties.items():
        # print(f"Adding luminosity uncertainty parameter {lumi_unc_name} with effect {effect}.")
        self.add_parameter(
            lumi_unc_name,
            type=ParameterType.rate_gauss,
            effect=effect,
            transformations=[ParameterTransformation.symmetrize],
        )

    for proc in self.config_insts[0].x.process_rates.keys():
        subprocesses = [
            subproc.name for subproc, _, _ in self.config_insts[0].get_process(proc).walk_processes(
                algo="bfs",
                include_self=True,
            )
        ]
        # print(f"Adding process rate parameter for {proc} with subprocesses: {subprocesses}.")
        self.add_parameter(
            f"xsec_{proc}",
            type=ParameterType.rate_gauss,
            effect=self.config_insts[0].x.process_rates[proc],
            process=[inference_processes.get(subproc, subproc) for subproc in subprocesses],
        )

    # systematic shifts (TODO: add others)
    uncertainty_shifts = shape_unc

    # old list left for reference
    # uncertainty_shifts = [
    #     # "pdf",
    #     # "mcscale",
    #     # "prefiring",
    #     # "minbias_xs",  # pileup

    #     # "muon",  # TODO: split?
    #     # "mu_id",
    #     # "mu_iso",
    #     # "mu_reco",
    #     # "mu_trigger",

    #     # b-tagging
    #     # "btag_cferr1",
    #     # "btag_cferr2",
    #     # "btag_hf",
    #     # "btag_hfstats1_2017",
    #     # "btag_hfstats2_2017",
    #     # "btag_lf",
    #     # "btag_lfstats1_2017",
    #     # "btag_lfstats2_2017",
    # ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }
    param_kwargs = {}

    for proc in processes:
        for unc in uncertainty_shifts:
            if proc == "mj" and unc in ["FSR", "ISR"]:
                continue
            if unc in ["mur", "muf"] and not (proc.startswith("st") or proc.startswith("tt")):
                continue
            par = inference_pars.get(unc, unc)
            param_kwargs["config_data"] = {
                config_inst.name: self.parameter_config_spec(
                    shift_source=unc,
                )
                for config_inst in self.config_insts
            }
            # print(f"Adding parameter {par} for process {proc} with uncertainty {unc}.")
            self.add_parameter(
                f"{par}",
                process=inference_processes.get(proc, proc),
                type=ParameterType.shape,
                **param_kwargs,
            )

    self.cleanup()
