# coding: utf-8

"""
TopSF inference model, based on UHH2 LegacyTopTagging analysis
"""
import itertools

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


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

    year = self.config_inst.campaign.x.year  # noqa; not used right now

    processes = self.config_inst.x.inference_processes

    #
    # regions/categories
    #

    #
    # categories
    #

    # category elements to combine in fit
    years = [
        year % 100,
    ]
    channels = self.config_inst.x.fit_setup["channels"]
    pt_bins = self.config_inst.x.fit_setup["pt_bins"]
    wp_names = self.config_inst.x.fit_setup["wp_names"]

    # tuples of inference categories and
    # corresponding config category names
    categories = [
        # (columnflow_name, combine_name)
        (f"{channel}__{pt_bin}__tau32_wp_{wp_name}_{region}",
        f"bin_{channel}__{year}__{pt_bin}__tau32_wp_{wp_name}_{region}")
        for channel, year, pt_bin, wp_name, region in itertools.product(
            channels,
            years,
            pt_bins,
            wp_names,
            ("pass", "fail"),
        )
    ]

    # add categories to inference model
    for config_cat, inference_cat in categories:
        self.add_category(
            inference_cat,
            config_category=config_cat,
            config_variable="probejet_msoftdrop_widebins",
            mc_stats=True,
            config_data_datasets=get_process_datasets(self.config_inst, "data"),
            # fake data from sum of MC processes
            data_from_processes=processes,
            # empty bins should stay empty!!
            empty_bin_value=0.0,
        )

    #
    # processes
    #

    # different naming convention in combine for some processes
    inference_processes = {}

    for proc in processes:

        # raise if process not defined in config
        if not self.config_inst.has_process(proc):
            raise ValueError(
                f"Process {proc} requested for inference, but is not "
                f"present in the config {self.config_inst.name}.",
            )

        # determine datasets for process
        datasets = get_process_datasets(self.config_inst, proc)

        # check if process is signal
        is_signal = self.config_inst.get_process(proc).has_tag("is_topsf_signal")

        # add process to inference model
        self.add_process(
            inference_processes.get(proc, proc),
            config_process=proc,
            is_signal=is_signal,
            config_mc_datasets=datasets,
        )

    #
    # parameters
    #

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    for proc in self.config_inst.x.process_rates.keys():
        subprocesses = [
            subproc.name for subproc, _, _ in self.config_inst.get_process(proc).walk_processes(
                algo="bfs",
                include_self=True,
            )
        ]
        self.add_parameter(
            f"xsec_{proc}",
            type=ParameterType.rate_gauss,
            effect=self.config_inst.x.process_rates[proc],
            process=[inference_processes.get(subproc, subproc) for subproc in subprocesses],
        )

    # systematic shifts (TODO: add others)
    uncertainty_shifts = [
        # "pdf",
        # "mcscale",
        # "prefiring",
        # "minbias_xs",  # pileup

        # "muon",  # TODO: split?
        # "mu_id",
        # "mu_iso",
        # "mu_reco",
        # "mu_trigger",

        # b-tagging
        # "btag_cferr1",
        # "btag_cferr2",
        # "btag_hf",
        # "btag_hfstats1_2017",
        # "btag_hfstats2_2017",
        # "btag_lf",
        # "btag_lfstats1_2017",
        # "btag_lfstats2_2017",
    ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)
            self.add_parameter(
                f"{par}_{proc}",
                process=inference_processes.get(proc, proc),
                type=ParameterType.shape,
                config_shift_source=unc,
            )

    self.cleanup()
