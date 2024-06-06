from __future__ import print_function

import itertools
import re

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

# https://hypernews.cern.ch/HyperNews/CMS/get/higgs-combination/1653.html
# https://gitlab.cern.ch/gouskos/boostedjetcalibration/-/blob/master/TagAndProbeExtended.py


class TopSFCombinePhysicsModel(PhysicsModel):

    # old naming convention
    RE_COMBINE_CHANNEL = "^bin_(?P<channel>\w+)__(?P<year>\w+)__(?P<pt_bin>\w+)__(?P<region>\w+)$"  # noqa: W605
    RE_COMBINE_PROCESS = "^(?P<process>\w+?)(_(?P<msc>(3q|2q|0o1q|bkg)))?$"  # noqa: W605

    # # simplified naming convention (no year, pt_bin)
    # RE_COMBINE_CHANNEL = "^(?P<process>\w+?)(__MSc_(?P<msc>\w+?))?$"
    # RE_COMBINE_PROCESS = "^(?P<process>\w+)(__(?P<msc>\w+))?$"

    def __init__(self):
        super().__init__()

        self.fit_merge_scenarios = {}

        # default options, possibly modified by '--PO' flags
        # FIXME set "wp_name" option and reset regions to pass and fail
        self.model_options = {
            "sf_naming_scheme": None,
            "sf_range": None,
            "pt_bins": set(),
            "years": set(),
            "regions": {"pass", "fail"},
        }

    # -- command-line options parsing

    def setPhysicsOptions(self, physOptions):
        """
        Set model options from parameters passed to combine via '--PO' flags.
        """
        for po in physOptions:
            # parse option
            try:
                name, value = po.split("=", 1)
            except ValueError:
                raise ValueError("physics option should be given as 'name=value'")

            # parse option value and set attributs
            if name == "merge_scenarios":
                merge_scenario_specs = value.split(",")
                for msc_spec in merge_scenario_specs:
                    try:
                        msc, mode = msc_spec.split(":", 1)
                    except ValueError:
                        raise ValueError("invalid merge scenario specification {msc_spec}")

                    self.fit_merge_scenarios[msc] = (mode == "TagAndProbe")

            elif name in ("sf_naming_scheme", "sf_range"):
                self.model_options[name] = value

            elif name in ("pt_bins", "years"):
                self.model_options[name] = set(value.split(","))

    # -- convenient access to options via properties

    @property
    def sf_naming_scheme(self):
        return self.model_options["sf_naming_scheme"]

    @property
    def sf_range(self):
        return self.model_options["sf_range"]

    @property
    def pt_bins(self):
        return self.model_options["pt_bins"]

    @property
    def years(self):
        return self.model_options["years"]

    @property
    def regions(self):
        return self.model_options["regions"]

    # -- extract information from channel/process names

    def parse_channel(self, channel):
        m = re.match(self.RE_COMBINE_CHANNEL, channel)
        if not m:
            raise ValueError(f"cannot parse channel name '{channel}'")
        return m.groupdict()

    def parse_process(self, process):
        m = re.match(self.RE_COMBINE_PROCESS, process)
        if not m:
            raise ValueError(f"cannot parse process name '{process}'")
        return m.groupdict()

    # -- implement combine PhysicsModel abstract methods

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        pois = []

        expected_yields = {}
        for channel in self.DC.bins:
            for process in self.DC.exp.get(channel):
                expected_yields.setdefault(channel, {})[process] = self.DC.exp.get(channel).get(process)

        channel_dicts = {
            channel: self.parse_channel(channel)
            for channel in self.DC.bins
        }

        # get sum of pass/fail expected yields for TTbar/ST for each channel/msc combo
        expected_yields_top = {}
        for msc, (channel, channel_dict) in itertools.product(
            self.fit_merge_scenarios,
            channel_dicts.items(),
        ):
            # skip unrequested combinations
            year = channel_dict["year"]
            pt_bin = channel_dict["pt_bin"]
            region = channel_dict["region"]
            if (
                year not in self.years or
                pt_bin not in self.pt_bins or
                region not in self.regions
            ):
                continue

            for process in self.DC.exp.get(channel):
                process_dict = self.parse_process(process)
                if not msc == process_dict["msc"]:
                    continue
                expected_yields_top.setdefault(msc, {}).setdefault(year, {}).setdefault(pt_bin, {}).setdefault(region, 0)  # noqa: E501
                expected_yields_top[msc][year][pt_bin][region] += self.DC.exp.get(channel).get(process)

        print(expected_yields_top)
        # FIXME: why is the CLI-supplied naming scheme being overridden here?
        # self.sf_naming_scheme = "__".join(["SF", r"{msc}", r"{year}", r"{pt_bin}"])

        for msc, fit_msc in self.fit_merge_scenarios.items():
            for year, pt_bin in itertools.product(self.years, self.pt_bins):

                # scale factors for pass region
                sf_name = self.sf_naming_scheme.format(msc=msc, year=year, pt_bin=pt_bin)
                sf_range = self.sf_range

                self.modelBuilder.doVar(sf_name + sf_range)
                pois.append(sf_name)

                # do not create the antisf variable if not needed
                if not fit_msc:
                    continue

                # scale factors for "fail" region ("anti-scale factors")
                antisf_name = f"Anti{sf_name}"
                yields = expected_yields_top[msc][year][pt_bin]
                n_pass, n_fail = yields["pass"], yields["fail"]
                self.modelBuilder.factory_(
                    f"expr::{antisf_name}(\"max(0.,1.+(1.-@0)*{n_pass}/{n_fail})\", {sf_name})"  # noqa: C812
                )
                # TODO: maybe guard against zero division on n_fail = 0?

        # set the scale factors as the POIs
        self.modelBuilder.doSet("POI", ",".join(pois))

    def getYieldScale(self, channel, process):
        """
        Return the name of a RooAbsReal with which to scale the yield of process *process* in channel *channel*,
        or the two special values 1 and 0 (don't scale, and set to zero).
        """
        # parse the names of the given channel and process
        channel_dict = self.parse_channel(channel)
        process_dict = self.parse_process(process)

        # was using get(), why? can a process be missing?
        if not self.DC.isSignal[process]:
            return 1

        msc = process_dict["msc"]
        sf_name = self.sf_naming_scheme.format(msc=msc, **channel_dict)
        if (not self.fit_merge_scenarios.get(msc)) or channel_dict["region"] == "Pass":
            return sf_name
        elif channel_dict["region"] == "Fail":
            return sf_name.replace("SF_", "AntiSF_")
        else:
            return 1


topsf_model = TopSFCombinePhysicsModel()
