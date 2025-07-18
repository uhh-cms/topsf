# coding: utf-8

"""
Tasks related to the creation of datacards for inference purposes.
"""

from collections import OrderedDict

import luigi
import law
import re

from columnflow.tasks.framework.base import AnalysisTask, wrapper_factory
from columnflow.tasks.framework.inference import SerializeInferenceModelBase
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import DotDict

from topsf.tasks.base import TopSFTask


def multi_string_repr(strings, max_len=1, sep="_"):
    """Create a unique representation of a sequence of strings."""
    if len(strings) <= max_len:
        return sep.join(strings)

    return f"{len(strings)}_{law.util.create_hash(sorted(strings))}"


class CreateDatacards(
    TopSFTask,
    SerializeInferenceModelBase,
):
    resolution_task_cls = MergeHistograms

    per_category = luigi.BoolParameter(
        default=False,
        significant=True,
        description="when True, create a separate datacard for each category",
    )

    wp_name = luigi.Parameter(
        significant=True,
        description="working point to use for the fit",
    )

    def _requires_cat_obj(self, cat_obj: DotDict, merge_variables: bool = False, **req_kwargs):
        """
        Custom helper to create the requirements for a single category object.
        The difference to the columnflow.tasks.framework.inference.SerializeInferenceModelBase
        implementation is the due to the different branch_map structure.

        :param cat_obj: category object from an InferenceModel
        :param merge_variables: whether to merge the variables from all requested category objects
        :return: requirements for the category object
        """
        reqs = {}
        for config_inst in self.config_insts:
            if not (config_data := cat_obj.config_data.get(config_inst.name)):
                continue

            if merge_variables:
                variables = tuple(
                    _cat_obj.config_data.get(config_inst.name).variable
                    for _cat_obj in list(self.branch_map.values())[0]["categories"]  # noqa: E501, different from cf implementation of SerializeInferenceModelBase
                )
            else:
                variables = (config_data.variable,)

            # add merged shifted histograms for mc
            reqs[config_inst.name] = {
                proc_obj.name: {
                    dataset: self.reqs.MergeShiftedHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        shift_sources=tuple(
                            param_obj.config_data[config_inst.name].shift_source
                            for param_obj in proc_obj.parameters
                            if (
                                config_inst.name in param_obj.config_data and
                                self.inference_model_inst.require_shapes_for_parameter(param_obj)
                            )
                        ),
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in self.get_mc_datasets(config_inst, proc_obj)
                }
                for proc_obj in cat_obj.processes
                if config_inst.name in proc_obj.config_data and not proc_obj.is_dynamic
            }
            # add merged histograms for data, but only if
            # - data in that category is not faked from mc, or
            # - at least one process object is dynamic (that usually means data-driven)
            if (
                (not cat_obj.data_from_processes or any(proc_obj.is_dynamic for proc_obj in cat_obj.processes)) and
                (data_datasets := self.get_data_datasets(config_inst, cat_obj))
            ):
                reqs[config_inst.name]["data"] = {
                    dataset: self.reqs.MergeHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in data_datasets
                }
        return reqs

    # custom function required to create the branch map
    # unchanged
    def create_branch_map(self):
        cats = list(self.inference_model_inst.categories)

        # Regular expression pattern to match the exact wp_name
        pattern = f"^(?P<channel>\w+)__(?P<pt_bin>\w+)__tau32_wp_(?P<wp_name>({self.wp_name}))_(?P<region>(pass|fail))?$"  # noqa: E501, W605

        # Filter categories to include only those with the specified wp in user input
        filtered_cats = []
        for cat in cats:
            config_category = cat.name
            if re.match(pattern, config_category):
                filtered_cats.append(cat)

        if self.per_category:
            return [
                {
                    "categories": [cat],
                }
                for cat in filtered_cats
            ]
        else:
            return [
                {
                    "categories": filtered_cats,
                },
            ]

    def workflow_requires(self):
        # why does this not work?
        # reqs = super().workflow_requires()
        reqs = law.util.InsertableDict()

        reqs["merged_hists"] = hist_reqs = {}
        for cat_obj in list(self.branch_map.values())[0]["categories"]:  # noqa: E501, different from cf implementation of SerializeInferenceModelBase
            cat_reqs = self._requires_cat_obj(cat_obj)
            for config_name, proc_reqs in cat_reqs.items():
                hist_reqs.setdefault(config_name, {})
                for proc_name, dataset_reqs in proc_reqs.items():
                    hist_reqs[config_name].setdefault(proc_name, {})
                    for dataset_name, task in dataset_reqs.items():
                        hist_reqs[config_name][proc_name].setdefault(dataset_name, set()).add(task)
        return reqs

    def requires(self):
        cat_objs = list(self.branch_map.values())[0]["categories"]
        reqs = {}
        for cat_obj in cat_objs:
            reqs[cat_obj.name] = self._requires_cat_obj(cat_obj, branch=-1, workflow="local")

        return reqs

    def output(self):
        cat_objs = self.branch_data["categories"]

        categories = {cat_obj.config_data[self.config_insts[0].name].category for cat_obj in cat_objs}
        variables = {cat_obj.config_data[self.config_insts[0].name].variable for cat_obj in cat_objs}
        categories_repr = multi_string_repr(categories)
        variables_repr = multi_string_repr(variables)

        basename = lambda name, ext: f"{name}__cat_{categories_repr}__var_{variables_repr}.{ext}"
        return {
            "card": self.target(basename("datacard", "txt")),
            "shapes": self.target(basename("shapes", "root")),
        }

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_before("version", "inf_model", f"inf__{self.inference_model}")
        parts.insert_after("inf_model", "wp_name", f"wp__{self.wp_name}")
        return parts

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import hist
        from columnflow.inference.cms.datacard import DatacardWriter
        print("Creating datacards ...")

        # prepare inputs
        inputs = self.input()

        cat_objs = self.branch_data["categories"]

        # reference config, assuming all configs use the same categories, processes, and fit variable
        ref_config_inst = self.config_insts[0]

        hists_all_cats = OrderedDict()
        for cat_obj in cat_objs:
            category_inst = ref_config_inst.get_category(cat_obj.config_data[ref_config_inst.name].category)
            variable_inst = ref_config_inst.get_variable(cat_obj.config_data[ref_config_inst.name].variable)
            leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

            # histogram data per process

            with self.publish_step(f"extracting {variable_inst.name} in {category_inst.name} ..."):
                inputs[cat_obj.name] = inputs[cat_obj.name].copy()

                for config_inst in self.config_insts:
                    hists = OrderedDict()
                    print(f"Processing config: {config_inst.name}")
                    for proc_obj_name, inp in inputs[cat_obj.name][config_inst.name].items():
                        if proc_obj_name == "data":
                            proc_obj = None
                            process_inst = config_inst.get_process("data")
                        else:
                            proc_obj = self.inference_model_inst.get_process(proc_obj_name, category=cat_obj.name)
                            process_inst = config_inst.get_process(proc_obj.config_data[config_inst.name].process)
                        sub_process_insts = [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]

                        h_proc = None
                        for dataset, _inp in inp.items():
                            dataset_inst = config_inst.get_dataset(dataset)

                            # skip when the dataset is already known to not contain any sub process
                            if not any(map(dataset_inst.has_process, sub_process_insts)):
                                self.logger.warning(
                                    f"dataset '{dataset}' does not contain process '{process_inst.name}' "
                                    "or any of its subprocesses which indicates a misconfiguration in the "
                                    f"inference model '{self.inference_model}'",
                                )
                                continue

                            # open the histogram and work on a copy
                            h = _inp["collection"][0]["hists"][variable_inst.name].load(formatter="pickle").copy()
                            # print(f"Loaded histograms for dataset '{dataset}' and variable '{variable_inst.name}'")

                            # axis selections
                            h = h[{
                                "process": [
                                    hist.loc(p.name)
                                    for p in sub_process_insts
                                    if p.name in h.axes["process"]
                                ],
                                "category": [
                                    hist.loc(c.name)
                                    for c in leaf_category_insts
                                    if c.name in h.axes["category"]
                                ],
                            }]

                            # axis reductions
                            h = h[{"process": sum, "category": sum}]

                            # add the histogram for this dataset
                            if h_proc is None:
                                h_proc = h
                            else:
                                h_proc += h

                        # there must be a histogram
                        if h_proc is None:
                            raise Exception(f"no histograms found for process '{process_inst.name}'")

                        # create the nominal hist
                        hists[proc_obj_name] = OrderedDict()
                        nominal_shift_inst = config_inst.get_shift("nominal")
                        hists[proc_obj_name]["nominal"] = h_proc[
                            {"shift": hist.loc(nominal_shift_inst.name)}
                        ]

                        # per shift
                        if proc_obj:
                            for param_obj in proc_obj.parameters:
                                # skip the parameter when varied hists are not needed
                                if not self.inference_model_inst.require_shapes_for_parameter(param_obj):
                                    continue
                                # store the varied hists
                                # hists[proc_obj_name] = {}
                                for d in ["up", "down"]:
                                    shift_inst = config_inst.get_shift(
                                        f"{param_obj.config_data[config_inst.name].shift_source}_{d}",
                                    )
                                    hists[proc_obj_name][(param_obj.name, d)] = h_proc[
                                        {"shift": hist.loc(shift_inst.name)}
                                    ]

                        # store hists for this category
                        if cat_obj.name not in hists_all_cats:
                            hists_all_cats[cat_obj.name] = OrderedDict()

                        for process_name, hist_dict in hists.items():
                            hists_all_cats[cat_obj.name].setdefault(process_name, OrderedDict())[config_inst.name] = hist_dict  # noqa: E501

        # forward objects to the datacard writer
        outputs = self.output()
        writer = DatacardWriter(self.inference_model_inst, hists_all_cats)
        with outputs["card"].localize("w") as tmp_card, outputs["shapes"].localize("w") as tmp_shapes:
            writer.write(tmp_card.path, tmp_shapes.path, shapes_path_ref=outputs["shapes"].basename)


CreateDatacardsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateDatacards,
    enable=["configs", "skip_configs"],
)
