# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

    lowest digit
              |
    +---+---+---+
    | P | M | C |
    +---+---+---+

    C = channel       (1: electron [1e], 2: muon [1m])
    M = merge type    (1: top fully merged [t3q], 2: top semi-merged [t2q],
                       3: top not merged [t1q], 4: top background [t0q])
    P = process

A digit group consisting entirely of zeroes ('0') represents the inclusive
category for the corresponding category group, i.e. no selection from that
group is applied.

This scheme encodes child/parent relations into the ID, making it easy
to check if categories overlap or are categories of each other. When applied
to a set of leaf categories, the sum of the category IDs is the ID of the
parent category.
"""

import order as od

from columnflow.util import maybe_import
from columnflow.config_util import create_category_combinations
from columnflow.selection import Selector, selector

from topsf.production.gen_top import probe_jet

np = maybe_import("numpy")
ak = maybe_import("awkward")


def name_fn(**groups):
    """Naming function for automatically generated combined categories."""
    return "__".join(
        cat_name for cat_name in groups.values()
        if cat_name is not None
    )


def kwargs_fn(categories: dict[str, od.Category]):
    """Customization function for automatically generated combined categories."""
    return {
        "id": sum(cat.id for cat in categories.values()),
        "selection": [cat.selection for cat in categories.values()],
        "label": ", ".join(
            cat.label for cat in categories.values()
        ),
    }


def add_categories(config: od.Config) -> None:
    """
    Adds categories to a *config* that are available after the selection step.
    """
    # inclusive category
    config.add_category(
        name="incl",
        id=0,
        selection="sel_incl",
        label="inclusive",
    )

    #
    # group 1: channel
    #

    # electron channel
    config.add_category(
        name="1e",
        id=1,
        selection="sel_1e",
        label="1e",
    )

    # muon channel
    config.add_category(
        name="1m",
        id=2,
        selection="sel_1m",
        label=r"1$\mu$",
    )

    #
    # group 2: process (with split by # of merged quarks for top processes)
    #

    # used for assigning labels for top process categories
    merge_labels = {
        0: "background",
        1: "not merged",
        2: "semi-merged",
        3: "fully merged",
    }

    # used for assigning process category IDs
    process_indexes = {
        "st": 1,
        "tt": 2,
        "dy_lep": 3,
        "w_lnu": 4,
        "vv": 5,
        "qcd": 6,
        "data": 7,
    }

    # build categories and corresponding selectors
    proc_categories = []
    for proc, _, _ in config.walk_processes():

        # don't add categories for proesses without a mapped index
        if proc.name not in process_indexes:
            continue

        # get index used to compute the category ID
        proc_idx = process_indexes[proc.name]

        # separate categories for top processes
        # split by # of quarks merged to probe jet
        if proc.has_tag("has_top"):
            for n_merged in range(4):

                cat_name = f"{proc.name}_{n_merged}q"
                sel_name = f"sel_{cat_name}"

                @selector(
                    uses={probe_jet},
                    produces={probe_jet},
                    cls_name=sel_name,
                )
                def sel_proc_n_merged(
                    self: Selector, events: ak.Array,
                    proc: od.Process = proc,
                    n_merged: int = n_merged,
                    **kwargs,
                ) -> ak.Array:
                    f"""
                    Select events from process '{proc}' with {n_merged} quarks merged to a hadronic probe jet.
                    """
                    # get process
                    if len(self.dataset_inst.processes) != 1:
                        raise NotImplementedError(
                            f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
                            "assigned, which is not supported",
                        )
                    this_proc = self.dataset_inst.processes.get_first()
                    sel_proc = this_proc.has_parent_process(proc)

                    # get probe jet
                    events = self[probe_jet](events, **kwargs)

                    return ak.fill_none(
                        (
                            sel_proc &
                            (events.ProbeJet.n_merged == n_merged) &
                            events.ProbeJet.is_hadronic_top
                        ),
                        False,
                    )

                n_merged_label = merge_labels[n_merged]
                cat = config.add_category(
                    name=f"{proc.name}_{n_merged}q",
                    id=100 * (proc_idx + 1) + 10 * (n_merged + 1),
                    selection=f"sel_{proc.name}_{n_merged}q",
                    label=f"{proc.label}, {n_merged_label} ({n_merged}q)",
                )

                proc_categories.append(cat)

        # one category per non-top processes
        else:

            cat_name = f"{proc.name}"
            sel_name = f"sel_{cat_name}"

            @selector(
                uses={"event"},
                cls_name=sel_name,
            )
            def sel_proc(
                self: Selector, events: ak.Array,
                proc: od.Process = proc,
                **kwargs,
            ) -> ak.Array:
                f"""
                Select events from process '{proc}'
                """
                # get process
                if len(self.dataset_inst.processes) != 1:
                    raise NotImplementedError(
                        f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
                        "assigned, which is not supported",
                    )
                this_proc = self.dataset_inst.processes.get_first()
                sel_proc = this_proc.has_parent_process(proc)

                return (
                    sel_proc &
                    ak.ones_like(events.event, dtype=bool)
                )

            cat = config.add_category(
                name=cat_name,
                id=100 * (proc_idx + 1),
                selection=sel_name,
                label=f"{proc.label}",
            )

            proc_categories.append(cat)

    # -- combined categories

    category_groups = {
        "lepton": [
            config.get_category(name)
            for name in ["1e", "1m"]
        ],
        "process": proc_categories,
    }

    create_category_combinations(config, category_groups, name_fn, kwargs_fn)
