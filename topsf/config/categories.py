# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

                   lowest digit
                              |
    +---+---+---+---+---+---+---+
    | W | W | T | T | P | M | C |
    +---+---+---+---+---+---+---+

    C = channel       (1: electron [1e], 2: muon [1m])
    M = merge type    (1: top fully merged [t3q], 2: top semi-merged [t2q],
                       3: top not merged [t1q], 4: top background [t0q])
    P = process
    T = pt bin
    W = working point bin

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
from columnflow.selection import Selector, selector

from topsf.production.gen_top import probe_jet
from topsf.config.util import create_category_combinations

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

    #
    # group 3: probe jet pt bins
    #

    pt_bins = [300, 400, 480, 600, None]
    pt_categories = []

    for cat_idx, (pt_min, pt_max) in enumerate(
        zip(pt_bins[:-1], pt_bins[1:]),
    ):
        pt_min_repr = f"{int(pt_min)}"
        if pt_max is None:
            pt_max_repr = "inf"
            cat_label = rf"{pt_min} $\leq$ $p_{{T}}$ < {pt_max} GeV"
        else:
            pt_max_repr = f"{int(pt_max)}"
            cat_label = rf"$p_{{T}}$ > {pt_min} GeV"

        cat_name = f"pt_{pt_min_repr}_{pt_max_repr}"
        sel_name = f"sel_{cat_name}"

        @selector(
            uses={probe_jet},
            cls_name=sel_name,
        )
        def sel_pt(
            self: Selector, events: ak.Array,
            pt_range: tuple = (pt_min, pt_max),
            **kwargs,
        ) -> ak.Array:
            f"""
            Select events with probe jet pt the range [{pt_min_repr}, {pt_max_repr})
            """
            events = self[probe_jet](events, **kwargs)
            return ak.fill_none(
                (events.ProbeJet.pt >= pt_range[0]) &
                ((events.ProbeJet.pt < pt_range[1]) if pt_range[1] is not None else True),
                False,
            )

        cat = config.add_category(
            name=cat_name,
            id=1000 * (cat_idx + 1),
            selection=sel_name,
            label=cat_label,
        )

        pt_categories.append(cat)

    #
    # group 4: probe jet tau3/tau2 bins
    #

    tau32_wps = [
        "very_tight",
        "tight",
        "medium",
        "loose",
        "very_loose",
    ]
    tau32_bins = [0] + [
        config.x.toptag_working_points["tau32"][wp]
        for wp in tau32_wps
    ] + [1]

    tau32_categories = []
    for cat_idx, (tau32_min, tau32_max) in enumerate(
        zip(tau32_bins[:-1], tau32_bins[1:]),
    ):
        tau32_min_repr = f"{int(tau32_min*100):03d}"
        tau32_max_repr = f"{int(tau32_max*100):03d}"
        cat_label = rf"{tau32_min} $\leq$ $\tau_{{32}}$ < {tau32_max}"

        cat_name = f"tau32_{tau32_min_repr}_{tau32_max_repr}"
        sel_name = f"sel_{cat_name}"

        @selector(
            uses={probe_jet},
            cls_name=sel_name,
        )
        def sel_tau32(
            self: Selector, events: ak.Array,
            tau32_range: tuple = (tau32_min, tau32_max),
            **kwargs,
        ) -> ak.Array:
            f"""
            Select events with probe jet pt the range [{tau32_min_repr}, {tau32_max_repr})
            """
            events = self[probe_jet](events, **kwargs)
            tau32 = events.ProbeJet.tau3 / events.ProbeJet.tau2
            return ak.fill_none(
                (tau32 >= tau32_range[0]) & (tau32 < tau32_range[1]),
                False,
            )

        cat = config.add_category(
            name=cat_name,
            id=100000 * (cat_idx + 1),
            selection=sel_name,
            label=cat_label,
        )

        tau32_categories.append(cat)

    # pass/fail categories from union of tau32 slices
    assert len(tau32_wps) + 2 == len(tau32_bins)
    for cat_idx, (tau32_wp, tau32_val) in enumerate(zip(tau32_wps, tau32_bins[1:-1])):
        assert cat_idx < 9, "no space for category, ID reassignement necessary"
        for i_pass_fail, (pass_fail, comp_symbol, cat_slice) in enumerate([
            ("pass", ">", slice(None, cat_idx + 1)),
            ("fail", "<", slice(cat_idx + 1, None)),
        ]):
            cat_label = rf"$\tau_{{32}}$ {comp_symbol} {tau32_val}"

            cat_name = f"tau32_wp_{tau32_wp}_{pass_fail}"
            sel_name = f"sel_{cat_name}"

            # create category and add individual tau32 intervals as child categories
            cat = config.add_category(
                name=cat_name,
                id=10000000 * (cat_idx + 1) + 1000000 * (i_pass_fail + 1),
                selection=None,
                label=cat_label,
            )
            child_cats = tau32_categories[cat_slice]
            for child_cat in child_cats:
                cat.add_category(child_cat)

    # -- combined categories

    category_groups = {
        "lepton": [
            config.get_category(name)
            for name in ["1e", "1m"]
        ],
        "process": proc_categories,
        "pt": pt_categories,
        "tau32": tau32_categories,
    }

    create_category_combinations(
        config,
        category_groups,
        name_fn,
        kwargs_fn,
        only_leaves=True,
    )
