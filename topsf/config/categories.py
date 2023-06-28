# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

           lowest digit
                      |
    +---+---+---+---+---+
    | S | W | P | P | C |
    +---+---+---+---+---+

+=======+===============+======================+================================+
| Digit | Description   | Values               | Category name                  |
+=======+===============+======================+================================+
| C     | channel       | 1: electron          | 1e                             |
|    x1 |               | 2: muon              | 1m                             |
+-------+---------------+----------------------+--------------------------------+
| P     | probe jet     | 1: e.g. 300-400 GeV  | pt_{pt_min}_{pt_max}           |
|   x10 | pt bin        | ...                  |                                |
+-------+---------------+----------------------+--------------------------------+
| W     | working point | 1: very tight        |                                |
|       | (wp)          | ...                  | tau32_wp_{wp_name}_{pass,fail} |
| x1000 |               | 5: very loose        | (if *S* is 1/pass or 2/fail)   |
+-------+---------------+----------------------+                                |
| S     | wp category   | 1: pass wp *W*       |              - or -            |
|       | type          | 2: fail wp *W*       |                                |
|       |               | 3: strict, i.e.      | tau32_{tau32_min}_{tau32_max}  |
|       |               |    pass wp *W*,      | (if *S* is 3/strict)           |
|       |               |    fail next-loosest |                                |
| x 1e4 |               |    wp *W-1*          |                                |
+-------+---------------+----------------------+--------------------------------+

A digit group consisting entirely of zeroes ('0') represents the inclusive
category for the corresponding category group, i.e. no selection from that
group is applied.

This scheme encodes child/parent relations into the ID, making it easy
to check if categories overlap or are subcategories of each other. When applied
to a set of categories from different groups, the sum of the category IDs is the
ID of the combined category.
"""

import itertools

import order as od

from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

from topsf.config.util import create_category_combinations
from topsf.production.probe_jet import probe_jet

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

    cat_idx_lsd = 0  # 10-power of least significant digit
    cat_idx_ndigits = 1  # number of digits to use for category group

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
    # group 2: probe jet pt bins
    #
    cat_idx_lsd += cat_idx_ndigits
    cat_idx_ndigits = 2

    # get pt bins from config
    pt_bins = config.x.jet_selection.ak8.pt_bins
    pt_categories = []

    for cat_idx, (pt_min, pt_max) in enumerate(
        zip(pt_bins[:-1], pt_bins[1:]),
    ):
        pt_min_repr = f"{int(pt_min)}"
        if pt_max is None:
            pt_max_repr = "inf"
            cat_label = rf"$p_{{T}}$ > {pt_min} GeV"
        else:
            pt_max_repr = f"{int(pt_max)}"
            cat_label = rf"{pt_min} $\leq$ $p_{{T}}$ < {pt_max} GeV"

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

        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignement necessary"
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * (cat_idx + 1)),
            selection=sel_name,
            label=cat_label,
        )

        pt_categories.append(cat)

    #
    # group 3: probe jet tau3/tau2 bins
    #
    cat_idx_lsd += cat_idx_ndigits
    cat_idx_ndigits = 2

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
        cat_label = rf"{tau32_min} $\leq$ $\tau_{{3}}/\tau_{{2}}$ < {tau32_max}"

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

        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignement necessary"
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * ((cat_idx + 1) + 10 * 3)),
            selection=sel_name,
            label=cat_label,
        )

        tau32_categories.append(cat)

    # pass/fail categories from union of tau32 bins
    assert len(tau32_wps) + 2 == len(tau32_bins)
    for cat_idx, (tau32_wp, tau32_val) in enumerate(zip(tau32_wps, tau32_bins[1:-1])):
        for cat_type_idx, (pass_fail, comp_symbol, cat_slice) in enumerate([
            ("pass", "<", slice(None, cat_idx + 1)),
            ("fail", ">", slice(cat_idx + 1, None)),
        ]):
            cat_label = rf"$\tau_{{3}}/\tau_{{2}}$ {comp_symbol} {tau32_val} ({pass_fail})"

            cat_name = f"tau32_wp_{tau32_wp}_{pass_fail}"
            sel_name = f"sel_{cat_name}"

            # create category and add individual tau32 intervals as child categories
            cat = config.add_category(
                name=cat_name,
                id=int(10**cat_idx_lsd * ((cat_idx + 1) + 10 * (cat_type_idx + 1))),
                selection=None,
                label=cat_label,
            )
            child_cats = tau32_categories[cat_slice]
            for child_cat in child_cats:
                cat.add_category(child_cat)

    # -- combined categories

    def add_combined_categories(config):
        if getattr(config, "has_combined_categories", False):
            return  # combined categories already added

        category_groups = {
            "lepton": [
                config.get_category(name)
                for name in ["1e", "1m"]
            ],
            "pt": pt_categories,
            "tau32": tau32_categories,
        }

        create_category_combinations(
            config,
            category_groups,
            name_fn,
            kwargs_fn,
            skip_existing=False,
        )

        config.has_combined_categories = True

    add_combined_categories(config)
