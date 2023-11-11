# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

    lowest digit
              |
    +---+---+---+
    | S | P | P |
    +---+---+---+

+=======+===============+======================+================================+
| Digit | Description   | Values               | Category name                  |
+=======+===============+======================+================================+
| P     | jet pt bin    | 1: e.g. 300-400 GeV  | pt_{pt_min}_{pt_max}           |
|   x10 |               | ...                  |                                |
+-------+---------------+----------------------+                                |
| S     | pt interval   | 1: lower bound       |                                |
|       | type          | 2: upper bound       |                                |
| x 1e3 |               | 3: strict            |                                |
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
from columnflow.categorization import Categorizer, categorizer

from topsf.config.util import create_category_combinations
from topsf.production.probe_jet import probe_jet

np = maybe_import("numpy")
ak = maybe_import("awkward")


def name_fn(categories: dict[str, od.Category]):
    """Naming function for automatically generated combined categories."""
    return "__".join(cat.name for cat in categories.values() if cat)


def kwargs_fn(categories: dict[str, od.Category]):
    """Customization function for automatically generated combined categories."""
    return {
        "id": sum(cat.id for cat in categories.values()),
        "selection": [cat.selection for cat in categories.values()],
        "label": "\n".join(
            cat.label for cat in categories.values()
        ),
    }


def skip_fn(categories: dict[str, od.Category]):
    """Custom function for skipping certain category combinations."""
    # skip if combination involves both `pt` and `pt_low` groups,
    # since these are not disjoint
    if all(group in categories for group in ["pt", "pt_combined"]):
        return True

    return False  # don't skip


def add_categories(config):
    # only inclusive category for now
    config.add_category(
        name="incl",
        id=0,
        selection="sel_incl",
        label="inclusive",
    )


def add_categories_per_jet(config):
    """
    (EXPERIMENTAL, not working yet) Define categories per-jet instead of per-event.
    """
    # inclusive category
    cat_incl = config.add_category(
        name="incl",
        id=0,
        label="inclusive",
    )

    #
    # group 1: jet pt bins
    #

    cat_idx_lsd = 0  # 10-power of least significant digit
    cat_idx_ndigits = 2  # number of digits to use for category group

    # get pt bins from config
    pt_bins = config.x.jet_selection.ak8.pt_bins
    pt_bin_reprs = [
        f"{int(pt)}" if pt is not None else "inf"
        for pt in pt_bins
    ]
    pt_categories = []

    for cat_idx, (pt_min, pt_max, pt_min_repr, pt_max_repr) in enumerate(
        zip(pt_bins[:-1], pt_bins[1:], pt_bin_reprs[:-1], pt_bin_reprs[1:]),
    ):
        assert pt_min_repr != "inf"
        if pt_max is None:
            cat_label = rf"$p_{{T}}^{{jet}}$ > {pt_min} GeV"
        else:
            cat_label = rf"{pt_min} $\leq$ $p_{{T}}^{{jet}}$ < {pt_max} GeV"

        cat_name = f"pt_{pt_min_repr}_{pt_max_repr}"
        sel_name = f"sel_{cat_name}"

        @categorizer(
            cls_name=sel_name,
        )
        def sel_pt(
            self: Categorizer, events: ak.Array,
            pt_range: tuple = (pt_min, pt_max),
            **kwargs,
        ) -> ak.Array:
            f"""
            Select fat jets with pt in the range [{pt_min_repr}, {pt_max_repr})
            """
            column = self.cfg.column
            return events, ak.fill_none(
                (events[column].pt >= pt_range[0]) &
                ((events[column].pt < pt_range[1]) if pt_range[1] is not None else True),
                False,
            )

        @sel_pt.init
        def sel_pt_init(self: Categorizer) -> None:
            # return immediately if config not yet loaded
            config_inst = getattr(self, "config_inst", None)
            if not config_inst:
                return

            # set config dict
            self.cfg = config_inst.x.jet_selection.get("ak8", {})

            # set input columns
            column = self.cfg.get("column", "FatJet")
            self.uses.add(f"{column}.pt")


        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignment necessary"
        cat_id = int(10**cat_idx_lsd * ((cat_idx + 1) + 300))
        print(f"{cat_name = }, {cat_id = }, {cat_idx_lsd = }")
        cat = cat_incl.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * ((cat_idx + 1) + 300)),
            selection=sel_name,
            label=cat_label,
        )

        pt_categories.append(cat)

    # pt categories resulting from from union
    # of two or more neighboring pt bins
    pt_combined_categories = []
    pt_lowest, pt_highest = pt_bins[0], pt_bins[-1]
    pt_lowest_repr, pt_highest_repr = pt_bin_reprs[0], pt_bin_reprs[-1]
    for cat_idx, pt_val in enumerate(pt_bins[1:-1]):
        for cat_type_idx, (bin_repr, comp_symbol, cat_slice) in enumerate([
            (f"{{pt_val}}_{pt_highest_repr}", ">", slice(cat_idx + 1, None)),  # 0/'low'
            (f"{pt_lowest_repr}_{{pt_val}}", "<", slice(None, cat_idx + 1)),  # 1/'high'
        ]):
            if comp_symbol == ">" and pt_highest is None:
                cat_label = rf"$p_{{T}}^{{jet}}$ > {pt_val} GeV"
            elif comp_symbol == ">":
                cat_label = rf"{pt_val} $\leq$ $p_{{T}}^{{jet}}$ < {pt_highest} GeV"
            elif comp_symbol == "<":
                cat_label = rf"{pt_lowest} $\leq$ $p_{{T}}^{{jet}}$ < {pt_val} GeV"
            else:
                assert False

            cat_name = f"pt_{bin_repr.format(pt_val=pt_val)}"
            sel_name = f"sel_{cat_name}"

            cat_id = int(10**cat_idx_lsd * ((cat_idx + 1) + 100 * (cat_type_idx + 1)))
            print(f"{cat_name = }, {cat_id = }, {cat_idx_lsd = }, {cat_idx = }, {cat_type_idx = }")
            # don't do anything for existing categories
            if config.has_category(cat_name):
                continue

            # create category and add individual pt intervals as child categories
            cat = cat_incl.add_category(
                name=cat_name,
                id=int(10**cat_idx_lsd * ((cat_idx + 1) + 100 * (cat_type_idx + 1))),
                selection=None,
                label=cat_label,
            )
            child_cats = pt_categories[cat_slice]
            for child_cat in child_cats:
                cat.add_category(child_cat)

            pt_combined_categories.append(cat)

    # -- combined categories

    def add_combined_categories(config):
        if getattr(config, "has_combined_categories", False):
            return  # combined categories already added

        category_groups = {
            "pt": pt_categories,
            "pt_combined": pt_combined_categories,
        }

        create_category_combinations(
            config,
            category_groups,
            name_fn,
            kwargs_fn,
            skip_fn=skip_fn,
            skip_existing=False,
        )

        # connect intermediary `pt_combined` and `pt` categories
        category_groups_no_pt = {}
        for n in range(1, len(category_groups_no_pt) + 1):
            for group_names in itertools.combinations(category_groups_no_pt, n):
                root_cats = [category_groups_no_pt[gn] for gn in group_names]
                for root_cats in itertools.product(*root_cats):
                    root_cats = dict(zip(group_names, root_cats))
                    for pt_combined_cat in pt_combined_categories:
                        root_cats_1 = dict(root_cats, **{"pt_combined": pt_combined_cat})
                        name_1 = name_fn(root_cats_1)
                        cat_1 = config.get_category(name_1)
                        for pt_cat in pt_combined_cat.categories:
                            # skip compound children
                            if "__" in pt_cat.name:
                                continue
                            root_cats_2 = dict(root_cats, **{"pt": pt_cat})
                            name_2 = name_fn(root_cats_2)
                            cat_2 = config.get_category(name_2)
                            cat_1.add_category(cat_2)

        config.has_combined_categories = True

    add_combined_categories(config)
