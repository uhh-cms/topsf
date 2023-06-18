# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

    lowest digit
          |
    +---+---+
    | M | C |
    +---+---+

    C = channel       (1: electron [1e], 2: muon [1m])
    M = merge type    (1: fully merged [3q], 2: not fully merged [le2q])

A digit group consisting entirely of zeroes ('0') represents the inclusive
category for the corresponding category group, i.e. no selection from that
group is applied.

This scheme encodes child/parent relations into the ID, making it easy
to check if categories overlap or are categories of each other. When applied
to a set of leaf categories, the sum of the category IDs is the ID of the
parent category.
"""

import order as od

from columnflow.config_util import create_category_combinations


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

    # top-level category for electron channel
    config.add_category(
        name="1e",
        id=1,
        selection="sel_1e",
        label="1e",
        #channel=config.get_channel("e"),  # noqa
    )

    # top-level category for muon channel
    config.add_category(
        name="1m",
        id=2,
        selection="sel_1m",
        label=r"1$\mu$",
        #channel=config.get_channel("mu"),  # noqa
    )

    # top jet merge types (fully merged, semi merged + unmerged)
    config.add_category(
        name="3q",
        id=10,
        selection="sel_3q",
        label=r"fully merged (3q)",
    )
    config.add_category(
        name="le2q",
        id=20,
        selection="sel_le2q",
        label=r"not merged ($\leq$2q)",
    )

    # -- combined categories

    category_groups = {
        "lepton": [
            config.get_category(name)
            for name in ["1e", "1m"]
        ],
        "merge_type": [
            config.get_category(name)
            for name in ["3q", "le2q"]
        ],
    }

    create_category_combinations(config, category_groups, name_fn, kwargs_fn)
