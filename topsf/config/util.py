# coding: utf-8

"""
Collection of general helpers and utilities.
"""

from __future__ import annotations

from columnflow.config_util import create_category_id

__all__ = [
    "create_category_combinations",
]

import itertools
from typing import Callable, Any

import order as od


def create_category_combinations(
    config: od.Config,
    categories: dict[str, list[od.Category]],
    name_fn: Callable[[Any], str],
    kwargs_fn: Callable[[Any], dict] | None = None,
    skip_existing: bool = True,
    skip_fn: Callable[[dict[str, od.Category], str], bool] | None = None,
    only_leaves: bool = False,
) -> int:
    """
    Given a *config* object and sequences of *categories* in a dict, creates all combinations of
    possible leaf categories at different depths, connects them with parent - child relations
    (see :py:class:`order.Category`) and returns the number of newly created categories.

    *categories* should be a dictionary that maps string names to sequences of categories that
    should be combined. The names are used as keyword arguments in a callable *name_fn* that is
    supposed to return the name of newly created categories (see example below).

    Each newly created category is instantiated with this name as well as arbitrary keyword
    arguments as returned by *kwargs_fn*. This function is called with the categories (in a
    dictionary, mapped to the sequence names as given in *categories*) that contribute to the newly
    created category and should return a dictionary. If the fields ``"id"`` and ``"selection"`` are
    missing, they are filled with reasonable defaults leading to a auto-generated, deterministic id
    and a list of all parent selection statements.

    If the name of a new category is already known to *config* it is skipped unless *skip_existing*
    is *False*. In addition, *skip_fn* can be a callable that receives a dictionary mapping group
    names to categories that represents the combination of categories to be added. In case *skip_fn*
    returns *True*, the combination is skipped.

    If *only_leaves* is *True*, only categories with at least one member per group are added.

    Example:

    .. code-block:: python

        categories = {
            "lepton": [cfg.get_category("e"), cfg.get_category("mu")],
            "n_jets": [cfg.get_category("1j"), cfg.get_category("2j")],
            "n_tags": [cfg.get_category("0t"), cfg.get_category("1t")],
        }

        def name_fn(categories):
            # simple implementation: join names in defined order if existing
            return "__".join(cat.name for cat in categories.values() if cat)

        def kwargs_fn(categories):
            # return arguments that are forwarded to the category init
            # (use id "+" here which simply increments the last taken id, see order.Category)
            # (note that this is also the default)
            return {"id": "+"}

        create_category_combinations(cfg, categories, name_fn, kwargs_fn)
    """
    n_created_categories = 0
    n_groups = len(categories)
    group_names = list(categories.keys())

    # nothing to do when there are less than 2 groups
    if n_groups < 2:
        return n_created_categories

    # check functions
    if not callable(name_fn):
        raise TypeError(f"name_fn must be a function, but got {name_fn}")
    if kwargs_fn and not callable(kwargs_fn):
        raise TypeError(f"when set, kwargs_fn must be a function, but got {kwargs_fn}")

    # start combining, considering one additional groups for combinatorics at a time
    min_n_groups = n_groups if only_leaves else 2
    for _n_groups in range(min_n_groups, n_groups + 1):

        # build all group combinations
        for _group_names in itertools.combinations(group_names, _n_groups):

            # build the product of all categories for the given groups
            _categories = [categories[group_name] for group_name in _group_names]
            for root_cats in itertools.product(*_categories):
                # build the name
                root_cats = dict(zip(_group_names, root_cats))
                cat_name = name_fn(root_cats)

                # skip when already existing
                if skip_existing and config.has_category(cat_name, deep=True):
                    continue

                # skip when manually triggered
                if callable(skip_fn) and skip_fn(root_cats):
                    continue

                # create arguments for the new category
                kwargs = kwargs_fn(root_cats) if callable(kwargs_fn) else {}
                if "id" not in kwargs:
                    kwargs["id"] = create_category_id(config, cat_name)
                if "selection" not in kwargs:
                    kwargs["selection"] = [c.selection for c in root_cats.values()]

                # create the new category
                cat = od.Category(name=cat_name, **kwargs)
                n_created_categories += 1

                if only_leaves:
                    # connect root cats directly to leaves
                    for root_cat in root_cats.values():
                        root_cat.add_category(cat)
                else:
                    # find direct parents and connect them
                    for _parent_group_names in itertools.combinations(_group_names, _n_groups - 1):
                        if len(_parent_group_names) == 1:
                            parent_cat_name = root_cats[_parent_group_names[0]].name
                        else:
                            parent_cat_name = name_fn({
                                group_name: root_cats[group_name]
                                for group_name in _parent_group_names
                            })
                        parent_cat = config.get_category(parent_cat_name, deep=True)
                        parent_cat.add_category(cat)

    return n_created_categories
