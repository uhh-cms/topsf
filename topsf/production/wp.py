# coding: utf-8

"""
Column producers related to defining categories.

The difference between the standard `category_id` producer
and the `wp_category_id` producer is that the latter computes
a category *for each fat jet* rather than per event.
"""

from __future__ import annotations

from collections import defaultdict

import law

from columnflow.categorization import Categorizer
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    produces={"category_ids"},
)
def wp_category_ids(
    self: Producer,
    events: ak.Array,
    target_events: ak.Array | None = None,
    **kwargs,
) -> ak.Array:
    """
    Assigns each fat jet an array of category ids.
    """
    category_ids = []

    for cat_inst in self.config_inst.get_leaf_categories():
        # start with a true mask
        cat_mask = np.ones(len(events), dtype=bool)

        # loop through selectors
        for categorizer in self.categorizer_map[cat_inst]:
            events, mask = self[categorizer](events, call_force=True, **kwargs)
            cat_mask = cat_mask & mask

        # covert to nullable array with the category ids or none, then apply ak.singletons
        ids = ak.where(cat_mask, np.float64(cat_inst.id), np.float64(np.nan))
        category_ids.append(ak.singletons(ak.nan_to_none(ids), axis=-1))

    # combine
    category_ids = ak.concatenate(category_ids, axis=-1)

    # save, optionally on a target events array
    if target_events is None:
        target_events = events
    target_events = set_ak_column(target_events, "category_ids", category_ids, value_type=np.int64)

    return target_events


@wp_category_ids.init
def wp_category_ids_init(self: Producer) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # store a mapping from leaf category to categorizer classes for faster lookup
    self.categorizer_map = defaultdict(list)

    # add all categorizers obtained from leaf category selection expressions to the used columns
    for cat_inst in self.config_inst.get_leaf_categories():
        # treat all selections as lists of categorizers
        for sel in law.util.make_list(cat_inst.selection):
            if Categorizer.derived_by(sel):
                categorizer = sel
            elif Categorizer.has_cls(sel):
                categorizer = Categorizer.get_cls(sel)
            else:
                raise Exception(
                    f"selection '{sel}' of category '{cat_inst.name}' cannot be resolved to an "
                    "existing Categorizer object",
                )

            # the categorizer must be exposed
            if not categorizer.exposed:
                raise RuntimeError(
                    f"cannot use unexposed categorizer '{categorizer}' to evaluate category "
                    f"{cat_inst}",
                )

            # update dependency sets
            self.uses.add(categorizer)
            self.produces.add(categorizer)

            self.categorizer_map[cat_inst].append(categorizer)

    # set config dict
    self.cfg = config_inst.x.jet_selection.get("ak8", {})

    # explicitly add dependence on fat jet column
    # (to get correct return array shape if no categorizers given)
    column = self.cfg.get("column", "FatJet")
    self.uses.add(f"{column}.pt")
