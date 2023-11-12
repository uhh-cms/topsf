# coding: utf-8

"""
Column production methods related defining categories.
"""

from collections import defaultdict

import law

from columnflow.categorization import Categorizer
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(produces={"category_ids"})
def category_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Assigns each event an array of category ids.
    """
    category_ids = None

    # TODO: we maybe don't want / need to loop through all leaf categories
    for cat_inst in self.config_inst.get_leaf_categories():
        # start with a true mask
        cat_mask = np.ones(len(events)) > 0

        # loop through categorizers
        for categorizer in self.category_to_categorizers[cat_inst]:
            # run the categorizer for events that still match the mask, then AND concat
            events, mask = self[categorizer](events, **kwargs)
            cat_mask = cat_mask & mask
            # _cat_mask = self[categorizer](events[cat_mask], **kwargs)
            # cat_mask[cat_mask] &= np.asarray(_cat_mask == 1)

            # stop if no events are left
            if not ak.any(cat_mask):
                break

        # make nullable array with the category id or none, then apply ak.singletons
        ids = ak.singletons(ak.mask(ak.ones_like(cat_mask, dtype=np.uint32) * cat_inst.id, cat_mask))

        # merge to other categories
        if category_ids is None:
            category_ids = ids
        else:
            category_ids = ak.concatenate([category_ids, ids], axis=1)

    # combine and save
    events = set_ak_column(events, "category_ids", category_ids, value_type=np.uint32)

    return events


@category_ids.init
def category_ids_init(self: Producer) -> None:
    # store a mapping from leaf category to categorizer classes for faster lookup
    self.category_to_categorizers = defaultdict(list)

    if not hasattr(self.config_inst, "cached_leaf_categories"):
        self.config_inst.cached_leaf_categories = self.config_inst.get_leaf_categories()

    # add all categorizers obtained from leaf category selection expressions to the used columns
    for cat_inst in self.config_inst.cached_leaf_categories:
        # treat all selections as lists
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

            # update dependency sets
            self.uses.add(categorizer)
            self.produces.add(categorizer)

            self.category_to_categorizers[cat_inst].append(categorizer)
