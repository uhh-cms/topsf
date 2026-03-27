"""
Collection of helpers
"""

from __future__ import annotations

import law

from columnflow.types import Any
from columnflow.util import maybe_import
from columnflow.columnar_util import ArrayFunction, deferred_column

np = maybe_import("numpy")

_logger = law.logger.get_logger(__name__)


def has_tag(tag, *container, operator: callable = any) -> bool:
    """
    Helper to check multiple container for a certain tag *tag*.
    Per default, booleans are combined with logical "or"

    :param tag: String of which tag to look for.
    :param container: Instances to check for tags.
    :param operator: Callable on how to combine tag existance values.
    :return: Boolean whether any (all) containter contains the requested tag.
    """
    values = [inst.has_tag(tag) for inst in container]
    return operator(values)


@deferred_column
def IF_MC(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.is_mc else None
