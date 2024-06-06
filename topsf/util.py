"""
Collection of helpers
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import

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
