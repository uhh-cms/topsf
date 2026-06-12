"""
Collection of helpers
"""

from __future__ import annotations

import law

from contextlib import contextmanager
from functools import wraps
import time

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


@contextmanager
def record_calls(inst, run_list):
    cls = type(inst)
    orig_getitem = cls.__getitem__

    def wrapped_getitem(self, key):
        prod = orig_getitem(self, key)
        name = getattr(key, "__name__", None) or str(key)

        @wraps(prod)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()

            result = prod(*args, **kwargs)

            duration = time.perf_counter() - start
            run_list.append(f"    {name:<30} {duration:7.3f}s")

            return result

        wrapped.__dict__.update(getattr(prod, "__dict__", {}))
        return wrapped

    cls.__getitem__ = wrapped_getitem

    try:
        yield
    finally:
        cls.__getitem__ = orig_getitem


def call_once_on_config(func=None, *, include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*.
    Can be used with or without parentheses.
    """
    if func is None:
        # If func is None, it means the decorator was called with arguments.
        def wrapper(f):
            return call_once_on_config(f, include_hash=include_hash)
        return wrapper
