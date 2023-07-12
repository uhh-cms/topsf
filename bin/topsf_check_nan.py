# coding: utf8
"""
utility script for checking the presence of
`nan` values in files

supported formats:
    - 'parquet' (columnflow array output)

"""
import awkward as ak
import numpy as np
import glob
import os
import sys

from columnflow.columnar_util import get_ak_routes


def _load_parquet(fname):
    return ak.from_parquet(fname)



def load(fname):
    """
    Load file contents based on file extension.
    """
    basename, ext = os.path.splitext(fname)
    if ext == ".pickle":
        return _load_pickle(fname)
    elif ext == ".parquet":
        return _load_parquet(fname)
    elif ext == ".root":
        return _load_nano_root(fname)
    elif ext == ".json":
        return _load_json(fname)
    else:
        raise NotImplementedError(
            f"No loader implemented for extension {ext}",
        )

def check(ak_array):
    nonfinite_routes = set()
    for route in get_ak_routes(ak_array):
        if ak.any(~np.isfinite(ak.flatten(route.apply(ak_array), axis=None))):
            nonfinite_routes.add(route.column)

    if nonfinite_routes:
        nonfinite_routes = ", ".join(sorted(nonfinite_routes))
        raise ValueError(
            f"found one or more non-finite values in columns: {nonfinite_routes}",
        )


if __name__ == "__main__":
    paths = sys.argv[1:]
    exceptions = {}
    for path in paths:
        fnames = glob.glob(path)
        for fname in fnames:
            arr = load(fname)
            try:
                check(arr)
            except Exception as e:
                exceptions[fname] = e
                print(f"FAIL: {fname} [{e}]")
            else:
                print(f"OK:   {fname}")
