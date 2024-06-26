# coding: utf8
"""
utility script for checking integrity of files

supported formats:
    - 'root' (NanoAOD ROOT files)
    - 'parquet' (columnflow array output)
    - 'pickle' (columnflow histogram output)
    - 'json'

"""
import awkward as ak
import coffea.nanoevents
import glob
import hist
import json
import os
import pickle
import sys
import uproot


def _load_json(fname):
    with open(fname, "r") as fobj:
        return json.load(fobj)


def _load_pickle(fname):
    with open(fname, "rb") as fobj:
        return pickle.load(fobj)


def _load_parquet(fname):
    return ak.from_parquet(fname)


def _load_nano_root(fname):
    source = uproot.open(fname)
    return coffea.nanoevents.NanoEventsFactory.from_root(
        source,
        runtime_cache=None,
        persistent_cache=None,
    ).events()


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


if __name__ == "__main__":
    paths = sys.argv[1:]
    exceptions = {}
    for path in paths:
        fnames = glob.glob(path)
        for fname in fnames:
            try:
                load(fname)
            except Exception as e:
                exceptions[fname] = e
                print(f"FAIL: {fname} [{e}]")
            else:
                print(f"OK:   {fname}")
