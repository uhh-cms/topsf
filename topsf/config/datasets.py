# coding: utf-8

"""
Configuration of datasets for the m(ttbar) analysis.
Reads data from a YAML file and provides it in a structured way to the analysis configs.
"""

from __future__ import annotations

import yaml

DATASETS_FILE = "/data/dust/user/matthiej/topsf/topsf/config/datasets.yaml"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


DATASETS = load_yaml(DATASETS_FILE)


def register_datasets(
    config,
    names,
    tags,
    limit_dataset_files=None,
):

    for name in names:

        ds = config.add_dataset(
            config.campaign.get_dataset(name)
        )

        if tags:
            ds.add_tag(tags)

        if limit_dataset_files:
            for info in ds.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)


def add_datasets_from_yaml(
    config,
    limit_dataset_files=None,
    dataset_types=None,
    log=False,
):
    """
    Add datasets defined in DATASETS (loaded from YAML) to config.

    Parameters:
      config: configuration object
      limit_dataset_files: optional int to limit files per dataset
      dataset_types: optional iterable of top-level dataset keys (e.g. ["data","tt","qcd"])
                     If provided, only those dataset types are added.
      log: if True, print a summary

    Returns:
      list of added dataset names
    """
    tag = config.x.cpn_tag
    if tag == "2024":
        tag = "2024full"

    # normalize requested types to a set for fast membership tests
    if dataset_types is None:
        requested = None
    else:
        requested = set(dataset_types)

    total = 0
    added = []

    for sample_type, info in DATASETS.items():
        if requested is not None and sample_type not in requested:
            continue

        try:
            names = info["eras"][tag]
        except KeyError:
            continue

        register_datasets(
            config,
            names,
            tags=set(info.get("tags", [])),
            limit_dataset_files=limit_dataset_files,
        )

        total += len(names)
        added.extend(names)

    if log:
        print(f"Added {total} datasets (types={', '.join(sorted(requested)) if requested else 'all'})")

    return added
