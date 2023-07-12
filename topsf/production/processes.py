# coding: utf-8

"""
Column production methods related detecting truth processes.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from topsf.production.gen_top import gen_top_decay_products, gen_top_decay_n_had
from topsf.production.probe_jet import probe_jet

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={gen_top_decay_products, gen_top_decay_n_had, probe_jet},
    produces={"process_id"},
)
def process_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Determines the process ID based on the dataset and, if applicable, subprocess
    information stored in the configuration.

    For ttbar & single top datasets, subprocesses are defined based on the decay
    mode of the top quark(s) and the number of quarks from the top decay that are
    merged to the probe jet, as follows:

    Process     Decay mode          # of merged quarks  Subprocess
    -------     ----------          ------------------  ----
    tt          semi-leptonic       3                   tt_3q
                                    2                   tt_2q
                                    1 or 0              tt_1o0q
                dileptonic or       any                 tt_bkg
                fully hadronic

    st          hadronic            3                   st_3q
                                    2                   st_2q
                                    1 or 0              st_1o0q
                leptonic            any                 st_bkg
    """
    # get process to which the dataset belongs
    if len(self.dataset_inst.processes) != 1:
        raise NotImplementedError(
            f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
            "assigned, which is not yet implemented",
        )

    process = self.dataset_inst.processes.get_first()
    process_id = np.ones(len(events), dtype=np.int32) * process.id

    if process.has_tag("has_subprocesses"):
        subprocess_ids = {
            p.x.subprocess_name: p.id
            for p in process.processes
            if p.has_tag("is_subprocess")
        }

        events = self[gen_top_decay_products](events, **kwargs)
        events = self[gen_top_decay_n_had](events, **kwargs)
        events = self[probe_jet](events, **kwargs)

        # tag events with exactly one hadronic top
        is_had_top = ak.fill_none(events.gen_top_decay_n_had == 1, False)

        # background: all events where number of hadronically
        #             decaying top quarks is not exactly 1
        #             or without probe jet
        process_id = ak.where(
            (~is_had_top) | ak.is_none(events.ProbeJet.n_merged),
            subprocess_ids["bkg"],
            process_id,
        )

        # nq: events where exactly n quarks are merged
        #     to the probe jet
        for n_merged in (3, 2):
            process_id = ak.where(
                is_had_top & ak.fill_none(events.ProbeJet.n_merged == n_merged, False),
                subprocess_ids[f"{n_merged}q"],
                process_id,
            )

        # 0o1q: events where 0 or 1 quarks are merged
        #       to the probe jet
        process_id = ak.where(
            is_had_top & ak.fill_none(
                ((events.ProbeJet.n_merged == 0) | (events.ProbeJet.n_merged == 1)),
                False,
            ),
            subprocess_ids["0o1q"],
            process_id,
        )

        # check that no events are left without a subprocess assigned
        assert ~ak.any(process_id == process.id), "subprocess definition does not cover all events"

    # store the column
    events = set_ak_column(events, "process_id", process_id, value_type=np.int32)

    return events
