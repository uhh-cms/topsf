# coding: utf-8

"""
Column production methods related detecting truth processes.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from topsf.production.gen_top import gen_top_decay_products
from topsf.production.probe_jet import probe_jet

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={gen_top_decay_products, probe_jet},
    produces={"process_id"},
)
def process_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Determine the process ID based on the dataset and, if applicable, subprocess
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
                leptonic            (see below)

    For single-top production in association with a W boson, the top quark may
    decay leptonically, but the associated W boson decay may be hadronic. In this
    case, the number of merged quarks is determined not from the top quark decay
    products, but from the decay products of the associated hadronic W boson.
    The corresponding subprocess is determined as follows:

    Process     Top decay  Assoc. W decay    # of merged quarks  Subprocess
    -------     ---------  --------------    ------------------  ----
    st_tW       leptonic   hadronic          3                   st_bkg*
                                             2                   st_2q
                                             1 or 0              st_1o0q
    st_tW       leptonic   leptonic          any                 st_bkg
    st_other    leptonic   --                any                 st_bkg

    *) This case arises only if an additional associated b quark is found in close
       proximity to the probe jet. Since this b quark is not related to the actual
       top process, it is classified as background.

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

        # get number of decay products merged to hadronic top
        events = self[gen_top_decay_products](events, **kwargs)
        events = self[probe_jet](events, **kwargs)
        has_probejet = ~ak.is_none(events.ProbeJet.pt)
        n_merged = ak.fill_none(events.ProbeJet.n_merged, 0)
        is_hadronic = ak.fill_none(events.ProbeJet.is_hadronic_decay, False)
        is_assoc = ak.fill_none(events.ProbeJet.is_associated_decay, False)

        # treat events with associated hadronic W but three merged
        # decay products (i.e. w/ unrelated b quark) as background
        n_merged = ak.where(
            is_hadronic & is_assoc & (n_merged == 3),
            0,
            n_merged,
        )

        # background: all events where there is no probe jet, or where
        #             the decay merged to the probe jet is not hadronic,
        #             or where an associated b quark and both associated
        #             W decay products are merged to the probe jet
        is_background = (
            (~has_probejet) |
            (~is_hadronic) |
            (is_hadronic & is_assoc & (n_merged == 3))
        )
        process_id = ak.where(
            is_background,
            subprocess_ids["bkg"],
            process_id,
        )

        # nq: events where exactly n quarks are merged
        #     to the probe jet
        for n_merged_comp in (3, 2):
            process_id = ak.where(
                (~is_background) & (n_merged == n_merged_comp),
                subprocess_ids[f"{n_merged_comp}q"],
                process_id,
            )

        # 0o1q: events where 0 or 1 quarks are merged
        #       to the probe jet
        process_id = ak.where(
            ak.fill_none(
                (~is_background) & (
                    (n_merged == 0) |
                    (n_merged == 1)
                ),
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
