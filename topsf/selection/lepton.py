# coding: utf-8

"""
Lepton-related selectors.
"""

from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from columnflow.selection import Selector, SelectionResult, selector

from topsf.selection.util import masked_sorted_indices
from topsf.production.lepton import choose_lepton

np = maybe_import("numpy")
ak = maybe_import("awkward")


class ChannelSelector(Selector):

    exposed = False

    def __init__(self, channel: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # return early if no config yet
        config_inst = getattr(self, "config_inst", None)
        if not config_inst:
            return

        # set the channel
        self.channel = self.config_inst.get_channel(channel)

        # set config dict
        self.cfg = config_inst.x.lepton_selection[self.channel.name]

        # set input columns
        column = self.cfg.column
        self.uses |= {
            f"{column}.pt",
            f"{column}.eta",
            f"{column}.{self.cfg.id}",
            f"{column}.{self.cfg.id_veto}",
        } | {
            f"HLT.{trigger}"
            for trigger in self.cfg.triggers
        }

        # optionally wire relative isolation
        rel_iso = self.cfg.get("rel_iso", None)
        if rel_iso is not None:
            self.uses |= {"f{column}.{rel_iso}"}


@selector(
    uses={
        "event",
    },
)
def channel_selection(
    self: ChannelSelector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Select leptons based on trigger, offline ID/Isolation.
    Veto events with more than one lepton.
    """

    # get channel and selection parameters from the config
    self.cfg = self.config_inst.x.lepton_selection[self.channel.name]

    # choose lepton column
    lepton = events[self.cfg.column]

    # baseline lepton selection
    lepton_mask = (
        (abs(lepton.eta) < self.cfg.max_abseta) &
        (lepton.pt > self.cfg.min_pt) &
        (lepton[self.cfg.id])
        # TODO: relative isolation
    )

    # optionally apply relative isolation
    rel_iso = self.cfg.get("rel_iso", None)
    if rel_iso is not None:
        max_rel_iso = self.cfg.max_rel_iso
        lepton_mask = lepton_mask | (lepton[rel_iso] < max_rel_iso)

    # lepton indices
    lepton_indices = masked_sorted_indices(lepton_mask, lepton.pt)

    # veto events if additional loose leptons present
    lepton_mask_veto = (
        (abs(lepton.eta) < self.cfg.max_abseta) &
        (lepton.pt > self.cfg.min_pt_addveto) &
        (lepton[self.cfg.id_addveto])
    )
    add_lepton_veto = (ak.sum(lepton_mask_veto, axis=-1) <= 1)

    # check missing triggers
    missing_triggers = {
        tn for tn in self.cfg.triggers
        if tn not in events.HLT.fields
    }
    if missing_triggers:
        missing_triggers = ", ".join(missing_triggers)
        raise ValueError(f"triggers not found in input sample: {missing_triggers}")

    # OR triggers
    trigger_mask = ak.zeros_like(events.event, dtype=bool)
    for trigger in self.cfg.triggers:
        trigger_mask = trigger_mask | events.HLT[trigger]

    # return selection result
    return SelectionResult(
        steps={
            "Lepton": (ak.num(lepton_indices) == 1),
            "LeptonTrigger": ak.fill_none(trigger_mask, False),
            "AddLeptonVeto": add_lepton_veto,
        },
        objects={
            self.cfg.column: {
                self.cfg.column: lepton_indices,
            },
        },
    )


# @channel_selection.init
# def channel_selection_init(self: ChannelSelector) -> None:
#     # add standard jec and jer for mc, and only jec nominal for dta
#     config_inst = getattr(self, "config_inst", None)
#     if not config_inst:
#         return
#
#     # set config dict
#     self.cfg = config_inst.x.lepton_selection[self.channel.name]
#
#     # set input columns
#     column = self.cfg.column
#     self.uses |= {
#         f"{column}.pt",
#         f"{column}.eta",
#         f"{column}.{self.cfg.id}",
#         f"{column}.{self.cfg.id_veto}",
#     } | {
#         f"HLT.{trigger}"
#         for trigger in self.cfg.triggers
#     }
#
#     # optionally wire relative isolation
#     rel_iso = self.cfg.get("rel_iso", None)
#     if rel_iso is not None:
#         self.uses |= {"f{column}.{rel_iso}"}


# instantiate selectors for channels
muon_selection = channel_selection.derive("muon_selection", cls_dict=dict(channel="mu"))
electron_selection = channel_selection.derive("electron_selection", cls_dict=dict(channel="e"))


def merge_selection_steps(step_dicts):
    """
    Merge two or more dictionaries of selection steps by concatenating the
    corresponding masks for the different selections steps along the innermost
    dimension (axis=-1).
    """

    step_names = {
        step_name
        for step_dict in step_dicts
        for step_name in step_dict
    }

    # check for step name incompatibilities
    if any(step_names != set(step_dict) for step_dict in step_dicts):
        raise ValueError(
            "Selection steps to merge must have identical "
            "selection step names!",
        )

    merged_steps = {
        step_name: ak.concatenate([
            ak.singletons(step_dict[step_name])
            for step_dict in step_dicts
        ], axis=-1)
        for step_name in step_names
    }

    return merged_steps


@selector(
    uses={
        "event",
        muon_selection,
        electron_selection,
        choose_lepton,
    },
    produces={
        "channel_id",
        choose_lepton,
    },
    exposed=True,
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Select muons/electrons and determine channel for event."""

    # get channels from the config
    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu")

    # selection results for each channel
    channel_results = {}

    # array of lists to keep track of matching channels
    channel_ids = ak.singletons(events.event)[:, :0]
    channel_indexes = ak.values_astype(
        ak.singletons(events.event)[:, :0],
        np.int8,
    )

    merged_objects = {}
    for ch_index, (channel, ch_selector, lepton_name, lepton_route) in enumerate([
        (ch_mu.id, muon_selection, "muon", "Muon"),
        (ch_e.id, electron_selection, "electron", "Electron"),
    ]):

        # selection results for channel
        channel_results[channel] = results = self[ch_selector](events, **kwargs)

        # add channel IDs based on selection result
        add_channel_id = ak.singletons(
            ak.where(
                results.steps["Lepton"],
                channel,
                0,
            ),
        )
        channel_ids = ak.concatenate([
            channel_ids,
            add_channel_id[add_channel_id != 0],
        ], axis=-1)

        # keep track of channel index
        add_channel_index = ak.singletons(
            ak.where(
                results.steps["Lepton"],
                ch_index,
                -1,
            ),
        )
        channel_indexes = ak.concatenate([
            channel_indexes,
            add_channel_index[add_channel_index != -1],
        ], axis=-1)

        # add the object indices to the selection
        merged_objects.update(results.objects)

    # concatenate selection results
    step_dicts = [r.steps for r in channel_results.values()]
    aux_dicts = [r.aux for r in channel_results.values()]
    merged_steps = merge_selection_steps(step_dicts)
    merged_aux = merge_selection_steps(aux_dicts)

    # decide channel and merge selection results
    merged_steps = {
        step: ak.fill_none(ak.firsts(
            selection[channel_indexes],
        ), False)
        for step, selection in merged_steps.items()
    }
    merged_aux = {
        var: ak.firsts(
            vals[channel_indexes],
        )
        for var, vals in merged_aux.items()
    }

    # invalidate event if multiple channel selections passed
    # (veto mixed e+mu events)
    channel_id = ak.where(
        ak.num(channel_ids, axis=-1) == 1,
        ak.firsts(channel_ids),
        0,
    )

    # set channel to 0 if undefined
    channel_id = ak.fill_none(channel_id, 0)

    # ensure integer type
    channel_id = ak.values_astype(channel_id, np.int8)

    # put channel in a column
    events = set_ak_column(events, "channel_id", channel_id)

    # multiplex Muon/Electron to a single Lepton collection
    # based on channel_id
    events = self[choose_lepton](events, **kwargs)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps=merged_steps,
        objects=merged_objects,
        aux=merged_aux,
    )
